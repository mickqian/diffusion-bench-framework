#!/usr/bin/env python3
"""No request a server sees may repeat an earlier request's conditioning inputs.

sglang-diffusion keeps an exact, content-hashed cross-request conditioning cache
(text-encoder outputs, VAE and image encodings). The harness sent every request of
a case with the same prompt and reference image, so from the second request on
sglang skipped its encoders -- text encoding is ~10% of a Z-Image request, the
image encode ~0.45s of a Qwen-Image-Edit one -- while the competitors, which have
no such cache, ran theirs. ComfyUI's node cache was already defeated by a nonce;
sglang's was not.

Every request now carries a prompt tag and a marked reference image of its own,
numbered across the server's whole lifetime: warmups, adaptive warmups, measured
repeats, the diagnostic request, then bench_serving's warmups and requests. The
last part drives every framework's real request builders against a fake
transport and checks what each server would have received.

The tag is two digits because, across tag values, digits keep the token count
and letters would not. On the Qwen2.5-VL, Qwen3, Qwen3-VL, Qwen3.5, Gemma-3,
Mistral-3.1, umT5 and CLIP tokenizers "(take 07)" costs the same for every value,
while a letter tag costs one token less than a digit tag in all of them but CLIP
(the Qwen, Gemma and Mistral pre-tokenizers split a digit off alone but join a
letter to its leading space), so a 0-9A-Z alphabet would move the count. T5 varies
even for one digit, but FLUX pads T5 to 512, so no shape moves.
"""
import base64
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from diffusion_bench import bench_serving as bs  # noqa: E402
from diffusion_bench import comfyui_client as cc  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402
from diffusion_bench.datasets import RequestFuncOutput  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def synthetic(mode, size):
    ramp = Image.linear_gradient("L")
    rgb = Image.merge(
        "RGB",
        (ramp.resize(size), ramp.transpose(Image.Transpose.ROTATE_90).resize(size), Image.new("L", size, 90)),
    )
    return rgb.convert(mode)


def encoded(image, fmt, **info):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **info)
    return buffer.getvalue()


# --- the tag -----------------------------------------------------------------
case = {"id": "c", "prompt": "Make the cat wear a red hat", "seed": 42, "num_inference_steps": 40}
requests_of_one_server = [rc._request_case(case, i) for i in range(rc.MAX_REQUEST_INPUTS)]
prompts = [r["prompt"] for r in requests_of_one_server]
check("every request gets its own prompt", len(set(prompts)) == len(prompts), f"{len(set(prompts))}")
check(
    "tags are fixed-width digits, so the token count cannot move",
    len({re.sub(r"\d", "#", p) for p in prompts}) == 1
    and all(re.fullmatch(r".* \(take \d\d\)", p) for p in prompts),
    f"{prompts[0]!r} .. {prompts[-1]!r}",
)
check(
    "sampling params and seed are the case's",
    all(r["seed"] == 42 and r["num_inference_steps"] == 40 for r in requests_of_one_server),
)
check("the case itself is left alone", case["prompt"] == "Make the cat wear a red hat" and "input_variant" not in case)
try:
    rc._request_case(case, rc.MAX_REQUEST_INPUTS)
    check("running out of tags is an error, not a repeat", False)
except ValueError:
    check("running out of tags is an error, not a repeat", True)

# --- the tag covers what a server is really sent --------------------------------
cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())


def most_requests(bench, is_video, throughput):
    warm = bench["warmup"]
    fixed = int(warm["num_requests"])
    if is_video:
        fixed = max(fixed, int(warm["video_num_requests"]))
    repeats = bench["single"]["video_repeats" if is_video else "image_repeats"]
    # + 1: sglang's diagnostic request
    total = fixed + rc.WARMUP_EXTRA_MAX + int(bench["single"].get("measured_repeats", repeats)) + 1
    if throughput:
        total += rc.BENCH_SERVING_WARMUPS + int(bench["throughput"]["num_requests"])
    return total


busiest = (0, "")
for c in cfg["cases"]:
    for fw, raw in c["frameworks"].items():
        blocks = [raw] + list((raw.get("command_profiles") or {}).values())
        for block in blocks:
            bench = rc._merge_nested(rc._benchmark_config(cfg, c), block.get("benchmark", {}))
            busiest = max(busiest, (most_requests(bench, bool(c.get("num_frames")), c.get("throughput")), f"{c['id']}/{fw}"))
check(
    f"the busiest configured server fits the tags ({busiest[0]} requests, {busiest[1]})",
    busiest[0] <= rc.MAX_REQUEST_INPUTS,
)
override = rc._merge_nested(rc._benchmark_config(cfg, cfg["cases"][0]), {"throughput": {"num_requests": 32}})
check(
    "so does the largest scripted throughput override (32 requests)",
    most_requests(override, False, True) <= rc.MAX_REQUEST_INPUTS,
    f"{most_requests(override, False, True)}",
)

# --- the reference image -----------------------------------------------------
exif = Image.Exif()
exif[0x0112] = 1
sources = {
    "RGBA png (the shared test image)": (synthetic("RGBA", (1024, 704)), "PNG", {"exif": exif.tobytes(), "icc_profile": b"icc"}),
    "RGB jpeg (Cosmos3's I2V input)": (synthetic("RGB", (758, 438)), "JPEG", {}),
    "L png": (synthetic("L", (320, 240)), "PNG", {}),
}
for label, (image, fmt, info) in sources.items():
    data = encoded(image, fmt, **info)
    with Image.open(io.BytesIO(data)) as original:
        original.load()
        variants = [Image.open(io.BytesIO(rc._ref_image_variant(data, v))) for v in (0, 7, 42, 99)]
        check(
            f"{label}: size and mode unchanged",
            all(v.size == original.size and v.mode == original.mode for v in variants),
            f"{original.mode} {original.size}",
        )
        check(
            f"{label}: lossless -- the decoded variant is the original plus the patch",
            all(v.tobytes() == rc._mark_reference(original, n).tobytes() for v, n in zip(variants, (0, 7, 42, 99))),
        )
        cx, cy, side = original.width // 2, original.height // 2, rc.REF_PATCH_PX
        patch = (cx - side, cy - side // 2, cx + side, cy + side // 2)
        changed = ImageChops.difference(original.convert("RGB"), variants[1].convert("RGB")).getbbox()
        check(
            f"{label}: only the centre patch changes",
            changed is not None
            and patch[0] <= changed[0]
            and patch[1] <= changed[1]
            and changed[2] <= patch[2]
            and changed[3] <= patch[3],
            f"changed {changed}, patch {patch}",
        )
        kept = {k for k in ("exif", "icc_profile") if k in original.info}
        check(
            f"{label}: EXIF and ICC profile survive ({sorted(kept) or 'none present'})",
            all(v.info.get(k) == original.info[k] for v in variants for k in kept),
        )


def cosmos3_resize_crop(image, width=1280, height=720):
    # sglang and vLLM-Omni Cosmos3 I2V: aspect-preserving resize, then center crop
    scale = max(width / image.width, height / image.height)
    size = (math.ceil(scale * image.width), math.ceil(scale * image.height))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return image.crop((left, top, left + width, top + height))


def h3_reference(image):
    # MiniMax-H3 Ref2VA: 2048px short edge, upscaling, each side to a multiple of 32
    scale = 2048 / min(image.size)
    return image.resize(tuple(max(32, round(s * scale / 32) * 32) for s in image.size), Image.Resampling.LANCZOS)


cat = synthetic("RGBA", (1024, 704))
cosmos = synthetic("RGB", (3034, 1754))
transforms = {
    "Cosmos3 I2V 3034x1754 -> resize + center crop 1280x720": (cosmos, cosmos3_resize_crop),
    "MiniMax-H3 reference -> 2048 short edge": (cat, h3_reference),
    "Qwen-Image-Edit condition -> 448x320": (cat, lambda im: im.resize((448, 320), Image.Resampling.LANCZOS)),
    "#41689's check -> 256x176 bicubic": (cat, lambda im: im.resize((256, 176), Image.Resampling.BICUBIC)),
}
for label, (image, transform) in transforms.items():
    seen = {hashlib.sha256(transform(rc._mark_reference(image, v)).tobytes()).digest() for v in range(rc.MAX_REQUEST_INPUTS)}
    check(f"{label}: every variant still distinct", len(seen) == rc.MAX_REQUEST_INPUTS, f"{len(seen)}")

# #41689 marks a corner; Cosmos3's crop removes it, which is why the patch is central.
corners = []
for level in (8, 32):
    marked = cosmos.copy()
    marked.paste((level, level, level), (0, 0, 8, 8))
    corners.append(cosmos3_resize_crop(marked).tobytes())
check("a corner patch would not survive Cosmos3's crop (why the centre)", corners[0] == corners[1])

# --- every framework's builders, every phase of a server -------------------------
# The shared test image and Cosmos3's own, faked so no request touches the network.
urls = {cfg["test_image_url"]: encoded(synthetic("RGBA", (1024, 704)), "PNG")}
for c in cfg["cases"]:
    if c.get("reference_image_url"):
        urls[c["reference_image_url"]] = encoded(synthetic("RGB", (758, 438)), "JPEG")
rc._cached_ref_images.update(urls)
source_shape = {url: Image.open(io.BytesIO(data)) for url, data in urls.items()}

received: list[dict] = []
current = {"workspace": None}  # the ComfyUI input directory of the cell being driven
# LightX2V writes its MP4 where the request says, and the harness checks its tracks.
FAKE_MP4 = b"\x00\x00\x00\x10ftypisom" + b"".join(b"hdlr" + bytes(8) + kind for kind in (b"vide", b"soun"))
mp4_written: list[str] = []


def data_uri_bytes(node):
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        return next((b for b in map(data_uri_bytes, node) if b), None)
    if isinstance(node, str) and node.startswith("data:image/"):
        return base64.b64decode(node.split(",", 1)[1])
    return None


def record(prompt, image, seed):
    received.append({"prompt": prompt, "image": image, "seed": None if seed is None else str(seed)})


class FakeResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload=None, content=b"media"):
        self._payload = payload if payload is not None else {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def box(kind, payload=b""):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


# FastVideo's sync endpoint answers with the MP4, which its sender checks for a video and an audio track.
MP4_WITH_AUDIO = box(b"ftyp", b"isom") + box(
    b"moov", b"".join(box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + h + bytes(12)))) for h in (b"vide", b"soun"))
)


def fake_post(url, **kwargs):
    body = kwargs.get("json") if kwargs.get("json") is not None else kwargs.get("data") or {}
    image = None
    if kwargs.get("files"):
        _, (_, handle, _) = next(iter(kwargs["files"].items()))
        image = handle.getvalue()
    elif body.get("image_path"):
        image = Path(body["image_path"]).read_bytes()
    else:
        image = data_uri_bytes(body)
    prompt = body.get("prompt")
    if prompt is None and "messages" in body:
        prompt = next(p["text"] for p in body["messages"][0]["content"] if p.get("type") == "text")
    record(prompt, image, body.get("seed", (body.get("extra_body") or {}).get("seed")))
    if body.get("save_result_path"):
        mp4_written.append(body["save_result_path"])
        Path(mp4_written[-1]).write_bytes(FAKE_MP4)
    if url.endswith("/v1/videos") or "/v1/tasks/video" in url:
        return FakeResponse({"id": "job", "task_id": "task"})
    if url.endswith("/v1/chat/completions"):
        return FakeResponse({"choices": [{}]})
    if url.endswith("/v1/videos/sync"):
        return FakeResponse(content=MP4_WITH_AUDIO)
    return FakeResponse({"data": [{"b64_json": "x"}]})


def fake_run_prompt(base_url, graph, *, timeout_s, fetch_outputs):
    loads = [n["inputs"]["image"] for n in graph.values() if n["class_type"] == "LoadImage"]
    image = (current["workspace"] / loads[0]).read_bytes() if loads else None
    texts = {v for n in graph.values() for v in n["inputs"].values() if isinstance(v, str) and "(take " in v}
    seeds = {v for n in graph.values() for k, v in n["inputs"].items() if k in ("seed", "noise_seed")}
    record(next(iter(texts)) if len(texts) == 1 else texts, image, next(iter(seeds)) if len(seeds) == 1 else seeds)
    return 0.01, {"cached_nodes": []}


async def fake_bench_request(input, session, pbar=None):
    image = Path(input.image_paths[0]).read_bytes() if input.image_paths else data_uri_bytes(input.extra_body)
    record(input.prompt, image, input.extra_body.get("seed"))
    return RequestFuncOutput(success=True, latency=0.01, output_count=1)


def fake_subprocess_run(cmd, **kwargs):
    # bench_serving in-process, so its real dataset and warmup selection run
    sys.argv = ["bench_serving", *cmd[3:]]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bs.main()
    return subprocess.CompletedProcess(cmd, 0, stdout="")


saved = (rc._post_drained, rc._poll_get, rc._read_perf_dump, cc.run_prompt, time.sleep, requests.get, subprocess.run, sys.argv)
rc._post_drained = fake_post
rc._poll_get = lambda url, deadline, timeout=30: FakeResponse(
    {"status": "completed", "task_status": "COMPLETED", "save_result_path": mp4_written[-1] if mp4_written else None}
)
rc._read_perf_dump = lambda path, timeout=10.0: 0.5
cc.run_prompt = fake_run_prompt
time.sleep = lambda seconds: None
requests.get = lambda url, timeout=None, **kwargs: FakeResponse({})
subprocess.run = fake_subprocess_run
for name in ("async_request_image_sglang", "async_request_video_sglang", "async_request_vllm_omni", "async_request_lightx2v"):
    setattr(bs, name, fake_bench_request)
bs.make_fastvideo_request_func = lambda expect_audio: fake_bench_request

covered: dict[str, set] = {}
with tempfile.TemporaryDirectory() as tmp:
    os.environ[rc.COMFYUI_WORKSPACE_ENV] = tmp
    log_dir = Path(tmp) / "logs"
    log_dir.mkdir()
    try:
        for c in cfg["cases"]:
            for fw, raw in c["frameworks"].items():  # the built config lists only runnable cells
                with contextlib.redirect_stdout(io.StringIO()):  # its profile-ambiguity warnings
                    fw_cfg = rc._resolve_framework_config(fw, raw, None, {"hardware_profile_override": "h200"})
                server_case = rc._case_for_framework(c, fw_cfg)
                current["workspace"] = rc._comfyui_workspace(c) / "input"
                current["workspace"].mkdir(parents=True, exist_ok=True)
                bench = rc._merge_nested(rc._benchmark_config(cfg, server_case), fw_cfg.get("benchmark", {}))
                received.clear()
                inputs = itertools.count()
                phases = []
                with contextlib.redirect_stdout(io.StringIO()):
                    rc._run_warmups("http://fake", server_case, fw, cfg, bench, inputs)
                    phases.append(("warmup", len(received)))
                    result = rc.run_single_request("http://fake", server_case, fw, log_dir, cfg, bench, inputs=inputs)
                    phases.append(("single_e2e", len(received)))
                    throughput = rc.run_throughput("http://fake", server_case, fw, cfg, bench, log_dir, inputs)
                    phases.append(("throughput", len(received)))
                where = f"{c['id']}/{fw}"
                errors = [r for r in (result, throughput) if r.get("error")]
                prompts = [r["prompt"] for r in received]
                images = [r["image"] for r in received]
                starts = [0] + [n for _, n in phases[:-1]]
                sent = {name: end - start for (name, end), start in zip(phases, starts)}
                for name, count in sent.items():
                    if count:
                        covered.setdefault(fw, set()).add(name)
                conditioned = c.get("reference_image") or "__TEST_IMAGE_URL__" in json.dumps(c.get("framework_request_extra") or {})
                ok = (
                    not errors
                    and all(isinstance(p, str) for p in prompts)
                    and len(set(prompts)) == len(prompts)
                    and len({r["seed"] for r in received}) == 1
                    and (not conditioned or (None not in images and len(set(images)) == len(images)))
                )
                shapes = ""
                if conditioned and ok and fw != "comfyui":
                    source = source_shape[rc._reference_image_url(cfg, c)]
                    if c.get("reference_short_edge"):
                        # every framework gets the reference at the case's official size
                        buffer = io.BytesIO()
                        source.save(buffer, format="PNG")
                        source = Image.open(io.BytesIO(rc._prepared_reference(buffer.getvalue(), c)))
                    got = {(im.size, im.mode) for im in (Image.open(io.BytesIO(b)) for b in images)}
                    ok = got == {(source.size, source.mode)}
                    shapes = f", images {sorted(got)}"
                check(
                    f"{where}: {len(received)} requests, every prompt{' and image' if conditioned else ''} its own, one seed",
                    ok,
                    f"{sent}{shapes}" + (f", errors {[r['error'] for r in errors]}" if errors else ""),
                )
    finally:
        rc._post_drained, rc._poll_get, rc._read_perf_dump, cc.run_prompt, time.sleep, requests.get, subprocess.run, sys.argv = saved
        for path in rc._cached_ref_image_paths.values():  # the harness's variant files, in the system temp dir
            os.remove(path)
        for path in set(mp4_written):
            os.remove(path)

check("drove every framework in scope", set(covered) == set(rc.FRAMEWORK_ORDER), f"{sorted(covered)}")
check(
    "through warmup, single_e2e and throughput each",
    all(phases == {"warmup", "single_e2e", "throughput"} for phases in covered.values()),
    f"{ {fw: sorted(p) for fw, p in covered.items()} }",
)

# --- recorded, so the data before and after can be told apart -------------------
metadata = rc._resolve_framework_config("sglang", cfg["cases"][0]["frameworks"]["sglang"], None, {})["_benchmark_metadata"]
check("every result's framework_metadata names the policy", metadata.get("request_inputs") == rc.REQUEST_INPUTS == "distinct-per-request")
src = (ROOT / "src" / "diffusion_bench" / "run_comparison.py").read_text()
check("and so does the run metadata", '"request_inputs": REQUEST_INPUTS' in src[src.index("artifact_meta = {"):])
spec = importlib.util.spec_from_file_location("publish_bench_run", ROOT / "scripts" / "publish_bench_run.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
old_row, new_row = {"framework_metadata": {"profile": "p"}}, {"framework_metadata": metadata}
published = {
    name: pub._policy_block({"results": rows, "throughput_results": []}, type("Args", (), {"note": []}))["request_inputs"]
    for name, rows in (("before", [old_row]), ("after", [new_row]), ("merged across", [old_row, new_row]))
}
check(
    "the published policy block says which, derived from the rows",
    published == {"before": "repeated", "after": "distinct-per-request", "merged across": "distinct-per-request + repeated"},
    f"{published}",
)

sys.exit(fail)
