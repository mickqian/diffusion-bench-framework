#!/usr/bin/env python3
"""LightX2V's MiniMax-H3 cells launch, configure and request what the case means.

LightX2V serves H3 differently from every other model it runs here, and each
difference is a way to measure the wrong thing without an error:

* No startup task. `--model-variant fl2av|ref2av` picks the weights and each
  request names its task; ref2av on the base variant would run a different
  transformer than the one sglang loads.
* Steps count DiT evaluations: `infer_steps` N samples linspace(1, 0, N + 1),
  while sglang's num_inference_steps 50 is the 50-point grid -- 49 forwards.
  Copying the case's 50 into the launch config would add a forward and move
  every sigma.
* The task API forbids unknown fields, and the H3 runner then rejects any field
  outside its task's list, so the throughput path (bench_serving) must send what
  single_e2e sends -- it used to send infer_steps/height/width/fps.
* It encodes an MP4 only when a request names an output path, so a completed
  task is checked, off the clock, for the MP4 with its audio track.

The request fields are LightX2V 0c2edc12's (2026-10-05): MiniMaxH3Runner's
`supported_request_fields_by_task`, all of them VideoTaskRequest fields except
return_result_tensor, which a request never sets.
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import huggingface_hub  # noqa: E402
import requests  # noqa: E402

from diffusion_bench import bench_serving as bs  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:400]) if detail and not cond else ''}")
    if not cond:
        fail = 1


RUNNER_FIELDS = {
    "t2av": {"task", "prompt", "seed", "size", "num_frames", "save_result_path", "return_result_tensor"},
}
RUNNER_FIELDS["ref2av"] = RUNNER_FIELDS["t2av"] | {"image_path", "video_path", "audio_path"}
VARIANT = {"t2av": "fl2av", "ref2av": "ref2av"}
TRANSFORMER = {"fl2av": "transformer/*", "ref2av": "transformer_ref/*"}
EXACT_ATTENTION = {"flash_attn2", "flash_attn3", "torch_sdpa"}

os.environ[rc.DISABLE_TORCH_COMPILE_ENV] = "0"  # the policy's mode: competitors compiled
cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
h3 = [c for c in cfg["cases"] if c["model"] == "MiniMaxAI/MiniMax-H3" and "lightx2v" in c["frameworks"]]
check(
    "T2V+A on 2 and 4 GPUs and Ref2V+A carry a LightX2V cell",
    sorted(c["id"] for c in h3) == ["minimax_h3_ref2va_5s", "minimax_h3_t2va_5s", "minimax_h3_t2va_5s_4gpu"],
    [c["id"] for c in h3],
)
v2 = next(c for c in cfg["cases"] if c["id"] == "minimax_h3_fasth3_v2_t2va_5s")
reason = v2.get("report_framework_reasons", {}).get("lightx2v", "")
check(
    "FastH3 V2 is unsupported, with a dated reason naming the missing VSA",
    v2.get("report_framework_statuses", {}).get("lightx2v") == "unsupported" and "2026-10-08" in reason and "VSA" in reason,
    reason,
)

# --- launch command and launch config, per hardware class ------------------------
downloads = []


def fake_snapshot_download(model_id, cache_dir=None, allow_patterns=None):
    downloads.append(allow_patterns)
    return "/models/MiniMax-H3"


saved_download = huggingface_hub.snapshot_download
huggingface_hub.snapshot_download = fake_snapshot_download
try:
    for case in h3:
        raw = case["frameworks"]["lightx2v"]
        task = case["framework_request_extra"]["lightx2v"]["task"]
        sglang = case["framework_request_extra"]["sglang"]
        grid_points = sglang.get("num_inference_steps", case["num_inference_steps"])
        for hw in ("h200", "b200", "gb300"):
            where = f"{case['id']}/lightx2v on {hw}"
            with contextlib.redirect_stdout(io.StringIO()):
                fw_cfg = rc._resolve_framework_config("lightx2v", raw, None, {"hardware_profile_override": hw})
            server_case = rc._case_for_framework(case, fw_cfg)
            downloads.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                cmd = rc._build_lightx2v_cmd(server_case, fw_cfg, 30000)
            launch_path = Path(cmd[cmd.index("--config_json") + 1])
            launch = json.loads(launch_path.read_text())
            launch_path.unlink()
            gpus = case["num_gpus"]
            check(
                f"{where}: torchrun over the case's {gpus} GPUs, --model-variant {VARIANT[task]}, no startup --task",
                cmd[:4] == ["torchrun", f"--nproc_per_node={gpus}", "-m", "lightx2v.server"]
                and fw_cfg["num_gpus"] == gpus
                and cmd[cmd.index("--model_cls") + 1] == "minimax_h3"
                and cmd[cmd.index("--model-variant") + 1] == VARIANT[task]
                and "--task" not in cmd,
                " ".join(cmd),
            )
            check(
                f"{where}: {launch['infer_steps']} evaluations on sglang's {grid_points}-point grid, shifts 12/3",
                launch["infer_steps"] == grid_points - 1
                and launch["video_flow_shift"] == sglang["flow_shift"]
                and launch["audio_flow_shift"] == sglang["audio_flow_shift"],
                launch,
            )
            parallel = launch["parallel"]
            check(
                f"{where}: compiled, resident, exact attention, parallel sizes cover the GPUs",
                launch["use_compile"] is True
                and not any(v for k, v in launch.items() if k.endswith("cpu_offload") or k in ("lazy_load", "unload_modules"))
                and launch["attn_type"] in EXACT_ATTENTION
                and not (hw != "h200" and launch["attn_type"] == "flash_attn3")
                and math.prod(parallel.get(k, 1) for k in ("tensor_p_size", "seq_p_size", "cfg_p_size")) == gpus,
                launch,
            )
            check(
                f"{where}: fetches the {VARIANT[task]} transformer and the shared components only",
                downloads == [[TRANSFORMER[VARIANT[task]], "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "audio_vae/*"]],
                downloads,
            )
finally:
    huggingface_hub.snapshot_download = saved_download

# --- what a server receives, single_e2e and throughput ----------------------------
test_image = io.BytesIO()
Image.new("RGBA", (1024, 704), (90, 120, 30, 255)).save(test_image, "PNG")
rc._cached_ref_images[cfg["test_image_url"]] = test_image.getvalue()  # no network
MP4 = {
    "av": b"\x00\x00\x00\x10ftypisom" + b"hdlr" + bytes(8) + b"vide" + b"hdlr" + bytes(8) + b"soun",
    "video-only": b"\x00\x00\x00\x10ftypisom" + b"hdlr" + bytes(8) + b"vide",
}
posted = []
written = []
reply = {"mp4": MP4["av"], "report_path": True}


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.content = b"{}"

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def fake_post(url, **kwargs):
    body = kwargs["json"]
    posted.append((url, body))
    if body.get("save_result_path"):
        written.append(body["save_result_path"])
        Path(written[-1]).write_bytes(reply["mp4"])
    return FakeResponse({"task_id": "task"})


def fake_poll(url, deadline, timeout=30):
    path = written[-1] if reply["report_path"] else None
    return FakeResponse({"status": "completed", "save_result_path": path})


class FakeSession:
    """aiohttp's session as bench_serving's LightX2V client uses it."""

    def post(self, url, json=None):
        fake_post(url, json=json)
        return _Reply({"task_id": "task"})

    def get(self, url):
        return _Reply({"status": "completed"})


class _Reply:
    status = 200

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self.body

    async def text(self):
        return json.dumps(self.body)


real_bench_request = bs.async_request_lightx2v


async def bench_request(input, session, pbar=None):
    return await real_bench_request(input, FakeSession(), pbar)


def fake_subprocess_run(cmd, **kwargs):
    sys.argv = ["bench_serving", *cmd[3:]]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bs.main()
    return subprocess.CompletedProcess(cmd, 0, stdout="")


saved = (rc._post_drained, rc._poll_get, time.sleep, subprocess.run, requests.get, bs.async_request_lightx2v, sys.argv)
rc._post_drained, rc._poll_get, time.sleep, subprocess.run = fake_post, fake_poll, lambda s: None, fake_subprocess_run
requests.get = lambda url, timeout=None, **kwargs: FakeResponse({})  # bench_serving's health and model-info probes
bs.async_request_lightx2v = bench_request
try:
    with tempfile.TemporaryDirectory() as log_dir:
        for case in h3:
            task = case["framework_request_extra"]["lightx2v"]["task"]
            where = f"{case['id']}/lightx2v"
            with contextlib.redirect_stdout(io.StringIO()):
                fw_cfg = rc._resolve_framework_config("lightx2v", case["frameworks"]["lightx2v"], None, {"hardware_profile_override": "h200"})
            server_case = rc._case_for_framework(case, fw_cfg)
            request = rc._request_case(server_case, 3)
            posted.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                rc.send_request_lightx2v("http://fake", request, cfg)
            url, body = posted[0]
            expected = {"task", "prompt", "seed", "size", "num_frames", "save_result_path"} | ({"image_path"} if task == "ref2av" else set())
            check(
                f"{where}: single_e2e posts exactly the fields its {task} task takes",
                url == "http://fake/v1/tasks/video/" and set(body) == expected and set(body) <= RUNNER_FIELDS[task],
                f"{url} {sorted(body)}",
            )
            check(
                f"{where}: the case's request -- size [h, w], frames, seed, task, its own prompt",
                body["size"] == [case["height"], case["width"]]
                and body["num_frames"] == case["num_frames"]
                and body["seed"] == case["seed"]
                and body["task"] == task
                and body["prompt"] == request["prompt"]
                and body["save_result_path"].endswith(".mp4"),
                body,
            )
            if task == "ref2av":
                with Image.open(body["image_path"]) as reference:
                    check(
                        f"{where}: the reference is the harness's prepared file (2048 px short edge)",
                        body["image_path"] == rc._get_ref_image_path(cfg, request) and min(reference.size) == case["reference_short_edge"],
                        f"{body['image_path']} {reference.size}",
                    )

            if case.get("throughput"):
                posted.clear()
                bench = rc._merge_nested(rc._benchmark_config(cfg, server_case), fw_cfg.get("benchmark", {}))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = rc.run_throughput("http://fake", server_case, "lightx2v", cfg, bench, Path(log_dir), iter(range(10, 40)))
                bodies = [b for _, b in posted]
                check(
                    f"{where}: throughput's warmup and requests post single_e2e's fields to the video endpoint",
                    not result.get("error")
                    and len(bodies) == rc.BENCH_SERVING_WARMUPS + int(bench["throughput"]["num_requests"])
                    and all(u == "http://fake/v1/tasks/video/" for u, _ in posted)
                    and all(set(b) == expected for b in bodies),
                    f"{result.get('error')} {[(u, sorted(b)) for u, b in posted][:2]}",
                )
                same = {k: v for k, v in body.items() if k not in ("prompt", "save_result_path", "image_path")}
                check(
                    f"{where}: ... with the same values, each request its own prompt",
                    all({k: b[k] for k in same} == same for b in bodies) and len({b["prompt"] for b in bodies}) == len(bodies),
                    bodies[:1],
                )

        # The output check, against a case that asks for audio.
        request = rc._request_case(rc._case_for_framework(h3[0], h3[0]["frameworks"]["lightx2v"]), 5)
        for label, mp4, report_path, ok in (
            ("an MP4 with video and audio passes", MP4["av"], True, True),
            ("a video-only MP4 for an audio-video task fails", MP4["video-only"], True, False),
            ("a completed task with no saved output fails", MP4["av"], False, False),
        ):
            reply.update(mp4=mp4, report_path=report_path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc.send_request_lightx2v("http://fake", request, cfg)
                raised = None
            except RuntimeError as exc:
                raised = exc
            check(label, (raised is None) == ok, raised)
finally:
    rc._post_drained, rc._poll_get, time.sleep, subprocess.run, requests.get, bs.async_request_lightx2v, sys.argv = saved
    for path in set(written) | set(rc._cached_ref_image_paths.values()):
        os.remove(path)

# The track reader against a real muxer (LightX2V's is PyAV over the same libavformat).
if shutil.which("ffmpeg"):
    with tempfile.TemporaryDirectory() as td:
        source = ["-f", "lavfi", "-i", "testsrc=size=128x64:rate=24"]
        audio = ["-f", "lavfi", "-i", "sine=sample_rate=32000", "-ac", "2", "-c:a", "aac"]
        for name, extra, tracks in (("av", audio, {"vide", "soun"}), ("video", [], {"vide"})):
            out = os.path.join(td, f"{name}.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", *source, *extra, "-t", "0.5", "-c:v", "libx264", out], check=True)
            check(f"ffmpeg's {name} MP4 reads as {sorted(tracks)}", rc._mp4_tracks(out) == tracks, rc._mp4_tracks(out))
else:
    print("  skip the track reader against a real muxer: no ffmpeg on PATH")
sys.exit(fail)
