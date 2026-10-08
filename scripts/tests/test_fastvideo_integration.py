#!/usr/bin/env python3
"""FastVideo's launch command, request and output check, against what FastVideo accepts.

FastVideo (hao-ai-lab/FastVideo) joined on 2026-10-08 for the MiniMax-H3 cases.
`fastvideo serve` takes a stub config plus `--dotted.key value` overrides, and its
video request model is `extra="forbid"`, so a misspelled override or an extra
field fails at launch or as a 400 -- on a GPU, after the model download. This
drives the harness's builders offline and checks what FastVideo would receive:

* the command: the harness's overrides, the profile's, no key twice, every
  offload switch false (FastVideo defaults them all to true), the route each
  hardware class selects, and the harness's compile-off mode;
* the request: only fields FastVideo's VideoGenerationRequest declares, the H3
  shape FastVideo admits, and the step count in FastVideo's convention -- sigma
  grid points, as on sglang: FastH3 V2 sends 9 (eight forwards; vLLM-Omni says 8)
  and base H3 sends the case's 50 (49 forwards, as on sglang);
* inputs: every request its own prompt, and for Ref2VA its own prepared reference;
* the output: an MP4 without the audio track the case asks for is an error;
* bench_serving's request body is the single_e2e one.
"""
import asyncio
import base64
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench import bench_serving as bs  # noqa: E402
from diffusion_bench import fastvideo_client as fv  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402
from diffusion_bench.datasets import RequestFuncInput  # noqa: E402
from diffusion_bench.page_export import framework_versions  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:400]) if detail and not cond else ''}")
    if not cond:
        fail = 1


def box(kind, payload=b""):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def mp4(*handlers):
    tracks = b"".join(box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + h + bytes(12)))) for h in handlers)
    return box(b"ftyp", b"isom") + box(b"moov", tracks) + box(b"mdat", b"\x00" * 16)


WITH_AUDIO = mp4(b"vide", b"soun")
SILENT = mp4(b"vide")

# fastvideo/entrypoints/openai/protocol.py VideoGenerationRequest at 2b164405 (extra="forbid").
REQUEST_FIELDS = {
    "prompt", "model", "seconds", "size", "image_reference", "video_reference", "audio_reference",
    "input_reference", "reference_url", "video_path", "video_url", "video_params", "user", "task", "width",
    "height", "fps", "num_frames", "aspect_ratio", "short_edge", "num_outputs_per_prompt", "n",
    "start_time_seconds", "quality", "negative_prompt", "num_inference_steps", "guidance_scale",
    "guidance_scale_2", "boundary_ratio", "flow_shift", "true_cfg_scale", "seed", "generate_sound",
    "sound_duration", "enable_teacache", "max_sequence_length", "enable_frame_interpolation",
    "frame_interpolation_exp", "frame_interpolation_scale", "frame_interpolation_model_path", "lora",
    "extra_params",
}
# Switches FastVideo documents as changing the output; the profiles keep them off.
OUTPUT_CHANGING_ENV = {
    "FASTVIDEO_MINIMAX_H3_FUSIONS": {"", "0", "none"},
    "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": {"0"},
    "FASTVIDEO_NVFP4_FA4": {"0"},
    "FASTVIDEO_H3_VSA_FP4": {"0"},
    "FASTVIDEO_H3_FP8_ATTENTION": {"0"},
}

cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
cases = {c["id"]: c for c in cfg["cases"] if "fastvideo" in c["frameworks"]}
check(
    "FastVideo serves the four MiniMax-H3 cases",
    set(cases) == {"minimax_h3_t2va_5s", "minimax_h3_t2va_5s_4gpu", "minimax_h3_ref2va_5s",
                   "minimax_h3_fasth3_v2_t2va_5s"},
    sorted(cases),
)


def resolved(case, hardware):
    with contextlib.redirect_stdout(io.StringIO()):
        fw_cfg = rc._resolve_framework_config(
            "fastvideo", case["frameworks"]["fastvideo"], None, {"hardware_profile_override": hardware}
        )
    return rc._case_for_framework(case, fw_cfg), fw_cfg


def dotted_keys(args):
    return [token[2:].partition("=")[0].replace("-", "_") for token in args if token.startswith("--")]


# --- the launch command ---------------------------------------------------------
os.environ[rc.DISABLE_TORCH_COMPILE_ENV] = "0"
for cid, case in cases.items():
    v2 = cid == "minimax_h3_fasth3_v2_t2va_5s"
    ref2va = cid == "minimax_h3_ref2va_5s"
    for hardware, blackwell in (("h200", False), ("b200", True), ("gb300", True)):
        case_fw, fw_cfg = resolved(case, hardware)
        where = f"{cid} on {hardware} ({fw_cfg['_benchmark_metadata']['profile']})"
        cmd = rc.build_server_cmd("fastvideo", case_fw, fw_cfg, 30123)
        check(f"{where}: `fastvideo serve --config <stub>`", cmd[:3] == ["fastvideo", "serve", "--config"], cmd[:4])
        check(f"{where}: the stub is only a generator mapping",
              json.loads(Path(cmd[3]).read_text()) == fv.SERVE_STUB)
        keys = dotted_keys(cmd[4:])
        check(f"{where}: no override twice", len(keys) == len(set(keys)),
              sorted(k for k in set(keys) if keys.count(k) > 1))
        o = {k: v.lower() for k, v in fv.serve_overrides(cmd[4:]).items()}
        num_gpus = case_fw["num_gpus"]
        check(f"{where}: model, GPUs, host, port from the harness",
              (o["generator.model_path"], o["generator.engine.num_gpus"], o["server.host"], o["server.port"])
              == (case["model"].lower(), str(num_gpus), rc.DEFAULT_HOST, "30123"), o)
        check(f"{where}: Ulysses over every GPU, no TP",
              (o["generator.engine.parallelism.sp_size"], o["generator.engine.parallelism.tp_size"])
              == (str(num_gpus), "1"))
        check(f"{where}: resident (every offload switch false, nothing reloaded per request)",
              all(o.get(f"generator.engine.offload.{p}") == "false" for p in fv.OFFLOAD_PARTS)
              and all(o.get(k) == "false" for k in fv.DEFERRED_LOAD_KEYS))
        check(f"{where}: DiT replicated on Blackwell, FSDP-sharded on Hopper",
              o["generator.engine.use_fsdp_inference"] == ("false" if blackwell else "true"))
        check(f"{where}: VAE uncompiled, so #1930's tile split runs",
              o["generator.engine.compile.vae_enabled"] == "false"
              and o["generator.pipeline.experimental.vae_parallel_decode"] == "true")
        env = fw_cfg["extra_env"]
        check(f"{where}: #1930/#1807's bit-identical switches on",
              (env["FASTVIDEO_MINIMAX_H3_EXACT_KERNELS"], env["FASTVIDEO_H3_VAE_TILE_PARALLEL"],
               env["FASTVIDEO_ULYSSES_A2A"]) == ("all", "1", "auto"), env)
        check(f"{where}: output-changing switches off",
              all(env.get(k, "0").lower() in allowed for k, allowed in OUTPUT_CHANGING_ENV.items())
              and o.get("generator.pipeline.experimental.video_decode_backend", "h3-vae") == "h3-vae"
              and not any(k.startswith("generator.engine.quantization") for k in o), env)
        check(f"{where}: FA4 on Blackwell only", env["FASTVIDEO_FA4"] == ("1" if blackwell else "0"))
        if v2:
            check(f"{where}: the checkpoint's trained VSA at the pinned revision",
                  (o["generator.pipeline.experimental.attention_backend"],
                   o["generator.pipeline.experimental.VSA_sparsity"],
                   o["generator.pipeline.experimental.VSA_tile_size"], o["generator.revision"])
                  == ("video_sparse_attn_h3", "0.8", "64", case["model_revision"]), o)
            route = {"FASTVIDEO_VSA_SM100A": "1", "FASTVIDEO_VSA_TK": "0"} if blackwell else {
                "FASTVIDEO_VSA_SM100A": "0", "FASTVIDEO_VSA_TK": "1"}
            check(f"{where}: sm_100a VSA on Blackwell, ThunderKittens on Hopper",
                  {k: env[k] for k in route} == route and env["FASTVIDEO_H3_VSA_HEADS_FIRST_TILE"] == "1")
            # VSA-H3 compiles only on sm_100a with 64-token tiles; elsewhere it degrades to eager.
            check(f"{where}: regional compile only where VSA-H3 has a compile-safe route",
                  o["generator.pipeline.experimental.inference_torch_compile"] == ("true" if blackwell else "false"))
        else:
            check(f"{where}: dense FlashAttention, regionally compiled",
                  (o["generator.pipeline.experimental.attention_backend"],
                   o["generator.pipeline.experimental.inference_torch_compile"]) == ("flash_attn", "true"))
        check(f"{where}: FastVideo's Ref2VA pipeline (transformer_ref) exactly when the case is Ref2VA",
              (o.get("generator.pipeline.components.override_pipeline_cls_name"),
               o["generator.pipeline.workload_type"])
              == (("minimaxh3ref2vamodularpipeline", "i2v") if ref2va else (None, "t2v")))

# The harness's compile-off mode wins over a profile's compile, last override first.
case_fw, fw_cfg = resolved(cases["minimax_h3_t2va_5s"], "b200")
os.environ[rc.DISABLE_TORCH_COMPILE_ENV] = "1"
off = fv.serve_overrides(rc.build_server_cmd("fastvideo", case_fw, fw_cfg, 30123)[4:])
os.environ[rc.DISABLE_TORCH_COMPILE_ENV] = "0"
check("compile-off mode turns every compile switch off",
      all(off[k] == "false" for k in (*fv.COMPILE_KEYS, "generator.engine.compile.vae_enabled")))
try:
    rc.build_server_cmd("fastvideo", case_fw, {**fw_cfg, "serve_args": "--server.port 1"}, 30123)
    check("a profile cannot set what the harness passes", False)
except ValueError as exc:
    check("a profile cannot set what the harness passes", "server.port" in str(exc), exc)

# --- the request ----------------------------------------------------------------
buffer = io.BytesIO()
Image.linear_gradient("L").resize((1024, 704)).convert("RGBA").save(buffer, format="PNG")
rc._cached_ref_images[cfg["test_image_url"]] = buffer.getvalue()

sent = []


class FakeResponse:
    def __init__(self, status_code=200, content=WITH_AUDIO, text=""):
        self.status_code, self.content, self.text = status_code, content, text
        self.headers = {"X-Inference-Time-S": "1.000"}


reply = {"response": FakeResponse()}


def fake_post(url, **kwargs):
    sent.append((url, kwargs))
    return reply["response"]


rc._post_drained = fake_post


def steps(case, framework):
    return (case.get("framework_request_extra") or {}).get(framework, {}).get(
        "num_inference_steps", case["num_inference_steps"]
    )


for cid, case in cases.items():
    case_fw, _ = resolved(case, "h200")
    sent.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(3):
            rc.send_request("http://fake", rc._request_case(case_fw, index), "fastvideo", cfg)
    urls = {url for url, _ in sent}
    bodies = [kwargs["json"] for _, kwargs in sent]
    body = bodies[0]
    check(f"{cid}: POST /v1/videos/sync, JSON", urls == {"http://fake/v1/videos/sync"}
          and all("json" in kwargs and "files" not in kwargs for _, kwargs in sent), urls)
    check(f"{cid}: only fields FastVideo's request declares",
          set().union(*bodies) <= REQUEST_FIELDS, sorted(set().union(*bodies) - REQUEST_FIELDS))
    w, h = (int(x) for x in body["size"].split("x"))
    check(f"{cid}: the case's shape, on FastVideo's H3 grid (24 fps, 17n+5 frames, multiples of 32)",
          (w, h, body["num_frames"], body["fps"]) == (case["width"], case["height"], case["num_frames"], 24)
          and body["num_frames"] % 17 == 5 and w % 32 == 0 and h % 32 == 0, body)
    check(f"{cid}: the case's seed, guidance 1, an explicit task, no flow_shift (H3's shifts are the checkpoint's)",
          body["seed"] == case["seed"] and body["guidance_scale"] == 1.0 and body["task"] in ("t2va", "ref2va")
          and "flow_shift" not in body, body)
    # FastVideo, like sglang, counts sigma grid points; vLLM-Omni counts forwards.
    check(f"{cid}: steps in FastVideo's convention, equal to sglang's ({steps(case, 'sglang')})",
          body["num_inference_steps"] == steps(case, "sglang"), body["num_inference_steps"])
    prompts = [b["prompt"] for b in bodies]
    check(f"{cid}: every request its own prompt", len(set(prompts)) == 3 and all("(take 0" in p for p in prompts))
    if cid == "minimax_h3_ref2va_5s":
        refs = [b["image_reference"] for b in bodies]
        images = [base64.b64decode(r[0]["image_url"].split(",", 1)[1]) for r in refs]
        sizes = {Image.open(io.BytesIO(data)).size for data in images}
        check(f"{cid}: the reference as its own data URI, prepared at the 2048px short edge",
              all(len(r) == 1 and r[0]["image_url"].startswith("data:image/png;base64,") for r in refs)
              and len(set(images)) == 3 and sizes == {(2976, 2048)}, sizes)

v2 = cases["minimax_h3_fasth3_v2_t2va_5s"]
check("FastH3 V2: 9 grid points = 8 forwards, vLLM-Omni's 8 evaluations",
      steps(v2, "fastvideo") == 9 == steps(v2, "vllm-omni") + 1 == steps(v2, "sglang"))

# --- the output -------------------------------------------------------------------
check("an MP4's tracks, in order", fv.mp4_handler_types(WITH_AUDIO) == ["vide", "soun"])
large = struct.pack(">I4sQ", 1, b"moov", 16 + len(box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + b"vide" + bytes(12))))))
large += box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + b"vide" + bytes(12))))
check("a 64-bit box size", fv.mp4_handler_types(large) == ["vide"])
check("a last box that runs to the end", fv.mp4_handler_types(struct.pack(">I4s", 0, b"moov") + box(b"trak")) == [])
fv.check_mp4(SILENT, audio=False)
for label, data in (("a silent MP4 when audio is expected", SILENT), ("bytes that are not an MP4", b'{"error": 1}'),
                    ("a truncated MP4", WITH_AUDIO[:30]), ("no bytes", b"")):
    try:
        fv.check_mp4(data, audio=True)
        check(f"{label} is an error", False)
    except RuntimeError as exc:
        check(f"{label} is an error", True, exc)

case_fw, _ = resolved(cases["minimax_h3_t2va_5s"], "h200")
for label, response, needle in (
    ("a silent MP4 fails the request", FakeResponse(content=SILENT), "soun"),
    ("an HTTP error carries FastVideo's message", FakeResponse(400, b"", '{"error": {"message": "fps=24"}}'),
     "fps=24"),
):
    reply["response"] = response
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc.send_request("http://fake", rc._request_case(case_fw, 0), "fastvideo", cfg)
        check(label, False)
    except RuntimeError as exc:
        check(label, needle in str(exc), exc)
reply["response"] = FakeResponse()


# --- bench_serving sends the single_e2e body ------------------------------------
class FakeAioResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, status=200, body=WITH_AUDIO):
        self.calls, self.status, self.body = [], status, body

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeAioResponse(self.status, self.body)


for cid in ("minimax_h3_ref2va_5s", "minimax_h3_fasth3_v2_t2va_5s"):
    case_fw, _ = resolved(cases[cid], "h200")
    request = rc._request_case(case_fw, 7)
    inputs = rc._bench_request_inputs(request, "fastvideo", cfg)
    bench_input = RequestFuncInput(
        prompt=inputs["prompt"], api_url="http://fake/v1/videos/sync", width=case_fw["width"],
        height=case_fw["height"], num_frames=case_fw["num_frames"], fps=case_fw["fps"],
        num_inference_steps=case_fw["num_inference_steps"], extra_body=inputs["extra_body"],
        image_paths=inputs["image_paths"],
    )
    session = FakeSession()
    out = asyncio.run(bs.make_fastvideo_request_func(True)(bench_input, session))
    check(f"{cid}: bench_serving's body is the single_e2e one",
          session.calls[0][1]["json"] == rc._fastvideo_payload(request, cfg) and inputs["image_paths"] is None)
    check(f"{cid}: bench_serving counts an MP4 with audio", out.success and out.output_count == 1, out.error)
    silent = asyncio.run(bs.make_fastvideo_request_func(True)(bench_input, FakeSession(body=SILENT)))
    check(f"{cid}: and fails a silent one", not silent.success and "soun" in silent.error, silent.error)

captured = {}


def fake_run(cmd, **kwargs):
    captured["cmd"] = cmd
    return subprocess.CompletedProcess(cmd, 1, stdout="")


rc.subprocess.run, saved_run = fake_run, rc.subprocess.run
case_fw, fw_cfg = resolved(cases["minimax_h3_t2va_5s"], "h200")
try:
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        rc.run_throughput("http://fake", case_fw, "fastvideo", cfg, rc._benchmark_config(cfg, case_fw), Path(tmp),
                          iter(range(50)))
finally:
    rc.subprocess.run = saved_run
cmd = captured["cmd"]
check("throughput runs bench_serving's FastVideo backend and asks for the audio track",
      cmd[cmd.index("--backend") + 1] == "fastvideo" and "--expect-audio" in cmd, cmd)

# --- the published version ------------------------------------------------------
versions = framework_versions({"framework_runtime": {"fastvideo": {
    "source_commit": "2b164405c3d15ed2d222d0582d6db09ffc5ee6b2",
    "packages": {"fastvideo": {"Version": "0.2.1"}, "fastvideo-kernel": {"Version": "0.3.5"}},
}}})
check("the published version is the checkout commit and its kernel",
      versions.get("fastvideo") == "FastVideo @ 2b164405c + fastvideo-kernel 0.3.5", versions)

sys.exit(fail)
