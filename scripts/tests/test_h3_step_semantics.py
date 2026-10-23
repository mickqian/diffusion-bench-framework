#!/usr/bin/env python3
"""Every MiniMax-H3 row runs the same number of DiT forwards on the same sigma grid.

The frameworks disagree on what a step count means. TRT-LLM's MiniMaxH3Scheduler and
FastVideo count sigma grid points (linspace(1, 0, N): N - 1 forwards); vLLM-Omni
(#7219, 2026-09-24), LightX2V (`infer_steps`) and sglang count forwards
(linspace(1, 0, N + 1)). sglang switched with sglang#38671 (2026-10-10), and the
harness kept sending it the case's 50 -- so the 2026-10-22 round ran sglang 50
forwards on a 51-point grid ('infer_steps: 50', '50/50' in every base-H3 sglang
log) against everyone else's 49 on the official 50-point grid: ~2% more work for
sglang and a different schedule. The official grid is 50 points, so every row must
run 49 forwards. This checks what each framework's request (or launch config)
says, translated through that framework's convention.

ComfyUI counts evaluations (BasicScheduler simple: steps + 1 sigmas) and is still
sent the case's 50 on the base cases: one forward more, its own extra work, kept
visible here rather than hidden.
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from diffusion_bench import comfyui_client as cc  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        fail = 1


cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
cases = {c["id"]: c for c in cfg["cases"] if c["id"].startswith("minimax_h3")}
# Ref2VA's request extras carry the reference; a local stand-in keeps this offline.
_png = io.BytesIO()
Image.new("RGB", (64, 48), (90, 120, 150)).save(_png, format="PNG")
rc._cached_ref_images[cfg["test_image_url"]] = _png.getvalue()


def resolved(case, fw, hw="b300"):
    with contextlib.redirect_stdout(io.StringIO()):
        fw_cfg = rc._resolve_framework_config(fw, case["frameworks"][fw], None, {"hardware_profile_override": hw})
    return rc._case_for_framework(case, fw_cfg), fw_cfg


def extra_steps(case, fw):
    return ((case.get("framework_request_extra") or {}).get(fw) or {}).get("num_inference_steps")


def forwards(case, fw):
    """DiT forwards one request of `case` runs on `fw`, from what the harness sends it."""
    if fw == "sglang":
        return int(rc._build_sglang_payload(rc._request_case(case, 0), cfg)["num_inference_steps"])
    if fw == "vllm-omni":  # the video form: case fields, then this framework's extras
        return int(extra_steps(case, fw) or case["num_inference_steps"])
    if fw == "lightx2v":
        server_case, fw_cfg = resolved(case, fw)
        os.environ.setdefault(rc.DISABLE_TORCH_COMPILE_ENV, "0")
        path = rc._write_lightx2v_config(server_case, fw_cfg, "/models/MiniMax-H3", case["model"])
        launch = json.loads(Path(path).read_text())
        os.remove(path)
        return int(launch["infer_steps"])
    if fw == "fastvideo":
        return int(rc._fastvideo_payload(rc._request_case(case, 0), cfg)["num_inference_steps"]) - 1
    if fw == "trtllm-visual":
        spec = rc._generic_http_request_spec(case, fw)
        return int(rc._generic_video_fields(rc._request_case(case, 0), cfg, fw, spec)["num_inference_steps"]) - 1
    if fw == "comfyui":
        server_case, _ = resolved(case, fw)
        return int(cc.workflow_params(server_case, server_case["_comfyui"])["steps"])
    raise ValueError(fw)


OFFICIAL = 49  # the official 50-point grid
KNOWN_EXTRA = {"comfyui": 1}  # counts evaluations and is sent 50; see the docstring
for cid in ("minimax_h3_t2va_5s", "minimax_h3_ref2va_5s", "minimax_h3_t2va_5s_4gpu"):
    case = cases[cid]
    check(f"{cid}: the case keeps the official 50-point count", case["num_inference_steps"] == 50)
    for fw in case["frameworks"]:
        got = forwards(case, fw)
        want = OFFICIAL + KNOWN_EXTRA.get(fw, 0)
        check(f"{cid}/{fw}: {got} forwards (want {want})", got == want)
    # warmups and throughput: the request extra outranks the reduced warmup step count,
    # as it already did for vLLM-Omni, so sglang's warmups run the measured 49 too
    warm = {**case, "num_inference_steps": 3}
    check(f"{cid}: sglang's warmup requests carry the measured 49 (the extra wins, as for vLLM-Omni)",
          rc._build_sglang_payload(rc._request_case(warm, 0), cfg)["num_inference_steps"] == OFFICIAL)
    body = rc._bench_request_inputs(rc._request_case(case, 1), "sglang", cfg)["extra_body"]
    check(f"{cid}: bench_serving's sglang body carries 49 (merged after its --num-inference-steps)",
          body.get("num_inference_steps") == OFFICIAL, body.get("num_inference_steps"))

src = (ROOT / "src" / "diffusion_bench" / "bench_serving.py").read_text()
video_fn = src[src.index("async def async_request_video_sglang"):src.index("async def async_request_vllm_omni")]
json_branch = video_fn[video_fn.index("# Use JSON"):]
check("bench_serving's sglang video JSON merges extra_body after num_inference_steps",
      json_branch.index('payload["num_inference_steps"]') < json_branch.index("payload.update(input.extra_body)"))

# The consumer-tier case: its only competitor, ComfyUI, counts evaluations like sglang,
# so both run the cookbook comparison's 20 forwards and sglang needs no override.
low = cases["minimax_h3_t2va_480p_1gpu"]
check("minimax_h3_t2va_480p_1gpu: sglang and ComfyUI both run 20 forwards, no override",
      extra_steps(low, "sglang") is None and forwards(low, "sglang") == forwards(low, "comfyui") == 20)

# FastH3 V2: 8 forwards on the checkpoint's rungs everywhere (sglang and vLLM-Omni say 8, FastVideo 9).
v2 = cases["minimax_h3_fasth3_v2_t2va_5s"]
check("FastH3 V2: 8 forwards on sglang, vLLM-Omni and FastVideo",
      {forwards(v2, fw) for fw in ("sglang", "vllm-omni", "fastvideo")} == {8})

sys.exit(fail)
