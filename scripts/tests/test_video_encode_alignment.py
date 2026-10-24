#!/usr/bin/env python3
"""Every video cell asks each configurable framework for its closest competitor's MP4 encode.

The rule (2026-10-24): within a cell, a framework whose encoder is configurable
encodes like the closest competitor; a hard-coded one is recorded in the case.
sglang defaults to x264 fast at CRF 25, vLLM-Omni to ultrafast at CRF 18 and
TRT-LLM to medium at CRF 23, so without it the container encode, which is
inside every video latency, was different work per framework (LTX-2.5:
~0.28 s of sglang's 6 s). Checked here at the point each framework reads it:
sglang's request fields, vLLM-Omni's extra_params.video_codec_options,
LightX2V's launch config, ComfyUI's SaveVideo inputs. Also: the `_note` that
records why travels with the setting and never reaches a server.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench import comfyui_client as cc  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:300]) if detail and not cond else ''}")
    if not cond:
        fail = 1


def sglang_crf(output_compression):  # sglang entrypoints/utils.py CudaVideoEncoder.open
    return int((1 - output_compression / 10 / 10.0) * 51)


ULTRA18 = ("ultrafast", 18)
MEDIUM23 = ("medium", 23)
# case -> (encode, which frameworks the case configures for it)
EXPECTED = {
    "wan22_t2v_a14b_720p": (MEDIUM23, {"sglang", "vllm-omni", "comfyui"}),
    "cosmos3_nano_t2v_720p_189f": (MEDIUM23, {"sglang", "vllm-omni"}),
    "cosmos3_nano_i2v_720p_189f": (ULTRA18, {"sglang"}),
    "minimax_h3_t2va_5s": (ULTRA18, {"sglang", "lightx2v"}),
    "minimax_h3_t2va_5s_4gpu": (ULTRA18, {"sglang", "lightx2v"}),
    "minimax_h3_ref2va_5s": (ULTRA18, {"sglang", "lightx2v"}),
    "ltx2.5_twostage_t2v_2gpus": (ULTRA18, {"sglang"}),
    "minimax_h3_fasth3_v2_t2va_5s": (ULTRA18, {"sglang"}),
    "minimax_h3_t2va_480p_1gpu": (("medium", 18), {"sglang"}),
}

config = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
cases = {c["id"]: c for c in config["cases"]}
video = {cid for cid, c in cases.items() if "video" in c["task"]}
check("every video case has an expected encode", video == set(EXPECTED), sorted(video ^ set(EXPECTED)))


def no_comment_keys(node, path=""):
    if isinstance(node, dict):
        return [f"{path}.{k}" for k in node if k.startswith("_")] + [
            bad for k, v in node.items() for bad in no_comment_keys(v, f"{path}.{k}")
        ]
    if isinstance(node, list):
        return [bad for i, v in enumerate(node) for bad in no_comment_keys(v, f"{path}[{i}]")]
    return []


for cid, ((preset, crf), configured) in EXPECTED.items():
    case = cases[cid]
    frameworks = case["frameworks"]
    for fw in frameworks:
        # _request_extra's merge, minus inlining the reference image (a download)
        extra = rc._deep_merge(rc._without_comments(case.get("request_extra", {})),
                               rc._without_comments((case.get("framework_request_extra") or {}).get(fw, {})))
        check(f"{cid}/{fw}: no `_` key reaches the request", not no_comment_keys(extra), no_comment_keys(extra))

    sgl = rc._without_comments(case["framework_request_extra"]["sglang"])
    check(f"{cid}: sglang encodes x264 {preset} at CRF {crf}",
          (sgl.get("x264_preset"), sglang_crf(sgl.get("output_compression", 50))) == (preset, crf), sgl)
    check(f"{cid}: sglang's encode carries its rationale",
          "_note" in case["framework_request_extra"]["sglang"])

    vo = rc._without_comments((case.get("framework_request_extra") or {}).get("vllm-omni", {}))
    options = (vo.get("extra_params") or {}).get("video_codec_options")
    if "vllm-omni" in configured:
        check(f"{cid}: vLLM-Omni encodes x264 {preset} at CRF {crf}, its threads kept",
              options == {"preset": preset, "crf": str(crf), "threads": "0"}, options)
    elif preset == "ultrafast" and crf == 18 and frameworks.get("vllm-omni", {}).get("status") == "supported":
        check(f"{cid}: vLLM-Omni keeps its own default (ultrafast, CRF 18)", options is None, options)

    if "lightx2v" in configured:
        for hw in ("b300", "h200"):
            fw_cfg = rc._resolve_framework_config("lightx2v", frameworks["lightx2v"], None,
                                                  {"hardware_profile_override": hw})
            path = rc._write_lightx2v_config(case, fw_cfg, "/models/x", case["model"])
            launched = json.loads(Path(path).read_text())
            check(f"{cid}/lightx2v[{hw}]: launch config encodes x264 {preset} at CRF {crf}",
                  launched.get("video_codec_options") == {"preset": preset, "crf": str(crf)},
                  launched.get("video_codec_options"))

    if "comfyui" in configured:
        fw = frameworks["comfyui"]
        blocks = [fw["comfyui"]] + [{**fw["comfyui"], **p["comfyui"]}
                                    for p in (fw.get("command_profiles") or {}).values() if p.get("comfyui")]
        for spec in blocks:
            save = [n["inputs"] for n in cc.build_graph(case, spec).values() if n["class_type"] == "SaveVideo"]
            check(f"{cid}/comfyui: SaveVideo re-encodes h264 at CRF {crf} (x264's medium: no preset input)",
                  save and save[0].get("format.codec") == "h264"
                  and save[0].get("format.codec.encoding") == "re-encode"
                  and save[0].get("format.codec.encoding.crf") == float(crf), save)

# the comment keys are dropped at both levels of the merge, and only there
probe = {
    "request_extra": {"a": 1, "_note": "x"},
    "framework_request_extra": {"sglang": {"b": {"_kept": 1}, "_note": "y"}},
}
merged = rc._request_extra(probe, "sglang")
check("top-level `_` keys of request_extra and framework extras are dropped", merged == {"a": 1, "b": {"_kept": 1}},
      merged)

# DIFFUSION_BENCH_KEEP_VIDEO_DIR keeps a reply, and nothing without it
with tempfile.TemporaryDirectory() as d:
    os.environ.pop(rc.KEEP_VIDEO_DIR_ENV, None)
    rc._keep_video_reply("vllm-omni", {"id": "c"}, b"mp4")
    check("no keep dir: nothing written", not os.listdir(d))
    os.environ[rc.KEEP_VIDEO_DIR_ENV] = os.path.join(d, "kept")
    rc._keep_video_reply("vllm-omni", {"id": "c"}, b"mp4")
    kept = os.listdir(os.path.join(d, "kept"))
    check("keep dir: the reply is written as <framework>_<case>_<ms>.mp4",
          len(kept) == 1 and kept[0].startswith("vllm-omni_c_") and kept[0].endswith(".mp4"), kept)
    os.environ.pop(rc.KEEP_VIDEO_DIR_ENV, None)

sys.exit(fail)
