#!/usr/bin/env python3
"""Every image cell answers the same lossless format: PNG.

sglang answers JPEG q75 when a request names no format (choose_output_image_ext),
except for models whose sampling params default to PNG (Qwen-Image-2.1, Cosmos3);
vLLM-Omni (`output_format or "png"`), TRT-LLM (`format` defaults to png),
LightX2V's sync endpoint and ComfyUI's SaveImage all answer PNG. So the 2026-10-22
Z-Image, FLUX.3 and Ideogram-4 sglang rows encoded a lossy JPEG while every
competitor encoded a PNG -- different work in the measured window. Every image
request now asks for PNG through the field that framework reads; this drives each
builder (single_e2e and throughput) against a fake transport and checks it.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench import comfyui_client as cc  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        fail = 1


cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
_png = io.BytesIO()
Image.new("RGBA", (64, 48), (90, 120, 150, 255)).save(_png, format="PNG")
rc._cached_ref_images[cfg["test_image_url"]] = _png.getvalue()

sent = []


class FakeResponse:
    status_code = 200
    content = b"\x89PNG"
    headers: dict = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"b64_json": "x"}]}


def fake_post(url, **kwargs):
    sent.append({"url": url, "body": kwargs.get("json") if kwargs.get("json") is not None else kwargs.get("data")})
    return FakeResponse()


saved = rc._post_drained
rc._post_drained = fake_post
image_cases = [c for c in cfg["cases"] if c["task"] in rc.IMAGE_TASKS]
check("the image tasks the harness knows cover every image case", len(image_cases) >= 7, len(image_cases))
seen = set()
try:
    for case in image_cases:
        for fw, raw in case["frameworks"].items():
            with contextlib.redirect_stdout(io.StringIO()):
                fw_cfg = rc._resolve_framework_config(fw, raw, None, {"hardware_profile_override": "b300"})
            server_case = rc._case_for_framework(case, fw_cfg)
            request = rc._request_case(server_case, 0)
            where = f"{case['id']}/{fw}"
            seen.add(fw)
            if fw == "comfyui":
                graph = cc.build_graph(request, request["_comfyui"])
                savers = sorted({n["class_type"] for n in graph.values() if n["class_type"].startswith("Save")})
                check(f"{where}: the graph saves through SaveImage (PNG)", savers == ["SaveImage"], savers)
                continue
            sent.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                rc.send_request("http://fake", request, fw, cfg)
            body, url = sent[-1]["body"], sent[-1]["url"]
            if fw == "lightx2v":
                check(f"{where}: the sync image endpoint (PNG bytes), no format field (the schema forbids one)",
                      url.endswith("/v1/tasks/image/sync") and not {"output_format", "format"} & set(body), url)
                continue
            field = rc.IMAGE_FORMAT_FIELD[fw]
            others = {"output_format", "format"} - {field}
            check(f"{where}: single_e2e asks for png via `{field}`",
                  body.get(field) == "png" and not others & set(body), {k: body.get(k) for k in ("output_format", "format")})
            inputs = rc._bench_request_inputs(rc._request_case(server_case, 1), fw, cfg)
            tp = inputs["http_request"]["json"] if inputs.get("http_request") else inputs["extra_body"]
            check(f"{where}: throughput asks for png via `{field}`",
                  tp.get(field) == "png" and not others & set(tp), {k: tp.get(k) for k in ("output_format", "format")})
finally:
    rc._post_drained = saved

check("covered sglang, vLLM-Omni, TRT-LLM, LightX2V and ComfyUI image cells",
      {"sglang", "vllm-omni", "trtllm-visual", "lightx2v", "comfyui"} <= seen, sorted(seen))
video = next(c for c in cfg["cases"] if c["task"] == "text-to-video" and "sglang" in c["frameworks"])
check("a video request carries no image format field",
      "output_format" not in rc._build_sglang_payload(rc._request_case(video, 0), cfg))
sys.exit(fail)
