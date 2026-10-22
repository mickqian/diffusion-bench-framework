#!/usr/bin/env python3
"""The generic HTTP path speaks TRT-LLM VisualGen: config file, request shape, sync MP4.

trtllm-serve reads its parallel/compile/attention settings from `--visual_gen_args
<yaml>`, rejects unknown top-level request fields with a 422 (so model knobs go
under `extra_params`), and its `/v1/videos/sync` answers with the MP4 itself.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def mp4(*handlers: str) -> bytes:
    traks = b"".join(
        box(b"trak", box(b"mdia", box(b"hdlr", b"\0" * 8 + h.encode() + b"\0" * 12))) for h in handlers
    )
    return box(b"ftyp", b"isom") + box(b"moov", traks)


class Reply:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200

    def json(self):
        return json.loads(self.content)


sent = []


def fake_post(url, **kwargs):
    sent.append((url, kwargs))
    return reply


rc._post_drained = fake_post
rc._raise_for_status = lambda resp: None

# --- server command: the profile's visual_gen_args become --visual_gen_args <file>
vga = {"parallel_config": {"cfg_size": 2, "ulysses_size": 2}, "torch_compile_config": {"enable": True}}
fw_cfg = {
    "num_gpus": 4,
    "http_server": {"bin": "trtllm-serve", "args_template": ["{model}", "--port", "{port}"]},
    "visual_gen_args": vga,
}
case = {"id": "wan_case", "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "num_gpus": 4}
cmd = rc._build_generic_http_cmd(case, fw_cfg, 61234)
check("the config flag is passed", "--visual_gen_args" in cmd, " ".join(cmd))
path = cmd[cmd.index("--visual_gen_args") + 1]
check("the file holds the profile's settings", json.loads(Path(path).read_text()) == vga, path)
check("no --tp_size without a tp_flag", "--tp_size" not in cmd)

# --- video request: JSON body, knobs under extra_params, sync MP4 checked
spec = {
    "endpoint": "/v1/videos/sync",
    "mode": "sync",
    "body": "json",
    "output": "mp4+audio",
    "extra_params_keys": ["guidance_scale_2", "flow_shift"],
    "omit_keys": ["model", "size", "extra_params.guardrails"],
}
video_case = {
    "id": "wan_case",
    "task": "text-to-video",
    "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "prompt": "a fox (take 01)",
    "width": 1280,
    "height": 720,
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 3.0,
    "seed": 42,
    "request_extra": {"flow_shift": 10.0, "extra_params": {"guardrails": False, "use_duration_template": False}},
    "frameworks": {"trtllm-visual": {"http_request": {"text-to-video": spec}}},
}
reply = Reply(mp4("vide", "soun"))
rc.send_request_generic_http("http://h", video_case, {}, "trtllm-visual")
url, kwargs = sent[-1]
body = kwargs.get("json")
check("posted to the sync endpoint", url == "http://h/v1/videos/sync", url)
check("as JSON, with native types", isinstance(body, dict) and body.get("num_frames") == 81, json.dumps(body))
check("moved keys live under extra_params",
      body["extra_params"] == {"guidance_scale_2": 3.0, "flow_shift": 10.0, "use_duration_template": False},
      json.dumps(body.get("extra_params")))
check("omitted keys, top-level and dotted, are gone",
      not {"model", "size", "guidance_scale_2", "flow_shift"} & set(body) and "guardrails" not in body["extra_params"])

reply = Reply(mp4("vide"))
try:
    rc.send_request_generic_http("http://h", video_case, {}, "trtllm-visual")
    check("an MP4 without its soundtrack is refused", False)
except RuntimeError as exc:
    check("an MP4 without its soundtrack is refused", "soun" in str(exc), str(exc))

reply = Reply(b'{"id": "x"}' + b"\0" * 4)
try:
    rc.send_request_generic_http("http://h", video_case, {}, "trtllm-visual")
    check("a reply that is not an MP4 is refused", False)
except RuntimeError as exc:
    check("a reply that is not an MP4 is refused", "MP4" in str(exc), str(exc))

# --- image request: the shared extras are reshaped the same way
image_case = dict(video_case, task="text-to-image", id="img_case")
image_case["frameworks"] = {"trtllm-visual": {"http_request": {"text-to-image": {
    "extra_params_keys": ["flow_shift"], "omit_keys": ["extra_params.guardrails"]}}}}
reply = Reply(b'{"data": []}')
rc.send_request_generic_http("http://h", image_case, {}, "trtllm-visual")
body = sent[-1][1]["json"]
check("image extras reshaped", "flow_shift" not in body and body["extra_params"].get("flow_shift") == 10.0
      and "guardrails" not in body["extra_params"], json.dumps(body.get("extra_params")))

sys.exit(fail)
