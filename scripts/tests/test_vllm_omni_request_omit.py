#!/usr/bin/env python3
"""vLLM-Omni video requests drop the fields the case's entry lists in `request_omit_keys`.

vLLM-Omni's LTX applies a top-level `guidance_scale` to BOTH video and audio CFG,
after `extra_params`, so the LTX-2.3 cell must not send it: audio CFG 7 would
become the case's 3.
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


MP4 = box(b"ftyp", b"isom") + box(b"moov", box(b"trak", box(b"mdia", box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12))))


class Reply:
    content = MP4
    status_code = 200


sent = []
rc._post_drained = lambda url, **kwargs: sent.append(kwargs) or Reply()
rc._raise_for_status = lambda resp: None

config = json.load(open(ROOT / "configs" / "comparison_configs.json"))
cases = {c["id"]: c for c in config["cases"]}

rc.send_request_vllm_omni("http://x", cases["ltx2.3_twostage_t2v_2gpus"], config)
data = sent[-1]["data"]
extra = json.loads(data["extra_params"])
check("the LTX-2.3 request carries no top-level guidance_scale", "guidance_scale" not in data, str(sorted(data)))
check("its extra_params pin the official LTX-2.3 guider",
      (extra["video_cfg_scale"], extra["audio_cfg_scale"], extra["video_stg_blocks"]) == (3.0, 7.0, [28]),
      str(extra))
check("steps and frames still go", (data["num_inference_steps"], data["num_frames"]) == ("30", "121"))

rc.send_request_vllm_omni("http://x", cases["ltx2_twostage_t2v"], config)
check("a case without request_omit_keys keeps guidance_scale", sent[-1]["data"].get("guidance_scale") == "4.0")

sys.exit(fail)
