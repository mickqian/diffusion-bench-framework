#!/usr/bin/env python3
"""MiniMax-H3 T2V+A client with the request shape and timing of InferenceX VideoGenX.

Mirrors video-generation/evaluator/runner.py (_payload, _multipart, the poll loop):
both engines get POST /v1/videos, a 0.1 s status poll and GET /v1/videos/<id>/content;
latency = submit to the last byte of the MP4. One request at a time.

  python3 vgx_client.py --backend sglang|omni --base-url http://127.0.0.1:61283 \
      --cases vgx_cases.json --n 20 --warmup 1 --out res.jsonl
"""
import argparse
import hashlib
import json
import statistics
import time
import urllib.request
import uuid

MODEL = "MiniMaxAI/MiniMax-H3"
POLL_INTERVAL_SECONDS = 0.1
ACTIVE = {"queued", "pending", "in_progress", "processing", "running"}
FAILED = {"failed", "cancelled", "canceled"}
TERMINAL = FAILED | {"completed", "succeeded", "success"}
WARMUP = {
    "case_id": "warmup-only",
    "prompt": "A single continuous shot of a hand rotating a metal key in a lock. The action is visible and the lock clicking is audible. Fixed camera, no speech, music, subtitles, or cuts.",
    "seed": 11,
}
GENERATION = {
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": 124,
    "duration_seconds": 5,
    "aspect_ratio": "16:9",
    "num_inference_steps": 49,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}
TARGET = {
    "duration_seconds": GENERATION["duration_seconds"],
    "aspect_ratio": GENERATION["aspect_ratio"],
    "short_edge": min(GENERATION["width"], GENERATION["height"]),
}


def payload(backend, case):
    common = {
        "model": MODEL,
        "prompt": case["prompt"],
        "seed": case["seed"],
        "num_inference_steps": GENERATION["num_inference_steps"],
        "flow_shift": GENERATION["flow_shift"],
    }
    if backend == "sglang":
        return {
            **common,
            "task": "t2va",
            "conditions": [],
            "target": TARGET,
            "audio_flow_shift": GENERATION["audio_flow_shift"],
            "x264_preset": "ultrafast",
            "output_compression": 63,  # CRF = int((1 - 0.63) * 51) = 18
        }
    return {
        **common,
        "width": GENERATION["width"],
        "height": GENERATION["height"],
        "fps": GENERATION["fps"],
        "num_frames": GENERATION["frame_count"],
        "num_outputs_per_prompt": 1,
        "extra_params": {
            "task": "t2va",
            "target": TARGET,
            "audio_flow_shift": GENERATION["audio_flow_shift"],
            "video_codec_options": {"preset": "ultrafast", "crf": "18"},
        },
    }


def multipart(fields):
    boundary = "vgbench-" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{text}\r\n'.encode())
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def http(method, url, body=None, content_type=None, timeout=3600):
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def run(backend, base, case):
    fields = payload(backend, case)
    if backend == "sglang":
        body, ctype = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode(), "application/json"
    else:
        body, ctype = multipart(fields)
    start = time.monotonic()
    reply = json.loads(http("POST", f"{base}/v1/videos", body, ctype))
    vid = reply["id"]
    while True:
        status = reply.get("status")
        if status in TERMINAL:
            break
        if status not in ACTIVE:
            raise RuntimeError(f"unsupported status {status!r}: {str(reply)[:300]}")
        time.sleep(POLL_INTERVAL_SECONDS)
        reply = json.loads(http("GET", f"{base}/v1/videos/{vid}"))
    terminal = time.monotonic() - start
    if status in FAILED:
        raise RuntimeError(f"job {status}: {str(reply)[:500]}")
    mp4 = http("GET", f"{base}/v1/videos/{vid}/content")
    latency = time.monotonic() - start
    rec = {
        "latency_s": round(latency, 3),
        "terminal_s": round(terminal, 3),
        "bytes": len(mp4),
        "md5": hashlib.md5(mp4).hexdigest(),
    }
    for key in ("inference_time_s", "peak_memory_mb", "stage_timings", "metrics", "usage"):
        if key in reply:
            rec[key] = reply[key]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sglang", "omni"], required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = json.load(open(a.cases))[: a.n]
    with open(a.out, "a") as f:
        for _ in range(a.warmup):
            rec = {"backend": a.backend, "case": "warmup", **run(a.backend, a.base_url, WARMUP)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"warmup {rec['latency_s']:.2f}s {rec['bytes']}B {rec['md5'][:8]}", flush=True)
        lats = []
        for c in cases:
            rec = {"backend": a.backend, "case": c["case_id"], **run(a.backend, a.base_url, c)}
            lats.append(rec["latency_s"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{c['case_id']:18s} {rec['latency_s']:.2f}s {rec['bytes']}B {rec['md5'][:8]}", flush=True)
        print(
            f"p50 {statistics.median(lats):.3f}s min {min(lats):.3f}s max {max(lats):.3f}s n={len(lats)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
