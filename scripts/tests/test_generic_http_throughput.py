#!/usr/bin/env python3
"""Throughput sends a generic HTTP framework the request single_e2e sends it.

bench_serving had no TRT-LLM VisualGen backend, so trtllm-visual throughput fell
through to sglang's multipart `/v1/videos` submit and every request got a 422
(2026-10-22, minimax_h3_t2va_5s on 4xB300: 0/4 completed) while single_e2e, which
posts the `http_request`-shaped JSON to `/v1/videos/sync`, measured 74.45 s.
run_throughput now prebuilds each request with single_e2e's builder and
bench_serving's generic backend posts it as is and checks the reply the same way.
This drives both through a fake transport and compares what the server receives.
"""
import contextlib
import copy
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from diffusion_bench import bench_serving as bs  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

FW = "trtllm-visual"
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:600]) if detail and not cond else ''}")
    if not cond:
        fail = 1


def box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def mp4(*handlers: bytes) -> bytes:
    traks = b"".join(box(b"trak", box(b"mdia", box(b"hdlr", bytes(8) + h + bytes(12)))) for h in handlers)
    return box(b"ftyp", b"isom") + box(b"moov", traks)


reply = {"status": 200, "body": mp4(b"vide", b"soun")}


class SyncReply:
    """requests' Response as single_e2e reads it."""

    def __init__(self):
        self.status_code = reply["status"]
        self.content = reply["body"]
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return json.loads(self.content or b"{}")


class AioReply:
    """aiohttp's response as bench_serving's generic backend reads it."""

    def __init__(self):
        self.status, self._body = reply["status"], reply["body"]

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


single_sent: list = []
bench_sent: list = []


class FakeSession:
    def post(self, url, **kwargs):
        bench_sent.append((url, kwargs))
        return AioReply()


def fake_post(url, **kwargs):
    single_sent.append((url, kwargs))
    return SyncReply()


real_generic = bs.make_generic_http_request_func


def generic_on_fake_session(framework):
    real = real_generic(framework)

    async def request(input, session, pbar=None):
        return await real(input, FakeSession(), pbar)

    return request


launched: list = []


def fake_subprocess_run(cmd, **kwargs):
    # bench_serving in-process, so its real dataset, warmups and backend dispatch run
    launched.append(cmd)
    sys.argv = ["bench_serving", *cmd[3:]]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bs.main()
    return subprocess.CompletedProcess(cmd, 0, stdout="")


class Probe:
    status_code = 200

    def json(self):
        return {}


cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
cases = {c["id"]: c for c in cfg["cases"]}


def resolved(cid):
    case = cases[cid]
    with contextlib.redirect_stdout(io.StringIO()):
        fw_cfg = rc._resolve_framework_config(FW, case["frameworks"][FW], None, {"hardware_profile_override": "b300"})
    server_case = rc._case_for_framework(case, fw_cfg)
    return server_case, rc._merge_nested(rc._benchmark_config(cfg, server_case), fw_cfg.get("benchmark", {}))


def single_e2e_request(case, index):
    """What single_e2e posts for request `index`: (url, kwargs)."""
    single_sent.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        rc.send_request_generic_http("http://fake", rc._request_case(case, index), cfg, FW)
    return single_sent[0]


def throughput(case, bench, first=10):
    bench_sent.clear()
    launched.clear()
    with tempfile.TemporaryDirectory() as log_dir, contextlib.redirect_stdout(io.StringIO()):
        result = rc.run_throughput("http://fake", case, FW, cfg, bench, Path(log_dir), iter(range(first, first + 40)))
    return result


def expected_requests(case, bench, first=10):
    warmup = {**case, "num_inference_steps": rc.BENCH_SERVING_WARMUP_STEPS}
    total = rc.BENCH_SERVING_WARMUPS + int(bench["throughput"]["num_requests"])
    return [
        single_e2e_request(warmup if i < rc.BENCH_SERVING_WARMUPS else case, first + i) for i in range(total)
    ]


saved = (rc._post_drained, subprocess.run, requests.get, bs.make_generic_http_request_func, sys.argv)
rc._post_drained, subprocess.run = fake_post, fake_subprocess_run
requests.get = lambda url, timeout=None, **kwargs: Probe()  # bench_serving's health and model-info probes
bs.make_generic_http_request_func = generic_on_fake_session
try:
    # --- the failing cell: MiniMax-H3 T2V+A, JSON to /v1/videos/sync, MP4 with audio
    case, bench = resolved("minimax_h3_t2va_5s")
    num_requests = int(bench["throughput"]["num_requests"])
    spec = case["frameworks"][FW]["http_request"]["text-to-video"]
    result = throughput(case, bench)
    expected = expected_requests(case, bench)
    cmd = launched[0]
    check("throughput runs bench_serving's trtllm-visual backend",
          cmd[cmd.index("--backend") + 1] == FW, cmd)
    check("and passes the warmup step count its prebuilt warmups carry",
          cmd[cmd.index("--warmup-inference-steps") + 1] == str(rc.BENCH_SERVING_WARMUP_STEPS), cmd)
    check(f"all {num_requests} requests complete", not result.get("error")
          and result["metrics"]["completed_requests"] == num_requests, result.get("error") or result["metrics"])
    check("warmup and requests all post JSON to /v1/videos/sync",
          len(bench_sent) == len(expected)
          and all(u == "http://fake/v1/videos/sync" and set(kw) == {"json"} for u, kw in bench_sent),
          [(u, sorted(kw)) for u, kw in bench_sent])
    check("single_e2e posts the same way", all(u == "http://fake/v1/videos/sync" and "json" in kw for u, kw in expected))
    bodies = [kw["json"] for _, kw in bench_sent]
    check("each body is single_e2e's for the same request, warmup included",
          bodies == [kw["json"] for _, kw in expected],
          json.dumps([b for b, (_, e) in zip(bodies, expected) if b != e["json"]][:1]))
    check("omit_keys applied (no model, size or guidance_scale)",
          not any(set(spec["omit_keys"]) & set(b) for b in bodies), [sorted(b) for b in bodies])
    check("native types: frames, steps and seed are numbers",
          all(isinstance(b[k], int) for b in bodies for k in ("num_frames", "num_inference_steps", "seed")))
    check("the warmup runs bench_serving's warmup steps, the requests the case's",
          [b["num_inference_steps"] for b in bodies]
          == [rc.BENCH_SERVING_WARMUP_STEPS] * rc.BENCH_SERVING_WARMUPS + [case["num_inference_steps"]] * num_requests)
    check("every request its own prompt",
          len({b["prompt"] for b in bodies}) == len(bodies) and all("(take " in b["prompt"] for b in bodies))
    check("seed is the case's, as in single_e2e", {b["seed"] for b in bodies} == {case["seed"]})

    # A reply without the soundtrack fails the request, as it fails single_e2e.
    reply["body"] = mp4(b"vide")
    result = throughput(case, bench)
    errors = [e["error"] for e in result["metrics"].get("errors", [])]
    check("an MP4 without its audio track fails each request",
          result.get("error") and len(errors) == num_requests and all("soun" in e for e in errors), errors)
    reply.update(status=422, body=b'{"detail": "extra fields not permitted"}')
    result = throughput(case, bench)
    errors = [e["error"] for e in result["metrics"].get("errors", [])]
    check("a 422 fails each request and keeps the server's reason",
          result.get("error") and errors and all(e.startswith("HTTP 422") and "extra fields" in e for e in errors), errors)
    reply.update(status=200, body=mp4(b"vide"))

    # --- extras: flow_shift moves under extra_params, dotted omit_keys drop nested ones
    case, bench = resolved("cosmos3_nano_t2v_720p_189f")
    result = throughput(case, bench)
    expected = expected_requests(case, bench)
    bodies = [kw["json"] for _, kw in bench_sent]
    check("Cosmos3 T2V: completes, single_e2e's bodies", not result.get("error")
          and bodies == [kw["json"] for _, kw in expected], result.get("error"))
    params = [b.get("extra_params") or {} for b in bodies]
    check("extra_params populated (flow_shift moved there, not top level)",
          all(p.get("flow_shift") == 10.0 and "flow_shift" not in b for p, b in zip(params, bodies)), params[:1])
    check("dotted omit_keys removed the nested guardrails and max_sequence_length",
          all("guardrails" not in p and "max_sequence_length" not in p for p in params), params[:1])

    # --- images: JSON to /v1/images/generations
    case, bench = resolved("cosmos3_nano_t2i_720p")
    reply["body"] = b'{"data": [{"b64_json": "x"}]}'
    result = throughput(case, bench)
    expected = expected_requests(case, bench)
    check("Cosmos3 T2I: completes, posting single_e2e's bodies to /v1/images/generations",
          not result.get("error")
          and [(u, kw["json"]) for u, kw in bench_sent] == [(u, kw["json"]) for u, kw in expected]
          and all(u == "http://fake/v1/images/generations" for u, _ in bench_sent),
          result.get("error"))
    reply["body"] = b""
    result = throughput(case, bench)
    errors = [e["error"] for e in result["metrics"].get("errors", [])]
    check("an empty image reply fails, in throughput and single_e2e alike",
          result.get("error") and errors and all("empty body" in e for e in errors), errors)
    try:
        single_e2e_request(case, 60)
        check("single_e2e refuses it too", False)
    except RuntimeError as exc:
        check("single_e2e refuses it too", "empty body" in str(exc), exc)
    reply["body"] = mp4(b"vide", b"soun")

    # --- a request bench_serving cannot replay is refused up front, single_e2e untouched
    case, bench = resolved("minimax_h3_t2va_5s")
    polled = copy.deepcopy(case)
    del polled["frameworks"][FW]["http_request"]["text-to-video"]["mode"]
    result = throughput(polled, bench)
    check("a polled request is a throughput error before bench_serving runs",
          not launched and "one JSON POST" in (result.get("error") or ""), result.get("error"))

    # --- bench_serving's backend needs the prebuilt requests
    sys.argv = ["bench_serving", "--backend", FW, "--base-url", "http://fake", "--dataset", "fixed",
                "--task", "text-to-video", "--num-prompts", "1"]
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            bs.main()
        check("--backend trtllm-visual without --request-inputs is refused", False)
    except ValueError as exc:
        check("--backend trtllm-visual without --request-inputs is refused", "--request-inputs" in str(exc), exc)
finally:
    rc._post_drained, subprocess.run, requests.get, bs.make_generic_http_request_func, sys.argv = saved

sys.exit(fail)
