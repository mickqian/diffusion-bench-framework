#!/usr/bin/env python3
"""sglang's diagnostic request must itemize the time no stage accounts for.

The perf dump's stages stop before the output is saved: `total_duration_ms` is
taken before the worker materialises and encodes it. So a stage breakdown alone
never shows the 1-3 s of MP4 (+audio) encode/save that a short sglang video
carries after its last stage -- 20-40% of those clients' latency, recorded in
the SKILL on 2026-09-25 and still invisible in every report after it, until an
external team found SGLang slower on few-step MiniMax-H3.

The tail is diagnostic client latency minus the sum of that same request's
stages. Both come from the one instrumented request, which is what makes the
difference a decomposition; the measured median is a different request. A
missing or empty dump must record nothing -- a zero sum would publish the whole
latency as "tail".
"""
import itertools
import json
import statistics
import sys
import tempfile
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


FIELDS = (
    "diagnostic_stages_s",
    "diagnostic_stage_sum_s",
    "diagnostic_client_latency_s",
    "unattributed_tail_s",
    "unattributed_tail_frac",
    "poll_interval_s",
)

# Shaped like sglang's dump_benchmark_report. denoise_steps_ms subdivides
# DenoisingStage, so adding it would count the denoise twice; the total also
# covers time between stages, so it is not the stage sum either.
DUMP = {
    "total_duration_ms": 4502.0,
    "steps": [
        {"name": "TextEncodingStage", "duration_ms": 180.0},
        {"name": "DenoisingStage", "duration_ms": 3650.0},
        {"name": "DecodingStage", "duration_ms": 601.0},
    ],
    "denoise_steps_ms": [{"step": i, "duration_ms": 912.5} for i in range(4)],
}
DIAG_CLIENT_S = 5.702

print("from a fake perf dump")
got = rc._stage_breakdown(DUMP, DIAG_CLIENT_S, rc.POLL_INTERVAL_S)
check(
    "stages recorded in seconds",
    got.get("diagnostic_stages_s")
    == {"TextEncodingStage": 0.18, "DenoisingStage": 3.65, "DecodingStage": 0.601},
    str(got.get("diagnostic_stages_s")),
)
check("their sum excludes the per-step list", got.get("diagnostic_stage_sum_s") == 4.431,
      str(got.get("diagnostic_stage_sum_s")))
check("tail = diagnostic client - sum", got.get("unattributed_tail_s") == 1.271,
      str(got.get("unattributed_tail_s")))
check("tail fraction of the diagnostic client", got.get("unattributed_tail_frac") == 0.223,
      str(got.get("unattributed_tail_frac")))
check("diagnostic client latency recorded", got.get("diagnostic_client_latency_s") == 5.702)
check("poll quantum recorded with it", got.get("poll_interval_s") == rc.POLL_INTERVAL_S)
check("a sync request carries no poll quantum",
      "poll_interval_s" not in rc._stage_breakdown(DUMP, DIAG_CLIENT_S, None))
check("no stages -> nothing", rc._stage_breakdown({**DUMP, "steps": []}, DIAG_CLIENT_S, None) == {})
check("an older dump without steps -> nothing",
      rc._stage_breakdown({"total_duration_ms": 4502.0}, DIAG_CLIENT_S, None) == {})


def run(case, framework="sglang", dump=DUMP, diag_client=DIAG_CLIENT_S, measured=(5.41, 5.46)):
    """run_single_request against a fake server; returns (metrics, calls)."""
    calls = []
    lats = iter(measured)

    def fake_send(base_url, case_, framework_, config=None, perf_dump_path=None):
        calls.append(perf_dump_path)
        if perf_dump_path is None:
            return rc.LatencyBreakdown(client_s=next(lats))
        if dump is None:  # the server wrote nothing
            return rc.LatencyBreakdown(client_s=diag_client)
        Path(perf_dump_path).write_text(json.dumps(dump))
        return rc.LatencyBreakdown(client_s=diag_client, server_s=dump["total_duration_ms"] / 1000)

    real = rc.send_request
    rc.send_request = fake_send
    try:
        with tempfile.TemporaryDirectory() as d:
            result = rc.run_single_request(
                "http://fake", case, framework, Path(d),
                bench_cfg={"single": {"video_repeats": len(measured), "image_repeats": len(measured)}},
                inputs=itertools.count(),
            )
    finally:
        rc.send_request = real
    return result["metrics"], calls


VIDEO = {"id": "ltx2", "model": "org/ltx2", "task": "text-to-video", "prompt": "a clip", "num_frames": 121}
IMAGE = {"id": "zimage", "model": "org/zimage", "task": "text-to-image", "prompt": "a picture"}

print("through run_single_request")
m, calls = run(VIDEO)
check("one diagnostic request, after the measured ones",
      calls[:-1] == [None, None] and calls[-1] is not None, str(calls))
check("the headline stays the measured median",
      m["client_latency_s"] == round(statistics.median([5.41, 5.46]), 3), str(m["client_latency_s"]))
check("the tail is taken on the diagnostic request, not the median",
      m.get("unattributed_tail_s") == 1.271, str(m.get("unattributed_tail_s")))
check("all tail fields recorded for a polled video", all(k in m for k in FIELDS),
      str(sorted(k for k in FIELDS if k not in m)))
check("server figure still comes from the same dump", m.get("server_latency_s") == 4.502)

m, _ = run(IMAGE)
check("an image request records the tail", m.get("unattributed_tail_s") == 1.271)
check("an image request records no poll quantum", "poll_interval_s" not in m)

m, _ = run(VIDEO, dump=None)
check("missing dump -> no tail fields", not any(k in m for k in FIELDS),
      str(sorted(k for k in FIELDS if k in m)))

m, _ = run(VIDEO, dump={**DUMP, "steps": []})
check("dump without stages -> no tail fields", not any(k in m for k in FIELDS),
      str(sorted(k for k in FIELDS if k in m)))
check("...but its total still annotates the cell", m.get("server_latency_s") == 4.502)

m, calls = run(VIDEO, framework="vllm-omni")
check("no competitor gets a diagnostic request", calls == [None, None], str(calls))
check("no competitor gets tail fields", not any(k in m for k in FIELDS))

sys.exit(fail)
