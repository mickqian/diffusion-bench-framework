#!/usr/bin/env python3
"""A dispersed median must say so in the report, not only in the JSON.

The harness flags repeats that spread 25% or more and publishes the median
anyway -- a framework that really produces two modes produces both, so the
median means something, it just does not describe the distribution. That
reasoning only holds if the reader is told. The flag sat in the artifact for a
while with nothing rendering it, so the dashboard showed a bare number and the
spread was invisible to everyone who read the report rather than the JSON.

`latency_unstable` is a different thing and stays as it was: the value is
withheld and the cell carries its own explanation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.generate_dashboard import generate_dashboard  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def result(case, fw, lat, samples=None, dispersed=False, unstable=False):
    metrics = {"client_latency_s": lat}
    if samples:
        metrics["latency_samples_s"] = samples
        lo, hi = min(samples), max(samples)
        metrics["latency_spread_pct"] = round((hi - lo) / lo * 100, 1)
    if dispersed:
        metrics["latency_dispersed"] = True
    if unstable:
        metrics["latency_unstable"] = True
    return {
        "case_id": case,
        "framework": fw,
        "model": f"org/{case}",
        "latency_s": lat,
        "metrics": metrics,
    }


run = {
    "timestamp": "2026-09-13T03:00:00+00:00",
    "commit_sha": "deadbeefcafe",
    "results": [
        # measured on 4xB200: the spread is real and the median still stands
        result("zimage", "sglang", 0.474,
               [0.695, 0.478, 0.474, 0.472, 0.473], dispersed=True),
        result("zimage", "vllm-omni", 0.487, [0.487, 0.483, 0.486, 0.491, 0.492]),
        # a tight cell must stay unmarked, or the marker means nothing
        result("qwen", "sglang", 2.587, [2.813, 2.587, 2.596, 2.576, 2.583]),
        # withheld: handled by the existing anomaly path, not by this one
        result("stalled", "vllm-omni", 55.4, [55.4, 55.5, 0.48],
               dispersed=True, unstable=True),
    ],
}

md, _ = generate_dashboard(run, history=[], charts_dir=None)
table = md.split("## Cross-Framework")[1].split("##")[0]

zimage_row = next(ln for ln in table.splitlines() if "zimage" in ln)
qwen_row = next(ln for ln in table.splitlines() if "qwen" in ln)

check("a dispersed cell is marked", "~" in zimage_row, zimage_row.strip())
check("a tight cell is not marked", "~" not in qwen_row, qwen_row.strip())
check("the median is still published", "0.47" in zimage_row)
check("the marker is explained", "spread 25% or more" in md)
check("the samples are printed", "0.695s" in md and "0.473s" in md)
check("the spread is stated", "47%" in md)
check(
    "the note names the case and framework",
    "`zimage`" in md and "**sglang**" in md,
)
check(
    "a withheld cell is not double-reported here",
    md.count("`stalled`") == 0,
    "latency_unstable has its own rendering",
)

sys.exit(fail)
