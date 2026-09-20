#!/usr/bin/env python3
"""A step detector has to be scaled to each case's own noise, not to a percentage.

Two live regressions sat on the nightly dashboard for a month uncalled:
cosmos3_super +3.5% and minimax_h3 +1.3%. A sweep with a ">= 7% is a step" rule
skipped both, and the dashboard's own run-over-run indicator showed "0.1%"
because the step had happened weeks earlier. Meanwhile zimage swings 14% between
two modes every few nights and means nothing.

The three properties that separate those cases are pinned here:

* a small step on a quiet case IS a finding -- 3.5% against 0.1% scatter is a
  hundred-sigma move;
* a large swing on a bimodal case is NOT -- the two levels interleave, so no
  single split separates them;
* a split whose two sides ran on different runner pools is suspect, because
  `h100-novita-temp-*` is 20-25% slower than its siblings and manufactures
  step-shaped artifacts on its own.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETECT = ROOT / "scripts" / "detect_nightly_steps.py"

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def make_runs(cases: dict[str, list[tuple[float, str]]]) -> dict:
    """cases: case_id -> [(latency, runner), ...] in nightly order."""
    n = max(len(v) for v in cases.values())
    runs = []
    for i in range(n):
        results = []
        for case, points in cases.items():
            if i < len(points):
                lat, runner = points[i]
                results.append(
                    {"case_id": case, "framework": "sglang", "latency_s": lat}
                )
        runs.append(
            {
                "timestamp": f"2026-08-{i + 1:02d}T00:00:00",
                "commit_sha": f"{i:09d}abc",
                "runner_name": next(iter(cases.values()))[min(i, n - 1)][1],
                "results": results,
            }
        )
    return {"runs": runs}


def run_detector(payload: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    out = subprocess.run(
        [sys.executable, str(DETECT), "--data", path],
        capture_output=True, text=True,
    )
    return out.stdout + out.stderr


POOL_A = "h100-novita10-gpu-0123"
POOL_B = "h100-novita-temp-gpu-4567"

# 1. cosmos3's shape: eleven points at 115.3 (0.1% scatter), then nineteen at 119.4.
quiet_step = [(115.30 + (i % 3) * 0.01, POOL_A) for i in range(11)] + [
    (119.38 + (i % 3) * 0.02, POOL_A) for i in range(19)
]
# 2. zimage's shape: two levels that interleave rather than step.
bimodal = [
    (0.47 if i % 2 else 0.68, POOL_A) for i in range(30)
]
# 3. a step that coincides exactly with a pool change.
pool_step = [(100.0 + (i % 3) * 0.05, POOL_A) for i in range(12)] + [
    (124.0 + (i % 3) * 0.05, POOL_B) for i in range(12)
]

out = run_detector(make_runs({"quiet_step_case": quiet_step}))
print(out.rstrip())
check("a 3.5% step on a quiet case is reported", "quiet_step_case" in out, out.strip()[:90])
check("and is called a regression", "REGRESSION" in out)

out = run_detector(make_runs({"bimodal_case": bimodal}))
check(
    "an interleaved bimodal case is not reported as a step",
    "bimodal_case" not in out,
    out.strip()[:90],
)

out = run_detector(make_runs({"pool_step_case": pool_step}))
check(
    "a step that coincides with a pool change is flagged suspect",
    "pool_step_case" not in out or "suspect" in out,
    out.strip()[:110],
)

# 4. against the real data, the two known live regressions must both appear.
real = run_detector(json.loads((ROOT / "docs" / "nightly-data.json").read_text()))
for case in ("cosmos3_super_t2v_2gpu", "minimax_h3_t2va_5s"):
    line = next((ln for ln in real.splitlines() if case in ln), "")
    check(f"{case} is caught on the real data", "REGRESSION" in line, line.strip()[:100])

sys.exit(fail)
