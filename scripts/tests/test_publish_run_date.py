#!/usr/bin/env python3
"""A published run is dated by its measurements, not by when it was merged.

merged.json's own timestamp is the merge time, and publish_bench_run.py used it
as the run date: re-merging the 2026-09-25 round the next morning would have
published it as 2026-09-26.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("publish_bench_run", ROOT / "scripts" / "publish_bench_run.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


merged = {
    "timestamp": "2026-09-26T01:29:05+00:00",
    "source_results": [
        {"timestamp": "2026-09-25T15:04:15+00:00"},
        {"timestamp": "2026-09-25T08:38:10+00:00"},
        {"timestamp": None},
    ],
}
got = pub._run_date(merged)
check("dated by the first measured result", got == "2026-09-25", got)
got = pub._run_date({"timestamp": "2026-09-26T01:29:05+00:00", "source_results": []})
check("falls back to the merge time when no result is dated", got == "2026-09-26", got)

sys.exit(fail)
