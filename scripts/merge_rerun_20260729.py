#!/usr/bin/env python3
"""Fold the 2026-07-29 clean-environment reruns into the h100 report's merged.json.

The reruns replace cells whose original numbers were invalidated by
shared-devbox contamination (wan22_i2v load "failure", wan22_t2v throughput
OOM) or measured under a different machine power state (wan21_i2v throughput,
re-measured same-window ABAB against lightx2v).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "h100x4-full-20260728"
RERUN = REPORT / "raw-rerun-20260729"
MERGED = REPORT / "merged.json"


def main() -> int:
    merged = json.loads(MERGED.read_text())
    replaced = []
    for path in sorted(RERUN.glob("*.json")):
        run = json.loads(path.read_text())
        for section in ("results", "throughput_results"):
            for entry in run.get(section, []):
                key = (entry["case_id"], entry["framework"], entry["mode"])
                rows = merged[section]
                idx = [
                    i
                    for i, r in enumerate(rows)
                    if (r["case_id"], r["framework"], r["mode"]) == key
                ]
                if idx:
                    rows[idx[0]] = entry
                    replaced.append((path.name, section, key, "replaced"))
                else:
                    rows.append(entry)
                    replaced.append((path.name, section, key, "appended"))
        merged["source_results"].append(
            {
                "path": f"raw-rerun-20260729/{path.name}",
                "run_id": run.get("run_id"),
                "commit_sha": run.get("commit_sha"),
                "timestamp": run.get("timestamp"),
            }
        )
    MERGED.write_text(json.dumps(merged, indent=1) + "\n")
    for r in replaced:
        print(*r)
    print(f"total: {len(replaced)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
