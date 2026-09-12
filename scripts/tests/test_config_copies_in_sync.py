#!/usr/bin/env python3
"""The editable and packaged benchmark configs must be committed together.

`scripts/build_benchmark_config.py` writes both copies, so a working tree is
always consistent -- which is exactly why this is easy to miss: `git add
configs/` stages only the editable one, and the packaged copy (what an install
of the package actually reads) silently falls behind. Eight consecutive commits
did that before anyone noticed.

So compare the two as COMMITTED, not as they sit on disk.
"""
import json
import subprocess
import sys

EDITABLE = "configs/comparison_configs.json"
PACKAGED = "src/diffusion_bench/comparison_configs.json"


def at_head(path: str) -> dict | None:
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return json.loads(out)


a, b = at_head(EDITABLE), at_head(PACKAGED)
if a is None or b is None:
    print(f"FAIL  missing at HEAD: {EDITABLE if a is None else PACKAGED}")
    sys.exit(1)

if a == b:
    cases = len(a.get("cases", []))
    print(f"ok    editable and packaged configs match at HEAD ({cases} cases)")
    sys.exit(0)

print("FAIL  the committed configs differ -- the packaged copy is stale")
ids_a = [c.get("id") for c in a.get("cases", [])]
ids_b = [c.get("id") for c in b.get("cases", [])]
if ids_a != ids_b:
    print(f"      case ids differ: only editable={sorted(set(ids_a) - set(ids_b))} "
          f"only packaged={sorted(set(ids_b) - set(ids_a))}")
else:
    differing = [
        cid for cid, ca, cb in zip(ids_a, a["cases"], b["cases"]) if ca != cb
    ]
    print(f"      same cases, differing content: {differing[:8]}")
print(f"      fix: python3 scripts/build_benchmark_config.py && git add {EDITABLE} {PACKAGED}")
sys.exit(1)
