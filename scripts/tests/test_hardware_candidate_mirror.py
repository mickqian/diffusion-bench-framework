#!/usr/bin/env python3
"""`config_guard.hardware_candidates` must agree with the runtime, exactly.

The runtime does not match profiles against the `--hardware-profile` string: it
joins the override, env, gpu_config, runner_labels and GPU names into one blob
and looks for known hardware tokens. So `--hardware-profile blackwell` on B200s
yields `b200`, and "blackwell" contributes nothing by itself.

config_guard used only the override string, so the command-drift check compared
published rows against a different profile than the one that ran -- it reported
every minimax row as drift because that case's Blackwell profile is named
`b200-2gpu`. False drift is worse than none: it trains you to ignore the check,
and `DIFFUSION_BENCH_STRICT_COMMANDS=1` turns it into a refusal to publish a
correct run.

Needs the runtime importable (i.e. `requests` installed), so it runs on a bench
box rather than a laptop.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.config_guard import hardware_candidates as mirror  # noqa: E402
from diffusion_bench.run_comparison import (  # noqa: E402
    _hardware_profile_candidates as runtime,
)

CASES = [
    {"hardware_profile_override": "blackwell",
     "gpus": ["NVIDIA B200, 183359 MiB, 580.126.09"] * 4},
    {"hardware_profile_override": "h200", "gpus": ["NVIDIA H200, 143771 MiB"] * 2},
    {"hardware_profile_override": None, "gpus": ["NVIDIA GeForce RTX 5090"]},
    {"gpu_config": "h100x4", "gpus": []},
    {"runner_labels": "gb200-cluster", "gpus": []},
    {"hardware_profile_override": "b300", "gpus": ["NVIDIA B300"]},
    {},
]

fail = 0
for metadata in CASES:
    expected = runtime(metadata)
    got = mirror(metadata, override=metadata.get("hardware_profile_override"))
    ok = expected == got
    if not ok:
        fail = 1
    print(f"  {'ok  ' if ok else 'FAIL'} runtime={expected} mirror={got}")

# The whole point: a blackwell override on B200s must yield a token that
# matches a profile named `b200-2gpu`, not the literal string "blackwell".
blackwell = mirror(
    {"hardware_profile_override": "blackwell", "gpus": ["NVIDIA B200"]},
    override="blackwell",
)
if blackwell != ["b200"]:
    print(f"  FAIL blackwell override on B200s -> {blackwell}, expected ['b200']")
    fail = 1
else:
    print("  ok   blackwell override on B200s resolves to ['b200']")

sys.exit(fail)
