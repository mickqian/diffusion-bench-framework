#!/usr/bin/env python3
"""A GB300 box selects GB300 profiles, and the builder lints what it selects.

Hardware matching was a substring test on both sides: `b300` is a substring of
`gb300`, so a GB300 box derived the candidates [gb300, b300] and ran whichever
B300-only profile came first in dict order, while a B300 box matched any profile
named `gb300-*`. The builder linted GB classes not at all. Now a token claims its
span (a GB300 is not also a B300), a profile applies to a class only if it names
it, and the builder resolves each policy class the way the runtime resolves a
box's nvidia-smi names -- checked here case by case, framework by framework.

Needs the runtime importable (i.e. `requests` installed).
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_benchmark_config as bbc  # noqa: E402
from diffusion_bench.config_guard import hardware_candidates, select_profile  # noqa: E402
from diffusion_bench.run_comparison import (  # noqa: E402
    HARDWARE_PROFILE_ENV,
    _hardware_profile_candidates,
    _select_command_profile,
)

# What `nvidia-smi --query-gpu=name,memory.total,driver_version` prints per class.
GPU_NAMES = {
    "h100": "NVIDIA H100 80GB HBM3, 81559 MiB, 580.126.09",
    "h200": "NVIDIA H200, 143771 MiB, 580.126.09",
    "b200": "NVIDIA B200, 183359 MiB, 580.126.09",
    "b300": "NVIDIA B300 SXM6 AC, 275040 MiB, 580.95.05",
    "gb200": "NVIDIA GB200, 189471 MiB, 580.95.05",
    "gb300": "NVIDIA GB300, 284208 MiB, 580.95.05",
    "rtx5090": "NVIDIA GeForce RTX 5090, 32607 MiB, 580.65.06",
    "rtx4090": "NVIDIA GeForce RTX 4090, 24564 MiB, 580.65.06",
}
os.environ.pop(HARDWARE_PROFILE_ENV, None)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        fail = 1


def runtime_pick(profiles, hw):
    with contextlib.redirect_stdout(io.StringIO()):  # ambiguity warnings
        return _select_command_profile("fw", profiles, None, {"gpus": [GPU_NAMES[hw]] * 4})[0]


def builder_pick(profiles, hw):
    return select_profile(profiles, hardware_candidates(None, override=hw))[0]


check("every policy class has an nvidia-smi name here", set(bbc.POLICY_HARDWARE) <= set(GPU_NAMES),
      f"missing {sorted(set(bbc.POLICY_HARDWARE) - set(GPU_NAMES))}")
check("GB200 and GB300 are policy hardware", {"gb200", "gb300"} <= set(bbc.DATACENTER_HARDWARE))
for hw in ("gb300", "gb200", "b300", "b200"):
    got = _hardware_profile_candidates({"gpus": [GPU_NAMES[hw]]})
    check(f"a {hw} box derives only {hw}", got == [hw], str(got))

b_only = {"default": {}, "b300-4gpu": {"hardware": ["b300"]}}
gb_named = {"default": {}, "gb300-4gpu": {}}
both = {"default": {}, "b300-x": {"hardware": ["b300"]}, "gb300-x": {"hardware": ["gb300"]}}
for label, profiles, hw, expected in (
    ("GB300 does not run a B300-only profile", b_only, "gb300", "default"),
    ("B300 does not run a profile named gb300-*", gb_named, "b300", "default"),
    ("GB300 picks its own profile over a B300 one listed first", both, "gb300", "gb300-x"),
    ("B300 picks its own profile over a GB300 one", dict(reversed(list(both.items()))), "b300", "b300-x"),
    ("a profile naming both classes serves both", {"default": {}, "bw": {"hardware": ["b300", "gb300"]}}, "gb300", "bw"),
):
    got = runtime_pick(profiles, hw)
    check(f"runtime: {label}", got == expected, f"selected {got!r}")
    check(f"builder: {label}", builder_pick(profiles, hw) == got, f"builder {builder_pick(profiles, hw)!r}")

compared = 0
for path in sorted((ROOT / "configs" / "benchmark" / "cases").rglob("*.json")):
    if path.name.startswith("_"):
        continue
    case = json.loads(path.read_text())
    for fw, entry in case["frameworks"].items():
        profiles = entry.get("command_profiles") or {}
        if entry.get("status") != "supported" or not profiles:
            continue
        for hw in bbc.POLICY_HARDWARE:
            compared += 1
            ours, theirs = builder_pick(profiles, hw), runtime_pick(profiles, hw)
            if ours != theirs:
                check(f"{case['id']}/{fw} on {hw}", False, f"builder {ours!r}, runtime {theirs!r}")
check(f"builder and runtime agree on all {compared} (case, framework, class) selections", compared > 0)

sys.exit(fail)
