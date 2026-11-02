#!/usr/bin/env python3
"""The command-drift gate checks a row against the profile that actually ran.

An explicitly requested profile (SGLANG_BENCH_SGLANG_PROFILE=dp2-throughput for
MiniMax-H3 throughput) has no `hardware` key on purpose, so hardware selection
never picks it. The gate compared every row with the hardware pick and refused
the H3 throughput row of the 2026-11-02 B300 supplement as drift. An explicit
row is now checked against its named profile -- which must still exist and
whose serve_args the run must still carry -- and hardware-selected rows keep the
old check.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench.config_guard import verify_merged_commands  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


CONFIG = {
    "cases": [
        {
            "id": "h3",
            "frameworks": {
                "sglang": {
                    "command_profiles": {
                        "b300-noexp": {"hardware": ["b300"], "serve_args": "--ulysses-degree 2"},
                        "dp2-throughput": {"serve_args": "--dp-size 2"},
                    }
                }
            },
        }
    ]
}


def drift(profile, source, command):
    row = {
        "case_id": "h3",
        "framework": "sglang",
        "server_command": command,
        "framework_metadata": {"profile": profile, "profile_source": source},
    }
    return verify_merged_commands({"throughput_results": [row]}, CONFIG, hardware=["b300"])


for label, args, expect_drift in (
    ("explicit row matching its profile passes", ("dp2-throughput", "explicit", "serve --dp-size 2"), False),
    ("explicit row missing its profile's args drifts", ("dp2-throughput", "explicit", "serve --ulysses-degree 2"), True),
    ("explicit row naming a removed profile drifts", ("dp4-throughput", "explicit", "serve --dp-size 4"), True),
    ("hardware row on the hardware pick passes", ("b300-noexp", "hardware", "serve --ulysses-degree 2"), False),
    ("hardware row off the hardware pick drifts", ("dp2-throughput", "hardware", "serve --dp-size 2"), True),
):
    got = drift(*args)
    check(label, bool(got) == expect_drift, f"warnings {got}")

sys.exit(fail)
