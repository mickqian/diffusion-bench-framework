#!/usr/bin/env python3
"""The B300 A/B script's arms exist, never auto-select, and WAVE=4 runs exactly its section.

scripts/run_b300_ab_20261022.sh names each arm by profile; a typo (or a profile
that was never added) surfaces on the box as an "Unknown ... command profile"
after a server start, per arm. A candidate whose name carries a hardware token is
auto-selected on that hardware before any A/B has measured it (h200-cudnn-vsa-
candidate, 2026-10-22). And AB_CASES, a regex on "<case> <framework>", cannot
select one wave's line when another wave has a line for the same cell -- the third
and fourth waves both run `minimax_h3_ref2va_5s trtllm-visual` -- so waves are
selected with WAVE. This dry-runs the script (AB_DRY_RUN=1) and checks the plan.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.config_guard import (  # noqa: E402
    HARDWARE_TOKENS,
    hardware_candidates,
    hardware_tokens,
    resolve_status,
    select_profile,
)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        fail = 1


SCRIPT = ROOT / "scripts" / "run_b300_ab_20261022.sh"
HW = "b300"
cases = {}
for path in (ROOT / "configs" / "benchmark" / "cases").rglob("*.json"):
    if not path.name.startswith("_"):
        case = json.loads(path.read_text())
        cases[case["id"]] = case


def plan(**env):
    out = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "AB_DRY_RUN": "1", "DBF_REPO_DIR": str(ROOT), **env},
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        m = re.fullmatch(r"PLAN wave=(\d+) port=(\d+) (\S+) (\S+) rounds=(\d+) arms=(.*)", line)
        if m:
            rows.append((int(m[1]), int(m[2]), m[3], m[4], int(m[5]), m[6].split()))
    return rows


every = plan()
wave4 = plan(WAVE="4")
lines = [ln.split() for ln in SCRIPT.read_text().splitlines() if ln.startswith("arms ")]
check("the dry run lists every arms line", len(every) == len(lines), f"{len(every)} vs {len(lines)}")
check("WAVE=4 lists only fourth-wave lines", wave4 and {r[0] for r in wave4} == {4}, {r[0] for r in wave4})
check("WAVE=4 lists every fourth-wave line", wave4 == [r for r in every if r[0] == 4])
check("ports do not depend on the selection, and stay 100 apart",
      [r[1] for r in every] == list(range(every[0][1], every[0][1] + 100 * len(every), 100)))
check("the last line's ports stay below 65535", every[-1][1] + 100 < 65535, every[-1][1])
narrowed = plan(WAVE="4", AB_CASES=" sglang$")
check("AB_CASES narrows inside WAVE", narrowed == [r for r in wave4 if r[3] == "sglang"])

for _, _, cid, fw, rounds, arms in every:
    profiles = (cases[cid]["frameworks"][fw].get("command_profiles") or {})
    missing = [a for a in arms if a != HW and a not in profiles]
    check(f"{cid} {fw}: every arm names a profile ({' '.join(arms)})", not missing, missing)

for wave, _, cid, fw, rounds, arms in wave4:
    case = cases[cid]
    entry = case["frameworks"][fw]
    image = case["task"] in ("text-to-image", "image-edit", "image-to-image")
    check(f"{cid} {fw}: the selected profile ({HW}) is an arm, rounds {2 if image else 1}",
          arms[0] == HW and rounds == (2 if image else 1), f"{arms} rounds={rounds}")
    status, _ = resolve_status(entry, hardware_candidates(None, override=HW))
    check(f"{cid} {fw}: runnable on {HW}", status == "supported", status)
    for arm in arms[1:]:
        prof = entry["command_profiles"][arm]
        if "wave 4" not in prof.get("notes", ""):
            continue  # an earlier wave's candidate, re-run here
        check(
            f"{cid} {fw} {arm}: a cand-* name without a hardware token, no hardware list, marked pending",
            arm.startswith("cand-") and not hardware_tokens(arm) and "hardware" not in prof
            and prof["description"].endswith("Never selected on its own: A/B arm.")
            and "VALIDATION PENDING (wave 4, 4xB300)" in prof["notes"],
            prof.get("description", ""),
        )
        if fw != "sglang":
            check(f"{cid} {fw} {arm}: carries an upstream_recipe", bool(prof.get("upstream_recipe")))

auto = []
for cid, case in cases.items():
    for fw, entry in case["frameworks"].items():
        profiles = entry.get("command_profiles") or {}
        for hw in HARDWARE_TOKENS:
            name, _, _ = select_profile(profiles, hardware_candidates(None, override=hw))
            if name and name.startswith("cand-"):
                auto.append(f"{cid}/{fw} on {hw}: {name}")
check("no cand-* profile is ever auto-selected", not auto, auto[:5])
sys.exit(fail)
