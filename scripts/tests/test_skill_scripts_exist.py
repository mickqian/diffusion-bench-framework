#!/usr/bin/env python3
"""Every path in THIS repo that a skill names must exist.

The skill is what a session reads at the start of a round, and its "Canonical
Scripts For A Round" table is there so the work is not improvised again. A
reference that has been renamed or never landed sends the next session hunting,
and it rots silently -- nothing else reads that table.

Scope is this repo's own trees. The skills also name paths in the sglang repo
(`tools/tune_dit_tp_plan.py`, the offline DiT-TP-plan tuner from sglang #30004);
those are correct references to a different checkout, and a test here cannot
resolve them.

Executability is deliberately NOT checked: everything in this repo is invoked as
`bash scripts/foo.sh`, and the core `rxrun.sh` / `rxpull.sh` have never carried
the bit. Asserting a convention the repo does not follow just produces noise.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Trees this repo owns. A named path outside them belongs to another checkout.
OWNED = ("scripts/", "src/", "configs/", "docs/", ".claude/")
SKILLS = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))

fail = 0


def check(name, cond, detail=""):
    global fail
    suffix = f" -- {detail}" if detail and not cond else ""
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{suffix}")
    if not cond:
        fail = 1


check("skill files found", bool(SKILLS), f"found {len(SKILLS)}")

checked = 0
for skill in SKILLS:
    text = skill.read_text()
    # Only paths written as code spans; prose mentions are not references.
    paths = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:sh|py|json|md))", text)))
    for rel in paths:
        if not rel.startswith(OWNED):
            continue
        checked += 1
        check(f"{skill.parent.name}: {rel}", (ROOT / rel).exists(), "does not exist")

check("the skills reference this repo at all", checked > 0, f"{checked} path(s)")

# The canonical-scripts table is the entry point for a round: it must still be
# there and still name the runner, the tuner, the publisher and the tests.
bench = ROOT / ".claude" / "skills" / "diffusion-framework-benchmarking" / "SKILL.md"
check("the benchmarking skill exists", bench.exists())
if bench.exists():
    text = bench.read_text()
    check("the canonical-scripts table exists", "## Canonical Scripts For A Round" in text)
    for expected in (
        "scripts/install_comparison_frameworks.sh",
        "scripts/upgrade_framework_stack.sh",
        "scripts/tune_sglang_serve_args.sh",
        "scripts/merge_and_publish_run.sh",
        "scripts/tests/run_all.sh",
    ):
        check(f"it still names {expected}", expected in text)

sys.exit(fail)
