#!/usr/bin/env python3
"""The generated config must be the one the sources currently produce.

`configs/benchmark/cases/*.json` is the source of truth; the harness reads the
generated `configs/comparison_configs.json`. Edit a case and forget to re-run
`scripts/build_benchmark_config.py` and the two drift -- with every existing
check still green, because `test_config_copies_in_sync.py` compares the two
GENERATED copies against each other, and they are equally stale.

That drift is not cosmetic. A case renamed its request field from
`request_overrides` to `request_extra` -- the name the harness actually reads --
and without a rebuild the stale config kept the dead key, so the benchmark sent
requests with no `output_format`, the RGBA checkpoint's encoder refused JPEG,
and four GPUs' worth of runs came back as a bare HTTP 500 that looked like a
model problem.

So rebuild from the sources into a scratch copy and require the result to match
what is committed, byte for byte after JSON normalisation.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATED = "configs/comparison_configs.json"

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          f"{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        fail = 1


with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    # The build resolves every path from its own __file__, so a copy that keeps
    # the layout rebuilds in isolation and touches nothing in the repo.
    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "build_benchmark_config.py", tmp / "scripts")
    shutil.copytree(ROOT / "configs" / "benchmark", tmp / "configs" / "benchmark")
    shutil.copytree(ROOT / "src", tmp / "src")

    proc = subprocess.run(
        [sys.executable, str(tmp / "scripts" / "build_benchmark_config.py")],
        capture_output=True, text=True,
    )
    check("the sources build cleanly", proc.returncode == 0,
          (proc.stdout + proc.stderr).strip()[-300:])

    if proc.returncode == 0:
        fresh = json.loads((tmp / GENERATED).read_text())
        committed = json.loads((ROOT / GENERATED).read_text())
        same = fresh == committed
        detail = ""
        if not same:
            fresh_cases = {c["id"]: c for c in fresh.get("cases", [])}
            comm_cases = {c["id"]: c for c in committed.get("cases", [])}
            drifted = sorted(
                cid for cid in set(fresh_cases) | set(comm_cases)
                if fresh_cases.get(cid) != comm_cases.get(cid)
            )
            detail = (f"run scripts/build_benchmark_config.py and commit; "
                      f"cases that differ: {drifted or 'none (top-level fields)'}")
        check("the committed config is what the sources produce", same, detail)

sys.exit(fail)
