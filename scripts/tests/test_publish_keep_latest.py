#!/usr/bin/env python3
"""--keep-latest adds a run to the history without taking over the headline.

publish_bench_run.py always pointed latest-cross-framework.json at the run it
published, so a side run (one RTX 5090 desktop, 2026-10-11) could only be
published by replacing the 4xGB300 round as the page's latest table.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def publish(work: Path, *extra: str) -> tuple[dict, dict]:
    hist, latest = work / "historical.json", work / "latest.json"
    shutil.copy(ROOT / "docs/data/historical-cross-framework.json", hist)
    shutil.copy(ROOT / "docs/data/latest-cross-framework.json", latest)
    argv = [
        "publish_bench_run.py",
        "--merged", str(ROOT / "reports/b200x4-full-20260925/merged.json"),
        "--run-id", "keep-latest-probe",
        "--label", "probe",
        "--gpu", "probe GPU",
        "--reproduce", "scripts/run_b200_cross_framework_20260912.sh",
        *extra,
    ]
    with (
        patch.object(pub, "HISTORICAL", hist),
        patch.object(pub, "LATEST", latest),
        patch.object(pub.subprocess, "run"),
        patch.object(sys, "argv", argv),
    ):
        rc = pub.main()
    check(f"publish {' '.join(extra) or '(default)'} exits 0", rc == 0, str(rc))
    return json.loads(hist.read_text()), json.loads(latest.read_text())


(ROOT / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as d:
    work = Path(d)
    before = json.loads((ROOT / "docs/data/latest-cross-framework.json").read_text())

    hist, latest = publish(work, "--keep-latest")
    check("history gains the run", any(s.get("run") == "keep-latest-probe" for s in hist["sections"]))
    check("latest keeps its run", latest == before, latest.get("id"))

    hist, latest = publish(work)
    check("without the flag latest moves to the run", latest.get("id") == "keep-latest-probe", latest.get("id"))

sys.exit(fail)
