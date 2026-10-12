#!/usr/bin/env python3
"""One bench round reads in one place: its hardware share a history tab.

A round runs on several machines (2026-10-10: 4xGB300, an RTX 5090 desktop),
and each published run used to get its own tab, so the round was split across
the strip by hardware. `--round` tags every section of a run; the page groups
sections by round, lists all the round's GPUs on the tab, and names the
hardware on each table inside it.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "index.html"
spec = importlib.util.spec_from_file_location("publish_bench_run", ROOT / "scripts" / "publish_bench_run.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


(ROOT / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as d:
    hist, latest = Path(d) / "historical.json", Path(d) / "latest.json"
    shutil.copy(ROOT / "docs/data/historical-cross-framework.json", hist)
    shutil.copy(ROOT / "docs/data/latest-cross-framework.json", latest)
    argv = [
        "publish_bench_run.py",
        "--merged", str(ROOT / "reports/b200x4-full-20260925/merged.json"),
        "--run-id", "round-probe-run",
        "--label", "probe",
        "--gpu", "1x NVIDIA GeForce RTX 5090 32GB",
        "--reproduce", "scripts/run_b200_cross_framework_20260912.sh",
        "--round", "round-probe",
        "--round-label", "Round probe",
        "--keep-latest",
    ]
    with (
        patch.object(pub, "HISTORICAL", hist),
        patch.object(pub, "LATEST", latest),
        patch.object(pub.subprocess, "run"),
        patch.object(sys, "argv", argv),
    ):
        check("publish --round exits 0", pub.main() == 0)
    mine = [s for s in json.loads(hist.read_text())["sections"] if s.get("run") == "round-probe-run"]
    check("every section of the run carries the round", bool(mine) and all(s.get("round") == "round-probe" for s in mine))
    check("and the round's label", all(s.get("round_label") == "Round probe" for s in mine))


def js_function(source: str, name: str) -> str:
    m = re.search(rf"\n( *)function {name}\(.*?\n\1}}\n", source, re.S)
    assert m, name
    return m.group(0)


page = PAGE.read_text()
node = shutil.which("node")
if node is None:
    print("  skip  node not installed: the page functions are not executed")
else:
    script = "\n".join(js_function(page, n) for n in ("shortGpu", "historyTabLabel", "historyRuns")) + """
const data = {sections: [
  {run: "gb", round: "r1", round_label: "Round 1", date: "2026-10-10", gpu: "4x NVIDIA GB300 288GB", mode: "single_e2e"},
  {run: "gb", round: "r1", round_label: "Round 1", date: "2026-10-10", gpu: "4x NVIDIA GB300 288GB", mode: "throughput"},
  {run: "pc", round: "r1", round_label: "Round 1", date: "2026-10-10", gpu: "1x NVIDIA GeForce RTX 5090 32GB", mode: "single_e2e"},
  {run: "old", date: "2026-09-25", gpu: "4x NVIDIA B200 183GB", mode: "single_e2e"},
]};
const runs = historyRuns(data);
console.log(JSON.stringify({n: runs.length, modes: runs[0].modes.length, gpus: runs[0].gpus, label: runs[0].label, tab: historyTabLabel(runs[0]), old: historyTabLabel(runs[1])}));
"""
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    check("a round's runs share one tab", out["n"] == 2 and out["modes"] == 3, str(out))
    check("the tab lists every GPU of the round", out["tab"] == "2026-10-10 · 4×GB300 + 1×RTX 5090", out["tab"])
    check("the tab is titled by the round", out["label"] == "Round 1", out["label"])
    check("a run without a round keeps its own tab", out["old"] == "2026-09-25 · 4×B200", out["old"])

check("each table in a mixed tab names its hardware", "a round tab holds every hardware it ran on" in page)

sys.exit(fail)
