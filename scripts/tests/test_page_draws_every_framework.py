#!/usr/bin/env python3
"""Every framework the data publishes must be drawn by the page.

The 2026-09-25 run was the first with ComfyUI. The data files carried its cells
and version, but the page draws from hardcoded framework lists (BNX_ORDER for the
charts, FRAMEWORK_META / FRAMEWORK_RANK for the latest table) and the latest
file's `framework_order`, which publishing inherited from the previous run. The
charts showed four frameworks, and nothing failed.

Also checks the footer's `source_report`, which kept naming the July H100 report
under every later run.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench.page_export import FRAMEWORK_LABELS  # noqa: E402

PAGE = (ROOT / "docs" / "index.html").read_text()
LATEST = json.loads((ROOT / "docs" / "data" / "latest-cross-framework.json").read_text())

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def js_block(name):
    m = re.search(rf"const {name} = (\[.*?\]|\{{.*?\}});", PAGE, re.S)
    return m.group(1) if m else ""


def js_keys(name):
    return {a or b for a, b in re.findall(r'(?:"([\w-]+)"|\b(\w+))\s*:', js_block(name))}


def js_list(name):
    return re.findall(r'"([\w-]+)"', js_block(name))


frameworks = list(FRAMEWORK_LABELS)
for name, have in (
    ("BNX_ORDER", set(js_list("BNX_ORDER"))),
    ("FRAMEWORK_RANK", set(js_list("FRAMEWORK_RANK"))),
    ("known", set(js_list("known"))),
    ("BNX_SHORT", js_keys("BNX_SHORT")),
    ("BNX_COLOR", js_keys("BNX_COLOR")),
    ("FRAMEWORK_META", js_keys("FRAMEWORK_META")),
    ("frameworkLabels", js_keys("frameworkLabels")),
):
    missing = [f for f in frameworks if f not in have]
    check(f"the page's {name} lists every framework", not missing, f"missing {missing}")

bars = re.findall(r'bar: "([\w-]+)"', js_block("FRAMEWORK_META"))
undefined = [b for b in bars if len(re.findall(rf"--bar-{b}:", PAGE)) < 3]
check("every framework colour is defined for light and both dark themes", not undefined, f"{undefined}")

in_rows = {fw for row in LATEST.get("rows") or [] for fw in (row.get("cells") or {})}
order = LATEST.get("framework_order") or []
check(
    "the latest table's framework_order covers every framework in its rows",
    in_rows <= set(order),
    f"rows have {sorted(in_rows - set(order))} that framework_order drops",
)

src = str(LATEST.get("source_report") or "")
report = ROOT / src
ids = {report.name}
if (report / "manifest.json").exists():
    ids.add(json.loads((report / "manifest.json").read_text()).get("id"))
check(
    "the footer's source report is this run's",
    (report.is_dir() and LATEST.get("id") in ids) if src.startswith("reports/") else src == LATEST.get("id"),
    f"{src!r} for run {LATEST.get('id')!r}",
)

sys.exit(fail)
