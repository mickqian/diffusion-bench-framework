#!/usr/bin/env python3
"""Published runs must read as a timeline: every one dated, and sorted.

The history tab strip used each run's free-text title. Those ranged from
"B200 cross-framework" to 74 characters, were shaped differently from each other
(some led with hardware, some with the topic), and carried no date -- so two
separate H200 runs were indistinguishable, and the strip gave no sign it was
chronological. Four of the nine sections had no `date` field at all, and the
strip was ordered by insertion, which ran 06-11, 05-14, 06-01, 06-10.

The label is now derived from `date` + `gpu` + `scope` rather than written by
hand, so it cannot drift again, and the descriptive title stays in the summary
line (and the tab's tooltip) where the length costs nothing.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "historical-cross-framework.json"
PAGE = ROOT / "docs" / "index.html"

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


sections = json.loads(DATA.read_text()).get("sections") or []
check("history data has sections", bool(sections), f"{len(sections)}")

undated = [s.get("id", "?")[:50] for s in sections if not s.get("date")]
check("every section carries a date", not undated, f"missing on {undated}")

bad = [
    (s.get("id", "?")[:40], s.get("date"))
    for s in sections
    if s.get("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(s["date"]))
]
check("every date is ISO yyyy-mm-dd", not bad, str(bad))

# One run may contribute several sections (single_e2e + throughput); the strip
# groups them, so check the grouped runs are what a reader ends up scanning.
runs = {}
for s in sections:
    runs.setdefault(s.get("run") or s.get("id"), s)
dates = [str(s.get("date")) for s in runs.values()]
check(
    "the runs can be ordered by date",
    all(dates),
    f"{len(runs)} run(s)",
)

page = PAGE.read_text()
check("the tab label is derived, not free text", "function historyTabLabel" in page)
check("it is date-first", 'parts = [(run.date || "").slice(0, 10)' in page)
check("the strip is sorted by date", "localeCompare" in page and "sort((a, b)" in page)
check(
    "the full title is still reachable",
    'title="${escapeHtml(run.label || "")}"' in page,
    "kept as the tab's tooltip",
)
check(
    "the summary still shows the descriptive title",
    'class="hs-title">${escapeHtml(run.label)}' in page,
)

# The inline snapshot backs file:// previews; a stale one shows different
# history than the deployed page.
inline = re.search(r'id="historicalDataInline"[^>]*>(.*?)</script>', page, re.S)
check("the inline snapshot exists", inline is not None)
if inline:
    embedded = json.loads(inline.group(1))
    check(
        "the inline snapshot matches the data file",
        len(embedded.get("sections") or []) == len(sections),
        f"{len(embedded.get('sections') or [])} vs {len(sections)}",
    )

sys.exit(fail)
