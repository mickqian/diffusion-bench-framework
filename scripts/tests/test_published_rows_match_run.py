#!/usr/bin/env python3
"""The published table must be the published run.

`docs/index.html` renders the main comparison from the artifact's top-level
`rows`; `sections` feed the history tabs. publish_bench_run.py replaced id,
title, hardware, policy and sections -- but never `rows`, so every run inherited
the previous one's table.

The 4xB200 artifact shipped that way: eighteen H100-era rows under a
"4x NVIDIA B200 183GB" heading, including five wan2.1/wan2.2 cases that run never
contained, and a Z-Image-Turbo cell attributed to `h100-h200-2gpu-tp-resident-eager`
-- a profile the B200 harness does not select. The sections underneath were
correct throughout, which is exactly why it passed review: the history tab showed
the real run while the headline table showed an older one.

So: the top-level rows must be the run's own single_e2e rows, and the inline
snapshot that backs file:// previews must agree with the data file.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "docs" / "data" / "latest-cross-framework.json"
PAGE = ROOT / "docs" / "index.html"

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def single_section(doc):
    return next(
        (s for s in (doc.get("sections") or []) if s.get("mode") == "single_e2e"), None
    )


data = json.loads(LATEST.read_text())
sec = single_section(data)
check("the artifact has a single_e2e section", sec is not None)

if sec is not None:
    rows = data.get("rows") or []
    sec_ids = [r.get("case_id") for r in sec.get("rows") or []]
    row_ids = [r.get("case_id") for r in rows]
    check(
        "the table has the run's cases, and only those",
        row_ids == sec_ids,
        f"table has {sorted(set(row_ids) - set(sec_ids))} that the run does not",
    )

    # A case can match by name while carrying a different measurement, which is
    # how a stale Z-Image cell survived: same case_id, older number and profile.
    def cell_sig(rows_):
        out = {}
        for r in rows_:
            for fw, c in (r.get("cells") or {}).items():
                out[(r["case_id"], fw)] = (
                    c.get("latency_s", c.get("client_latency_s")),
                    c.get("profile"),
                    c.get("status"),
                )
        return out

    mismatched = [
        k
        for k, v in cell_sig(sec["rows"]).items()
        if cell_sig(rows).get(k) != v
    ]
    check(
        "every published cell matches the run's own",
        not mismatched,
        f"{len(mismatched)} differ, e.g. {mismatched[:3]}",
    )

    profiles = {
        c.get("profile")
        for r in rows
        for c in (r.get("cells") or {}).values()
        if c.get("profile")
    }
    label = str((data.get("hardware") or {}).get("label", "")).lower()
    if "b200" in label:
        stale_hw = sorted(p for p in profiles if "h100" in p or "h200" in p)
        check(
            "no h100/h200 profile is published under a B200 heading",
            not stale_hw,
            str(stale_hw),
        )

    # The headline block is top-level too, and was inherited the same way: it
    # claimed "18 cases ... on H100" with 14 wins over a 15-case B200 run.
    summary = data.get("summary") or {}
    measured = [
        r
        for r in rows
        if len(
            [
                1
                for c in (r.get("cells") or {}).values()
                if c.get("status") == "ok" and c.get("latency_s") is not None
            ]
        )
        >= 2
    ]
    real_wins = sum(1 for r in measured if r.get("winner") == "sglang")
    check(
        "the headline counts match the table",
        (summary.get("cases"), summary.get("comparable_rows"),
         summary.get("sglang_diffusion_wins"))
        == (len(rows), len(measured), real_wins),
        f"{summary.get('cases')}/{summary.get('comparable_rows')}/"
        f"{summary.get('sglang_diffusion_wins')} vs "
        f"{len(rows)}/{len(measured)}/{real_wins}",
    )
    note = str(summary.get("note", "")).lower()
    if "b200" in label:
        check(
            "the headline does not describe other hardware",
            "h100" not in note and "h200" not in note,
            note[:90],
        )

page = PAGE.read_text()
m = re.search(r'id="latestDataInline"[^>]*>(.*?)</script>', page, re.S)
check("the inline snapshot exists", m is not None)
if m:
    inline = json.loads(m.group(1))
    check(
        "the inline snapshot's table matches the data file",
        [r.get("case_id") for r in (inline.get("rows") or [])]
        == [r.get("case_id") for r in (data.get("rows") or [])],
        "file:// previews would show a different run",
    )

sys.exit(fail)
