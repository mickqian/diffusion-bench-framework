#!/usr/bin/env python3
"""Publish a finished benchmark run to the Pages data files.

    python3 scripts/publish_bench_run.py \
        --merged tmp/report/merged.json \
        --run-id h200x2-fair-20260819 \
        --label "H200 cross-framework (latest-vs-latest)" \
        --gpu "2x NVIDIA H200 143GB"

Appends the run's sections to docs/data/historical-cross-framework.json, points
docs/data/latest-cross-framework.json at it, and refreshes the inline snapshots
that back file:// previews. Re-publishing the same --run-id replaces its
sections rather than appending duplicates, so a re-run is idempotent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date as date_cls
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.page_export import FRAMEWORK_LABELS, build_sections  # noqa: E402

HISTORICAL = ROOT / "docs" / "data" / "historical-cross-framework.json"
LATEST = ROOT / "docs" / "data" / "latest-cross-framework.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", required=True, type=Path, help="merged artifact from build_report_artifacts")
    ap.add_argument("--config", default=ROOT / "configs" / "comparison_configs.json", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument(
        "--date",
        default=None,
        help="run date; defaults to the run's own timestamp, not today",
    )
    ap.add_argument("--reproduce", default="scripts/biweekly_fair_bench.sh")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    merged = _load(args.merged)
    config = _load(args.config)
    # Publishing days later must not relabel when the run happened.
    run_date = args.date or (merged.get("timestamp") or "")[:10] or date_cls.today().isoformat()
    sections = build_sections(
        merged,
        config,
        run_id=args.run_id,
        run_label=args.label,
        gpu=args.gpu,
        date=run_date,
        reproduce=args.reproduce,
    )
    if not sections:
        print("error: run produced no publishable sections", file=sys.stderr)
        return 1

    for sec in sections:
        measured = sum(
            1
            for r in sec["rows"]
            for c in r["cells"].values()
            if "client_latency_s" in c or "qps" in c
        )
        print(f"{sec['id']}: {len(sec['rows'])} rows, {measured} measured cells")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    hist = _load(HISTORICAL)
    # bench2Html labels frameworks from data.frameworks; a framework missing
    # there renders as its raw key ("trtllm-visual" appeared verbatim on screen).
    # data.frameworks maps key -> display label (flat strings, not objects)
    fwmap = hist.setdefault("frameworks", {})
    for key, label in FRAMEWORK_LABELS.items():
        fwmap.setdefault(key, label)
    keep = [s for s in hist["sections"] if s.get("run") != args.run_id]
    dropped = len(hist["sections"]) - len(keep)
    if dropped:
        print(f"replacing {dropped} existing section(s) for run {args.run_id}")
    # newest first: the page shows sections in file order
    hist["sections"] = sections + keep
    hist["updated_at"] = date_cls.today().isoformat()
    _dump(HISTORICAL, hist)
    print(f"wrote {HISTORICAL.relative_to(ROOT)} ({len(hist['sections'])} sections)")

    if LATEST.exists():
        latest = _load(LATEST)
        # carry the run's identity across too; replacing only `sections` left an
        # H200 run described by the previous H100 run's id, title and hardware.
        latest["id"] = args.run_id
        latest["title"] = args.label
        latest["hardware"] = {"label": args.gpu}
        latest["frameworks"] = dict(FRAMEWORK_LABELS)
        latest["sections"] = sections
        latest["updated_at"] = date_cls.today().isoformat()
        _dump(LATEST, latest)
        print(f"wrote {LATEST.relative_to(ROOT)}")

    # Guard the mistake that published a whole run as "n/a": the live renderer
    # (bnxVal) ignores any cell without status == "ok".
    bad = [
        f"{sec['id']}/{row['case_id']}/{fw}"
        for sec in sections
        for row in sec["rows"]
        for fw, cell in row["cells"].items()
        if ("latency_s" in cell or "qps" in cell) and cell.get("status") != "ok"
    ]
    if bad:
        print(f"error: {len(bad)} measured cell(s) lack status 'ok' and would render as n/a: {bad[:3]}", file=sys.stderr)
        return 1

    subprocess.run([sys.executable, str(ROOT / "scripts" / "refresh_docs_data.py")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
