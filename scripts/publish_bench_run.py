#!/usr/bin/env python3
"""Publish a finished benchmark run to the Pages data files.

    python3 scripts/publish_bench_run.py \
        --merged tmp/report/merged.json \
        --run-id h200x2-fair-20260819 \
        --label "H200 cross-framework (latest-vs-latest)" \
        --gpu "2x NVIDIA H200 143GB" \
        --reproduce scripts/biweekly_fair_bench.sh

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


def _run_date(merged: dict) -> str:
    """The day the run's first result was measured.

    merged["timestamp"] is when the artifact was merged, so dating by it
    relabelled a run every time it was re-merged.
    """
    started = min(
        (s["timestamp"] for s in merged.get("source_results") or [] if s.get("timestamp")),
        default=merged.get("timestamp") or "",
    )
    return started[:10] or date_cls.today().isoformat()


def _policy_block(merged: dict, args) -> dict:
    """What this run actually did, rebuilt per publication.

    Only the invariants are hardcoded; anything run-specific is derived or
    passed in with --note, so a caveat cannot outlive the run it describes.
    """
    compile_off = bool(merged.get("torch_compile_disabled"))
    policy = {
        "latency_source": (
            "client-side wall clock, steady-state median of back-to-back "
            "measured requests after warmup; server-side timers kept only as "
            "per-framework diagnostics"
        ),
        "selection": (
            "best lossless profile per case for the hardware this ran on "
            "(see each row's profile; configs/benchmark/SELECTED.md lists what "
            "the harness selects)"
        ),
        "cache": "no response cache, no Cache-DiT, no quantized checkpoints",
        "torch_compile": (
            "OFF for every framework"
            if compile_off
            else "ON for the competitors; sglang runs compile-off by policy "
            "(its fused kernels match or beat compiler fusion, measured)"
        ),
        "attention": (
            "each framework runs its fastest exact-precision attention; no "
            "quantized or approximate substitution"
        ),
        "version_policy": (
            "latest-vs-latest: sglang runs origin/main HEAD, every competitor "
            "runs its newest main/release line (no pinned snapshots)"
        ),
    }
    revisions = merged.get("model_revisions") or {}
    if revisions:
        policy["model_revisions"] = {k: str(v)[:9] for k, v in sorted(revisions.items())}
    for note in args.note:  # validated in main()
        key, _, text = note.partition("=")
        policy[key.strip()] = text.strip()
    return policy


def _headline_summary(section: dict, args) -> dict:
    """Counts derived from the rows the page actually shows.

    The inherited block said "18 cases ... on H100" over a 15-case B200 run, with
    14 wins where there were 12. Deriving it means the headline cannot disagree
    with the table beneath it.
    """
    rows = section.get("rows") or []
    wins: dict[str, int] = {}
    comparable = 0
    for row in rows:
        measured = [
            fw
            for fw, cell in (row.get("cells") or {}).items()
            if cell.get("status") == "ok" and cell.get("latency_s") is not None
        ]
        if len(measured) < 2:  # sglang on its own is not a comparison
            continue
        comparable += 1
        winner = row.get("winner")
        if winner:
            wins[winner] = wins.get(winner, 0) + 1
    sglang_wins = wins.get("sglang", 0)
    return {
        "cases": len(rows),
        "comparable_rows": comparable,
        "sglang_diffusion_wins": sglang_wins,
        "other_wins": comparable - sglang_wins,
        "wins": wins,
        "note": (
            f"Steady-state single-request latency on {args.gpu}, best lossless "
            f"config per framework. SGLang-Diffusion fastest in "
            f"{sglang_wins}/{comparable} comparable cases across "
            f"{len(rows)} cases. Throughput is reported separately."
        ),
    }


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
        help="run date; defaults to the day of the run's first result, not today",
    )
    # No default: the link is the promise that this file produced these
    # numbers, and defaulting it silently attributed every hand-driven run to
    # the biweekly script, which had not run in weeks.
    ap.add_argument(
        "--reproduce",
        required=True,
        help="repo-relative path of the script that actually produced this run",
    )
    ap.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=TEXT",
        help="run-specific caveat for the published policy block, repeatable",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Validate --note here, not where the policy block is built: that happens
    # after the --dry-run return, so a malformed note survived every rehearsal
    # and only failed on the real publish.
    for note in args.note:
        if "=" not in note or not note.partition("=")[2].strip():
            print(f"error: --note must be KEY=TEXT, got {note!r}", file=sys.stderr)
            return 1

    if not (ROOT / args.reproduce).exists():
        print(
            f"error: --reproduce {args.reproduce} does not exist in the repo; "
            f"the published link would 404",
            file=sys.stderr,
        )
        return 1

    merged = _load(args.merged)
    config = _load(args.config)
    run_date = args.date or _run_date(merged)
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
        # Must use the same predicate as the status guard below. It counted
        # `client_latency_s`, a key build_sections does not emit, so it printed
        # "0 measured cells" for a perfectly good run -- a progress line that
        # reads zero whether the run is empty or fine tells you nothing, and a
        # run did once get published entirely as "n/a" without anyone noticing.
        measured = sum(
            1
            for r in sec["rows"]
            for c in r["cells"].values()
            if "latency_s" in c or "qps" in c
        )
        print(f"{sec['id']}: {len(sec['rows'])} rows, {measured} measured cells")
        if not measured:
            print(
                f"  warning: {sec['id']} has no measured cells — it would "
                f"publish as all n/a",
                file=sys.stderr,
            )

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
        # The page draws the latest table's columns from framework_order, so an
        # inherited list silently dropped a framework new to this run (ComfyUI).
        latest["framework_order"] = list(FRAMEWORK_LABELS)
        # Named in the page footer; inherited, it kept citing the July H100 report.
        report_dir = args.merged.resolve().parent
        latest["source_report"] = (
            f"{report_dir.relative_to(ROOT).as_posix()}/"
            if report_dir.is_relative_to(ROOT / "reports")
            else args.run_id
        )
        # Rebuild the policy block instead of inheriting it. It carries
        # run-specific prose (a bimodal-latency note from July, a harness bug
        # fixed in July, an h100-specific "selection" line) that publishing
        # silently carried into every later run -- and its torch_compile line
        # still claimed compile was ENABLED for every framework, months after
        # sglang moved to compile-off. A stale policy block misdescribes the
        # numbers underneath it.
        latest["policy"] = _policy_block(merged, args)
        latest["sections"] = sections
        # The main table renders `rows`, NOT `sections` -- and publishing never
        # replaced it, so every run since inherited the previous one's table. The
        # 4xB200 artifact carried eighteen H100-era rows under a "4x NVIDIA B200"
        # heading, including five wan2.1/wan2.2 cases that run has never had, and
        # a Z-Image number from an h100-h200 profile the B200 harness does not
        # select. The sections underneath were right the whole time, which is why
        # it survived review: the history tab showed the real run.
        single = next(
            (sec for sec in sections if sec.get("mode") == "single_e2e"), None
        )
        if single is not None:
            latest["rows"] = single["rows"]
            latest["summary"] = _headline_summary(single, args)
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
