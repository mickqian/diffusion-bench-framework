#!/usr/bin/env python3
"""Find level changes in the nightly track, scaled to each case's own noise.

Two live regressions sat on the dashboard for a month without being called one:

    cosmos3_super_t2v_2gpu  115.3 -> 119.4 s  (+3.5%) from 2026-08-18
    minimax_h3_t2va_5s       77.2 ->  78.2 s  (+1.3%) from early September

Both were missed twice over. The dashboard's own indicator compares a run with
the one before it, so a step that happened weeks ago always reads as "0.1%". And
a sweep with a fixed percentage threshold (">= 7% is a step") skipped them,
because a fixed threshold asks the wrong question: cosmos3_super repeats to 0.11%
and moved 3.5% -- a 120-sigma event -- while zimage swings 14% between two modes
and means nothing at all.

So score a split by how far apart the two sides are RELATIVE TO THEIR OWN
SCATTER, and report sigma alongside the percentage. Then check the runner: this
track runs on several self-hosted pools and `h100-novita-temp-*` is 20-25% slower
than the others, which manufactures step-shaped artifacts whenever the pool
assignment happens to change. A split whose two sides share no pool is suspect;
one that holds within a single pool is real.

    scripts/detect_nightly_steps.py [--data docs/nightly-data.json] [--min-sigma 8]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def spread(values: list[float]) -> float:
    """Typical within-level scatter: median absolute deviation, floored.

    The floor keeps a case whose repeats are byte-identical from dividing by
    zero and reporting infinite significance.
    """
    if len(values) < 2:
        return float("inf")
    mid = statistics.median(values)
    mad = statistics.median([abs(v - mid) for v in values])
    # MAD alone collapses to exactly zero whenever more than half the points are
    # equal, and the floor then turns that into astronomical significance. An
    # alternating bimodal series hits this at every odd split, which is how a
    # case that means nothing scored higher than a real step. Keep a standard
    # deviation term so a side containing two levels can never look quiet.
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return max(mad, 0.5 * sd, 1e-4 * mid)


def pool_of(runner: str) -> str:
    """`h100-novita-temp-gpu-0123` -> `h100-novita-temp`; the slice does not matter."""
    return runner.rsplit("-gpu-", 1)[0] if runner else "?"


def series_from(data: dict, framework: str = "sglang") -> dict[str, list[tuple]]:
    runs = sorted(data.get("runs") or [], key=lambda r: r.get("timestamp") or "")
    out: dict[str, list[tuple]] = {}
    for run in runs:
        for row in run.get("results") or []:
            if row.get("framework") != framework or row.get("latency_s") is None:
                continue
            out.setdefault(row["case_id"], []).append(
                (
                    str(run.get("timestamp"))[:10],
                    str(run.get("commit_sha"))[:9],
                    float(row["latency_s"]),
                    pool_of(str(run.get("runner_name") or "")),
                )
            )
    return out


def best_split(values: list[float]) -> tuple[float, int, float]:
    """The split maximising (gap between sides) / (scatter within them)."""
    best = (0.0, 0, 0.0)
    for i in range(3, len(values) - 2):
        before, after = values[:i], values[i:]
        gap = statistics.median(after) - statistics.median(before)
        noise = (spread(before) + spread(after)) / 2
        sigma = abs(gap) / noise if noise else 0.0
        if sigma > best[0]:
            best = (sigma, i, gap)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "docs" / "nightly-data.json"))
    ap.add_argument("--min-sigma", type=float, default=8.0)
    ap.add_argument("--min-pct", type=float, default=0.5)
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    found = []
    for case, points in sorted(series_from(data).items()):
        values = [v for _, _, v, _ in points]
        if len(values) < 8:
            continue
        sigma, i, gap = best_split(values)
        if sigma < args.min_sigma:
            continue
        pct = gap / statistics.median(values[:i]) * 100
        if abs(pct) < args.min_pct:
            continue
        shared_pool = {p for *_, p in points[:i]} & {p for *_, p in points[i:]}
        found.append(
            {
                "case": case,
                "pct": pct,
                "sigma": sigma,
                "from": points[i - 1],
                "to": points[i],
                "pool_confounded": not shared_pool,
            }
        )

    if not found:
        print("no level changes above the threshold")
        return 0

    print(f"  {'case':28s} {'step':>7s} {'sigma':>6s}  window                 verdict")
    for f in sorted(found, key=lambda x: -abs(x["sigma"])):
        d0, s0, _, _ = f["from"]
        d1, s1, _, _ = f["to"]
        if f["pool_confounded"]:
            verdict = "runner pool differs across the split -- suspect, not a finding"
        elif f["pct"] > 0:
            verdict = "REGRESSION, same pool both sides"
        else:
            verdict = "improvement"
        print(
            "  %-28s %+6.1f%% %6.0f  %s %s..%s  %s"
            % (f["case"], f["pct"], f["sigma"], d0, s0, s1, verdict)
        )
    regressions = [f for f in found if f["pct"] > 0 and not f["pool_confounded"]]
    print()
    print(f"  {len(regressions)} regression(s) not explained by the runner pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
