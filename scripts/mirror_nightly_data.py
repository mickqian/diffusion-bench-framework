#!/usr/bin/env python3
"""Aggregate recent nightly comparison runs into docs/nightly-data.json.

The dashboard's nightly track used to list sgl-project/ci-data-diffusion via
the GitHub contents API from the browser, which shares the anonymous 60/hour
rate limit across every user behind the same egress IP. CI runs this script
with a token instead and publishes one same-origin snapshot next to the site.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "sgl-project/ci-data-diffusion"
SGLANG = "sgl-project/sglang"
DIFF_JOB = "nightly-test-diffusion"
DIR = "diffusion-comparisons"
MAX_RUNS = 30
OUT = Path(__file__).resolve().parent.parent / "docs" / "nightly-data.json"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "diffusion-bench-mirror"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()



def _runner_names(runs: list[dict], previous: dict[str, str]) -> None:
    """Attach the self-hosted runner each nightly landed on.

    This is the single most useful field for reading the chart, and it is the
    one the run records do not carry. The pools are not equivalent: over twelve
    consecutive nightlies `h100-novita-temp-*` came out 20-25% slower than
    `h100-novita10-*` / `h100-novita5-*` on the heavyweight cases, with no
    exceptions -- which is every "regression" those cases have shown. Without
    this field each alert costs a trip through the Actions API to rule out.

    Runs never change, so a name already mirrored is reused; a fresh nightly
    costs one request, and a failure leaves the field absent rather than
    breaking the mirror.
    """
    for run in runs:
        run_id = run.get("run_id")
        if not run_id:
            continue
        known = previous.get(str(run_id))
        if known:
            run["runner_name"] = known
            continue
        try:
            jobs = json.loads(
                _get(f"https://api.github.com/repos/{SGLANG}/actions/runs/{run_id}/jobs")
            )
            for job in jobs.get("jobs") or []:
                if job.get("name") == DIFF_JOB and job.get("runner_name"):
                    run["runner_name"] = job["runner_name"]
                    break
        except Exception as exc:  # noqa: BLE001 - the mirror matters more than the field
            print(f"no runner for run {run_id}: {exc}", file=sys.stderr)


def main() -> int:
    listing = json.loads(
        _get(f"https://api.github.com/repos/{REPO}/contents/{DIR}?per_page=100")
    )
    names = sorted(
        f["name"]
        for f in listing
        if f["type"] == "file"
        and f["name"].endswith(".json")
        and f["name"][:10].count("-") == 2
    )[-MAX_RUNS:]
    if not names:
        print("no nightly runs found", file=sys.stderr)
        return 1

    runs = []
    for name in names:
        raw = f"https://raw.githubusercontent.com/{REPO}/main/{DIR}/{name}"
        try:
            runs.append(json.loads(_get(raw)))
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the mirror
            print(f"skip {name}: {exc}", file=sys.stderr)
    if not runs:
        print("no runs readable", file=sys.stderr)
        return 1

    previous = {}
    if OUT.exists():
        try:
            for old in json.loads(OUT.read_text()).get("runs") or []:
                if old.get("run_id") and old.get("runner_name"):
                    previous[str(old["run_id"])] = old["runner_name"]
        except Exception:  # noqa: BLE001 - a corrupt snapshot just costs refetches
            previous = {}
    _runner_names(runs, previous)

    runs.sort(key=lambda r: r.get("timestamp") or "")
    bundle = {"source": f"{REPO}/{DIR}", "run_count": len(runs), "runs": runs}
    OUT.write_text(json.dumps(bundle, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} with {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
