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

    runs.sort(key=lambda r: r.get("timestamp") or "")
    bundle = {"source": f"{REPO}/{DIR}", "run_count": len(runs), "runs": runs}
    OUT.write_text(json.dumps(bundle, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} with {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
