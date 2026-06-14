#!/usr/bin/env python3
"""Refresh the inline data snapshots embedded in docs/index.html.

The GitHub Pages site always reads docs/data/*.json via fetch(), so updating
those JSON files is enough for the deployed page. The inline snapshots only
serve local file:// previews (where fetch is blocked by the browser). Run this
after updating the JSON files if you want offline previews to stay current:

    python3 scripts/refresh_docs_data.py
"""

import json
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
SNAPSHOTS = {
    "latestDataInline": "latest-cross-framework.json",
    "historicalDataInline": "historical-cross-framework.json",
}


def main() -> int:
    html_path = DOCS / "index.html"
    html = html_path.read_text(encoding="utf-8")

    for tag_id, filename in SNAPSHOTS.items():
        data = json.loads((DOCS / "data" / filename).read_text(encoding="utf-8"))
        # `</` would terminate the script tag early; `<\/` is valid JSON inside strings.
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
        pattern = re.compile(
            rf'(<script type="application/json" id="{tag_id}">).*?(</script>)',
            re.S,
        )
        html, count = pattern.subn(lambda m: m.group(1) + payload + m.group(2), html, count=1)
        if count != 1:
            print(f"error: marker for #{tag_id} not found in docs/index.html", file=sys.stderr)
            return 1
        print(f"embedded {filename} -> #{tag_id} ({len(payload):,} bytes)")

    html_path.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
