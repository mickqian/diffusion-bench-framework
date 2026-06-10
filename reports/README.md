# Benchmark Report Data

Committed benchmark data uses one directory per report:

```text
reports/<report-id>/
  manifest.json   # stable metadata, version pins, result files, reproduce commands
  summary.md      # compact human-readable result tables
  raw/            # selected raw result JSONs used by the report
  failures/       # failed attempts kept only to explain missing rows
  images/         # generated report figures
```

Keep the format simple:

- `raw/` files must be original runner output, not edited summaries.
- `failures/` files do not count as performance data unless `summary.md` explicitly says so.
- `manifest.json` is the index for machines, framework versions, commands, and included files.
- `summary.md` should contain clear comparison data and short reproducibility notes, not debug analysis.
- Do not run or record benchmark data from CI, runner, build, GitHub Actions, Jenkins, or other shared automation machines.

Update the GitHub Pages one-pager from a formal report with:

```bash
node scripts/export_onepager_data.mjs reports/<report-id>
```
