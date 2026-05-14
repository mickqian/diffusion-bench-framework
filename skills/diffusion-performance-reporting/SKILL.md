---
name: diffusion-performance-reporting
description: Use when generating formal benchmark reports, dashboards, summaries, GitHub issue comments, or historical trend writeups for diffusion-bench-framework. Formal reports append data-only comments to the fixed tracker issue.
---

# Diffusion Performance Reporting

## Report Shape

A formal report should append one comment to the fixed tracker issue, not open a new issue. The comment should contain only benchmark data:

- run timestamp, benchmark commit, SGLang commit/version, run id, GPU count, and GPU model
- one grouped comparison block per case, with model, task, dimensions, steps, CFG fields, framework, and GPU count
- single-request latency and status
- throughput p50/p95/RPS and status
- ratio-to-SGLang columns for same-case comparison

## Issue Workflow

Use the fixed tracker issue for formal reports. For this repo the canonical tracker is `mickqian/diffusion-bench-framework#1`.

1. Generate or collect the source result JSONs.
2. Regenerate merged artifacts from a fixed script, for example `scripts/generate_h200_report_artifacts.sh`.
3. Review the generated issue Markdown and PNG/SVG image locally.
4. Delete stale formal report comments on the fixed tracker issue, then append exactly one new data-only comment.
5. Put debug analysis, root-cause notes, blocked-run details, and action items outside the tracker issue.

The preferred H200 artifact set is:

- merged JSON: `tmp/report/h200-framework-comparison-merged-local.json`
- data-only issue Markdown: `tmp/report/h200-framework-comparison-merged-local.issue.md`
- image report: `tmp/report/h200-framework-comparison-merged-local.png` and `.svg`

## Tables

The formal issue comment uses one grouped comparison block per case and the same table layout across runs.

Case metadata:

| model | task | dims | steps | cfg |

Framework comparison:

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |

## Interpretation

- Compare throughput with throughput and single-request latency with single-request latency.
- Do not put interpretation in the formal report comment.
- If a run looks unfair or regressed, write a separate investigation note or issue.
- Keep exact comparison semantics in case config so the report table remains self-contained.

## Final Output

Return the fixed issue comment URL, generated artifact paths, the commit that produced the framework/reporting change, and any verification that was actually run.
