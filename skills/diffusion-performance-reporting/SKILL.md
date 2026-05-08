---
name: diffusion-performance-reporting
description: Use when generating formal benchmark reports, dashboards, summaries, GitHub issue comments, or historical trend writeups for diffusion-bench-framework. Formal reports append data-only comments to the fixed tracker issue.
---

# Diffusion Performance Reporting

## Report Shape

A formal report should append one comment to the fixed tracker issue, not open a new issue. The comment should contain only benchmark data:

- run timestamp, benchmark commit, SGLang commit/version, run id, GPU count, and GPU model
- case, model, task, dimensions, steps, CFG fields, framework, and GPU count
- single-request latency and status
- throughput p50/p95/RPS and status

## Issue Workflow

1. Generate `comparison-results.json`.
2. Run `diffusion-bench-dashboard --results comparison-results.json --report-repo <owner/repo> --report-issue <fixed-number>`.
3. Keep the generated issue comment data-only.
4. Put debug analysis, root-cause notes, blocked-run details, and action items outside the tracker issue.

## Tables

The formal issue comment uses one normalized table:

| case | model | task | dims | steps | cfg | framework | gpus | single_e2e_s | single_status | throughput_p50_s | throughput_p95_s | throughput_rps | throughput_status |

## Interpretation

- Compare throughput with throughput and single-request latency with single-request latency.
- Do not put interpretation in the formal report comment.
- If a run looks unfair or regressed, write a separate investigation note or issue.
- Keep exact comparison semantics in case config so the report table remains self-contained.

## Final Output

Return the fixed issue URL, the commit that produced the framework/reporting change, and any verification that was actually run.
