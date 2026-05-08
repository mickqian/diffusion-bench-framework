---
name: diffusion-performance-reporting
description: Use when generating benchmark reports, dashboards, summaries, GitHub issue/PR notes, or historical trend writeups for diffusion-bench-framework. Reports should explain whether SGLang-Diffusion is keeping up with open-source frameworks and make invalid or unfair comparisons explicit.
---

# Diffusion Performance Reporting

## Report Shape

A useful report should normally be filed as a GitHub issue in this repository. Use the performance report issue template when available. A useful report should answer:

- what was compared
- exact hardware and git/package versions
- exact case parameters and selected SGLang command profile
- single-request and throughput results, kept separate
- stage breakdown or best available bottleneck evidence
- whether the comparison is fair
- artifact paths needed to audit the conclusion

## Issue Workflow

1. Generate or inspect `comparison-results.json`, server logs, bench logs, and dashboard output.
2. Draft a GitHub issue with a short title: `[perf] <model/case> <framework summary>`.
3. Use labels such as `performance`, `benchmark`, `sgld`, and either `regression` or `blocked` when appropriate.
4. Put the verdict in the first paragraph: SGLang-Diffusion ahead, tied, behind, or no valid comparison.
5. Link local/remote artifacts and include the selected SGLang command profile.
6. Close the issue only when the result is superseded, the regression is fixed, or the blocked run has a valid replacement.

## Tables

Use separate tables for:

- valid performance results
- invalid or blocked runs
- stage breakdown
- fairness gaps

Do not hide failed runs. Mark them as invalid with the blocker: import error, OOM, compile stall, timeout, client bug, or dependency conflict.

## Interpretation

- Compare throughput with throughput and single-request latency with single-request latency.
- Call out when server-side timing and client E2E timing differ.
- Distinguish startup/compile time from steady-state inference.
- State when Qwen/image results are valid but Wan/video results are not.
- Include a precision or alignment estimate only if there was a real output or parameter alignment check.

## Final Output

Keep the summary direct: issue URL, where SGLang-Diffusion is ahead, tied, or behind; what is blocking a fair comparison; and the next engineering action.
