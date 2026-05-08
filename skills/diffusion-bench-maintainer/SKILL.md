---
name: diffusion-bench-maintainer
description: Use when maintaining the diffusion-bench-framework repository for long-term open-source diffusion framework performance tracking, including benchmark config changes, SGLang-Diffusion command profiles, result schema updates, dependency isolation, dashboards, and keeping SGLang-Diffusion competitive against vLLM-Omni, LightX2V, diffusers, or other serving stacks.
---

# Diffusion Bench Maintainer

## Purpose

This repo is a long-running performance guardrail for SGLang-Diffusion. Treat every change as something that future benchmark results must be able to explain and reproduce.

## Core Rules

- Preserve comparability before improving convenience.
- Do not use response caching, Cache-DiT, precomputed outputs, or hidden warm-cache shortcuts as performance wins.
- Record enough context to explain a number later: git commits, package versions, hardware, selected command profile, actual server command, sampling params, shape, frames, concurrency, and artifacts.
- Keep external frameworks isolated in their own envs when dependencies can conflict with SGLang.
- Do not treat failed startup, compile stalls, OOMs, or import errors as valid latency data.
- SGLang failures are regressions or invalid command profiles. Fix the backend or add a hardware-specific stable command profile, then rerun before using the result as a comparison baseline.

## Maintenance Workflow

1. Inspect `configs/comparison_configs.json`, `src/diffusion_bench/comparison_configs.json`, runner code, README, and recent results before editing.
2. If changing config semantics, update both the editable config and packaged config.
3. If changing result JSON shape, update dashboard/reporting code and README examples.
4. For SGLang backend changes, add or update a per-case `command_profiles` entry instead of overwriting historical intent. Split H100/H200 commands with profile `hardware` selectors when capacity or best args differ.
5. For runner changes, preserve fail-fast behavior and keep server logs plus bench JSON paths discoverable.
6. After edits, inspect diffs and run only lightweight static checks unless the user asks for a real benchmark.

## Expected Output

For a maintenance task, report:

- what changed
- why it helps long-term tracking
- whether benchmark comparability changed
- what was verified
- what still needs GPU validation
