---
name: diffusion-framework-benchmarking
description: Use when planning, running, extending, or interpreting diffusion framework benchmarks across SGLang-Diffusion, vLLM-Omni, LightX2V, diffusers, or similar systems, especially when comparing single-request latency, throughput, command profiles, framework versions, model support, fairness, failure reasons, or report/image generation in diffusion-bench-framework.
---

# Diffusion Framework Benchmarking

## Goal

Produce reproducible, fair, and actionable performance data for open-source diffusion serving frameworks. The benchmark is a guardrail for SGLang-Diffusion, so invalid comparisons are worse than missing data.

## Non-Negotiables

- Do not use response cache, Cache-DiT, precomputed outputs, quantized checkpoints, reduced steps, or distilled weights unless the case explicitly compares those semantics.
- Disable torch compile for all frameworks unless the user explicitly asks for compile-inclusive numbers.
- Keep external frameworks in isolated virtualenvs when dependencies conflict with SGLang.
- Never treat startup failure, import failure, OOM, timeout, compile stall, or NaN as latency data.
- Compare single-request latency with single-request latency, and throughput with throughput. Do not mix the two in one conclusion.
- Track actual framework version/ref, selected command profile, hardware, GPU count, sampling params, dimensions, frames, concurrency, and actual server command in result artifacts.

## Fairness Checklist

Before accepting a number, verify:

- same model weights or an explicitly documented equivalent model path
- same task type, prompt, seed, resolution, `num_frames`, FPS, steps, guidance/CFG, scheduler semantics, dtype, and VAE path
- same output count and response format expectations
- intended GPU class and GPU count are recorded; H100/H200 may need different profiles
- warmup is comparable and not dominated by cold model download or compile
- framework-specific fast paths are fair: fast attention is fine; hidden caches or lower-quality models are not
- single request uses one request; throughput records `num_requests`, concurrency, p50, p99, and QPS

## Command Profiles

Every framework entry should carry `command_profiles`, not just inline args.

- Use `sglang_ref` or `framework_ref` to pin the release, commit, package version, or meaningful upstream line.
- Split profiles by hardware when best commands differ, for example `h100-80gb-2gpu` vs `h200-2gpu`.
- For SGLang failures, fix the backend or add a stable hardware-specific profile before using the comparison.
- For non-SGLang frameworks, seek the fastest fair command too; do not leave a slow default if upstream has a documented faster path.
- When upstream support changes, update install specs and profiles before claiming unsupported. Example: latest LightX2V supports LTX-2/LTX-2.3 even if an older pinned commit did not.

## Running Benchmarks

Use repo scripts when available instead of ad hoc one-liners:

- single request: `scripts/run_h200_single_e2e_*.sh`
- throughput: `scripts/run_h200_throughput_*.sh`
- targeted reruns: set `CASES` / `FRAMEWORKS` env vars when the script supports them, or add a dedicated script for repeatable reruns

For throughput, use enough warmup to avoid first-request artifacts, then record at least:

- `num_requests`
- max concurrency
- p50 latency
- p99 latency
- QPS
- failure count

For fast image models, throughput is often more informative than a single request. For long video models, one single request per model can be enough unless the user asks for concurrency data.

## Failure Classification

Classify every failed or missing cell:

- `not_configured`: framework has no harness entry for this case
- `unsupported`: upstream does not support this model/task
- `supported_not_run`: upstream supports it, but this benchmark version/profile has not run it yet
- `failed`: server or request failed; include the short root cause
- `invalid`: command used wrong shape, frames, model, dtype, cache, compile, or sampling params

If the failure is SGLang, treat it as a bug or bad profile and fix before finalizing the report. If the failure is another framework, verify the official docs/scripts before marking unsupported.

## Reports And Images

Formal issue comments should be data-only: run metadata, framework versions, case tables, ratios, statuses, and reasons. Put debug analysis elsewhere.

For comparison images:

- group rows by case so SGLang and other frameworks are adjacent
- separate single-request and throughput sections
- show missing/failed cells explicitly
- show ratios relative to SGLang for the same case
- include source result JSON names and filter rules in the footer
- regenerate from a fixed script so future reports are reproducible

## Output Discipline

When reporting back, include:

- which result JSONs were used
- which framework versions/commits were compared
- which cases were excluded and why
- whether data is benchmarked, failed, unsupported, or supported but not run
- exact paths to generated report/image artifacts
