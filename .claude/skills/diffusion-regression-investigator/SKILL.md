---
name: diffusion-regression-investigator
description: Use when SGLang-Diffusion is slower than vLLM-Omni, LightX2V, diffusers, or another baseline; when benchmark numbers look unfair; when logs show hangs, compile stalls, OOMs, NaNs, dtype mismatches, VAE bottlenecks, scheduler issues, or unexpected throughput regressions.
---

# Diffusion Regression Investigator

## First Principle

Do not optimize before proving the comparison is fair. Separate invalid runs from real performance regressions.

## Fairness Checks

Verify these before comparing latency:

- same model or officially equivalent weights
- same resolution, `num_frames`, steps, seed, guidance, scheduler, dtype, and VAE path
- same output count and response format expectations
- same GPU class and intended GPU count
- same warmup policy and no hidden cache advantage
- single-request vs throughput metrics are not mixed

## Diagnosis Workflow

1. Read result JSON, server logs, bench logs, command profile, and actual server command.
2. Classify the run: valid steady-state, startup/compile, OOM, import failure, request timeout, scheduler stall, or client polling issue.
3. Use stage breakdown when available. Attribute time to text/image encoding, denoising, VAE encode/decode, transport, or client overhead.
4. For hangs, use `py-spy` on server and worker processes and compare with GPU utilization.
5. For surprising VAE or decode cost, check dtype, offload, tiling/chunking, output format, and CPU transfer.
6. If SGLang is actually slower, propose a targeted code or config fix and state why it should affect the measured stage.

## Timing Pitfalls (2026-07 qwen-vs-vLLM case)

- Stage timers without a device sync steal time from each other: an async
  denoise loop spills its GPU tail into the next stage, so "VAE 250ms" was
  really VAE 67ms + denoise tail. Before optimizing any stage number, add a
  `torch.cuda.synchronize()` bracket and reconcile the sum against client e2e.
- "average time per step" is a CPU-side figure; treat it as a lower bound.
- Baseline versions drift: the bench metadata's `install_specs` records the
  *requested* pins, but `packages`/`direct_urls` record what was actually
  installed (a source install can silently upgrade its deps — vllm-omni 0.18
  spec actually ran vllm 0.24). Reproduce the denominator from
  `packages`/`direct_urls`, never from install_specs, and re-measure it on the
  same box before chasing a gap.
- Before patching a "hot path", prove it executes (one-shot log in the
  function): qwen's Ulysses A2A goes through USPAttention's tail-pad branch
  (`_usp_input_all_to_all` x3 + `_usp_output_all_to_all`), not AllToAll4D and
  not the varlen pair. Two "zero-effect optimizations" were just patches on
  dead code.

## Reporting

Lead with the verdict:

- fair or not fair
- valid data or invalid run
- root cause category
- next action

Never present a failed startup, compile stall, OOM, or dependency error as a performance number.

