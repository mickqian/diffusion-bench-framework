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

## Prove The Path Executed Before Judging It (2026-07-31, three misfires in one day)

A comparison whose two arms never diverged in the code under test passes while
proving nothing, and it reads exactly like a clean result. This burned three
separate conclusions in one session:

- **"dp is byte-identical"** — the request used `guidance_scale 1.0`, so the
  encode batch was 1 and the dp gate refused in *both* arms. The identical
  bytes were two runs of the same replicated path.
- **"spatial split is a 32% pessimisation on small activations"** — image VAEs
  return `False` from `auto_parallel_decode_prefers_spatial_shard()`, so
  toggling `--vae-config.use-parallel-decode` changed nothing; the "difference"
  was 3-sample noise (arms overlapped: 0.091/0.061/0.106 vs 0.093/0.061/0.062).
- **"AllToAll4D parity holds"** — the arm never reached the IPC branch. Caught
  only because the staged-buffer count stayed at 6, which new shapes could not
  have done.

Make the evidence part of the assertion, not an afterthought:

- Require a **side effect that only the path under test produces** — a cache/
  staging key count that must grow, a one-shot log line, a counter. Assert it.
- For a flag A/B, first confirm the flag reaches a decision: log the resolved
  policy, or check the gate's own predicate for your inputs.
- Prefer a **crash/shape test over a numeric-equivalence test** when the change
  is structural; a wrong layout cannot hide behind a tolerance.
- Two or three samples cannot support a percentage claim. If the arms' ranges
  overlap, there is no effect to report yet.

## Reporting

Lead with the verdict:

- fair or not fair
- valid data or invalid run
- root cause category
- next action

Never present a failed startup, compile stall, OOM, or dependency error as a performance number.

