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
- **"AllToAll4D parity holds"** — suspected the arm never reached the IPC branch,
  because the staged-buffer count stayed at 6 where new shapes should have added
  keys. See the next section: that evidence was itself invalid.

Make the evidence part of the assertion, not an afterthought:

- Require a **side effect that only the path under test produces** — a cache/
  staging key count that must grow, a one-shot log line, a counter. Assert it.
- For a flag A/B, first confirm the flag reaches a decision: log the resolved
  policy, or check the gate's own predicate for your inputs.
- Prefer a **crash/shape test over a numeric-equivalence test** when the change
  is structural; a wrong layout cannot hide behind a tolerance.
- Two or three samples cannot support a percentage claim. If the arms' ranges
  overlap, there is no effect to report yet.

## Then Prove The Evidence Itself (same day, the fix that misfired)

The "prove the path ran" guard above immediately produced its own false alarm, so
it needs a second rule. The evidence chosen for the AllToAll4D arm was *a new
staging key must appear*. It never appeared, and the reported verdict was
"AllToAll4D never took the IPC path". A probe that called only that entry point
disproved it in one line — the hook ran, the transport was ready, it returned a
tensor, staging went 0→1:

```
PROBE ipc_fn_calls=1 ready=True returned=tensor staging 0->1
```

The staging key is `(n_local, n_peer, dtype)` with **no call-site component**, so
the second entry point reuses the buffers the first one allocated. The premise —
"a new call site with these shapes must allocate" — was false, and it accused
working code. Two rules:

- **Count the event, not a resource derived from it.** A monotone per-call
  counter (`IPC_A2A.calls`) cannot collide; a cache key deliberately can, since
  reuse is the cache's whole purpose. Before asserting on a container's size, ask
  what its key is and whether two different call sites can hash to the same one.
- **When an evidence assertion fires, probe the narrow path before believing it.**
  Isolating one entry point costs a few minutes and distinguishes "the code is
  broken" from "my assertion is wrong" — a distinction that reasoning about the
  gate conditions did not settle in either direction.
- **An evidence assertion must also separate "did not run" from "cannot run
  here."** The same guard that proves a CUDA-IPC path executed failed the test on
  MI300, where the transport correctly refuses and both arms fall back — a true
  report of "never engaged" on a platform that can never engage. Device count is
  not the discriminator (`torch.cuda.device_count()` is 2 under HIP); the platform
  is (`current_platform.is_cuda()`). Any test whose evidence is "the accelerated
  path ran" needs a platform skip next to it, or it becomes a false failure on
  every other backend in the matrix.

## A NCCL Watchdog Timeout Means One Rank Took A Different Path

`Watchdog caught collective operation timeout` is almost never a NCCL bug and
almost never a slow network. It means one rank posted a collective the others did
not, so read it as **"find the branch that is per-rank"**.

The instance that taught this (sgl-project/sglang#31854, `wan2_2_t2v_a14b_2gpu`):
a custom IPC transport bounded its GPU-side spin, and on expiry the rank that
timed out disabled the transport **for itself** and fell back to NCCL. Its peer
never timed out, stayed on IPC, and posted nothing. Ten minutes later the
watchdog took the process down. Both ends were in the CI log — the transport's
own "timed out waiting for the peer" line, then the abort with the transport's
file in the stack — so the chain was evidence, not inference.

- Any **fallback, capability gate, or feature flag that a single rank can resolve
  differently must be decided collectively**, or the collective it guards will
  eventually be posted by a subset of ranks. Per-rank device state (a flag a
  kernel sets locally) is exactly this hazard.
- Prefer telling the peer over adding a collective to ask it. A rank that already
  maps the peer's memory can write the decision there; an extra all-reduce at a
  boundary introduces a *second* way for ranks to disagree (one rank skipping the
  all-reduce because its own init failed leaves the other hanging in it).
- A bound that "gives up and degrades" is only safe if giving up is **symmetric**
  and the incomplete result is **not consumed**. Otherwise it converts a hang into
  a corrupted output plus a later, misattributed crash.
- Sanity-check the bound against the model, not the kernel: layerwise offload and
  wan2.2's expert-tower swap stall a rank for **seconds**. A 200 ms "generous"
  budget was simply wrong; a deadlock backstop belongs in the seconds.

## CI Blame Needs The Checkout Time, Not The Log Time (2026-08-06)

A flux 2-GPU perf regression was pinned on the only attention commit in the
window, "seven minutes before the failure" — but the seven minutes were measured
against the mid-job failure **log** timestamp. The job's `started_at` (which is
when the merge ref is checked out) was 31 minutes **before** the commit landed:
the failing code could not contain it. Two public comments had to be retracted.

- Blame windows are bounded by **`started_at` of the runs' checkouts**, never by
  log timestamps inside the job, and a rerun (`run_attempt > 1`) reuses the
  run's ORIGINAL merge commit — a green attempt-2 executed later still tests the
  old code.
- An independent A/B of the accused commit (ABAB on the case it allegedly broke)
  is cheap insurance before accusing publicly; here it showed both arms
  identical, which is what forced the timeline recheck.
- Library-version bumps (sgl-kernel) belong in every perf-blame window: kernels
  ship from there, and a dispatch change matches a constant per-step factor.
- Resolution of this case: the same-commit rerun (attempt 2) passed — there was
  **no code regression at all**, only runner state. The cheapest discriminator
  for any CI perf blame is a rerun at the same merge commit; run it before any
  bisect, and treat a runner with same-day OOM/cache incidents as suspect #1.

## Reporting

Lead with the verdict:

- fair or not fair
- valid data or invalid run
- root cause category
- next action

Never present a failed startup, compile stall, OOM, or dependency error as a performance number.

