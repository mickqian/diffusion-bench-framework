# CUDA-IPC all-to-all: does the 2-rank win survive more ranks?

**Date**: 2026-07-30 · **Machine**: `bench-full` devbox, 4x H100 80GB, single node, NVLink
**Subject**: sgl-project/sglang#31854 (CUDA-IPC transport for the Ulysses all-to-all)
**Script**: `scripts/bench_ipc_nrank.py` (also left on the devbox at `/tmp/bench_ipc_nrank.py`)

## Question

#31854 replaces NCCL with direct peer writes for the **2-rank** Ulysses
all-to-all, measured at −10% per denoise step on Qwen-Image-2512 (2xH100, ~87 →
~79 ms/step). The transport is gated on `world_size == 2`. Should it be
generalised to 4 or 8 ranks?

This had been attempted once before and reverted: the generalisation introduced
an illegal memory access that broke even the 2-rank path, and the root cause was
not found at the time. So the first question is not "how do we make it work" but
"would it pay at all".

## Method

A standalone microbenchmark, deliberately independent of the shipped code, so a
positive result would double as a clean N-rank reference implementation:

- **IPC arm**: every rank writes its chunk directly into each peer's mapped
  staging buffer (`R-1` remote writes) plus its own (1 local write), then
  publishes a sequence number into every peer's flag and spins until all peers
  have published. Same structure the shipped 2-rank path uses, generalised.
  Peer access is enabled for **every** peer, not just `1 - dev`.
- **NCCL arm**: `dist.all_to_all_single` on the same buffers.
- **Timing**: 10 warmup, then 30 iterations of `barrier(); synchronize(); t0;
  exchange(); synchronize()`, median reported. Per-call, not queue throughput.
- **Correctness**: after each configuration the IPC result is compared against
  the NCCL result with `torch.equal`, and the verdict is all-reduced. Every
  number below carries `correct=R/R`.

## Result

IPC latency relative to NCCL (higher is better for IPC):

| ranks | 1 MB | 4 MB | 16 MB |
|---|---|---|---|
| **R=2** | 1.00x | 1.05x | **1.19x** |
| **R=4** | **0.69x** | **0.75x** | **0.78x** |

Absolute medians, for reference:

| config | NCCL | IPC |
|---|---|---|
| R=2 1MB / 4MB / 16MB | 0.036 / 0.040 / 0.069 ms | 0.036 / 0.038 / 0.058 ms |
| R=4 1MB / 4MB / 16MB | 0.034 / 0.043 / 0.085 ms | 0.049 / 0.058 / 0.109 ms |

**At R=4 the direct-IPC exchange is 25-45% slower than NCCL, at every size.**

### Blackwell (B200, sm100, NVLink5)

Re-run on 2/4x B200 to check the gate is not Hopper-specific:

| ranks | 1 MB | 4 MB | 16 MB |
|---|---|---|---|
| R=2 | 0.91x | 1.08x | **1.15x** |
| R=4 | 0.70x | 0.85x | 0.94x |

Same shape as H100: R=2 breaks even and pulls ahead with size, R=4 loses at
every size. The R=4 deficit is smaller than on H100 (0.94x vs 0.78x at 16MB),
consistent with NVLink5's higher bandwidth making each per-peer write relatively
cheaper — but it still does not turn positive. So `world_size == 2` is the right
gate on both architectures and needs no per-architecture branch.

The parity test (`test_ipc_a2a_2_gpu.py`) also passes bitwise on B200.

**A portability bug this surfaced**: the spin's timeout was converted from
milliseconds using `cudaDevAttrClockRate`, which reports 1980 MHz on H100 (its
peak) but **120 MHz on B200**. A 200 ms budget therefore became ~13 ms of real
time there, so a legitimately slow peer would trip the watchdog, return
incomplete data and disable the transport. Fixed by timing the spin with PTX
`%globaltimer`, a nanosecond wall clock that needs no conversion. Worth
remembering for any GPU-side deadline: **do not derive time from a clock-rate
attribute.**

## Why

The IPC scheme's cost is linear in rank count on two axes that NCCL amortises:

- **`R-1` separate copies.** Each peer gets its own `copy_`, so each is its own
  scheduling and copy-engine event. NCCL issues one fused kernel whose channels
  cover all peers concurrently.
- **`R-1` flags polled serially** inside the wait kernel, so the sync tail grows
  with R too.

At R=2 both reduce to a single remote write and a single flag, which is where the
approach is competitive — and even then the raw-transfer win is small (1.00-1.19x).

**This means the in-situ −10%/step does not come from moving bytes faster.** It
comes from what surrounds the transfer, at 120 exchanges per step: NCCL's per-op
launch and protocol overhead disappears, `batched qkv` folds three exchanges into
one, and the zero-copy gather removes the post-exchange concat. Plus the property
that mattered most: plain kernels are **CUDA-graph capturable**, whereas a
captured NCCL collective bakes a per-op sequence number and deadlocks on replay —
that is what makes the full-forward graph (#31852, 63 ms/step) possible at all.

## Conclusion

**Do not generalise.** `world_size != 2 → fall back to NCCL` is the correct
boundary, not a stopgap. NCCL is simply better at the many-peer case; IPC's value
at 2 ranks is the removed overhead and the capture-safety, neither of which
improves with more ranks.

Side result: the earlier crash's prime cause is now identified. The shipped code
calls `cudaDeviceEnablePeerAccess(1 - dev, 0)`, a hardcoded 2-rank peer. With
R>2 a kernel dereferencing a third device's mapping is an illegal access. The
generic implementation here enables peer access for all peers and passes
`correct=4/4`. Since generalisation does not pay, the fix is not worth landing.

## Method note worth keeping

The first version of this benchmark timed 50 iterations with a single sync at the
end, which measures **launch-queue throughput, not per-call latency**. Its output
was non-monotonic in message size (NCCL R=4: 1MB 0.018 → 4MB 0.116 → 16MB
0.066 ms) — that inconsistency is what exposed the flaw. Two rules for
communication microbenchmarks:

1. Sync on both sides of each call and report a median, or the number describes
   the queue rather than the operation.
2. **Verify the result.** Spin-flag synchronisation passes trivially if the flag
   is already at or above the target from an earlier iteration, so a broken
   protocol yields fast, wrong numbers rather than an error.
