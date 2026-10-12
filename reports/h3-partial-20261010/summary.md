# MiniMax-H3 cross-platform attempts (2026-10-10, partial)

Two H3 runs from the 2026-10-10 round that ended with sglang-only cells. Neither is
published to the Pages data: one framework is not a comparison. Both run sglang main
`3831e7e09`, the round's pin.

| host | case | sglang | other frameworks |
|---|---|---:|---|
| AMD MI355X (rx `h3bench-mi355x`, shared slurm node), 2 GPUs | `minimax_h3_t2va_5s` | 121.09 s (throughput p50 242.5 s) | vLLM-Omni: 500 on every request |
| 4x RTX 5090 32GB (rx `probe-menu-5090`), 2 GPUs | `minimax_h3_t2va_5s` | 553.48 s (throughput p50 1099.9 s) | not run (released early) |

- **MI355X.** sglang runs with `MINIMAX_H3_VAE_DECODER_FUSED_NORM=0`, because
  the fused VAE norm is inline PTX and decodes NaN on ROCm. vLLM-Omni's +rocm723
  torch linked the system ROCm 10 and hit a hipBLASLt GPU memory fault on its
  first request. Harness 77eda45 adds a private ROCm 7.2.3 userspace for it.
  The node then degraded: a 128 MB two-GPU all-reduce took ~600 ms (~0.2 GB/s)
  with every XGMI link up, and an outside job took ~250 GB of HBM. The node was
  released with this single valid cell. `gpu_sampler.log` is the GPU telemetry
  of that window.
- **4x RTX 5090.** Released at the user's request after the sglang cell. The
  runner checkpoint is marked partial.
