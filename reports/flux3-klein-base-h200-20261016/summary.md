# FLUX.3 Image Klein Base on 2xH200

2026-10-16 · sglang `c15edf31` (0.5.22) · `black-forest-labs/flux-3-image-klein-base` rev `f66af34f`
· 1024×1024, 50 steps, CFG 3.5, seed 42, 1 output
· eager, native BF16, no cache · dbf `7fc85967`

This is the first measurement of FLUX.3 Image in this repo. sglang has served it since 2026-10-15
(sgl-project/sglang#44418). The box image predates that, so sglang ran from a `main`
checkout. This is the Base checkpoint; the step-distilled `flux-3-image-klein` is
a reduced-step model and stays out under the lossless policy.

This report covers one framework only. vLLM-Omni, LightX2V, TensorRT-LLM VisualGen,
ComfyUI and FastVideo have no FLUX.3 pipeline on their main lines (checked
2026-10-15).

## Single request

`scripts/devbox_run_cases.sh` runs `run_comparison --modes single_e2e`. The latency is the
median of 5 measured requests after warmup, timed by the client wall clock.

| GPU | GPUs | p50 | min–max | server | selected profile |
|---|---|---|---|---|---|
| H200 141 GB | 2 | **3.631 s** | 3.626–3.640 | 3.626 s | `h200-2gpu-cfg-parallel` |

| stage | time |
|---|---|
| text encode (Qwen3-VL) | 0.020 s |
| denoise (50 steps, CFG pair split across GPUs) | 3.386 s (0.068 s/step) |
| VAE decode | 0.217 s |
| unattributed tail (e2e − stages) | 0.015 s |

Served command:
`sglang serve --model-path black-forest-labs/flux-3-image-klein-base --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server --enable-cfg-parallel --performance-mode speed`

## Two-GPU layout

The layouts were compared with `scripts/tune_sglang_serve_args.sh`: 3 interleaved rounds, 5 requests per arm, and a
fresh server per arm. Each cell is that round's median.

| layout | round 1 | round 2 | round 3 | vs CFG parallel |
|---|---|---|---|---|
| CFG parallel | **3.637 s** | **3.633 s** | **3.634 s** | 1.00× |
| TP2 | 4.018 s | 4.027 s | 4.036 s | 1.11× |
| Ulysses-2 | 4.428 s | 4.441 s | 4.431 s | 1.22× |

Every CFG-parallel sample is faster than every TP2 and every Ulysses-2 sample, and
each arm's spread is 0.3–1.1%. The Base model runs two DiT branches per step,
and CFG parallelism gives each GPU one branch with no per-layer communication.
The cookbook's one-GPU H200 figure for this checkpoint is 7.09 s, so this
layout roughly halves it.

## Notes

- The runtime picked FlashAttention by default.
- `--performance-mode speed` keeps every component resident. Under the default auto
  policy, sglang layerwise-offloads the text encoder (0/32 layers resident) even
  with 139 GiB free on the card.
- The FLUX.3 profile on other hardware has not been measured yet. Its `default` uses the same
  command.
- `images/` holds the last measured request's output.
