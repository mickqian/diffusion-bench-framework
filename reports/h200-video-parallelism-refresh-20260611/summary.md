# H200 Video Parallelism Refresh

Scope: SGLang-Diffusion, vLLM-Omni, and LightX2V on two Wan video cases, 2xH200 devbox, client-side `single_e2e` latency. Torch compile is disabled for all refreshed rows. No response cache, no Cache-DiT, no temporal cache.

## Versions

| Framework | Version |
| --- | --- |
| SGLang-Diffusion | `sglang==0.0.0.dev13850+g7adca0d34`, commit `7adca0d34ab324cbe3a926aee63e827c2f17de35`, benchmark repo commit `bb012bfc6f0d95ff71caeb5cf9052658a0feb32e` |
| vLLM-Omni | `vllm==0.22.0`, `vllm-omni==0.22.1.dev31+g73df8326d`, commit `73df8326d76fe3bba0b7b5a6abf6ad68976f24e8` |
| LightX2V | `ModelTC/LightX2V@3db87106bafd980f0eeaffdb0d61dd26b364290c`, `flash-attn==2.8.3`, FA3 from `varunneal/flash-attention-3@de87b9b5af06dd9984df595bef90b2eba44b181a`, `flashinfer-python==0.6.11` |

## Precision Policy

Wan official native VAE defaults to fp32, so the earlier SGLang-Diffusion fp32 Wan rows are the official-parity precision baseline. LightX2V's Wan serving path uses BF16 through its global `GET_DTYPE()` path, so the refreshed SGLang-Diffusion rows use `h200-2gpu-cfg-bf16-vae-resident`: BF16 VAE, CFG2, DiT/text/image encoder/VAE resident, no cache, and torch compile disabled.

## Results

| Case | SGLang-Diffusion resident BF16 VAE | vLLM TP | vLLM CFG | vLLM Ulysses/SP | LightX2V FA3 | Fastest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.1 T2V 1.3B 480p, 81f, 50 steps, cfg 6 | 26.041s | 38.597s | 29.502s | 32.107s | 29.058s | SGLang-Diffusion |
| Wan2.2 TI2V 5B 704p, 81f, 50 steps, cfg 5 | 33.059s | 48.118s | 36.087s | 39.771s | 35.177s | SGLang-Diffusion |

## SGLang Stage Breakdown

| Case | Client | Server / generation | Text encode | Denoise | VAE decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wan2.1 T2V 1.3B 480p | 26.041s | 24.687s | 0.121s | 22.154s | 2.217s |
| Wan2.2 TI2V 5B 704p | 33.059s | 32.79s | 0.125s | 27.084s | 4.358s |

## Takeaways

vLLM-Omni TP-only was not the fastest fair video profile. `--ulysses-degree 2` activates SP and beats TP, but `--cfg-parallel-size 2` is fastest among the tested vLLM-Omni profiles for these two cases.

LightX2V FA2 was not a fair H200 setting once FA3 is available. The H200 FA3 profile is much faster than the older FA2 rows in the May full matrix and is the fastest refreshed external profile in both cases.

With VAE precision and high-memory resident/offload settings aligned to LightX2V's BF16 serving path, SGLang-Diffusion is faster on both refreshed Wan rows. The previous near-tie on Wan2.2 was largely explained by SGLang using fp32 VAE decode while LightX2V used BF16 VAE decode.

## Reproduce

Run on devboxes, not CI/runner/build/GitHub Actions machines. External frameworks use independent persistent venvs under `/scratch/framework-venvs`.

```bash
cd /path/to/diffusion-bench-framework
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/scratch/framework-venvs
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export HF_HOME=/cluster-storage/models/huggingface
export DIFFUSION_BENCH_HF_CACHE_DIR=/cluster-storage/models/huggingface/hub

RUN_LIGHTX2V=0 RUN_VLLM_OMNI=1 \
VIDEO_REFRESH_CASES="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p" \
RUN_ID_ROOT=h200-video-parallelism-refresh-20260611 \
REPORT_DIR=/scratch/flux1_tp_shard_20260611/report/video_parallelism_refresh \
PORT_BASE=31800 \
bash scripts/run_h200_video_parallelism_refresh_20260611.sh

RUN_VLLM_OMNI=0 RUN_LIGHTX2V=1 \
VIDEO_REFRESH_CASES="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p" \
RUN_ID_ROOT=h200-video-parallelism-refresh-20260611 \
REPORT_DIR=/scratch/flux1_tp_shard_20260611/report/video_parallelism_refresh \
PORT_BASE=31900 \
bash scripts/run_h200_video_parallelism_refresh_20260611.sh

RUN_SGLANG=1 RUN_VLLM_OMNI=0 RUN_LIGHTX2V=0 \
SGLANG_VIDEO_REFRESH_PROFILE=h200-2gpu-cfg-bf16-vae-resident \
VIDEO_REFRESH_CASES="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p" \
RUN_ID_ROOT=h200-video-parallelism-refresh-20260611 \
REPORT_DIR=/scratch/flux1_tp_shard_20260611/report/video_parallelism_refresh \
PORT_BASE=32300 \
bash scripts/run_h200_video_parallelism_refresh_20260611.sh
```

Raw result files are in `raw/`.
