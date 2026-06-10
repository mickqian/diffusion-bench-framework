# H200 Video Parallelism Refresh

Scope: vLLM-Omni and LightX2V on two Wan video cases, 2xH200 devbox, client-side `single_e2e` latency. Torch compile is disabled for all refreshed rows. No response cache, no Cache-DiT, no temporal cache. SGLang-Diffusion values below are the existing H200 full-matrix baseline, included only as reference.

## Versions

| Framework | Version |
| --- | --- |
| vLLM-Omni | `vllm==0.22.0`, `vllm-omni==0.22.1.dev31+g73df8326d`, commit `73df8326d76fe3bba0b7b5a6abf6ad68976f24e8` |
| LightX2V | `ModelTC/LightX2V@3db87106bafd980f0eeaffdb0d61dd26b364290c`, `flash-attn==2.8.3`, FA3 from `varunneal/flash-attention-3@de87b9b5af06dd9984df595bef90b2eba44b181a`, `flashinfer-python==0.6.11` |

## Results

| Case | SGLang baseline | vLLM TP | vLLM CFG | vLLM Ulysses/SP | LightX2V FA3 | Fastest refreshed external profile |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.1 T2V 1.3B 480p, 81f, 50 steps, cfg 6 | 25.618s | 38.597s | 29.502s | 32.107s | 29.058s | LightX2V FA3 |
| Wan2.2 TI2V 5B 704p, 81f, 50 steps, cfg 5 | 35.085s | 48.118s | 36.087s | 39.771s | 35.177s | LightX2V FA3 |

## Takeaways

vLLM-Omni TP-only was not the fastest fair video profile. `--ulysses-degree 2` activates SP and beats TP, but `--cfg-parallel-size 2` is fastest among the tested vLLM-Omni profiles for these two cases.

LightX2V FA2 was not a fair H200 setting once FA3 is available. The H200 FA3 profile is much faster than the older FA2 rows in the May full matrix and is the fastest refreshed external profile in both cases.

The stored SGLang-Diffusion baseline remains faster on Wan2.1 1.3B. On Wan2.2 TI2V 5B, the stored SGLang baseline and refreshed LightX2V FA3 result are effectively tied within this single-run spot check.

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
```

Raw result files are in `raw/`.
