# H200 Aligned SGLang-Diffusion vs vLLM-Omni Spot-Check

Scope: latest SGLang-Diffusion `origin/main` (`b0d888a195c3ee30350a7681d36efb0dde5fd4a9`) vs vLLM-Omni from `vllm-project/vllm-omni@73df8326d76fe3bba0b7b5a6abf6ad68976f24e8`. Latency uses client-side `single_e2e`; SGLang rows also include server-side latency when available. No response cache, no Cache-DiT, no temporal cache. Runs used a non-CI 2xH200 devbox.

## Versions

| Framework | Version |
| --- | --- |
| SGLang-Diffusion | `b0d888a195c3ee30350a7681d36efb0dde5fd4a9`, `sglang==0.0.0.dev0` from `/scratch/flux_fastest_20260610/sglang-archive-b0d888a` |
| vLLM-Omni | `vllm==0.22.0`, `vllm-omni==0.22.1.dev31+g73df8326d`, commit `73df8326d76fe3bba0b7b5a6abf6ad68976f24e8` |

## Results

| Case | GPUs | Compile | SGLang-Diffusion | vLLM-Omni | Fastest |
| --- | ---: | --- | ---: | ---: | --- |
| FLUX.1-dev T2I, 1024x1024, 50 steps, cfg 3.5 | 1 | on | 6.566s client, 6.392s server | 7.288s client | SGLang-Diffusion, 1.11x |
| FLUX.1-dev T2I, 1024x1024, 50 steps, cfg 3.5 | 2 TP | on | 6.400s client, 6.239s server | 5.912s client | vLLM-Omni, 1.08x |
| Cosmos3 Nano T2I, 1280x720, 35 steps, cfg 6 | 1 | off | 1.859s client, 1.732s server | 1.915s client | SGLang-Diffusion, 1.03x |
| Cosmos3 Nano T2V, 1280x720x189f, 35 steps, cfg 6 | 2 CFG | off | 107.159s client, 104.602s server | 117.820s client | SGLang-Diffusion, 1.10x |

Interpretation: SGLang-Diffusion wins 3 of 4 aligned same-GPU cases. The exception is FLUX.1 2GPU TP, where vLLM-Omni scales better by 1.08x. On FLUX.1 1GPU and Cosmos3 image/video, the earlier Hopper concern that SGLang-Diffusion is behind vLLM-Omni does not reproduce under aligned latest-main settings.

## Reproduce

Run on devboxes, not CI/runner/build/GitHub Actions machines. External frameworks use independent persistent venvs under `/scratch/framework-venvs`.

```bash
cd /path/to/diffusion-bench-framework
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/scratch/framework-venvs
export SGLANG_DIFFUSION_PIP_TMPDIR=/scratch/pip-tmp
export VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git@73df8326d76fe3bba0b7b5a6abf6ad68976f24e8

RUN_ID_ROOT=h200b-main-b0d888a-final \
REPORT_DIR=/scratch/h200b-main-b0d888a-final/report \
PORT_BASE=31000 \
scripts/run_h200_flux1_cosmos3_spotcheck_20260610.sh
```

Raw result files are in `raw/`. The formal one-pager data block is stored in `manifest.json` under `onepager`.
