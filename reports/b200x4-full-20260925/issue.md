## Diffusion Benchmark Data - 2026-09-26T01:46:47.567873+00:00

| item | value |
| --- | --- |
| run_id | b200x4-full-20260925 |
| data | single_e2e, throughput |
| bench_commit | aec9d38091785f268b7eb5f219f17410d69dcdf4 |
| gpu | 4 x NVIDIA B200, 183359 MiB, 580.126.09 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: run_b200full0925_cosmos3_nano_i2v_720p_189f_cosmos3_nano_i2v_720p_189f.json, run_b200full0925_cosmos3_nano_t2i_720p_cosmos3_nano_t2i_720p.json, run_b200full0925_cosmos3_nano_t2v_720p_189f_cosmos3_nano_t2v_720p_189f.json, run_b200full0925_flux1_dev_t2i_1024_flux1_dev_t2i_1024.json, run_b200full0925_flux2_dev_t2i_1024_flux2_dev_t2i_1024.json, run_b200full0925_ideogram4_t2i_1024_2gpu_tp_ideogram4_t2i_1024_2gpu_tp.json, run_b200full0925_ltx2.3_twostage_t2v_2gpus_ltx2.3_twostage_t2v_2gpus.json, run_b200full0925_ltx2_twostage_t2v_ltx2_twostage_t2v.json, run_b200full0925_minimax_h3_ref2va_5s_minimax_h3_ref2va_5s.json, run_b200full0925_minimax_h3_t2va_5s_minimax_h3_t2va_5s.json, run_b200full0925_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_b200full0925_qwen_image_2512_t2i_1024_qwen_image_2512_t2i_1024.json, run_b200full0925_qwen_image_2512_t2i_1024_truecfg_qwen_image_2512_t2i_1024_truecfg.json, run_b200full0925_qwen_image_edit_2511_qwen_image_edit_2511.json, run_b200full0925_wan22_t2v_a14b_720p_wan22_t2v_a14b_720p.json, run_b200full0925_zimage_turbo_t2i_1024_zimage_turbo_t2i_1024.json, run_b200full0925r_ltx2b_ltx2_twostage_t2v.json, run_b200full0925r_ltx23b_ltx2.3_twostage_t2v_2gpus.json, run_b200full0925r_ref2va_minimax_h3_ref2va_5s.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.0.0.dev1+g8ca82118e (434c2e3) |
| vLLM-Omni | vllm-omni 0.30.0rc2.dev69+g69de153f6; vllm 0.30.0 |
| LightX2V | 0.5.0 (a4b8ce3) |
| trtllm-visual | 1.3.0rc28 |
| ComfyUI | master @ 88ab4a0 |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### flux1_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.1-dev | text-to-image | 1024x1024 | 50 | gs=3.5 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-eager | 2 | 2.392 | 1.000x | ok | 4/4 | 2 | 4.776 | 1.000x | 4.786 | 4.786 | 0.4187 | 1.000x | ok | - |
| vLLM-Omni | blackwell-2gpu-tp-eager | 2 | 3.352 | 1.401x | ok | 4/4 | 2 | 6.502 | 1.361x | 6.579 | 6.590 | 0.3054 | 0.729x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no FLUX.1 serving path; FLUX coverage starts at FLUX.2. |
| trtllm-visual | default | 2 | 3.550 | 1.484x | ok | 4/4 | 2 | 6.903 | 1.445x | 6.977 | 6.987 | 0.2880 | 0.688x | ok | - |
| ComfyUI | blackwell-gpuonly | 1 | 4.727 | 1.976x | ok | 4/4 | 2 | 9.676 | 2.026x | 9.896 | 9.896 | 0.2067 | 0.494x | ok | - |

### flux2_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.2-dev | text-to-image | 1024x1024 | 50 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-eager | 2 | 6.950 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-2gpu-tp-eager | 2 | 9.815 | 1.412x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2-flashinfer | 2 | 16.548 | 2.381x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | default | 2 | 11.022 | 1.586x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | blackwell-gpuonly | 1 | 14.563 | 2.095x | ok | - | - | - | - | - | - | - | - | not_run | - |

### qwen_image_2512_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2512 | text-to-image | 1024x1024 | 50 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 2.776 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 2.638 | 0.950x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image text-to-image serving path. |
| trtllm-visual | default | 2 | 3.480 | 1.254x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | blackwell-gpuonly | 1 | 4.927 | 1.775x | ok | - | - | - | - | - | - | - | - | not_run | - |

### qwen_image_2512_t2i_1024_truecfg

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2512 | text-to-image | 1024x1024 | 50 | gs=1.0,true=4.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-cfg-parallel | 2 | 3.820 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 5.125 | 1.342x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image text-to-image serving path. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | true_cfg_scale honoring unverified for TensorRT-LLM VisualGen; excluded from the true-CFG quality-path comparison to avoid a mismatched single- vs two-pass row. |
| ComfyUI | blackwell-gpuonly | 2 | 7.065 | 1.849x | ok | - | - | - | - | - | - | - | - | not_run | - |

### qwen_image_edit_2511

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-Edit-2511 | image-edit | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 4.255 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 4.393 | 1.032x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image-Edit serving path. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 1 | 12.016 | 2.824x | ok | - | - | - | - | - | - | - | - | not_run | - |

### qwen_image_21_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-1gpu-resident-fa | 1 | 2.462 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image-2.1 serving path on main HEAD, which is the version policy's target for this framework. vllm-omni's docs/models/supported_models.md lists Qwen-Image, Qwen-Image-2512, Qwen-Image-Edit{,-2509,-2511} and Qwen-Image-Layered but not Qwen-Image-2.1, and a code search for QwenImage21 / Qwen-Image-2.1 across the repo returns nothing. Support lives only on vllm-project/vllm-omni#7759 ('[New model] Qwen-image-2.1 support', opened 2026-09-18, still OPEN as of 2026-09-21); recipes.vllm.ai/Qwen/Qwen-Image-2.1 documents `vllm serve Qwen/Qwen-Image-2.1 --omni` against that PR checkout and states support is not in a tagged release. Re-classify as soon as #7759 merges. |
| LightX2V | blackwell-1gpu-fa2 | 1 | 3.008 | 1.222x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | blackwell-gpuonly | 1 | 3.474 | 1.411x | ok | - | - | - | - | - | - | - | - | not_run | - |

### zimage_turbo_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Tongyi-MAI/Z-Image-Turbo | text-to-image | 1024x1024 | 9 | gs=0.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 0.469 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-flashinfer | 2 | 0.486 | 1.036x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 2 | 1.007 | 2.147x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 1 | 0.823 | 1.755x | ok | - | - | - | - | - | - | - | - | not_run | - |

### ideogram4_t2i_1024_2gpu_tp

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| ideogram-ai/ideogram-4-fp8 | text-to-image | 1024x1024 | 20 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 3.427 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest vLLM-Omni has no Ideogram-4 serving path (image-gen coverage is FLUX/Qwen-Image/Wan). |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest LightX2V has no Ideogram-4 serving path. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest TensorRT-LLM VisualGen has no Ideogram-4 serving path (coverage is FLUX/Wan/LTX-2/Qwen-Image/Cosmos3). |
| ComfyUI | blackwell-gpuonly | 1 | 3.178 | 0.927x | ok | - | - | - | - | - | - | - | - | not_run | - |

### wan22_t2v_a14b_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | text-to-video | 1280x720x81 | 40 | gs=4.0,gs2=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 106.393 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 129.057 | 1.213x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 4 | 258.343 | 2.428x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 2 | 541.085 | 5.086x | ok | - | - | - | - | - | - | - | - | not_run | - |

### ltx2_twostage_t2v

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2 | text-to-video | 768x512x121 | 40 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 6.048 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 15.573 | 2.575x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 2 | 34.573 | 5.716x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 2 | 24.511 | 4.053x | ok | - | - | - | - | - | - | - | - | not_run | - |

### ltx2.3_twostage_t2v_2gpus

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2.3 | text-to-video | 768x512x121 | 30 | gs=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 7.753 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No compatible vLLM-Omni LTX2.3 serving profile is configured. |
| LightX2V | - | - | - | - | failed | - | - | - | - | - | - | - | - | failed | OOM on H100 80GB: LTX-2.3 22B two-stage 121-frame t2av peak exceeds 80GB even with every lossless lever (seq_p_size=2 activation sharding + block cpu_offload + Gemma/VAE offload + exact flash_attn3). Upstream profiles target H200 141GB. THIS VERDICT IS H100-SPECIFIC AND IS BEING APPLIED GLOBALLY -- `status` has no per-hardware form, so the cell is skipped everywhere, including the 4xB200 box (183359 MiB/GPU, ~179GB) where the stated constraint does not hold and upstream's own H200 target is exceeded. Retest on Blackwell before trusting the classification; if it runs, `status` needs a per-hardware form before this can be flipped, or H100 runs will spend ~40 min per attempt re-discovering the OOM. command_profiles below are the best-attempted lossless config, kept for provenance. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | ComfyUI loads LTX-2.3, but no official workflow reproduces this case: sglang runs LTX-2.3's multimodal guidance in stage 1 (video CFG 3 + STG 1.0 on block 28 + modality 3, audio CFG 7 + STG + modality -- up to four DiT passes per step, LTX23SamplingParams), while ComfyUI's LTX-2.3 T2V template (and the LTX-2.5 one) is the distilled-LoRA fast path at cfg 1. A graph built from LTXVSpatioTemporalGuidance / LTXVModalityGuidance / LTXVDualCFGGuider could match it but has no upstream reference and is not validated. |

### cosmos3_nano_t2i_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-image | 1280x720 | 35 | gs=6.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-cfg-parallel | 2 | 0.567 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 0.631 | 1.113x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | failed | - | - | - | - | - | - | - | - | failed | Hangs at startup on the current Cosmos3-Nano checkpoint: trtllm-serve reaches 'Orchestrator is creating IPC executor' / 'using MpiPoolSession to spawn MPI processes' and then produces no output at all -- 40 min in the matrix, and 20 min each at tp=1 and tp=2 in a dedicated probe, so it is not a multi-GPU path issue. The startup log warns 'You are using a model of type cosmos3_omni to instantiate a model of type cosmos3': the checkpoint was re-uploaded 2026-08-26 and declares cosmos3_omni, which tensorrt-llm 1.3.0rc26 does not know. Marked failed rather than left supported so a run does not spend 40 minutes per attempt reaching the same hang. Re-test when trtllm adds cosmos3_omni; the profile is kept below. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |

### cosmos3_nano_t2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 31.715 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 33.039 | 1.042x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |

### cosmos3_nano_i2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | image-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 31.821 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 33.397 | 1.050x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |

### minimax_h3_t2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-2gpu | 2 | 72.878 | 1.000x | ok | 4/4 | 2 | 145.843 | 1.000x | 146.362 | 146.363 | 0.0137 | 1.000x | ok | - |
| vLLM-Omni | b200-2gpu | 2 | 78.640 | 1.079x | ok | 4/4 | 2 | 155.828 | 1.068x | 156.343 | 156.346 | 0.0128 | 0.934x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No MiniMax-H3 pipeline in LightX2V as of 2026-08; a turbo LoRA effort for H3 was blocked by the model license's territory clause (excludes EU/UK/South Korea/US), not by missing engineering support. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | blackwell-gpuonly | 1 | 319.129 | 4.379x | ok | 4/4 | 2 | 638.370 | 4.377x | 638.433 | 638.435 | 0.0031 | 0.226x | ok | - |

### minimax_h3_ref2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-2gpu | 2 | 111.627 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | b200-2gpu | 2 | 118.990 | 1.066x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No MiniMax-H3 pipeline in LightX2V as of 2026-08; a turbo LoRA effort for H3 was blocked by the model license's territory clause (excludes EU/UK/South Korea/US), not by missing engineering support. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | blackwell-gpuonly | 1 | 534.701 | 4.790x | ok | - | - | - | - | - | - | - | - | not_run | - |
