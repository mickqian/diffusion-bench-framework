## Diffusion Benchmark Data - 2026-10-10T18:46:54.639292+00:00

| item | value |
| --- | --- |
| run_id | gb300x4-20261010 |
| data | single_e2e, throughput |
| bench_commit | 1d771ecb0d5300a262f9d91f78fa6ec4153cb181 |
| gpu | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: run_gb300img1010_cosmos3_nano_i2v_720p_189f_cosmos3_nano_i2v_720p_189f.json, run_gb300img1010_cosmos3_nano_t2i_720p_cosmos3_nano_t2i_720p.json, run_gb300img1010_cosmos3_nano_t2v_720p_189f_cosmos3_nano_t2v_720p_189f.json, run_gb300img1010_flux1_dev_t2i_1024_flux1_dev_t2i_1024.json, run_gb300img1010_flux2_dev_t2i_1024_flux2_dev_t2i_1024.json, run_gb300img1010_ideogram4_t2i_1024_2gpu_tp_ideogram4_t2i_1024_2gpu_tp.json, run_gb300img1010_ltx2.3_twostage_t2v_2gpus_ltx2.3_twostage_t2v_2gpus.json, run_gb300img1010_ltx2_twostage_t2v_ltx2_twostage_t2v.json, run_gb300img1010_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_gb300img1010_qwen_image_21_t2i_1024_2gpu_qwen_image_21_t2i_1024_2gpu.json, run_gb300img1010_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_gb300img1010_zimage_turbo_t2i_1024_zimage_turbo_t2i_1024.json, run_gb300img1010b_cosmos3_nano_t2v_720p_189f_cosmos3_nano_t2v_720p_189f.json, run_gb300img1010fix_cosmos3_nano_i2v_720p_189f_cosmos3_nano_i2v_720p_189f.json, run_gb300img1010fix_cosmos3_nano_t2i_720p_cosmos3_nano_t2i_720p.json, run_gb300img1010fix_cosmos3_nano_t2v_720p_189f_cosmos3_nano_t2v_720p_189f.json, run_gb300img1010fix_flux1_dev_t2i_1024_flux1_dev_t2i_1024.json, run_gb300img1010fix_flux2_dev_t2i_1024_flux2_dev_t2i_1024.json, run_gb300img1010fix_ideogram4_t2i_1024_2gpu_tp_ideogram4_t2i_1024_2gpu_tp.json, run_gb300img1010fix_ltx2.3_twostage_t2v_2gpus_ltx2.3_twostage_t2v_2gpus.json, run_gb300img1010fix_ltx2_twostage_t2v_ltx2_twostage_t2v.json, run_gb300img1010fix_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_gb300img1010fix_qwen_image_21_t2i_1024_2gpu_qwen_image_21_t2i_1024_2gpu.json, run_gb300img1010fix_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_gb300img1010fix_zimage_turbo_t2i_1024_zimage_turbo_t2i_1024.json, run_gb300img1010vfix_qwen_image_21_t2i_1024_2gpu_qwen_image_21_t2i_1024_2gpu.json, run_gb300vid1010_minimax_h3_fasth3_v2_t2va_5s_minimax_h3_fasth3_v2_t2va_5s.json, run_gb300vid1010_wan22_t2v_a14b_720p_wan22_t2v_a14b_720p.json, run_gb300vid1010b_minimax_h3_t2va_5s_minimax_h3_t2va_5s.json, run_gb300vid1010fix_minimax_h3_fasth3_v2_t2va_5s_minimax_h3_fasth3_v2_t2va_5s.json, run_gb300vid1010fix_minimax_h3_ref2va_5s_minimax_h3_ref2va_5s.json, run_gb300vid1010fix_minimax_h3_t2va_5s_4gpu_minimax_h3_t2va_5s_4gpu.json, run_gb300vid1010sfix_minimax_h3_t2va_5s_minimax_h3_t2va_5s.json, run_gb300vid1010sfix_wan22_t2v_a14b_720p_wan22_t2v_a14b_720p.json, run_gb300vid1010f3fix_minimax_h3_fasth3_v2_t2va_5s_minimax_h3_fasth3_v2_t2va_5s.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.5.21 (284d91e) |
| vLLM-Omni | vllm-omni 0.31.0rc2.dev98+gf69b1f2b1; vllm 0.31.0 |
| LightX2V | 0.5.0 (b6d3828) |
| trtllm-visual | 1.3.0rc29 |
| ComfyUI | master @ 0df64eb |
| FastVideo | main @ d5253fd |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### Thinnest lead - single_e2e

| case | hardware | sglang_gpus | sglang_single_e2e_s | fastest_competitor | competitor_gpus | competitor_single_e2e_s | competitor/SGLang-Diffusion | diagnostic_client_s | stage_sum_s | unattributed_tail_s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flux1_dev_t2i_1024 | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 | 2 | 4.141 | trtllm-visual | 2 | 3.104 | 0.750x | 4.118 | 4.050 | 0.068 |
| minimax_h3_ref2va_5s | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 | 2 | 88.421 | FastVideo | 2 | 90.460 | 1.023x | 88.458 | 87.222 | 1.236 |
| minimax_h3_t2va_5s | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 | 2 | 57.529 | vLLM-Omni | 2 | 60.947 | 1.059x | 57.576 | 56.872 | 0.704 |

### Thinnest lead - throughput

| case | hardware | sglang_gpus | sglang_qps | fastest_competitor | competitor_gpus | competitor_qps | competitor/SGLang-Diffusion |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| flux1_dev_t2i_1024 | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 | 2 | 0.2445 | trtllm-visual | 2 | 0.3312 | 1.355x |
| minimax_h3_t2va_5s | 4 x NVIDIA GB300, 284208 MiB, 580.159.04 | 2 | 0.0173 | vLLM-Omni | 2 | 0.0164 | 0.948x |

### flux1_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.1-dev | text-to-image | 1024x1024 | 50 | gs=3.5 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-eager | 2 | 4.141 | 1.000x | ok | 4/4 | 2 | 8.130 | 1.000x | 8.198 | 8.203 | 0.2445 | 1.000x | ok | - |
| vLLM-Omni | blackwell-2gpu-tp-eager | 2 | 5.307 | 1.282x | ok | 4/4 | 2 | 10.396 | 1.279x | 10.522 | 10.538 | 0.1909 | 0.781x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no FLUX.1 serving path; FLUX coverage starts at FLUX.2. |
| trtllm-visual | default | 2 | 3.104 | 0.750x | ok | 4/4 | 2 | 5.969 | 0.734x | 6.084 | 6.099 | 0.3312 | 1.355x | ok | - |
| ComfyUI | blackwell-gpuonly | 1 | 4.410 | 1.065x | ok | 4/4 | 2 | 10.268 | 1.263x | 10.338 | 10.348 | 0.1940 | 0.793x | ok | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers FLUX.1-dev, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### flux2_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.2-dev | text-to-image | 1024x1024 | 50 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-eager | 2 | 5.138 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-2gpu-tp-eager | 2 | 8.368 | 1.629x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2-flashinfer | 2 | 14.536 | 2.829x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | default | 2 | 9.288 | 1.808x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | blackwell-gpuonly | 1 | 12.377 | 2.409x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers FLUX.2-dev, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### qwen_image_21_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-1gpu-resident-fa | 1 | 2.018 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 1 | 2.147 | 1.064x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-1gpu-fa2 | 1 | 3.009 | 1.491x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | blackwell-gpuonly | 1 | 3.221 | 1.596x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### qwen_image_21_t2i_1024_2gpu

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | gb300-2gpu-tp2 | 2 | 1.728 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 3.081 | 1.783x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No validated two-GPU LightX2V command for Qwen-Image-2.1 yet (its upstream configs for this model are single-GPU) (2026-10-10). |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | ComfyUI's only multi-GPU path splits CFG branches, and this case runs without CFG, so a two-GPU cell would be its one-GPU cell. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### qwen_image_21_edit_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | image-edit | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-1gpu-resident-fa | 1 | 2.424 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 1 | 2.598 | 1.072x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | LightX2V's Qwen-Image-2.1 integration is validated in this harness for t2i only (lightx2v_task t2i); its image-conditioned path has no validated profile yet (2026-10-10). |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | The harness's Qwen-Image-2.1 graph is the official t2i template; an edit graph wiring the reference into its images.image_1 input has not been built and validated yet (2026-10-10). |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### zimage_turbo_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Tongyi-MAI/Z-Image-Turbo | text-to-image | 1024x1024 | 9 | gs=0.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 0.505 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-flashinfer | 2 | 0.729 | 1.444x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 2 | 1.504 | 2.978x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 1 | 0.792 | 1.568x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers Z-Image-Turbo, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### ideogram4_t2i_1024_2gpu_tp

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| ideogram-ai/ideogram-4-fp8 | text-to-image | 1024x1024 | 20 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 2.515 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest vLLM-Omni has no Ideogram-4 serving path (image-gen coverage is FLUX/Qwen-Image/Wan). |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest LightX2V has no Ideogram-4 serving path. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest TensorRT-LLM VisualGen has no Ideogram-4 serving path (coverage is FLUX/Wan/LTX-2/Qwen-Image/Cosmos3). |
| ComfyUI | blackwell-gpuonly | 1 | 3.618 | 1.439x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Ideogram pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### wan22_t2v_a14b_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | text-to-video | 1280x720x81 | 40 | gs=4.0,gs2=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 90.016 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 108.988 | 1.211x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 4 | 225.850 | 2.509x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 2 | 484.031 | 5.377x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers Wan2.2-T2V-A14B (examples/serving/openai_wan22_t2v_a14b.yaml), but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### ltx2_twostage_t2v

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2 | text-to-video | 768x512x121 | 40 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | gb300-cfg-parallel-compile | 2 | 8.563 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 12.755 | 1.490x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2 | 2 | 20.070 | 2.344x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 2 | 29.740 | 3.473x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers LTX-2, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### ltx2.3_twostage_t2v_2gpus

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2.3 | text-to-video | 768x512x121 | 30 | gs=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 11.749 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No compatible vLLM-Omni LTX2.3 serving profile is configured. |
| LightX2V | - | - | - | - | failed | - | - | - | - | - | - | - | - | failed | OOM on H100 80GB: LTX-2.3 22B two-stage 121-frame t2av peak exceeds 80GB even with every lossless lever (seq_p_size=2 activation sharding + block cpu_offload + Gemma/VAE offload + exact flash_attn3). Upstream profiles target H200 141GB. THIS VERDICT IS H100-SPECIFIC AND IS BEING APPLIED GLOBALLY -- `status` has no per-hardware form, so the cell is skipped everywhere, including the 4xB200 box (183359 MiB/GPU, ~179GB) where the stated constraint does not hold and upstream's own H200 target is exceeded. Retest on Blackwell before trusting the classification; if it runs, `status` needs a per-hardware form before this can be flipped, or H100 runs will spend ~40 min per attempt re-discovering the OOM. command_profiles below are the best-attempted lossless config, kept for provenance. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | ComfyUI loads LTX-2.3, but no official workflow reproduces this case: sglang runs LTX-2.3's multimodal guidance in stage 1 (video CFG 3 + STG 1.0 on block 28 + modality 3, audio CFG 7 + STG + modality -- up to four DiT passes per step, LTX23SamplingParams), while ComfyUI's LTX-2.3 T2V template (and the LTX-2.5 one) is the distilled-LoRA fast path at cfg 1. A graph built from LTXVSpatioTemporalGuidance / LTXVModalityGuidance / LTXVDualCFGGuider could match it but has no upstream reference and is not validated. |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers LTX-2.3, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### cosmos3_nano_t2i_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-image | 1280x720 | 35 | gs=6.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | gb300-2gpu-cfg-parallel-compile | 2 | 0.868 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 2 | 0.935 | 1.077x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | failed | - | - | - | - | - | - | - | - | failed | Hangs at startup on the current Cosmos3-Nano checkpoint: trtllm-serve reaches 'Orchestrator is creating IPC executor' / 'using MpiPoolSession to spawn MPI processes' and then produces no output at all -- 40 min in the matrix, and 20 min each at tp=1 and tp=2 in a dedicated probe, so it is not a multi-GPU path issue. The startup log warns 'You are using a model of type cosmos3_omni to instantiate a model of type cosmos3': the checkpoint was re-uploaded 2026-08-26 and declares cosmos3_omni, which tensorrt-llm 1.3.0rc26 does not know. Marked failed rather than left supported so a run does not spend 40 minutes per attempt reaching the same hang. Re-test when trtllm adds cosmos3_omni; the profile is kept below. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### cosmos3_nano_t2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | gb-4gpu-cfg-ulysses-speed | 4 | 22.982 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 26.048 | 1.133x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### cosmos3_nano_i2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | image-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 23.170 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn | 4 | 26.391 | 1.139x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### minimax_h3_t2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 57.529 | 1.000x | ok | 4/4 | 2 | 115.335 | 1.000x | 115.359 | 115.361 | 0.0173 | 1.000x | ok | - |
| vLLM-Omni | blackwell-2gpu | 2 | 60.947 | 1.059x | ok | 4/4 | 2 | 122.297 | 1.060x | 122.315 | 122.316 | 0.0164 | 0.948x | ok | - |
| LightX2V | blackwell-tp2-fa2 | 2 | 141.040 | 2.452x | ok | 4/4 | 2 | 281.811 | 2.443x | 282.610 | 282.722 | 0.0071 | 0.410x | ok | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | blackwell-gpuonly | 1 | 283.481 | 4.928x | ok | 4/4 | 2 | 566.772 | 4.914x | 567.043 | 567.070 | 0.0035 | 0.202x | ok | - |
| FastVideo | blackwell-2gpu | 2 | 61.148 | 1.063x | ok | 4/4 | 2 | 122.319 | 1.061x | 122.428 | 122.439 | 0.0163 | 0.942x | ok | - |

### minimax_h3_t2va_5s_4gpu

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 31.682 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | default | 4 | 34.078 | 1.076x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-sp4-fa2 | 4 | 73.292 | 2.313x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | ComfyUI has no sequence or tensor parallelism (its one multi-GPU path splits CFG branches, and H3 runs without CFG), so a four-GPU cell would be its one-GPU number; that number is in minimax_h3_t2va_5s. |
| FastVideo | blackwell-4gpu | 4 | 34.077 | 1.076x | ok | - | - | - | - | - | - | - | - | not_run | - |

### minimax_h3_ref2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 88.421 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-2gpu | 2 | 91.910 | 1.039x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-tp2-fa2 | 2 | 229.212 | 2.592x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | blackwell-gpuonly | 1 | 479.477 | 5.423x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | blackwell-2gpu | 2 | 90.460 | 1.023x | ok | - | - | - | - | - | - | - | - | not_run | - |

### minimax_h3_fasth3_v2_t2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| FastVideo/FastVideo-FastH3-8-Step-V2 | text-to-video | 1344x768x124 | 8 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 6.411 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | default | 4 | 7.439 | 1.160x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Checked 2026-10-08 against LightX2V main 0c2edc12 (2026-10-05): LightX2V cannot serve this checkpoint under its trained contract (fastvideo_inference.json). V2 is trained for VIDEO_SPARSE_ATTN_H3 at 0.8 sparsity on 64-token tiles, and LightX2V has no VSA backend (its sparse attentions -- Sparge, SVG, radial, NBHD, Sol-Attn, sparse Sage/FA4 -- are other methods; dense attention would run a different model). Its H3 scheduler builds only a uniform shifted grid from infer_steps, so the explicit rungs [999, 874, ..., 125] cannot be given (an 8-step grid starts at 1000). LightX2V's own few-step H3 paths (configs/minimax_h3/dmd/: 4- and 8-step LoRAs on the base checkpoint, mostly FP8/INT8) are a different distillation, not this checkpoint, and are not substituted. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No official ComfyUI workflow for the FastH3 8-Step V2 checkpoint and its VSA attention as of 2026-09-30. |
| FastVideo | blackwell-4gpu | 4 | 7.170 | 1.118x | ok | - | - | - | - | - | - | - | - | not_run | - |
