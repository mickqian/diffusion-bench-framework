## Diffusion Benchmark Data - 2026-10-11T16:19:54.097208+00:00

| item | value |
| --- | --- |
| run_id | rtx5090-desktop-20261011 |
| data | single_e2e |
| bench_commit | d20d8a5957e6a4434a639a23591e8712144110e5 |
| gpu | 1 x NVIDIA GeForce RTX 5090, 32607 MiB, 595.99.02 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: run_pa5090qwen1010_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_pa5090qwen1010_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090qwen1010b_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_pa5090qwen1010b_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090qwen1010c_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_pa5090qwen1010c_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090h31010b_minimax_h3_t2va_480p_1gpu_minimax_h3_t2va_480p_1gpu.json, run_pa5090main1011_minimax_h3_t2va_480p_1gpu_minimax_h3_t2va_480p_1gpu.json, run_pa5090main1011_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_pa5090main1011_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090ab1011comfy1_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090ab1011sgl1_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090ab1011comfy2_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_pa5090ab1011sgl2_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.0.0.dev0 |
| vLLM-Omni | vllm-omni 0.31.0rc2.dev98+gf69b1f2b1; vllm 0.31.0 |
| LightX2V | 0.5.0 (b6d3828) |
| trtllm-visual | - |
| ComfyUI | master @ 0df64eb |
| FastVideo | - |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### Thinnest lead - single_e2e

| case | hardware | sglang_gpus | sglang_single_e2e_s | fastest_competitor | competitor_gpus | competitor_single_e2e_s | competitor/SGLang-Diffusion | diagnostic_client_s | stage_sum_s | unattributed_tail_s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_image_21_t2i_1024 | 1 x NVIDIA GeForce RTX 5090, 32607 MiB, 595.99.02 | 1 | 13.298 | ComfyUI | 1 | 13.827 | 1.040x | 13.369 | 13.140 | 0.229 |
| qwen_image_21_edit_1024 | 1 x NVIDIA GeForce RTX 5090, 32607 MiB, 595.99.02 | 1 | 15.744 | vLLM-Omni | 1 | 17.218 | 1.094x | 15.812 | 15.592 | 0.221 |

### qwen_image_21_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | rtx5090-1gpu-encoder-layerwise-offload | 1 | 13.298 | 1.000x | ok | - |
| vLLM-Omni | rtx5090-cpu-offload | 1 | 14.707 | 1.106x | ok | - |
| LightX2V | rtx5090-1gpu-fa2 | 1 | 14.037 | 1.056x | ok | - |
| trtllm-visual | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | default | 1 | 13.827 | 1.040x | ok | - |
| FastVideo | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### qwen_image_21_edit_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | image-edit | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | rtx5090-1gpu-encoder-layerwise-offload | 1 | 15.744 | 1.000x | ok | - |
| vLLM-Omni | rtx5090-cpu-offload | 1 | 17.218 | 1.094x | ok | - |
| LightX2V | - | - | - | - | no_profile | LightX2V's Qwen-Image-2.1 integration is validated in this harness for t2i only (lightx2v_task t2i); its image-conditioned path has no validated profile yet (2026-10-10). |
| trtllm-visual | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | - | - | - | - | no_profile | The harness's Qwen-Image-2.1 graph is the official t2i template; an edit graph wiring the reference into its images.image_1 input has not been built and validated yet (2026-10-10). |
| FastVideo | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### minimax_h3_t2va_480p_1gpu

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 864x480x124 | 20 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 1 | 103.277 | 1.000x | ok | - |
| vLLM-Omni | - | - | - | - | no_profile | MiniMax-H3 holds about 108 GB of weights (61.73 GB DiT + 46.18 GB text encoder, per the sglang cookbook) and this case's host has 60 GB of RAM. vLLM-Omni's MiniMax-H3 recipe documents no single-32 GB-GPU command, and its offload keeps the streamed weights in host RAM, which cannot hold them here. Not attempted in the 2026-10-10 round. |
| LightX2V | - | - | - | - | no_profile | MiniMax-H3 holds about 108 GB of weights (61.73 GB DiT + 46.18 GB text encoder, per the sglang cookbook) and this case's host has 60 GB of RAM. LightX2V documents no single-32 GB-GPU command, and its offload keeps the streamed weights in host RAM, which cannot hold them here. Not attempted in the 2026-10-10 round. |
| trtllm-visual | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | default | 1 | - | - | failed | ComfyUI execution failed in node 11 (SamplerCustomAdvanced): AttributeError: 'torch.Stream' object has no attribute 'cuda_stream'  |
| FastVideo | - | - | - | - | no_profile | MiniMax-H3 holds about 108 GB of weights (61.73 GB DiT + 46.18 GB text encoder, per the sglang cookbook) and this case's host has 60 GB of RAM. FastVideo documents no single-32 GB-GPU command, and its offload keeps the streamed weights in host RAM, which cannot hold them here. Not attempted in the 2026-10-10 round. |
