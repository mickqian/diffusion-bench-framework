## Diffusion Benchmark Data - 2026-07-01T11:19:49.004114+00:00

| item | value |
| --- | --- |
| run_id | b200-cross-framework-20260701 |
| data | single_e2e, throughput |
| bench_commit | 97bbeee92fb3d936f81be05c2542cb05e988f125 |
| gpu | 4 x NVIDIA B200, 183359 MiB, 580.126.09 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: b200-sglang-single.json, b200-sglang-tput-image.json, b200-sglang-tput-video.json, b200-vllm-omni-single.json, b200-vllm-omni-tput-image.json, b200-vllm-omni-tput-video.json, b200-lightx2v-single.json, b200-lightx2v-tput-image.json, b200-lightx2v-tput-video.json, b200-trtllm-visual-single.json, b200-trtllm-visual-tput-image.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.5.14 (1c2523b) |
| vLLM-Omni | vllm-omni 0.24.0rc1; vllm 0.24.0 |
| LightX2V | 0.1.0 (16d7202) |
| trtllm-visual | 1.3.0rc18 |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### flux1_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.1-dev | text-to-image | 1024x1024 | 50 | gs=3.5 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-compile | 2 | 2.776 | 1.000x | ok | 32/32 | 4 | 10.872 | 1.000x | 10.927 | 10.935 | 0.3677 | 1.000x | ok | - |
| vLLM-Omni | blackwell-2gpu-tp-compile | 2 | - | - | anomaly | 32/32 | 4 | 16.696 | 1.536x | 16.736 | 16.744 | 0.2393 | 0.651x | ok | single-request latency 59.2s is 3.5x the steady-state throughput p50 (16.7s) -- isolated-request stall, not compute; see throughput for steady-state |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no FLUX.1 serving path; FLUX coverage starts at FLUX.2. |
| trtllm-visual | default | 2 | 5.276 | 1.901x | ok | 32/32 | 4 | 21.119 | 1.943x | 21.202 | 21.333 | 0.1893 | 0.515x | ok | - |

### flux2_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.2-dev | text-to-image | 1024x1024 | 50 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-tp-compile | 2 | 7.795 | 1.000x | ok | 32/32 | 4 | 29.617 | 1.000x | 29.669 | 29.684 | 0.1350 | 1.000x | ok | - |
| vLLM-Omni | blackwell-2gpu-tp-compile | 2 | - | - | anomaly | 32/32 | 4 | 55.628 | 1.878x | 55.778 | 55.797 | 0.0719 | 0.533x | ok | single-request latency did not converge across 5 back-to-back requests (samples: 66.575s, 66.568s, 66.574s, 14.371s, 66.58s); intermittent stall -- see throughput for steady-state |
| LightX2V | blackwell-fa2-flashinfer | 2 | 17.034 | 2.185x | ok | 32/32 | 4 | 66.218 | 2.236x | 67.246 | 67.251 | 0.0602 | 0.446x | ok | - |
| trtllm-visual | default | 2 | 13.918 | 1.786x | ok | 32/32 | 4 | 55.752 | 1.882x | 55.948 | 55.957 | 0.0718 | 0.532x | ok | - |

### qwen_image_2512_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2512 | text-to-image | 1024x1024 | 50 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 3.601 | 1.000x | ok | 32/32 | 4 | 13.400 | 1.000x | 14.551 | 14.553 | 0.2953 | 1.000x | ok | - |
| vLLM-Omni | blackwell-flashinfer | 2 | - | - | anomaly | 32/32 | 4 | 18.475 | 1.379x | 18.491 | 18.532 | 0.2164 | 0.733x | ok | single-request latency did not converge across 5 back-to-back requests (samples: 5.107s, 59.612s, 59.607s, 5.107s, 59.621s); intermittent stall -- see throughput for steady-state |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image text-to-image serving path. |
| trtllm-visual | default | 2 | 5.691 | 1.580x | ok | 32/32 | 4 | 22.744 | 1.697x | 22.781 | 22.920 | 0.1758 | 0.595x | ok | - |

### zimage_turbo_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Tongyi-MAI/Z-Image-Turbo | text-to-image | 1024x1024 | 9 | gs=0.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 0.484 | 1.000x | ok | 32/32 | 4 | 1.659 | 1.000x | 1.834 | 1.834 | 2.3804 | 1.000x | ok | - |
| vLLM-Omni | blackwell-flashinfer | 2 | - | - | anomaly | 32/32 | 4 | 3.918 | 2.362x | 3.946 | 3.979 | 1.0172 | 0.427x | ok | single-request latency did not converge across 5 back-to-back requests (samples: 55.976s, 1.459s, 55.972s, 55.976s, 1.467s); intermittent stall -- see throughput for steady-state |
| LightX2V | blackwell-fa2 | 2 | 1.004 | 2.074x | ok | 32/32 | 4 | 4.015 | 2.420x | 4.017 | 4.019 | 0.9962 | 0.419x | ok | - |
| trtllm-visual | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | no validated aligned serving profile in this benchmark |

### wan21_t2v_1_3b_480p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | text-to-video | 832x480x81 | 50 | gs=6.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 14.025 | 1.000x | ok | 8/8 | 2 | 26.064 | 1.000x | 27.074 | 27.078 | 0.0753 | 1.000x | ok | - |
| vLLM-Omni | blackwell-flashinfer | 2 | 42.071 | 3.000x | ok | 8/8 | 2 | 83.168 | 3.191x | 83.914 | 84.127 | 0.0240 | 0.319x | ok | - |
| LightX2V | blackwell-fa2 | 2 | 32.058 | 2.286x | ok | 8/8 | 2 | 62.148 | 2.384x | 62.801 | 63.079 | 0.0321 | 0.426x | ok | - |
| trtllm-visual | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | no validated aligned serving profile in this benchmark |

### wan22_ti2v_5b_704p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | text-image-to-video | 1280x704x81 | 50 | gs=5.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 19.041 | 1.000x | ok | 8/8 | 2 | 37.088 | 1.000x | 37.095 | 37.096 | 0.0539 | 1.000x | ok | - |
| vLLM-Omni | blackwell-flashinfer | 2 | 41.604 | 2.185x | ok | 8/8 | 2 | 80.930 | 2.182x | 85.816 | 86.545 | 0.0244 | 0.453x | ok | - |
| LightX2V | blackwell-fa2 | 2 | 29.054 | 1.526x | ok | 8/8 | 2 | 59.145 | 1.595x | 59.169 | 59.175 | 0.0340 | 0.631x | ok | - |
| trtllm-visual | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | no validated aligned serving profile in this benchmark |

### ltx2.3_twostage_t2v_2gpus

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2.3 | text-to-video | 768x512x121 | 30 | gs=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 9.018 | 1.000x | ok | 8/8 | 2 | 15.036 | 1.000x | 15.044 | 15.044 | 0.1352 | 1.000x | ok | - |
| vLLM-Omni | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No compatible vLLM-Omni LTX2.3 serving profile is configured. |
| LightX2V | default | 1 | 57.116 | 6.334x | ok | 8/8 | 2 | 113.293 | 7.535x | 113.314 | 113.316 | 0.0177 | 0.131x | ok | - |
| trtllm-visual | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | no validated aligned serving profile in this benchmark |

### cosmos3_nano_t2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 29.055 | 1.000x | ok | 8/8 | 2 | 56.130 | 1.000x | 56.792 | 57.075 | 0.0355 | 1.000x | ok | - |
| vLLM-Omni | blackwell-flashinfer | 4 | 76.220 | 2.623x | ok | 8/8 | 2 | 150.624 | 2.683x | 152.371 | 152.542 | 0.0133 | 0.375x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | no validated aligned serving profile in this benchmark |
