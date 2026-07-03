## Diffusion Benchmark Data - 2026-07-03T07:58:35.570301+00:00

| item | value |
| --- | --- |
| run_id | h100x4-fastest-20260703 |
| data | single_e2e, throughput |
| bench_commit | unknown |
| gpu | 4 x NVIDIA H100 80GB HBM3, 81559 MiB, 580.126.20 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: base_filtered.json, rebench_zimage_turbo_t2i_1024.json, rebench_wan21_t2v_1_3b_480p.json, rebench_wan22_ti2v_5b_704p.json, rebench_cosmos3_nano_t2v_720p_189f.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.0.0 (f89f4b3) |
| vLLM-Omni | vllm-omni 0.24.0rc2.dev8+g4d2ee1515; vllm 0.24.0 |
| LightX2V | 0.1.0 (7efd05f) |
| trtllm-visual | 1.3.0rc18 |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### flux1_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.1-dev | text-to-image | 1024x1024 | 50 | gs=3.5 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu-tp-compile | 2 | 4.409 | 1.000x | ok | 4/4 | 2 | 8.809 | 1.000x | 8.829 | 8.831 | 0.2266 | 1.000x | ok | - |
| vLLM-Omni | h100-h200-2gpu-tp-compile | 2 | 6.075 | 1.378x | ok | 4/4 | 2 | 12.018 | 1.364x | 12.098 | 12.104 | 0.1660 | 0.733x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no FLUX.1 serving path; FLUX coverage starts at FLUX.2. |
| trtllm-visual | h100-h200-2gpu-tp-speed | 2 | 7.254 | 1.645x | ok | 4/4 | 2 | 14.531 | 1.650x | 14.540 | 14.540 | 0.1376 | 0.607x | ok | - |

### flux2_dev_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/FLUX.2-dev | text-to-image | 1024x1024 | 50 | gs=4.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu-tp-compile | 2 | 13.232 | 1.000x | ok | 4/4 | 2 | 26.419 | 1.000x | 26.441 | 26.444 | 0.0757 | 1.000x | ok | - |
| vLLM-Omni | h100-h200-2gpu-tp-compile | 2 | 18.278 | 1.381x | ok | 4/4 | 2 | 36.321 | 1.375x | 36.420 | 36.431 | 0.0550 | 0.727x | ok | - |
| LightX2V | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | configured framework entry, but no result in this artifact |
| trtllm-visual | h100-h200-2gpu-tp-speed | 2 | - | - | failed | - | - | - | - | - | - | - | - | failed | trtllm-visual server exited before health check passed (exit 1) |

### qwen_image_2512_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2512 | text-to-image | 1024x1024 | 50 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu-tp-speed-compile | 2 | 5.043 | 1.000x | ok | 4/4 | 2 | 10.083 | 1.000x | 10.110 | 10.112 | 0.1982 | 1.000x | ok | - |
| vLLM-Omni | default | 2 | 5.168 | 1.025x | ok | 4/4 | 2 | 10.166 | 1.008x | 10.242 | 10.253 | 0.1959 | 0.988x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image text-to-image serving path. |
| trtllm-visual | h100-h200-2gpu-tp-speed | 2 | 7.528 | 1.493x | ok | 4/4 | 2 | 15.083 | 1.496x | 15.110 | 15.113 | 0.1325 | 0.669x | ok | - |

### qwen_image_2512_t2i_1024_truecfg

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2512 | text-to-image | 1024x1024 | 50 | gs=1.0,true=4.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu-tp-speed-compile | 2 | 6.496 | 1.000x | ok | 4/4 | 2 | 13.329 | 1.000x | 13.631 | 13.636 | 0.1500 | 1.000x | ok | - |
| vLLM-Omni | default | 2 | 10.145 | 1.562x | ok | 4/4 | 2 | 20.095 | 1.508x | 20.180 | 20.191 | 0.0993 | 0.662x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Tracked LightX2V version has no Qwen-Image text-to-image serving path. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | true_cfg_scale honoring unverified for TensorRT-LLM VisualGen; excluded from the true-CFG quality-path comparison to avoid a mismatched single- vs two-pass row. |

### zimage_turbo_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Tongyi-MAI/Z-Image-Turbo | text-to-image | 1024x1024 | 9 | gs=0.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu-tp-resident-eager | 2 | 0.696 | 1.000x | ok | 4/4 | 2 | 1.426 | 1.000x | 1.466 | 1.468 | 1.3937 | 1.000x | ok | - |
| vLLM-Omni | default | 2 | 1.245 | 1.789x | ok | 4/4 | 2 | 2.338 | 1.640x | 2.417 | 2.428 | 0.8395 | 0.602x | ok | - |
| LightX2V | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | configured framework entry, but no result in this artifact |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |

### wan21_t2v_1_3b_480p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | text-to-video | 832x480x81 | 50 | gs=6.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-2gpu-cfg-speed-compile | 2 | 25.049 | 1.000x | ok | 4/4 | 2 | 50.116 | 1.000x | 50.127 | 50.127 | 0.0399 | 1.000x | ok | - |
| vLLM-Omni | default | 2 | 31.048 | 1.239x | ok | 4/4 | 2 | 60.949 | 1.216x | 62.081 | 62.173 | 0.0326 | 0.817x | ok | - |
| LightX2V | default | 2 | 29.058 | 1.160x | ok | 4/4 | 2 | 56.777 | 1.133x | 57.291 | 57.292 | 0.0349 | 0.875x | ok | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |

### wan22_ti2v_5b_704p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | text-image-to-video | 1280x704x81 | 50 | gs=5.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-h200-2gpu | 2 | 31.062 | 1.000x | ok | 4/4 | 2 | 60.133 | 1.000x | 61.007 | 61.130 | 0.0330 | 1.000x | ok | - |
| vLLM-Omni | default | 2 | 38.149 | 1.228x | ok | 4/4 | 2 | 71.107 | 1.182x | 73.382 | 73.686 | 0.0276 | 0.836x | ok | - |
| LightX2V | default | 2 | 36.239 | 1.167x | ok | 4/4 | 2 | 68.747 | 1.143x | 70.307 | 70.444 | 0.0289 | 0.876x | ok | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |

### ltx2.3_twostage_t2v_2gpus

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2.3 | text-to-video | 768x512x121 | 30 | gs=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-80gb-2gpu | 2 | 12.025 | 1.000x | ok | 4/4 | 2 | 23.561 | 1.000x | 24.067 | 24.068 | 0.0849 | 1.000x | ok | - |
| vLLM-Omni | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No compatible vLLM-Omni LTX2.3 serving profile is configured. |
| LightX2V | h100-1gpu | 1 | - | - | failed | - | - | - | - | - | - | - | - | failed | LightX2V task FAILED: {'task_id': 'CNU2-5061-48JW-FAF3-WHNG', 'status': 'failed', 'start_time': '2026-07-02T09:07:45.073092', 'end_time': '2026-07-02T09:07:48.840916', 'error': 'CUDA out of memory. Tried to allocate 30.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 6.19 MiB is free. Including non-PyTorch memory, this process has 79.09 GiB memory in use. Of the allocated memory 78.16 GiB is allocated by PyTorch, and 198.54 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)', 'error_type': 'OutOfMemoryError', 'save_result_path': '/tmp/lightx2v_ltx2.3_twostage_t2v_2gpus_1782983264225.mp4'}; OOM on H100 80GB: LTX-2.3 22B two-stage 121-frame t2av peak exceeds 80GB even with every lossless lever (seq_p_size=2 activation sharding + block cpu_offload + Gemma/VAE offload + exact flash_attn3). Upstream profiles target H200 141GB. command_profiles below are the best-attempted lossless config, kept for provenance. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |

### cosmos3_nano_t2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | h100-4gpu-cfg-ulysses-speed-compile | 4 | 54.106 | 1.000x | ok | 4/4 | 2 | 106.241 | 1.000x | 107.120 | 107.242 | 0.0187 | 1.000x | ok | - |
| vLLM-Omni | default | 4 | 65.933 | 1.219x | ok | 4/4 | 2 | 122.802 | 1.156x | 126.611 | 127.144 | 0.0160 | 0.856x | ok | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No tracked LightX2V Cosmos3 serving path in this benchmark. |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
