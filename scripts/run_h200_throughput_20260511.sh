#!/usr/bin/env bash
set -euo pipefail

# Reproduce the 2026-05-11 H200 throughput report for faster image cases.
# Run inside the benchmark repo on the H200 container where sglang is installed.

RUN_ID="${RUN_ID:-h200-throughput-20260511-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30420}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
DASHBOARD="${DASHBOARD:-${OUTPUT%.json}.dashboard.md}"
THROUGHPUT_NUM_REQUESTS="${THROUGHPUT_NUM_REQUESTS:-32}"
THROUGHPUT_MAX_CONCURRENCY="${THROUGHPUT_MAX_CONCURRENCY:-4}"
THROUGHPUT_REQUEST_RATE="${THROUGHPUT_REQUEST_RATE:-inf}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-1}"
export DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="${DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS:---batching-max-size ${THROUGHPUT_MAX_CONCURRENCY} --batching-delay-ms 0}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
export LIGHTX2V_FLASH_ATTN_INSTALL_SPEC="${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
export LIGHTX2V_FA3_HF_REPO="${LIGHTX2V_FA3_HF_REPO:-varunneal/flash-attention-3}"
export LIGHTX2V_FA3_HF_REVISION="${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
export LIGHTX2V_FA3_HF_SUBDIR="${LIGHTX2V_FA3_HF_SUBDIR:-build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3}"
export LIGHTX2V_FLASHINFER_INSTALL_SPEC="${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"

if [[ -n "${THROUGHPUT_CASES:-}" ]]; then
  read -r -a CASES <<< "${THROUGHPUT_CASES}"
else
  CASES=(
    zimage_turbo_t2i_1024
    qwen_image_2512_t2i_1024
    qwen_image_edit_2511
    flux1_dev_t2i_1024
  )
fi

if [[ -n "${THROUGHPUT_FRAMEWORKS:-}" ]]; then
  read -r -a FRAMEWORKS <<< "${THROUGHPUT_FRAMEWORKS}"
else
  FRAMEWORKS=(sglang vllm-omni lightx2v)
fi

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes throughput \
  --frameworks "${FRAMEWORKS[@]}" \
  --case-ids "${CASES[@]}" \
  --hardware-profile h200 \
  --throughput-num-requests "${THROUGHPUT_NUM_REQUESTS}" \
  --throughput-max-concurrency "${THROUGHPUT_MAX_CONCURRENCY}" \
  --throughput-request-rate "${THROUGHPUT_REQUEST_RATE}" \
  --run-id "${RUN_ID}" \
  --port "${PORT}" \
  --output "${OUTPUT}"

diffusion-bench-dashboard \
  --results "${OUTPUT}" \
  --output "${DASHBOARD}"

if [[ "${PUBLISH_ISSUE:-0}" == "1" ]]; then
  diffusion-bench-dashboard \
    --results "${OUTPUT}" \
    --output "${DASHBOARD}" \
    --report-repo mickqian/diffusion-bench-framework \
    --report-issue 1
fi
