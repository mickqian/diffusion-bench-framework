#!/usr/bin/env bash
set -euo pipefail

# Reproduce the 2026-05-10 H200 single-request report shape.
# Run inside the benchmark repo on the H200 container where sglang is installed.

RUN_ID="${RUN_ID:-h200-single-e2e-20260510-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30320}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
DASHBOARD="${DASHBOARD:-${OUTPUT%.json}.dashboard.md}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-1}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@573b9613adb0c1d33894b0920b5e12c87e42d280}"
export LIGHTX2V_FLASH_ATTN_INSTALL_SPEC="${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"

CASES=(
  flux1_dev_t2i_1024
  flux2_dev_t2i_1024
  qwen_image_2512_t2i_1024
  qwen_image_edit_2511
  zimage_turbo_t2i_1024
  wan21_t2v_1_3b_480p
  wan21_i2v_14b_480p
  wan22_t2v_a14b_720p
  wan22_ti2v_5b_704p
  wan21_i2v_14b_720p
  wan22_i2v_a14b_720p
  ltx2_twostage_t2v
  ltx2.3_twostage_t2v_2gpus
)

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang vllm-omni lightx2v \
  --case-ids "${CASES[@]}" \
  --hardware-profile h200 \
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
