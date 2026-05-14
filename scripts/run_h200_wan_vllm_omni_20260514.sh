#!/usr/bin/env bash
set -euo pipefail

# Targeted H200 rerun for Wan cells that used to be not_configured in vLLM-Omni.

RUN_ID="${RUN_ID:-h200-wan-vllm-omni-20260514-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30420}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
DASHBOARD="${DASHBOARD:-${OUTPUT%.json}.dashboard.md}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"

if [[ -n "${WAN_VLLM_CASES:-}" ]]; then
  read -r -a CASES <<< "${WAN_VLLM_CASES}"
else
  CASES=(
    wan21_t2v_1_3b_480p
    wan21_i2v_14b_480p
    wan22_t2v_a14b_720p
    wan22_ti2v_5b_704p
    wan21_i2v_14b_720p
    wan22_i2v_a14b_720p
  )
fi

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids "${CASES[@]}" \
  --hardware-profile h200 \
  --run-id "${RUN_ID}" \
  --port "${PORT}" \
  --output "${OUTPUT}"

diffusion-bench-dashboard \
  --results "${OUTPUT}" \
  --output "${DASHBOARD}"
