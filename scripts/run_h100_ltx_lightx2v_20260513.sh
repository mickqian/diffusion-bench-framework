#!/usr/bin/env bash
set -euo pipefail

# Targeted H100 rerun for latest upstream LightX2V LTX-2/LTX-2.3 support.

RUN_ID="${RUN_ID:-h100-ltx-lightx2v-20260513-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30520}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
DASHBOARD="${DASHBOARD:-${OUTPUT%.json}.dashboard.md}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
export SINGLE_E2E_CASES="${SINGLE_E2E_CASES:-ltx2_twostage_t2v ltx2.3_twostage_t2v_2gpus}"
export SINGLE_E2E_FRAMEWORKS="${SINGLE_E2E_FRAMEWORKS:-lightx2v}"

read -r -a CASES <<< "${SINGLE_E2E_CASES}"
read -r -a FRAMEWORKS <<< "${SINGLE_E2E_FRAMEWORKS}"

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks "${FRAMEWORKS[@]}" \
  --case-ids "${CASES[@]}" \
  --hardware-profile h100 \
  --run-id "${RUN_ID}" \
  --port "${PORT}" \
  --output "${OUTPUT}"

diffusion-bench-dashboard \
  --results "${OUTPUT}" \
  --output "${DASHBOARD}"
