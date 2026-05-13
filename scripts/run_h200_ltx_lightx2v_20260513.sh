#!/usr/bin/env bash
set -euo pipefail

# Targeted H200 rerun for latest upstream LightX2V LTX-2/LTX-2.3 support.

RUN_ID="${RUN_ID:-h200-ltx-lightx2v-20260513-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30520}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"

export RUN_ID
export PORT
export OUTPUT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
export SINGLE_E2E_CASES="${SINGLE_E2E_CASES:-ltx2_twostage_t2v ltx2.3_twostage_t2v_2gpus}"
export SINGLE_E2E_FRAMEWORKS="${SINGLE_E2E_FRAMEWORKS:-lightx2v}"

scripts/run_h200_single_e2e_20260510.sh
