#!/usr/bin/env bash
# The 2026-10-10 round's one-GPU cells on a single RTX 5090 desktop
# (rdxa-5090-pa: 32 GB card, Ryzen 9 9950X, 60 GB RAM, PCIe 5.0 NVMe, Ubuntu
# 26.04, driver 595.99.02 open kernel module, CUDA 13.1 toolkit).
#
# Wraps scripts/run_b200_cross_framework_20260912.sh twice, one batch after the
# other on the one card:
#   1. Qwen-Image-2.1 t2i and edit: every framework with a 5090 path. These
#      cases move here from the rx 4x5090 box so they run alongside its two-GPU
#      H3 batch on a different host.
#   2. MiniMax-H3 at the consumer tier (minimax_h3_t2va_480p_1gpu): the sglang
#      cookbook's 60 GB-host recipe against ComfyUI --fast-disk --cache-none,
#      the comparison the cookbook measured on this class of machine.
#
# Bare host, no container: sglang lives in a uv-managed Python 3.12 venv
# ($DBF_STATE_DIR/base) at the round's pin, competitors in their own venvs.
# Versions resolved 2026-10-10 ~05:30 UTC (same as the round's other boxes):
#   sglang origin/main 3831e7e09 (torch 2.14.1) | vllm 0.31.0 + vllm-omni main f69b1f2b
#   ComfyUI master 0df64eb2 | LightX2V main b6d38283 (torch 2.11)
#
#   bash scripts/run_rtx5090_pa_20261010.sh
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-${HOME}/h3bench}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-rtx5090}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-21600}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"
export GPUS="${GPUS:-0}"
export PATH="${HOME}/.local/bin:/usr/local/cuda/bin:${PATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# shellcheck disable=SC1091
source "${DBF_STATE_DIR}/base/bin/activate"
runner="${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"

DBF_FRAMEWORKS="${QWEN_FRAMEWORKS:-sglang vllm-omni comfyui lightx2v}" TAG="${QWEN_TAG:-pa5090qwen1010}" \
BASE_PORT=48001 CASES="${QWEN_CASES:-qwen_image_21_t2i_1024 qwen_image_21_edit_1024}" bash "$runner"

DBF_FRAMEWORKS="${H3_FRAMEWORKS:-sglang comfyui}" TAG="${H3_TAG:-pa5090h31010}" \
BASE_PORT=48101 CASES="${H3_CASES:-minimax_h3_t2va_480p_1gpu}" bash "$runner"
