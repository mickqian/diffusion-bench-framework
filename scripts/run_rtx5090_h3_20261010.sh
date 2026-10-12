#!/usr/bin/env bash
# The 2026-10-10 round on 2x RTX 5090 (32 GB): MiniMax-H3 at its consumer
# operating point, then the one-GPU Qwen-Image-2.1 cases.
#
# Wraps scripts/run_b200_cross_framework_20260912.sh twice, one batch after the
# other on the same cards (two batches at once on one host contaminate each
# other's latency through the shared host and PCIe):
#   1. MiniMax-H3 T2VA and Ref2VA, two GPUs: sglang `rtx5090-2gpu` (the
#      cookbook's TP2 + 20 resident DiT blocks), vLLM-Omni `rtx5090-2gpu` (its
#      recipe's current two-GPU command: TP2 + DLO 20 resident + text-encoder TP2
#      + VAE patch parallel 2, eager, cuDNN), ComfyUI on one card. FastVideo and
#      LightX2V have no 32 GB profile for H3 and are left out of this batch, as
#      are the four-GPU H3 cases (the resident datacenter commands do not fit 4x32 GB).
#   2. Qwen-Image-2.1 t2i and edit, one GPU: the frameworks with a 5090 path.
#
# Box: rx devbox probe-menu-5090 (4x RTX 5090 on a shared 5090-novita-ci node),
# image lmsysorg/sglang:latest with /sgl-workspace/sglang at origin/main.
# Versions resolved 2026-10-10 ~05:30 UTC:
#   sglang origin/main 3831e7e09 | vllm 0.31.0 + vllm-omni main f69b1f2b
#   ComfyUI master 0df64eb2 | LightX2V main b6d38283 (torch 2.11)
#
#   bash scripts/run_rtx5090_h3_20261010.sh
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/scratch/h3bench}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-rtx5090}"
# H3 at 50 steps is ~9-10 min per request on two 5090s; a framework cell is
# four or five of them.
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-28800}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"
export GPUS="${GPUS:-0,1}"
runner="${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"

DBF_FRAMEWORKS="${H3_FRAMEWORKS:-sglang vllm-omni comfyui}" TAG="${H3_TAG:-rtx5090h31010}" \
BASE_PORT=47001 CASES="${H3_CASES:-minimax_h3_t2va_5s minimax_h3_ref2va_5s}" bash "$runner"

DBF_FRAMEWORKS="${QWEN_FRAMEWORKS:-sglang vllm-omni comfyui lightx2v}" TAG="${QWEN_TAG:-rtx5090qwen1010}" \
BASE_PORT=47101 CASES="${QWEN_CASES:-qwen_image_21_t2i_1024 qwen_image_21_edit_1024}" bash "$runner"
