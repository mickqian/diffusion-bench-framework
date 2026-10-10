#!/usr/bin/env bash
# The 2026-10-10 MiniMax-H3 round on AMD Instinct MI355X: sglang against
# vLLM-Omni, the one competitor with a documented ROCm path for H3.
#
# Wraps scripts/run_b200_cross_framework_20260912.sh (the case list and the
# per-case runner) with this box's settings. What is specific to AMD:
#   * profiles: sglang `mi355x-*` (AITER attention, SGLANG_USE_AITER=1, the
#     cookbook builder's AMD policy), vLLM-Omni `rocm-*` (its recipe's "AMD ROCm"
#     section). The hardware token `mi355x` selects them; before 2026-10-10 the
#     token list had no AMD class and every framework fell back to its CUDA default
#   * FastH3 V2 is not in the list: its VSA kernel is CUDA-only in both engines
#   * vLLM-Omni installed with VLLM_OMNI_TARGET_DEVICE=rocm from the vLLM ROCm
#     wheel index (scripts/install_comparison_frameworks.sh), into a venv that
#     does not inherit the sglang image's PIP_CONSTRAINT torch pin
#   * GPU cleanup between cases finds this runner's processes by /dev/kfd and
#     their CUDA_VISIBLE_DEVICES (amd-smi reports host pids, useless in the pod)
#
# Box: rx devbox h3bench-mi355x (8x MI355X, mi355x-amd-slurm), image
# lmsysorg/sglang:v0.5.21-rocm10-mi35x with /sgl-workspace/sglang moved to
# origin/main (keeping the image's ROCm pyproject transform).
# Versions resolved 2026-10-10 ~05:30 UTC:
#   sglang origin/main 3831e7e09 | vllm 0.31.0+rocm723 + vllm-omni main f69b1f2b
#
#   bash scripts/run_mi355x_h3_20261010.sh
#
# Then merge and publish -- dry run first:
#   scripts/merge_and_publish_run.sh "<DBF_LOG_DIR>/run_mi355xh31010_*.json" \
#       mi355x-h3-20261010 "MiniMax-H3 on 8x MI355X (latest-vs-latest)" \
#       "8x AMD Instinct MI355X 288GB" scripts/run_mi355x_h3_20261010.sh
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/root/h3bench}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-mi355x}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-21600}"
export TAG="${TAG:-mi355xh31010}"
export GPUS="${GPUS:-0,1,2,3}"
export BASE_PORT="${BASE_PORT:-46001}"
export CASES="${CASES:-
minimax_h3_t2va_5s
minimax_h3_ref2va_5s
minimax_h3_t2va_5s_4gpu
qwen_image_21_t2i_1024
qwen_image_21_edit_1024
}"

exec bash "${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"
