#!/usr/bin/env bash
# Competitor A/B pass of the 2026-10-22 H200 round, before the matrix: each arm
# is a candidate command from the round's recipe audit (vLLM-Omni, LightX2V,
# ComfyUI, FastH3 V2), measured against the profile H200 currently selects
# ("h200"). Winners are promoted into the case's H200 profile, then the matrix
# (scripts/run_h200_full_20261022.sh) measures every cell through its profile.
#
#   ROLE=image bash scripts/run_h200_ab_20261022.sh   # 2xH200 box
#   ROLE=video bash scripts/run_h200_ab_20261022.sh   # 4xH200 box
#
# Arms print as `ARM <case> <framework> <arm> r<k> RESULT ...` lines.
set -u

ROLE="${ROLE:?ROLE=image or ROLE=video}"
export DBF_STATE_DIR="${DBF_STATE_DIR:-/scratch/r1022}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs-ab}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_SGLANG_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:-${DBF_STATE_DIR}/sgl/python}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"

# shellcheck source=scripts/gpu_job_lock.sh
source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
gpu_lock_acquire

arms() {  # <case> <framework> <rounds> <arms...>
  local case_id=$1 fw=$2 rounds=$3
  shift 3
  ARMS_CASE="${case_id}" ARMS_FW="${fw}" ARMS="$*" ROUNDS="${rounds}" \
    bash "${DBF_REPO_DIR}/scripts/run_competitor_arms.sh" "${TAG}_${fw}" "${port}"
  port=$((port + 100))
}

echo "=== AB ${ROLE} START $(date -Is) | $(git -C "${DBF_REPO_DIR}" log --oneline -1) ==="
case "${ROLE}" in
  image)
    export GPUS=0,1 TAG=ab1022img
    port=63000
    arms zimage_turbo_t2i_1024 vllm-omni 2 h200 cand-usp2 cand-usp2-permute
    arms zimage_turbo_t2i_1024 lightx2v 2 h200 cand-fusion
    arms cosmos3_nano_t2i_720p vllm-omni 2 h200 cand-dynamic
    arms qwen_image_21_t2i_1024 vllm-omni 2 h200 cand-static
    arms qwen_image_21_t2i_1024 lightx2v 2 h200 cand-vae-compile
    arms qwen_image_21_t2i_1024 comfyui 2 h200 cand-gpuonly cand-compile-gpuonly
    arms qwen_image_21_t2i_1024_2gpu vllm-omni 2 h200 cand-static
    arms qwen_image_21_edit_1024 vllm-omni 2 h200 cand-static
    arms ideogram4_t2i_1024_2gpu_tp comfyui 2 h200 cand-gpuonly cand-compile-gpuonly
    arms ltx2_twostage_t2v comfyui 1 h200 cand-gpuonly
    arms minimax_h3_t2va_5s vllm-omni 1 h200 cand-static
    ;;
  video)
    export GPUS=0,1,2,3 TAG=ab1022vid
    port=64000
    arms minimax_h3_fasth3_v2_t2va_5s vllm-omni 1 h200 cudnn-vsa-candidate
    arms cosmos3_nano_t2v_720p_189f vllm-omni 1 h200 cand-permute
    arms wan22_t2v_a14b_720p vllm-omni 1 h200 cand-hsdp cand-permute cand-devpost
    arms wan22_t2v_a14b_720p lightx2v 1 h200 cand-tuned
    arms wan22_t2v_a14b_720p comfyui 1 h200 cand-gpuonly
    ;;
  *) echo "ROLE must be image or video" >&2; exit 1 ;;
esac
echo "=== AB ${ROLE} DONE $(date -Is) ==="
