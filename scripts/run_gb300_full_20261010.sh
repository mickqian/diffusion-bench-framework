#!/usr/bin/env bash
# The 2026-10-10 round on GB300 (aarch64 Grace + Blackwell Ultra, 4 GPUs per
# node): every Hopper/Blackwell-class case, one harness revision, latest-vs-latest.
#
# Wraps scripts/run_b200_cross_framework_20260912.sh. The box is one rx devbox
# with two 4-GPU pods on different nodes (mm-vmm-lifecycle-1010, gb300-tml-k8s),
# so the matrix is split in two and both halves run at once, one per pod.
# Separate nodes share no host or PCIe, so neither half's latency sees the other:
#   ROLE=image (rank 0): the image cases, then LTX-2/2.3 and the Cosmos3 videos.
#     The gated checkpoints (FLUX.1/FLUX.2-dev, ideogram-4) come last, so the
#     token can arrive while the rest runs.
#   ROLE=video (rank 1): the four MiniMax-H3 cases, then Wan2.2.
# Qwen-Image 2512 / Edit 2511 are retired for Qwen-Image-2.1 this round; the
# single-32 GB H3 case is consumer-only and not run here.
#
# Each pod: image lmsysorg/sglang:latest (aarch64), /sgl-workspace/sglang moved
# to the pin; framework venvs, HF cache and logs on node-local /scratch/gb300.
# aarch64 install notes (scripts/install_comparison_frameworks.sh):
#   * LightX2V installs without decord (no aarch64 wheel; imported lazily, only
#     by video-input runners none of these cases use)
#   * FastVideo with FASTVIDEO_INSTALL_FA3=0: the aarch64 FA3 artifact links
#     libcudart.so.12, and FA3 is Hopper-only; Blackwell runs FA4 / sm100 VSA
# Versions resolved 2026-10-10 ~05:30 UTC (same as the round's other boxes):
#   sglang origin/main 3831e7e09 | vllm 0.31.0 + vllm-omni main f69b1f2b
#   LightX2V main b6d38283 (torch 2.11) | ComfyUI master 0df64eb2
#   FastVideo main d5253fd7 | tensorrt-llm 1.3.0rc29
#
#   ROLE=image bash scripts/run_gb300_full_20261010.sh   # on rank 0
#   ROLE=video bash scripts/run_gb300_full_20261010.sh   # on rank 1
#
# Then merge both pods' results and publish -- dry run first:
#   scripts/merge_and_publish_run.sh "<logs>/run_gb300*1010_*.json" \
#       gb300x4-full-20261010 "4xGB300 cross-framework (latest-vs-latest)" \
#       "4x NVIDIA GB300 288GB" scripts/run_gb300_full_20261010.sh
set -u

ROLE="${ROLE:?ROLE=image or ROLE=video}"
export DBF_STATE_DIR="${DBF_STATE_DIR:-/scratch/gb300}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual comfyui fastvideo}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-gb300}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-21600}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"
export GPUS="${GPUS:-0,1,2,3}"
case "${ROLE}" in
  image)
    export TAG="${TAG:-gb300img1010}" BASE_PORT="${BASE_PORT:-49001}"
    export CASES="${CASES:-
qwen_image_21_t2i_1024
qwen_image_21_t2i_1024_2gpu
qwen_image_21_edit_1024
zimage_turbo_t2i_1024
cosmos3_nano_t2i_720p
ltx2_twostage_t2v
ltx2.3_twostage_t2v_2gpus
cosmos3_nano_t2v_720p_189f
cosmos3_nano_i2v_720p_189f
flux1_dev_t2i_1024
flux2_dev_t2i_1024
ideogram4_t2i_1024_2gpu_tp
}" ;;
  video)
    export TAG="${TAG:-gb300vid1010}" BASE_PORT="${BASE_PORT:-49501}"
    export CASES="${CASES:-
minimax_h3_t2va_5s
minimax_h3_ref2va_5s
minimax_h3_t2va_5s_4gpu
minimax_h3_fasth3_v2_t2va_5s
wan22_t2v_a14b_720p
}" ;;
  *) echo "ROLE must be image or video" >&2; exit 1 ;;
esac

cd "${DBF_REPO_DIR}" || exit 1
echo "=== ${ROLE} | $(git log --oneline -1) | sglang $(git -C /sgl-workspace/sglang log --oneline -1) ==="
exec bash "${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"
