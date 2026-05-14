#!/usr/bin/env bash
set -euo pipefail

# H200 throughput suite for cases with SGLang-Diffusion, vLLM-Omni, and LightX2V profiles.

RUN_ID="${RUN_ID:-h200-throughput-common-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30420}"
OUTPUT="${OUTPUT:-runs/${RUN_ID}.json}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
DASHBOARD="${DASHBOARD:-${OUTPUT%.json}.dashboard.md}"
ISSUE_MD="${ISSUE_MD:-${OUTPUT%.json}.issue.md}"
IMAGE_PNG="${IMAGE_PNG:-${OUTPUT%.json}.png}"
IMAGE_SVG="${IMAGE_SVG:-${OUTPUT%.json}.svg}"
IMAGE_OUTPUT="${IMAGE_OUTPUT:-${OUTPUT%.json}.image.json}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-${OUTPUT%.json}.video.json}"
IMAGE_NUM_REQUESTS="${IMAGE_NUM_REQUESTS:-32}"
IMAGE_MAX_CONCURRENCY="${IMAGE_MAX_CONCURRENCY:-4}"
VIDEO_NUM_REQUESTS="${VIDEO_NUM_REQUESTS:-8}"
VIDEO_MAX_CONCURRENCY="${VIDEO_MAX_CONCURRENCY:-2}"
THROUGHPUT_REQUEST_RATE="${THROUGHPUT_REQUEST_RATE:-inf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-1}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
export LIGHTX2V_FLASH_ATTN_INSTALL_SPEC="${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
export LIGHTX2V_FA3_HF_REPO="${LIGHTX2V_FA3_HF_REPO:-varunneal/flash-attention-3}"
export LIGHTX2V_FA3_HF_REVISION="${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
export LIGHTX2V_FA3_HF_SUBDIR="${LIGHTX2V_FA3_HF_SUBDIR:-build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3}"
export LIGHTX2V_FLASHINFER_INSTALL_SPEC="${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"

if [[ -n "${THROUGHPUT_IMAGE_CASES:-}" ]]; then
  read -r -a IMAGE_CASES <<< "${THROUGHPUT_IMAGE_CASES}"
else
  IMAGE_CASES=(
    zimage_turbo_t2i_1024
    flux2_dev_t2i_1024
  )
fi

if [[ -n "${THROUGHPUT_VIDEO_CASES:-}" ]]; then
  read -r -a VIDEO_CASES <<< "${THROUGHPUT_VIDEO_CASES}"
else
  VIDEO_CASES=(
    wan21_t2v_1_3b_480p
    wan22_ti2v_5b_704p
  )
fi

if [[ -n "${THROUGHPUT_FRAMEWORKS:-}" ]]; then
  read -r -a FRAMEWORKS <<< "${THROUGHPUT_FRAMEWORKS}"
else
  FRAMEWORKS=(sglang vllm-omni lightx2v)
fi

run_group() {
  local output="$1"
  local group_run_id="$2"
  local num_requests="$3"
  local max_concurrency="$4"
  shift 4
  local cases=("$@")

  export DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--batching-max-size ${max_concurrency} --batching-delay-ms 0"

  diffusion-bench-compare \
    --config "${CONFIG}" \
    --modes throughput \
    --frameworks "${FRAMEWORKS[@]}" \
    --case-ids "${cases[@]}" \
    --hardware-profile h200 \
    --throughput-num-requests "${num_requests}" \
    --throughput-max-concurrency "${max_concurrency}" \
    --throughput-request-rate "${THROUGHPUT_REQUEST_RATE}" \
    --run-id "${group_run_id}" \
    --port "${PORT}" \
    --output "${output}"
}

run_group "${IMAGE_OUTPUT}" "${RUN_ID}-image" "${IMAGE_NUM_REQUESTS}" "${IMAGE_MAX_CONCURRENCY}" "${IMAGE_CASES[@]}"
run_group "${VIDEO_OUTPUT}" "${RUN_ID}-video" "${VIDEO_NUM_REQUESTS}" "${VIDEO_MAX_CONCURRENCY}" "${VIDEO_CASES[@]}"

"${PYTHON_BIN}" -m diffusion_bench.build_report_artifacts \
  --config "${CONFIG}" \
  --results "${IMAGE_OUTPUT}" "${VIDEO_OUTPUT}" \
  --output-json "${OUTPUT}" \
  --dashboard-md "${DASHBOARD}" \
  --issue-md "${ISSUE_MD}" \
  --image-png "${IMAGE_PNG}" \
  --image-svg "${IMAGE_SVG}" \
  --run-id "${RUN_ID}"

if [[ "${PUBLISH_ISSUE:-0}" == "1" ]]; then
  diffusion-bench-dashboard \
    --results "${OUTPUT}" \
    --output "${DASHBOARD}" \
    --report-repo mickqian/diffusion-bench-framework \
    --report-issue 1
fi
