#!/usr/bin/env bash
set -euo pipefail

# Reproduce the H200 Cosmos3 Nano comparison between SGLang-Diffusion PR #26926
# and vLLM-Omni PR #3454. Run inside the benchmark repo on a non-CI H200/H100
# container where the selected SGLang checkout is installed.

RUN_ID="${RUN_ID:-h200-cosmos3-20260601-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30420}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
SINGLE_OUTPUT="${SINGLE_OUTPUT:-${OUTPUT_DIR}/${RUN_ID}-single.json}"
THROUGHPUT_OUTPUT="${THROUGHPUT_OUTPUT:-${OUTPUT_DIR}/${RUN_ID}-throughput.json}"
REPORT_DIR="${REPORT_DIR:-tmp/report}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${REPORT_DIR}/${RUN_ID}.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.21.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/MaciejBalaNV/vllm-omni.git@453bd09ab3d1e2f10e3d1b63ff05f090ab49a7cc}"
export VLLM_OMNI_SERVER_BIN="${VLLM_OMNI_SERVER_BIN:-vllm}"
export VLLM_OMNI_REQUIRED_HELP_ARGS="${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni --model-class-name --stage-init-timeout --no-guardrails --diffusion-attention-backend --cache-backend --cfg-parallel-size --ulysses-degree}"
export SGLANG_COSMOS3_EXPECTED_REF="${SGLANG_COSMOS3_EXPECTED_REF:-sgl-project/sglang#26926@b7718f8bf87cb676802c60fcf3c71664d243d9c7}"

if [[ -n "${COSMOS3_SINGLE_CASES:-}" ]]; then
  read -r -a SINGLE_CASES <<< "${COSMOS3_SINGLE_CASES}"
else
  SINGLE_CASES=(
    cosmos3_nano_t2i_720p
    cosmos3_nano_t2v_720p_189f
    cosmos3_nano_i2v_720p_189f
  )
fi

if [[ -n "${COSMOS3_THROUGHPUT_CASES:-}" ]]; then
  read -r -a THROUGHPUT_CASES <<< "${COSMOS3_THROUGHPUT_CASES}"
else
  THROUGHPUT_CASES=(cosmos3_nano_t2i_720p)
fi

if [[ -n "${COSMOS3_FRAMEWORKS:-}" ]]; then
  read -r -a FRAMEWORKS <<< "${COSMOS3_FRAMEWORKS}"
else
  FRAMEWORKS=(sglang vllm-omni)
fi

mkdir -p "${OUTPUT_DIR}" "${REPORT_DIR}"

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks "${FRAMEWORKS[@]}" \
  --case-ids "${SINGLE_CASES[@]}" \
  --hardware-profile h200 \
  --run-id "${RUN_ID}-single" \
  --port "${PORT}" \
  --output "${SINGLE_OUTPUT}"

RESULT_FILES=("${SINGLE_OUTPUT}")

if [[ "${COSMOS3_RUN_THROUGHPUT:-1}" == "1" ]]; then
  diffusion-bench-compare \
    --config "${CONFIG}" \
    --modes throughput \
    --frameworks "${FRAMEWORKS[@]}" \
    --case-ids "${THROUGHPUT_CASES[@]}" \
    --hardware-profile h200 \
    --run-id "${RUN_ID}-throughput" \
    --port "$((PORT + 10))" \
    --output "${THROUGHPUT_OUTPUT}"
  RESULT_FILES+=("${THROUGHPUT_OUTPUT}")
fi

RESULTS="${RESULT_FILES[*]}" \
OUTPUT_JSON="${MERGED_OUTPUT}" \
DASHBOARD_MD="${MERGED_OUTPUT%.json}.dashboard.md" \
ISSUE_MD="${MERGED_OUTPUT%.json}.issue.md" \
IMAGE_PNG="${MERGED_OUTPUT%.json}.png" \
IMAGE_SVG="${MERGED_OUTPUT%.json}.svg" \
RUN_ID="${RUN_ID}" \
scripts/generate_h200_report_artifacts.sh
