#!/usr/bin/env bash
set -euo pipefail

RUN_ID_ROOT="${RUN_ID_ROOT:-h200-cosmos3-latest-main-2gpu-$(date -u +%Y%m%d-%H%M%S)}"
PORT_BASE="${PORT_BASE:-30570}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
REPORT_DIR="${REPORT_DIR:-tmp/report}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-/tmp/diffusion-bench-pip-tmp}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.22.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git@40b2959183d7043cd9d35f5f6387fbb8d1375c57}"
export VLLM_OMNI_SERVER_BIN="${VLLM_OMNI_SERVER_BIN:-vllm}"
export VLLM_OMNI_REQUIRED_HELP_ARGS="${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni --model-class-name --stage-init-timeout --stage-configs-path --diffusion-attention-backend --cache-backend --cfg-parallel-size --ulysses-degree}"
export SGLANG_COSMOS3_EXPECTED_REF="${SGLANG_COSMOS3_EXPECTED_REF:-sgl-project/sglang@3d540563916394115f9612f30601ae3fa97ada7c}"

mkdir -p "${REPORT_DIR}" runs

SGLANG_T2I_SINGLE="${REPORT_DIR}/${RUN_ID_ROOT}-sglang-t2i-single.json"
SGLANG_VIDEO_SINGLE="${REPORT_DIR}/${RUN_ID_ROOT}-sglang-video-single.json"
SGLANG_T2I_THROUGHPUT="${REPORT_DIR}/${RUN_ID_ROOT}-sglang-t2i-throughput.json"
VLLM_T2I_SINGLE="${REPORT_DIR}/${RUN_ID_ROOT}-vllm-t2i-single.json"
VLLM_VIDEO_SINGLE="${REPORT_DIR}/${RUN_ID_ROOT}-vllm-video-single.json"
VLLM_T2I_THROUGHPUT="${REPORT_DIR}/${RUN_ID_ROOT}-vllm-t2i-throughput.json"

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --sglang-profile h200-1gpu-fa \
  --run-id "${RUN_ID_ROOT}-sglang-t2i-single" \
  --port "${PORT_BASE}" \
  --output "${SGLANG_T2I_SINGLE}"

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang \
  --case-ids cosmos3_nano_t2v_720p_189f cosmos3_nano_i2v_720p_189f \
  --hardware-profile h200 \
  --sglang-profile h200-2gpu-cfg \
  --run-id "${RUN_ID_ROOT}-sglang-video-single" \
  --port "$((PORT_BASE + 10))" \
  --output "${SGLANG_VIDEO_SINGLE}"

diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes throughput \
  --frameworks sglang \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --sglang-profile h200-1gpu-fa-batch4 \
  --throughput-num-requests "${THROUGHPUT_NUM_REQUESTS:-8}" \
  --throughput-max-concurrency "${THROUGHPUT_MAX_CONCURRENCY:-4}" \
  --throughput-request-rate "${THROUGHPUT_REQUEST_RATE:-inf}" \
  --run-id "${RUN_ID_ROOT}-sglang-t2i-throughput" \
  --port "$((PORT_BASE + 20))" \
  --output "${SGLANG_T2I_THROUGHPUT}"

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h200-1gpu-fa \
diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --run-id "${RUN_ID_ROOT}-vllm-t2i-single" \
  --port "$((PORT_BASE + 30))" \
  --output "${VLLM_T2I_SINGLE}"

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h200-2gpu-cfg \
diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids cosmos3_nano_t2v_720p_189f cosmos3_nano_i2v_720p_189f \
  --hardware-profile h200 \
  --run-id "${RUN_ID_ROOT}-vllm-video-single" \
  --port "$((PORT_BASE + 40))" \
  --output "${VLLM_VIDEO_SINGLE}"

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h200-1gpu-fa \
diffusion-bench-compare \
  --config "${CONFIG}" \
  --modes throughput \
  --frameworks vllm-omni \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --throughput-num-requests "${THROUGHPUT_NUM_REQUESTS:-8}" \
  --throughput-max-concurrency "${THROUGHPUT_MAX_CONCURRENCY:-4}" \
  --throughput-request-rate "${THROUGHPUT_REQUEST_RATE:-inf}" \
  --run-id "${RUN_ID_ROOT}-vllm-t2i-throughput" \
  --port "$((PORT_BASE + 50))" \
  --output "${VLLM_T2I_THROUGHPUT}"

RESULTS="${SGLANG_T2I_SINGLE} ${SGLANG_VIDEO_SINGLE} ${SGLANG_T2I_THROUGHPUT} ${VLLM_T2I_SINGLE} ${VLLM_VIDEO_SINGLE} ${VLLM_T2I_THROUGHPUT}" \
OUTPUT_JSON="${REPORT_DIR}/${RUN_ID_ROOT}.json" \
DASHBOARD_MD="${REPORT_DIR}/${RUN_ID_ROOT}.dashboard.md" \
ISSUE_MD="${REPORT_DIR}/${RUN_ID_ROOT}.issue.md" \
IMAGE_PNG="${REPORT_DIR}/${RUN_ID_ROOT}.png" \
IMAGE_SVG="${REPORT_DIR}/${RUN_ID_ROOT}.svg" \
RUN_ID="${RUN_ID_ROOT}" \
scripts/generate_h200_report_artifacts.sh
