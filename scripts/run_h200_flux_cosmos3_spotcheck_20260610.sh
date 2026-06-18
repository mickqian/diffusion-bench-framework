#!/usr/bin/env bash
set -euo pipefail

# Targeted H200 spot-check:
# 1. rerun FLUX SGLang-Diffusion/vLLM-Omni with compile allowed
# 2. run one Cosmos3 image case and one Cosmos3 video case with compile disabled

RUN_ID_ROOT="${RUN_ID_ROOT:-h200-flux-cosmos3-spotcheck-$(date -u +%Y%m%d-%H%M%S)}"
PORT_BASE="${PORT_BASE:-30800}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
REPORT_DIR="${REPORT_DIR:-tmp/report/${RUN_ID_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/scratch/framework-venvs}"
export SGLANG_DIFFUSION_PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-/scratch/pip-tmp}"
export HF_HOME="${HF_HOME:-/scratch/hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DIFFUSION_BENCH_HF_CACHE_DIR}}"
export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.22.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git@73df8326d76fe3bba0b7b5a6abf6ad68976f24e8}"
export VLLM_OMNI_SERVER_BIN="${VLLM_OMNI_SERVER_BIN:-vllm}"
export VLLM_OMNI_REQUIRED_HELP_ARGS="${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni --model-class-name --stage-init-timeout --no-guardrails --diffusion-attention-backend --cache-backend --cfg-parallel-size --ulysses-degree}"

mkdir -p "${REPORT_DIR}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_FILES=()
status=0
port="${PORT_BASE}"

prepare_framework() {
  local framework="$1"
  scripts/install_comparison_frameworks.sh "${framework}"
}

run_compare() {
  local output="$1"
  shift
  set +e
  "${PYTHON_BIN}" -m diffusion_bench.run_comparison "$@" --output "${output}"
  local rc=$?
  set -e
  if [[ -f "${output}" ]]; then
    RESULT_FILES+=("${output}")
  fi
  if [[ "${rc}" -ne 0 ]]; then
    status="${rc}"
  fi
  port=$((port + 10))
}

export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0

run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-flux-sgld-2gpu.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang \
  --case-ids flux1_dev_t2i_1024 flux2_dev_t2i_1024 \
  --hardware-profile h200 \
  --sglang-profile h100-h200-2gpu-tp-speed-compile \
  --run-id "${RUN_ID_ROOT}-flux-sgld-2gpu" \
  --port "${port}"

prepare_framework vllm-omni
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h100-h200-2gpu-tp-compile \
run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-flux-vllm-2gpu.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids flux1_dev_t2i_1024 flux2_dev_t2i_1024 \
  --hardware-profile h200 \
  --run-id "${RUN_ID_ROOT}-flux-vllm-2gpu" \
  --port "${port}"

export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=1

run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-cosmos3-image-sgld.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --sglang-profile h200-1gpu-fa-speed \
  --run-id "${RUN_ID_ROOT}-cosmos3-image-sgld" \
  --port "${port}"

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h200-1gpu-fa \
run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-cosmos3-image-vllm.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids cosmos3_nano_t2i_720p \
  --hardware-profile h200 \
  --run-id "${RUN_ID_ROOT}-cosmos3-image-vllm" \
  --port "${port}"

run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-cosmos3-video-sgld.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks sglang \
  --case-ids cosmos3_nano_t2v_720p_189f \
  --hardware-profile h200 \
  --sglang-profile h200-2gpu-cfg-speed \
  --run-id "${RUN_ID_ROOT}-cosmos3-video-sgld" \
  --port "${port}"

DIFFUSION_BENCH_VLLM_OMNI_PROFILE=h200-2gpu-cfg \
run_compare "${REPORT_DIR}/${RUN_ID_ROOT}-cosmos3-video-vllm.json" \
  --config "${CONFIG}" \
  --modes single_e2e \
  --frameworks vllm-omni \
  --case-ids cosmos3_nano_t2v_720p_189f \
  --hardware-profile h200 \
  --run-id "${RUN_ID_ROOT}-cosmos3-video-vllm" \
  --port "${port}"

printf '%s\n' "${RESULT_FILES[@]}" > "${REPORT_DIR}/${RUN_ID_ROOT}.results.txt"
printf 'result files:\n'
printf '  %s\n' "${RESULT_FILES[@]}"

exit "${status}"
