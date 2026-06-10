#!/usr/bin/env bash
set -euo pipefail

# Refresh H200 video baselines where the fastest framework profile may differ
# from the older TP-only / non-FA3 comparison rows. Run on a non-CI H200 devbox.

RUN_ID_ROOT="${RUN_ID_ROOT:-h200-video-parallelism-refresh-20260611}"
REPORT_DIR="${REPORT_DIR:-tmp/report/h200-video-parallelism-refresh}"
PORT_BASE="${PORT_BASE:-31800}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/scratch/framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-1}"
export HF_HOME="${HF_HOME:-/cluster-storage/models/huggingface}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.22.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git@73df8326d76fe3bba0b7b5a6abf6ad68976f24e8}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@3db87106bafd980f0eeaffdb0d61dd26b364290c}"

mkdir -p "${REPORT_DIR}"

run_compare() {
  local framework="$1"
  local profile="$2"
  local output="$3"
  local run_id="$4"
  local port="$5"
  shift 5

  if [[ "${framework}" == "vllm-omni" ]]; then
    DIFFUSION_BENCH_VLLM_OMNI_PROFILE="${profile}" diffusion-bench-compare \
      --config "${CONFIG}" \
      --modes single_e2e \
      --frameworks vllm-omni \
      --case-ids "$@" \
      --hardware-profile h200 \
      --run-id "${run_id}" \
      --port "${port}" \
      --output "${output}"
  elif [[ "${framework}" == "lightx2v" ]]; then
    DIFFUSION_BENCH_LIGHTX2V_PROFILE="${profile}" diffusion-bench-compare \
      --config "${CONFIG}" \
      --modes single_e2e \
      --frameworks lightx2v \
      --case-ids "$@" \
      --hardware-profile h200 \
      --run-id "${run_id}" \
      --port "${port}" \
      --output "${output}"
  else
    echo "unknown framework ${framework}" >&2
    return 2
  fi
}

if [[ -n "${VIDEO_REFRESH_CASES:-}" ]]; then
  read -r -a CASES <<< "${VIDEO_REFRESH_CASES}"
else
  CASES=(
    wan21_t2v_1_3b_480p
    wan22_ti2v_5b_704p
  )
fi

if [[ "${RUN_VLLM_OMNI:-1}" == "1" ]]; then
  port="${PORT_BASE}"
  for profile in h200-2gpu-tp h200-2gpu-cfg h200-2gpu-ulysses; do
    output="${REPORT_DIR}/${RUN_ID_ROOT}-vllm-omni-${profile}.json"
    run_compare vllm-omni "${profile}" "${output}" "${RUN_ID_ROOT}-vllm-omni-${profile}" "${port}" "${CASES[@]}"
    port=$((port + 10))
  done
fi

if [[ "${RUN_LIGHTX2V:-1}" == "1" ]]; then
  output="${REPORT_DIR}/${RUN_ID_ROOT}-lightx2v-h200-fa3.json"
  run_compare lightx2v h200-fa3 "${output}" "${RUN_ID_ROOT}-lightx2v-h200-fa3" "$((PORT_BASE + 100))" "${CASES[@]}"
fi
