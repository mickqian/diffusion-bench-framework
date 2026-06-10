#!/usr/bin/env bash
set -euo pipefail

# FLUX fastest-profile probe/final run for Hopper and Blackwell.
# The selected SGLang checkout must already be installed editable in this env.

HARDWARE_PROFILE="${HARDWARE_PROFILE:-h200}"
RUN_ID_ROOT="${RUN_ID_ROOT:-${HARDWARE_PROFILE}-flux-fastest-$(date -u +%Y%m%d-%H%M%S)}"
PORT_BASE="${PORT_BASE:-30680}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
REPORT_DIR="${REPORT_DIR:-tmp/report/flux-fastest}"
CASE_IDS="${CASE_IDS:-flux1_dev_t2i_1024 flux2_dev_t2i_1024}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-/tmp/diffusion-bench-pip-tmp}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE="${DIFFUSION_BENCH_DISABLE_TORCH_COMPILE:-0}"

export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.22.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git@73df8326d76fe3bba0b7b5a6abf6ad68976f24e8}"
export VLLM_OMNI_SERVER_BIN="${VLLM_OMNI_SERVER_BIN:-vllm}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@3db87106bafd980f0eeaffdb0d61dd26b364290c}"
export LIGHTX2V_FLASH_ATTN_INSTALL_SPEC="${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
export LIGHTX2V_FA3_HF_REPO="${LIGHTX2V_FA3_HF_REPO:-varunneal/flash-attention-3}"
export LIGHTX2V_FA3_HF_REVISION="${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
export LIGHTX2V_FA3_HF_SUBDIR="${LIGHTX2V_FA3_HF_SUBDIR:-build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3}"
export LIGHTX2V_FLASHINFER_INSTALL_SPEC="${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"

mkdir -p "${REPORT_DIR}" runs

case "${HARDWARE_PROFILE}" in
  b200|b300|gb200|gb300|blackwell)
    DEFAULT_SGLANG_PROFILES="blackwell-1gpu-speed-compile blackwell-2gpu-tp-speed-compile"
    DEFAULT_VLLM_PROFILES="blackwell-1gpu-compile blackwell-2gpu-tp-compile"
    DEFAULT_LIGHTX2V_PROFILES="blackwell-fa3-flashinfer"
    ;;
  *)
    DEFAULT_SGLANG_PROFILES="h100-h200-1gpu-speed-compile h100-h200-2gpu-tp-speed-compile"
    DEFAULT_VLLM_PROFILES="h100-h200-1gpu-compile h100-h200-2gpu-tp-compile"
    DEFAULT_LIGHTX2V_PROFILES="h200-fa3-flashinfer"
    ;;
esac

read -r -a CASE_ARRAY <<< "${CASE_IDS}"
read -r -a FRAMEWORK_ARRAY <<< "${FRAMEWORKS:-sglang vllm-omni lightx2v}"
read -r -a SGLANG_PROFILE_ARRAY <<< "${SGLANG_PROFILES:-${DEFAULT_SGLANG_PROFILES}}"
read -r -a VLLM_PROFILE_ARRAY <<< "${VLLM_OMNI_PROFILES:-${DEFAULT_VLLM_PROFILES}}"
read -r -a LIGHTX2V_PROFILE_ARRAY <<< "${LIGHTX2V_PROFILES:-${DEFAULT_LIGHTX2V_PROFILES}}"

RESULT_FILES=()
status=0
port="${PORT_BASE}"

run_compare() {
  local framework="$1"
  local profile="$2"
  local output="$3"
  local run_id="$4"
  local current_port="$5"
  shift 5
  local cases=("$@")

  set +e
  if [[ "${framework}" == "sglang" ]]; then
    diffusion-bench-compare \
      --config "${CONFIG}" \
      --modes single_e2e \
      --frameworks sglang \
      --case-ids "${cases[@]}" \
      --hardware-profile "${HARDWARE_PROFILE}" \
      --sglang-profile "${profile}" \
      --run-id "${run_id}" \
      --port "${current_port}" \
      --output "${output}"
  elif [[ "${framework}" == "vllm-omni" ]]; then
    DIFFUSION_BENCH_VLLM_OMNI_PROFILE="${profile}" \
    diffusion-bench-compare \
      --config "${CONFIG}" \
      --modes single_e2e \
      --frameworks vllm-omni \
      --case-ids "${cases[@]}" \
      --hardware-profile "${HARDWARE_PROFILE}" \
      --run-id "${run_id}" \
      --port "${current_port}" \
      --output "${output}"
  else
    DIFFUSION_BENCH_LIGHTX2V_PROFILE="${profile}" \
    diffusion-bench-compare \
      --config "${CONFIG}" \
      --modes single_e2e \
      --frameworks lightx2v \
      --case-ids flux2_dev_t2i_1024 \
      --hardware-profile "${HARDWARE_PROFILE}" \
      --run-id "${run_id}" \
      --port "${current_port}" \
      --output "${output}"
  fi
  local rc=$?
  set -e

  if [[ -f "${output}" ]]; then
    RESULT_FILES+=("${output}")
  fi
  if [[ "${rc}" -ne 0 ]]; then
    status="${rc}"
  fi
}

contains_framework() {
  local needle="$1"
  local item
  for item in "${FRAMEWORK_ARRAY[@]}"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

if contains_framework sglang; then
  for profile in "${SGLANG_PROFILE_ARRAY[@]}"; do
    output="${REPORT_DIR}/${RUN_ID_ROOT}-sglang-${profile}.json"
    run_compare sglang "${profile}" "${output}" "${RUN_ID_ROOT}-sglang-${profile}" "${port}" "${CASE_ARRAY[@]}"
    port=$((port + 10))
  done
fi

if contains_framework vllm-omni; then
  for profile in "${VLLM_PROFILE_ARRAY[@]}"; do
    output="${REPORT_DIR}/${RUN_ID_ROOT}-vllm-omni-${profile}.json"
    run_compare vllm-omni "${profile}" "${output}" "${RUN_ID_ROOT}-vllm-omni-${profile}" "${port}" "${CASE_ARRAY[@]}"
    port=$((port + 10))
  done
fi

if contains_framework lightx2v; then
  for profile in "${LIGHTX2V_PROFILE_ARRAY[@]}"; do
    output="${REPORT_DIR}/${RUN_ID_ROOT}-lightx2v-${profile}.json"
    run_compare lightx2v "${profile}" "${output}" "${RUN_ID_ROOT}-lightx2v-${profile}" "${port}" flux2_dev_t2i_1024
    port=$((port + 10))
  done
fi

printf '%s\n' "${RESULT_FILES[@]}" > "${REPORT_DIR}/${RUN_ID_ROOT}.results.txt"
echo "${REPORT_DIR}/${RUN_ID_ROOT}.results.txt"
printf '  %s\n' "${RESULT_FILES[@]}"

exit "${status}"
