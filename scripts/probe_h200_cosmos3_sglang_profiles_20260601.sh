#!/usr/bin/env bash
set -euo pipefail

# Probe SGLang-Diffusion Cosmos3 launch profiles on a non-CI H200/H100 host.
# Defaults compare the formal profile against the matching explicit speed-mode profile.

RUN_ID="${RUN_ID:-h200-cosmos3-sglang-profile-probe-20260601-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30620}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/root/.cache/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export HF_HOME="${HF_HOME:-/root/diffusion-bench-hf-cache}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.21.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/MaciejBalaNV/vllm-omni.git@453bd09ab3d1e2f10e3d1b63ff05f090ab49a7cc}"
export VLLM_OMNI_SERVER_BIN="${VLLM_OMNI_SERVER_BIN:-vllm}"
export VLLM_OMNI_REQUIRED_HELP_ARGS="${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni --model-class-name --stage-init-timeout --no-guardrails --diffusion-attention-backend --cache-backend --cfg-parallel-size --ulysses-degree}"
export SGLANG_COSMOS3_EXPECTED_REF="${SGLANG_COSMOS3_EXPECTED_REF:-sgl-project/sglang@9a8ab2d22b4f85f840701e8ed717f663e1c4662b}"

if [[ -n "${COSMOS3_SGLANG_PROBE_CASES:-}" ]]; then
  read -r -a PROBE_CASES <<< "${COSMOS3_SGLANG_PROBE_CASES}"
else
  PROBE_CASES=(
    cosmos3_nano_t2i_720p
    cosmos3_nano_t2v_720p_189f
    cosmos3_nano_i2v_720p_189f
  )
fi

mkdir -p "${OUTPUT_DIR}"

status=0
next_port="${PORT}"
for case_id in "${PROBE_CASES[@]}"; do
  if [[ "${case_id}" == "cosmos3_nano_t2i_720p" ]]; then
    profiles="${COSMOS3_SGLANG_IMAGE_PROBE_PROFILES:-h200-1gpu-fa h200-1gpu-fa-speed h200-1gpu-fa-batch4}"
  else
    profiles="${COSMOS3_SGLANG_VIDEO_PROBE_PROFILES:-h200-4gpu-cfg-ulysses h200-4gpu-cfg-ulysses-speed}"
  fi
  read -r -a PROBE_PROFILES <<< "${profiles}"
  for profile in "${PROBE_PROFILES[@]}"; do
    output="${OUTPUT_DIR}/${RUN_ID}-${case_id}-${profile}.json"
    echo "==> case=${case_id} profile=${profile} output=${output}"
    set +e
    SGLANG_BENCH_SGLANG_PROFILE="${profile}" \
      diffusion-bench-compare \
        --config "${CONFIG}" \
        --modes single_e2e \
        --frameworks sglang \
        --case-ids "${case_id}" \
        --hardware-profile h200 \
        --run-id "${RUN_ID}-${case_id}-${profile}" \
        --port "${next_port}" \
        --output "${output}"
    ret=$?
    set -e
    if [[ "${ret}" != "0" ]]; then
      status=1
      echo "profile failed: case=${case_id} profile=${profile} exit=${ret}" >&2
    fi
    next_port=$((next_port + 1))
    sleep "${COSMOS3_PROBE_COOLDOWN_SECONDS:-15}"
  done
done

exit "${status}"
