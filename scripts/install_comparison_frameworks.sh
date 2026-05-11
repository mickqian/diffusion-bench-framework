#!/usr/bin/env bash
set -euxo pipefail

FRAMEWORK="${1:?usage: install_comparison_frameworks.sh <vllm-omni|lightx2v>}"
VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/tmp/sglang-diffusion-framework-venvs}"
VENV_PATH="${VENV_ROOT}/${FRAMEWORK}"
PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-${VENV_ROOT}/pip-tmp}"

mkdir -p "${VENV_ROOT}"
mkdir -p "${PIP_TMPDIR}"
export TMPDIR="${PIP_TMPDIR}"
python3 -m venv --clear "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python3 -m pip install --upgrade pip wheel setuptools

case "${FRAMEWORK}" in
  vllm-omni)
    python3 -m pip install --upgrade --force-reinstall "${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
    constraints="${VENV_PATH}/vllm_omni_constraints.txt"
    python3 -m pip freeze \
      | grep -E '^(llvmlite|numba|numpy|setuptools|tokenizers|torch|torchaudio|torchvision|transformers|triton)==' \
      > "${constraints}"
    python3 -m pip install --upgrade --force-reinstall -c "${constraints}" "${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
    ;;
  lightx2v)
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@573b9613adb0c1d33894b0920b5e12c87e42d280}"
    python3 -m pip install --upgrade ninja packaging matplotlib
    export MAX_JOBS="${MAX_JOBS:-8}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    python3 -m pip install --upgrade --no-build-isolation --no-deps "${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
    python3 -m pip install --upgrade --no-build-isolation --no-deps "${LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC:-git+https://github.com/Dao-AILab/flash-attention.git@ab66326aaa4fe3529fbc00f3156f3a762dd3141b#subdirectory=hopper}"
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"
    python3 -m pip install --upgrade --force-reinstall pyzmq
    ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac
