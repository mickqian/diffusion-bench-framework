#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="${1:?usage: install_comparison_frameworks.sh <vllm-omni|lightx2v>}"
VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/tmp/sglang-diffusion-framework-venvs}"
VENV_PATH="${VENV_ROOT}/${FRAMEWORK}"
PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-${VENV_ROOT}/pip-tmp}"
STAMP_PATH="${VENV_PATH}/.diffusion-bench-install-stamp"
STAMP_VERSION="20260514-v1"
FORCE_REINSTALL="${FORCE_FRAMEWORK_REINSTALL:-${SGLANG_DIFFUSION_FORCE_FRAMEWORK_REINSTALL:-0}}"

mkdir -p "${VENV_ROOT}"
mkdir -p "${PIP_TMPDIR}"
export TMPDIR="${PIP_TMPDIR}"

case "${FRAMEWORK}" in
  vllm-omni|lightx2v) ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac

write_desired_stamp() {
  local path="$1"
  local python_version
  local platform_id
  python_version="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  platform_id="$(python3 -c 'import platform; print(platform.platform())')"
  {
    echo "stamp_version=${STAMP_VERSION}"
    echo "framework=${FRAMEWORK}"
    echo "python=${python_version}"
    echo "platform=${platform_id}"
    case "${FRAMEWORK}" in
      vllm-omni)
        echo "vllm_install_spec=${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
        echo "vllm_omni_install_spec=${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
        ;;
      lightx2v)
        echo "lightx2v_install_spec=${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
        echo "lightx2v_transformers_install_spec=${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC:-transformers<5}"
        echo "lightx2v_safetensors_install_spec=${LIGHTX2V_SAFETENSORS_INSTALL_SPEC:-safetensors>=0.8.0rc0}"
        echo "lightx2v_flash_attn_install_spec=${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
        echo "lightx2v_flash_attn3_install_spec=${LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC:-}"
        echo "lightx2v_fa3_hf_repo=${LIGHTX2V_FA3_HF_REPO:-varunneal/flash-attention-3}"
        echo "lightx2v_fa3_hf_revision=${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
        echo "lightx2v_fa3_hf_subdir=${LIGHTX2V_FA3_HF_SUBDIR:-build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3}"
        echo "lightx2v_sageattention_install_spec=${LIGHTX2V_SAGEATTENTION_INSTALL_SPEC:-sageattention==1.0.6}"
        echo "lightx2v_flashinfer_install_spec=${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"
        echo "lightx2v_hf_xet_install_spec=${LIGHTX2V_HF_XET_INSTALL_SPEC:-hf-xet}"
        ;;
    esac
  } > "${path}"
}

framework_health_check() {
  [[ -x "${VENV_PATH}/bin/python3" ]] || return 1
  case "${FRAMEWORK}" in
    vllm-omni)
      "${VENV_PATH}/bin/python3" -c 'import importlib.metadata as m; import vllm; m.version("vllm"); m.version("vllm-omni")'
      ;;
    lightx2v)
      "${VENV_PATH}/bin/python3" -c 'import importlib.util; assert importlib.util.find_spec("lightx2v"); import flash_attn_interface; assert hasattr(flash_attn_interface, "flash_attn_func")'
      ;;
  esac
}

desired_stamp="$(mktemp "${PIP_TMPDIR}/${FRAMEWORK}.stamp.XXXXXX")"
write_desired_stamp "${desired_stamp}"
if [[ "${FORCE_REINSTALL}" != "1" && -f "${STAMP_PATH}" ]] && cmp -s "${desired_stamp}" "${STAMP_PATH}"; then
  if framework_health_check >/dev/null 2>&1; then
    echo "Reusing ${FRAMEWORK} venv at ${VENV_PATH}"
    rm -f "${desired_stamp}"
    exit 0
  fi
  echo "${FRAMEWORK} venv stamp matches but health check failed; reinstalling" >&2
fi
rm -f "${desired_stamp}"

echo "Installing ${FRAMEWORK} venv at ${VENV_PATH}"
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
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC:-transformers<5}"
    python3 -m pip install --upgrade --pre --upgrade-strategy only-if-needed "${LIGHTX2V_SAFETENSORS_INSTALL_SPEC:-safetensors>=0.8.0rc0}"
    python3 -m pip install --upgrade ninja packaging matplotlib
    export MAX_JOBS="${MAX_JOBS:-8}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    python3 -m pip install --upgrade --no-cache-dir --no-build-isolation --no-deps --no-binary flash-attn "${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
    if [[ -n "${LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC:-}" ]]; then
      python3 -m pip install --upgrade --no-build-isolation --no-deps "${LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC}"
    else
      python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_HF_XET_INSTALL_SPEC:-hf-xet}"
      python3 - <<'PY'
import os
import shutil
import site
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ.get("LIGHTX2V_FA3_HF_REPO", "varunneal/flash-attention-3")
revision = os.environ.get(
    "LIGHTX2V_FA3_HF_REVISION", "de87b9b5af06dd9984df595bef90b2eba44b181a"
)
subdir = os.environ.get(
    "LIGHTX2V_FA3_HF_SUBDIR",
    "build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3",
)
snapshot = Path(
    snapshot_download(repo, revision=revision, allow_patterns=[subdir + "/*"])
)
site_dir = Path(site.getsitepackages()[0])
dst = site_dir / "flash_attention_3"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(snapshot / subdir, dst, symlinks=False)
(site_dir / "flash_attn_interface.py").write_text(
    "from flash_attention_3.flash_attn_interface import *\n"
)
PY
    fi
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_SAGEATTENTION_INSTALL_SPEC:-sageattention==1.0.6}"
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"
    python3 -m pip install --upgrade --force-reinstall pyzmq
    ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac

framework_health_check
write_desired_stamp "${STAMP_PATH}"
