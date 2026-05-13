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
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC:-transformers<5}"
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
