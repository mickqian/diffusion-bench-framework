#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="${1:?usage: install_comparison_frameworks.sh <vllm-omni|lightx2v|trtllm-visual>}"
VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/tmp/sglang-diffusion-framework-venvs}"
VENV_PATH="${VENV_ROOT}/${FRAMEWORK}"
PIP_TMPDIR="${SGLANG_DIFFUSION_PIP_TMPDIR:-${VENV_ROOT}/pip-tmp}"
STAMP_PATH="${VENV_PATH}/.diffusion-bench-install-stamp"
STAMP_VERSION="20260610-v1"
FORCE_REINSTALL="${FORCE_FRAMEWORK_REINSTALL:-${SGLANG_DIFFUSION_FORCE_FRAMEWORK_REINSTALL:-0}}"

mkdir -p "${VENV_ROOT}"
mkdir -p "${PIP_TMPDIR}"
export TMPDIR="${PIP_TMPDIR}"

case "${FRAMEWORK}" in
  vllm-omni|lightx2v|trtllm-visual) ;;
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
        echo "vllm_omni_server_bin=${VLLM_OMNI_SERVER_BIN:-vllm}"
        echo "vllm_omni_required_help_args=${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni}"
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
        echo "lightx2v_librosa_install_spec=${LIGHTX2V_LIBROSA_INSTALL_SPEC:-librosa}"
        echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-9.0}"
        ;;
      trtllm-visual)
        echo "trtllm_install_spec=${TRTLLM_INSTALL_SPEC:-tensorrt-llm==1.3.0rc18}"
        echo "trtllm_torch_install_spec=${TRTLLM_TORCH_INSTALL_SPEC:-torch==2.10.0}"
        echo "trtllm_torch_index_url=${TRTLLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
        echo "trtllm_pip_extra_index_url=${TRTLLM_PIP_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"
        echo "trtllm_visual_server_bin=${TRTLLM_VISUAL_SERVER_BIN:-trtllm-serve}"
        ;;
    esac
  } > "${path}"
}

framework_health_check() {
  [[ -x "${VENV_PATH}/bin/python3" ]] || return 1
  case "${FRAMEWORK}" in
    vllm-omni)
      "${VENV_PATH}/bin/python3" -c 'import importlib.metadata as m; import vllm, vllm_omni; m.version("vllm"); m.version("vllm-omni")'
      local server_bin="${VLLM_OMNI_SERVER_BIN:-vllm}"
      local help_output
      help_output="$("${VENV_PATH}/bin/${server_bin}" serve --omni --help=all 2>&1)"
      for required_arg in ${VLLM_OMNI_REQUIRED_HELP_ARGS:---omni}; do
        grep -q -- "${required_arg}" <<< "${help_output}" || return 1
      done
      ;;
    lightx2v)
      "${VENV_PATH}/bin/python3" -c 'import importlib.util; assert importlib.util.find_spec("lightx2v"); import lightx2v.server.main; import flash_attn_interface; assert hasattr(flash_attn_interface, "flash_attn_func")'
      ;;
    trtllm-visual)
      "${VENV_PATH}/bin/python3" -c 'import importlib.util; assert importlib.util.find_spec("tensorrt_llm")'
      local trtllm_bin="${TRTLLM_VISUAL_SERVER_BIN:-trtllm-serve}"
      [[ -x "${VENV_PATH}/bin/${trtllm_bin}" ]] || return 1
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
      | grep -E '^(llvmlite|numba|numpy|setuptools|tokenizers|torch|torchaudio|torchvision|triton)==' \
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
      python3 "$(dirname "$0")/install_lightx2v_fa3_from_hf.py"
    fi
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_SAGEATTENTION_INSTALL_SPEC:-sageattention==1.0.6}"
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_FLASHINFER_INSTALL_SPEC:-flashinfer-python==0.6.11}"
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed "${LIGHTX2V_LIBROSA_INSTALL_SPEC:-librosa}"
    python3 -m pip install --upgrade --force-reinstall pyzmq
    ;;
  trtllm-visual)
    # TensorRT-LLM VisualGen, served via `trtllm-serve`. Wheels live on the
    # NVIDIA PyPI index; override the spec/index per release as needed.
    #
    # NOTE 1: VisualGen (diffusion serving + /v1/images/generations, with
    #   get_is_diffusion_model auto-detection in trtllm-serve) is NOT in the
    #   1.2.x stable line — it lands in the 1.3.0 release candidates. A plain
    #   `tensorrt-llm` resolves to 1.2.x and serves FLUX through the LLM path,
    #   which fails. Pin a 1.3.0rc (or newer) here.
    # NOTE 2: 1.3.0rc has an UNRESOLVABLE pip conflict on a clean index —
    #   it requires `cuda-python>=13` (→ cuda-bindings 13.x) AND `torch>=2.10`,
    #   but PyPI's torch 2.10.0 pins `cuda-bindings==12.9.4`. The fix is to
    #   install torch 2.10.0 from PyTorch's cu130 index FIRST (its cuda-bindings
    #   is 13.x compatible), then install tensorrt-llm with only-if-needed so it
    #   keeps that torch. Verified working: torch 2.10.0+cu130 + tensorrt-llm
    #   1.3.0rc18 on H200.
    python3 -m pip install \
      "${TRTLLM_TORCH_INSTALL_SPEC:-torch==2.10.0}" torchvision \
      --index-url "${TRTLLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    python3 -m pip install --upgrade --upgrade-strategy only-if-needed \
      --extra-index-url "${TRTLLM_PIP_EXTRA_INDEX_URL:-https://pypi.nvidia.com}" \
      "${TRTLLM_INSTALL_SPEC:-tensorrt-llm==1.3.0rc18}"
    # Local patch: let VisualGen run in eager mode (TORCH_COMPILE_DISABLE=1) for
    # a same-policy compile-off comparison. See the patcher's docstring.
    python3 "$(dirname "$0")/patches/apply_trtllm_visual_patches.py"
    ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac

framework_health_check
write_desired_stamp "${STAMP_PATH}"
