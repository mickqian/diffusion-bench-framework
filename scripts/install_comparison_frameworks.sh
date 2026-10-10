#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="${1:?usage: install_comparison_frameworks.sh <vllm-omni|lightx2v|trtllm-visual|comfyui|fastvideo>}"
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
  vllm-omni|lightx2v|trtllm-visual|comfyui|fastvideo) ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac

# One FastVideo venv serves Hopper and Blackwell boxes (the venv root is shared
# across devboxes): sm_90a builds the ThunderKittens VSA kernel, sm_100a/sm_103a
# the MiniMax-H3 VSA forward. TK never builds on aarch64 (GB200/GB300 hosts).
fastvideo_arch_list() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "${TORCH_CUDA_ARCH_LIST}"
  elif [[ "$(uname -m)" == "aarch64" ]]; then
    echo "10.0a;10.3a"
  else
    echo "9.0a;10.0a;10.3a"
  fi
}

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
        echo "vllm_omni_vsa_kernel_spec=${VLLM_OMNI_VSA_KERNEL_SPEC:-<unset>}"
        echo "vllm_omni_target_device=${VLLM_OMNI_TARGET_DEVICE:-cuda} vllm_rocm_extra_index_url=${VLLM_ROCM_EXTRA_INDEX_URL:-<unset>} vllm_rocm_userspace_release=${VLLM_ROCM_USERSPACE_RELEASE:-<unset>}"
        ;;
      lightx2v)
        echo "lightx2v_install_spec=${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
        echo "lightx2v_transformers_install_spec=${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC:-<unset: latest, as LightX2V itself declares>}"
        echo "lightx2v_safetensors_install_spec=${LIGHTX2V_SAFETENSORS_INSTALL_SPEC:-safetensors>=0.8.0rc0}"
        echo "lightx2v_flash_attn_install_spec=${LIGHTX2V_FLASH_ATTN_INSTALL_SPEC:-flash-attn==2.8.3}"
        echo "lightx2v_flash_attn3_install_spec=${LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC:-}"
        echo "lightx2v_fa3_hf_repo=${LIGHTX2V_FA3_HF_REPO:-varunneal/flash-attention-3}"
        echo "lightx2v_fa3_hf_revision=${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
        echo "lightx2v_fa3_hf_subdir=${LIGHTX2V_FA3_HF_SUBDIR:-auto}"
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
      comfyui)
        echo "comfyui_install_spec=${COMFYUI_INSTALL_SPEC:-https://github.com/comfyanonymous/ComfyUI.git@master}"
        echo "comfyui_torch_install_spec=${COMFYUI_TORCH_INSTALL_SPEC:-torch torchvision torchaudio}"
        echo "comfyui_torch_index_url=${COMFYUI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
        ;;
      fastvideo)
        echo "fastvideo_install_spec=${FASTVIDEO_INSTALL_SPEC:-https://github.com/hao-ai-lab/FastVideo.git@main}"
        echo "fastvideo_install_extras=${FASTVIDEO_INSTALL_EXTRAS:-fasth3}"
        echo "fastvideo_uv_torch_backend=${FASTVIDEO_UV_TORCH_BACKEND:-cu130}"
        echo "fastvideo_install_fa3=${FASTVIDEO_INSTALL_FA3:-1}"
        echo "fastvideo_fa3_hf_revision=${LIGHTX2V_FA3_HF_REVISION:-de87b9b5af06dd9984df595bef90b2eba44b181a}"
        echo "torch_cuda_arch_list=$(fastvideo_arch_list)"
        ;;
    esac
  } > "${path}"
}

framework_health_check() {
  [[ -x "${VENV_PATH}/bin/python3" ]] || return 1
  # Same rule as run_comparison._isolate_rocm_userspace: a ROCm-built venv runs on
  # the ROCm in <venv>/rocm, else on its own amdsmi library, never on the SDK a
  # ROCm host image points ROCM_HOME at.
  local smi
  smi="$(compgen -G "${VENV_PATH}/lib/python3*/site-packages/amdsmi/libamd_smi.so" | head -1)"
  if [[ -d "${VENV_PATH}/rocm/lib" ]]; then
    export ROCM_HOME="${VENV_PATH}/rocm" ROCM_PATH="${VENV_PATH}/rocm"
    export LD_LIBRARY_PATH="${VENV_PATH}/rocm/lib"
  elif [[ -n "${smi}" ]]; then
    unset ROCM_HOME ROCM_PATH
    export LD_LIBRARY_PATH="$(dirname "${smi}")"
  fi
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
    comfyui)
      # --quick-test-for-ci imports every node (core + comfy_extras) and exits,
      # so a missing dependency of any model family fails here, not mid-round.
      ( cd "${VENV_PATH}/ComfyUI" && "${VENV_PATH}/bin/python3" main.py --quick-test-for-ci --cpu )
      ;;
    fastvideo)
      [[ -d "${VENV_PATH}/FastVideo/.git" ]] || return 1
      "${VENV_PATH}/bin/python3" -c 'import fastvideo, flash_attn.cute; from fastvideo.entrypoints.openai.api_server import create_app; from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd'
      if [[ "${FASTVIDEO_INSTALL_FA3:-1}" == "1" ]]; then
        "${VENV_PATH}/bin/python3" -c 'import flash_attn_interface'
      fi
      "${VENV_PATH}/bin/fastvideo" serve --help >/dev/null
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
    # Upstream's documented flow (vllm-omni docs/getting_started/installation/
    # gpu/cuda.inc.md): install the matching vLLM minor, then install omni
    # editable from a clone with uv. Omni's dev line tracks vLLM's -- "the 0.29
    # development line uses vLLM 0.29.x" -- so a spec that pins them to
    # different lines cannot resolve.
    #
    # The previous recipe pinned pip-freeze constraints from the vLLM install
    # onto omni's. That froze `tokenizers` at whatever vLLM chose (0.23.2 for
    # vLLM 0.29.0), while omni's transformers 5.x caps it at <=0.23.0, so pip
    # reported ResolutionImpossible and the nightly went red. Left to resolve,
    # uv picks tokenizers 0.22.2 and both are satisfied.
    # ROCm (recipes/MiniMaxAI/MiniMax-H3.md "AMD ROCm"): vLLM comes from its ROCm wheel index
    # (e.g. VLLM_INSTALL_SPEC=vllm==0.31.0+rocm723 with
    # VLLM_ROCM_EXTRA_INDEX_URL=https://wheels.vllm.ai/rocm/0.31.0/rocm723) and omni builds with
    # VLLM_OMNI_TARGET_DEVICE=rocm and no build isolation, so it compiles against that torch.
    omni_rocm=0
    [[ "${VLLM_OMNI_TARGET_DEVICE:-}" == rocm ]] && omni_rocm=1
    if (( omni_rocm )); then
      # sglang's ROCm images export PIP_CONSTRAINT pinning the IMAGE's torch
      # (/etc/sglang/constraints/torch-rocm.txt, torch==2.11.0+rocm10.0.0); inherited by this
      # venv it makes the +rocm vLLM wheel, which needs its own torch, ResolutionImpossible.
      unset PIP_CONSTRAINT
      python3 -m pip install --upgrade --force-reinstall "${VLLM_INSTALL_SPEC:?ROCm needs an explicit +rocm VLLM_INSTALL_SPEC}" \
        --extra-index-url "${VLLM_ROCM_EXTRA_INDEX_URL:?ROCm needs VLLM_ROCM_EXTRA_INDEX_URL}"
      # That torch links the system ROCm, which on a sglang ROCm image is a newer release
      # than the wheel's (ROCm 10 vs 7.2.3): give the venv its own copy of the right one.
      if [[ -n "${VLLM_ROCM_USERSPACE_RELEASE:-}" ]]; then
        bash "$(dirname "${BASH_SOURCE[0]}")/install_rocm_userspace.sh" "${VLLM_ROCM_USERSPACE_RELEASE}" "${VENV_PATH}/rocm"
      fi
    else
      python3 -m pip install --upgrade --force-reinstall "${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
    fi
    omni_spec="${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
    if [[ "${omni_spec}" == git+* ]]; then
      omni_src="${VENV_PATH}/src/vllm-omni"
      omni_url="${omni_spec#git+}"
      omni_ref="main"
      if [[ "${omni_url}" == *"@"* ]]; then
        omni_ref="${omni_url##*@}"
        omni_url="${omni_url%@*}"
      fi
      rm -rf "${omni_src}"
      mkdir -p "$(dirname "${omni_src}")"
      git clone -q --depth 1 --branch "${omni_ref}" "${omni_url}" "${omni_src}" \
        || git clone -q "${omni_url}" "${omni_src}"
      ( cd "${omni_src}" && git checkout -q "${omni_ref}" 2>/dev/null || true )
      echo "vllm-omni source at $(cd "${omni_src}" && git rev-parse --short=12 HEAD)"
      if (( omni_rocm )); then
        VLLM_OMNI_TARGET_DEVICE=rocm python3 -m pip install -e "${omni_src}" --no-build-isolation
      elif command -v uv >/dev/null 2>&1; then
        VIRTUAL_ENV="${VENV_PATH}" uv pip install -e "${omni_src}"
      else
        python3 -m pip install -e "${omni_src}"
      fi
    else
      python3 -m pip install --upgrade --force-reinstall "${omni_spec}"
    fi
    # FASTVIDEO_VSA (FastH3) needs fastvideo-kernel. The `vsa` extra's 0.3.4 dies with "illegal
    # instruction" on H200; 0.3.5 declares torch==2.12.0 and, resolved normally, downgrades vLLM's
    # torch 2.13 so torchvision::nms vanishes (2026-09-30). --no-deps keeps the torch vLLM pinned.
    if [[ -n "${VLLM_OMNI_VSA_KERNEL_SPEC:-}" ]]; then
      torch_before="$(python3 -c 'import torch; print(torch.__version__)')"
      python3 -m pip install --no-deps "${VLLM_OMNI_VSA_KERNEL_SPEC}"
      python3 -c 'import fastvideo_kernel'
      [[ "$(python3 -c 'import torch; print(torch.__version__)')" == "${torch_before}" ]]
    fi
    ;;
  lightx2v)
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@7efd05f8e1425b83321fd4f1cef779ef6504076f}"
    # LightX2V pins neither transformers nor diffusers, so "LightX2V latest" is
    # whatever pip resolves today. Our own `transformers<5` pin (added
    # 2026-05-13 for an LTX model-resolution problem) broke that: transformers
    # 4.x caps huggingface_hub<1.0, diffusers 0.40 needs hub>=1.23, so every
    # `diffusers.pipelines.*` import raised ImportError. LightX2V's flux2
    # scheduler imports three names from diffusers in ONE try block, so the two
    # pipeline imports failing took the scheduler down with them -- it became
    # None and flux2 died at `NoneType.from_pretrained`, published as a failed
    # cell. Verified on the latest consistent set (transformers 5.17,
    # diffusers 0.40, hub 1.31): flux2's scheduler resolves and all 39 runner
    # modules import, ltx2 and wan included. Override the spec to reproduce a
    # dated historical report; leave it unset for a fresh latest-vs-latest run.
    if [[ -n "${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC:-}" ]]; then
      python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_TRANSFORMERS_INSTALL_SPEC}"
    fi
    python3 -m pip install --upgrade --pre --upgrade-strategy only-if-needed "${LIGHTX2V_SAFETENSORS_INSTALL_SPEC:-safetensors>=0.8.0rc0}"
    python3 -m pip install --upgrade ninja packaging matplotlib
    # flash-attn is a source build here (~70 translation units x 4 GPU archs --
    # its setup.py picks archs from the CUDA version and ignores
    # TORCH_CUDA_ARCH_LIST). A flat MAX_JOBS=8 cost 25 minutes on a 240-core
    # box; each nvcc peaks around 4GB, so scale with whichever of cores/RAM
    # runs out first and keep a ceiling so a huge host does not thrash.
    if [[ -z "${MAX_JOBS:-}" ]]; then
      _cores="$(nproc 2>/dev/null || echo 8)"
      _mem_gb="$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 16)"
      MAX_JOBS=$(( _cores / 2 )); [[ $(( _mem_gb / 4 )) -lt ${MAX_JOBS} ]] && MAX_JOBS=$(( _mem_gb / 4 ))
      [[ ${MAX_JOBS} -gt 48 ]] && MAX_JOBS=48
      [[ ${MAX_JOBS} -lt 1 ]] && MAX_JOBS=1
      echo "flash-attn build: MAX_JOBS=${MAX_JOBS} (${_cores} cores, ${_mem_gb}GB avail)"
    fi
    export MAX_JOBS
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
    # flash-attn==2.8.3's cute submodule statically references
    # cutlass.cute.core.ThrMma at import time; nvidia-cutlass-dsl (pulled
    # transitively via flashinfer-python, unpinned >=4.5.0) removed that
    # attribute in 4.6.0, and the last 4.5.x release predating the removal
    # (4.5.3) is itself missing a module (`cutlass.utils.ampere_helpers`)
    # flash-attn's cute code also needs -- no released cutlass-dsl version
    # satisfies both. lightx2v's own attn modules already wrap this import in
    # `except ImportError`, but the AttributeError from the incompatible
    # cutlass-dsl slips past that guard. Drop the unused cute submodule so the
    # import fails as a plain ModuleNotFoundError instead, which the existing
    # fallback handles correctly.
    python3 -c "
import pathlib
import flash_attn
cute_dir = pathlib.Path(flash_attn.__file__).parent / 'cute'
if cute_dir.exists():
    import shutil
    shutil.rmtree(cute_dir)
    print(f'removed unusable {cute_dir}')
"
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
  comfyui)
    # ComfyUI is not a pip package: the server is main.py in a checkout, so the
    # checkout lives inside the venv directory and the harness finds it there.
    # Its requirements.txt leaves torch unpinned and upstream's README installs
    # the newest stable wheel from PyTorch's CUDA index, so do the same first
    # and let requirements.txt keep it.
    comfy_spec="${COMFYUI_INSTALL_SPEC:-https://github.com/comfyanonymous/ComfyUI.git@master}"
    comfy_url="${comfy_spec%@*}"
    comfy_ref="${comfy_spec##*@}"
    [[ "${comfy_spec}" == *"@"* ]] || { comfy_url="${comfy_spec}"; comfy_ref="master"; }
    git clone -q --filter=blob:none "${comfy_url}" "${VENV_PATH}/ComfyUI"
    git -C "${VENV_PATH}/ComfyUI" checkout -q "${comfy_ref}"
    echo "ComfyUI source at $(git -C "${VENV_PATH}/ComfyUI" rev-parse --short=12 HEAD)"
    # shellcheck disable=SC2086 -- the spec is a package list on purpose
    python3 -m pip install ${COMFYUI_TORCH_INSTALL_SPEC:-torch torchvision torchaudio} \
      --index-url "${COMFYUI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    python3 -m pip install --upgrade-strategy only-if-needed -r "${VENV_PATH}/ComfyUI/requirements.txt"
    ;;
  fastvideo)
    # FastVideo's documented H3 install: `UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"` in
    # a checkout. uv is required, not preferred: its [tool.uv.sources] table builds fastvideo-kernel
    # from the same checkout (the sm_100a VSA route, the fused Ulysses all-to-all) and pins
    # flash-attn-4 to the revision its CuTe kernels need; pip ignores the table.
    fv_spec="${FASTVIDEO_INSTALL_SPEC:-https://github.com/hao-ai-lab/FastVideo.git@main}"
    fv_url="${fv_spec%@*}"
    fv_ref="${fv_spec##*@}"
    [[ "${fv_spec}" == *"@"* ]] || { fv_url="${fv_spec}"; fv_ref="main"; }
    fv_src="${VENV_PATH}/FastVideo"
    git clone -q --filter=blob:none "${fv_url}" "${fv_src}"
    git -C "${fv_src}" checkout -q "${fv_ref}"
    # The kernel build needs ThunderKittens and CUTLASS; the eval submodules are not needed.
    git -C "${fv_src}" submodule update -q --init --depth 1 fastvideo-kernel/include/tk fastvideo-kernel/include/cutlass
    echo "FastVideo source at $(git -C "${fv_src}" rev-parse --short=12 HEAD)"
    # fastvideo-kernel compiles against the torch the backend's index resolves (FastVideo pins
    # 2.12.0), so nvcc must be the same CUDA major; otherwise the build fails after the downloads.
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    fv_backend="${FASTVIDEO_UV_TORCH_BACKEND:-cu130}"
    fv_want="${fv_backend#cu}"
    fv_have="$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' || true)"
    if [[ "${fv_have}" != "${fv_want%?}" ]]; then
      echo "fastvideo: ${CUDA_HOME}/bin/nvcc is CUDA ${fv_have:-<missing>} but the torch index is ${fv_backend}; set CUDA_HOME or FASTVIDEO_UV_TORCH_BACKEND" >&2
      exit 1
    fi
    TORCH_CUDA_ARCH_LIST="$(fastvideo_arch_list)"
    export TORCH_CUDA_ARCH_LIST
    [[ -z "${MAX_JOBS:-}" ]] || export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
    python3 -m pip install --upgrade uv
    ( cd "${fv_src}" && UV_TORCH_BACKEND="${fv_backend}" uv pip install -e ".[${FASTVIDEO_INSTALL_EXTRAS:-fasth3}]" )
    # Dense H3 attention on Hopper: FastVideo's FLASH_ATTN loads FA3 when FASTVIDEO_FA4=0 and
    # flash_attn_interface imports, and falls back to SDPA without a word otherwise. The prebuilt
    # artifact LightX2V uses carries torch212-cu130 builds; Blackwell profiles use FA4 instead.
    if [[ "${FASTVIDEO_INSTALL_FA3:-1}" == "1" ]]; then
      python3 "$(dirname "$0")/install_lightx2v_fa3_from_hf.py"
    fi
    # Which kernel routes this build carries, for the record; a missing one falls back silently.
    python3 - <<'PY'
import importlib.metadata as metadata

routes = {}
try:
    from fastvideo_kernel.block_sparse_attn import _get_sm90_ops
    routes["tk_sm90a"] = all(_get_sm90_ops())
except Exception as exc:  # noqa: BLE001 - a record, not a gate
    routes["tk_sm90a"] = f"unknown ({type(exc).__name__})"
try:
    from fastvideo_kernel import block_sparse_attn_sm100a
    routes["vsa_sm100a"] = bool(block_sparse_attn_sm100a._HAS_VSA_SM100A)
except Exception as exc:  # noqa: BLE001
    routes["vsa_sm100a"] = f"unknown ({type(exc).__name__})"
print("fastvideo-kernel", metadata.version("fastvideo-kernel"), "torch", metadata.version("torch"), routes)
PY
    ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac

framework_health_check
write_desired_stamp "${STAMP_PATH}"
