#!/usr/bin/env bash
# Prepare the round's box (RunPod 4xB300; /workspace is its container disk) for scripts/run_b300_full_20261022.sh: harness and sglang
# at the round's pins, every competitor installed at its resolved latest, an HF
# token for the gated checkpoints, then the one-minute GPU health probe.
#
#   SGLANG_PIN=<sha> DBF_PIN=<sha> FRAMEWORKS="vllm-omni lightx2v ..." \
#     setsid -f bash setup_round_20261022.sh > /workspace/r1022/setup.log 2>&1 < /dev/null
#
# Competitor versions resolved 2026-10-22 ~04:00 UTC (override to re-resolve):
#   vllm 0.32.0 (vllm-omni main's docker/Dockerfile.ci) + vllm-omni main 0da1d831e07f
#   LightX2V main 19c19614b067 | tensorrt-llm 1.4.0rc1 (requires torch 2.14.0)
#   ComfyUI master 1a82e6d01f39 | FastVideo main fa2d1324c600
# Frameworks install one at a time: parallel source builds plus model downloads
# have OOMKilled pods before (2026-10-10).
set -u

STATE="${DBF_STATE_DIR:-/workspace/r1022}"
: "${SGLANG_PIN:?SGLANG_PIN}" "${DBF_PIN:?DBF_PIN}"
FRAMEWORKS="${FRAMEWORKS:-vllm-omni lightx2v trtllm-visual comfyui fastvideo}"
export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.32.0}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git@0da1d831e07fadccc3ab51ef27b7c21623b693c2}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git@19c19614b067f1746ac24350dfd097e1c0b48591}"
export TRTLLM_INSTALL_SPEC="${TRTLLM_INSTALL_SPEC:-tensorrt-llm==1.4.0rc1}"
export TRTLLM_TORCH_INSTALL_SPEC="${TRTLLM_TORCH_INSTALL_SPEC:-torch==2.14.0}"
export COMFYUI_INSTALL_SPEC="${COMFYUI_INSTALL_SPEC:-https://github.com/comfyanonymous/ComfyUI.git@1a82e6d01f395b43275c77acd289af1eb7090bb9}"
export FASTVIDEO_INSTALL_SPEC="${FASTVIDEO_INSTALL_SPEC:-https://github.com/hao-ai-lab/FastVideo.git@fa2d1324c60050cd65209c6aaeac470e7fb412f5}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${STATE}/fw-venvs"
export HF_HOME="${STATE}/hf" HF_HUB_CACHE="${STATE}/hf/hub" HUGGINGFACE_HUB_CACHE="${STATE}/hf/hub"

mkdir -p "${STATE}"/{logs,fw-venvs,hf,comfy-ws}
echo "S start $(date -Is) frameworks=[${FRAMEWORKS}]"

fetch_pin() {  # <url> <dir> <sha>: a shallow checkout of one commit
  local url=$1 dir=$2 sha=$3
  if [[ ! -d "${dir}/.git" ]]; then
    git init -q "${dir}" && git -C "${dir}" remote add origin "${url}"
  fi
  git -C "${dir}" -c http.https://github.com/.extraheader= fetch -q --depth 1 origin "${sha}" \
    && git -C "${dir}" checkout -q FETCH_HEAD
}
fetch_pin https://github.com/mickqian/diffusion-bench-framework.git "${STATE}/dbf" "${DBF_PIN}" \
  || { echo "S FATAL harness fetch"; exit 1; }
fetch_pin https://github.com/sgl-project/sglang.git "${STATE}/sgl" "${SGLANG_PIN}" \
  || { echo "S FATAL sglang fetch"; exit 1; }
echo "S harness $(git -C "${STATE}/dbf" log --oneline -1)"
echo "S sglang $(git -C "${STATE}/sgl" log --oneline -1) imports $(PYTHONPATH="${STATE}/sgl/python" python3 -c 'import sglang; print(sglang.__file__)')"

# Only HF_TOKEN, from the env file `rx devbox secrets push` wrote; never echoed.
( umask 077; set -a; . ~/.config/radixark/secrets.env; set +a
  [[ -n "${HF_TOKEN:-}" ]] && printf '%s' "${HF_TOKEN}" > "${STATE}/.hftoken" ) \
  && [[ -s "${STATE}/.hftoken" ]] && echo "S hf token ok" || echo "S WARN no HF token"

# LightX2V runs at its own image's torch (pytorch:2.11.0-cuda13.0, dockerfiles/Dockerfile_cu130):
# flash-attn 2.8.3 pins -std=c++17 and torch 2.14's headers need C++20 (fails on
# `std::strong_ordering`, 2026-10-22). The SKILL's "Installing a competitor when
# the box's torch has moved on" has the rule.
printf 'torch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\n' > "${STATE}/lightx2v_torch_constraints.txt"

cd "${STATE}/dbf" || exit 1
for fw in ${FRAMEWORKS}; do
  t0=$(date +%s)
  if [[ "${fw}" == lightx2v ]]; then
    PIP_CONSTRAINT="${STATE}/lightx2v_torch_constraints.txt" \
      bash scripts/install_comparison_frameworks.sh "${fw}" > "${STATE}/logs/install_${fw}.log" 2>&1
  else
    bash scripts/install_comparison_frameworks.sh "${fw}" > "${STATE}/logs/install_${fw}.log" 2>&1
  fi
  rc=$?
  stamp="${STATE}/fw-venvs/${fw}/.diffusion-bench-install-stamp"
  echo "S install ${fw} rc=${rc} stamp=$([[ -f ${stamp} ]] && echo yes || echo NO) $(( $(date +%s) - t0 ))s"
  [[ -f "${stamp}" ]] && grep -E "version|commit|_spec=" "${stamp}" | head -6 | sed 's/^/S   /'
done

python3 scripts/gpu_health_probe.py 2>&1 | sed 's/^/S probe /'
echo "=== S_DONE $(date -Is)"
