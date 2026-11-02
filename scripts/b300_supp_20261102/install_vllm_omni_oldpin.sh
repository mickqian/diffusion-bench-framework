#!/usr/bin/env bash
# r1102 wave 3: the previous round's vLLM-Omni pin (0da1d831, with vllm 0.32.0, as in setup_round_20261022.sh) in its own
# FRESH venv root, to tell a vLLM-Omni main change apart from a box difference. Nothing else on the box is touched.
set -u
STATE=/workspace/r1102
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${STATE}/fw-venvs-0da1"
export VLLM_INSTALL_SPEC=vllm==0.32.0
export VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git@0da1d831e07fadccc3ab51ef27b7c21623b693c2
export VLLM_OMNI_VSA_KERNEL_SPEC=fastvideo-kernel==0.3.5
export HF_HOME="${STATE}/hf" HF_HUB_CACHE="${STATE}/hf/hub" HUGGINGFACE_HUB_CACHE="${STATE}/hf/hub"
mkdir -p "${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT}"
cd "${STATE}/dbf" || exit 1
echo "I start $(date -Is)"
bash scripts/install_comparison_frameworks.sh vllm-omni > "${STATE}/logs/install_vllm-omni_0da1.log" 2>&1
rc=$?
echo "I install rc=${rc} $(date -Is)"
"${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT}/vllm-omni/bin/python" - <<PY
import importlib.metadata as m
for p in ["vllm", "vllm-omni", "torch", "flashinfer-python", "nvidia-cudnn-frontend", "fastvideo-kernel", "transformers", "diffusers", "triton"]:
    try: print("I", p, m.version(p))
    except Exception: print("I", p, "n/a")
PY
echo "I_DONE rc=${rc}"
