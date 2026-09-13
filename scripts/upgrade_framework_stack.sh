#!/usr/bin/env bash
# Move an already-installed competitor venv onto the latest dependency stack,
# and refuse to proceed if that would invalidate what is installed.
#
#   scripts/upgrade_framework_stack.sh <vllm-omni|lightx2v|trtllm-visual> [pkg ...]
#
# A fresh install already gets the latest stack -- scripts/install_comparison_frameworks.sh
# no longer pins LightX2V's transformers. This is for a box whose venvs already
# exist, where reinstalling would mean another flash-attn source build.
#
# Why the torch guard: flash-attn (and FA3) are compiled against the installed
# torch. pip's default upgrade strategy is only-if-needed, so torch SHOULD stay
# put -- but if a dependency drags it, every later cell dies on an import error
# hours away from the cause. Verify rather than assume.
#
# Why it matters at all: LightX2V's FLUX.2 cell published as `failed` for weeks
# because a `transformers<5` pin capped huggingface_hub below 1.0, which breaks
# every `diffusers.pipelines.*` import, which left its flux2 scheduler as None.
# Nothing in the artifact named those three packages. Upgrading the stack fixed
# the cell and left the working ones untouched (ltx2: 34.685s -> 34.67s).
set -u

FW="${1:?usage: upgrade_framework_stack.sh <vllm-omni|lightx2v|trtllm-visual> [pkg ...]}"
shift || true
PKGS=("$@")
[ "${#PKGS[@]}" -gt 0 ] || PKGS=(transformers diffusers huggingface_hub)

VENV_ROOT="${DBF_VENV_ROOT:-${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/personal/bench0912/fw-venvs}}"
V="${VENV_ROOT}/${FW}"
[ -x "${V}/bin/python3" ] || { echo "no venv at ${V}" >&2; exit 1; }

echo "=== ${FW}: before ==="
TORCH_BEFORE="$("${V}/bin/python3" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  torch=${TORCH_BEFORE}"
"${V}/bin/pip" list 2>/dev/null | grep -iE "^($(IFS='|'; echo "${PKGS[*]}")) " | sed 's/^/  /'

echo "=== upgrading: ${PKGS[*]} ==="
"${V}/bin/pip" install --upgrade "${PKGS[@]}" 2>&1 | tail -6

echo "=== after ==="
TORCH_AFTER="$("${V}/bin/python3" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  torch=${TORCH_AFTER}"
"${V}/bin/pip" list 2>/dev/null | grep -iE "^($(IFS='|'; echo "${PKGS[*]}")) " | sed 's/^/  /'

if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
  echo "ABORT: torch moved ${TORCH_BEFORE} -> ${TORCH_AFTER}; the compiled attention"
  echo "       extensions in this venv are built against the old one. Reinstall the"
  echo "       framework instead of upgrading in place."
  exit 1
fi

echo "=== health ==="
case "${FW}" in
  lightx2v)
    "${V}/bin/python3" - <<'PY' || exit 1
import sys
from lightx2v.models.schedulers.flux2 import scheduler as s
import flash_attn_interface
# The multi-import try/except in that module turns ONE unsatisfiable dependency
# into several unrelated Nones, so check the names, not just that it imported.
names = {
    "FlowMatchEulerDiscreteScheduler": s.FlowMatchEulerDiscreteScheduler,
    "compute_empirical_mu": s.compute_empirical_mu,
    "retrieve_timesteps": s.retrieve_timesteps,
}
for k, v in names.items():
    print(f"  {k} = {v}")
ok = all(v is not None for v in names.values()) and hasattr(flash_attn_interface, "flash_attn_func")
print(f"  flash_attn_func: {hasattr(flash_attn_interface, 'flash_attn_func')}")
sys.exit(0 if ok else 1)
PY
    ;;
  vllm-omni)
    "${V}/bin/python3" -c 'import vllm, vllm_omni; print("  vllm + vllm_omni import")' || exit 1
    ;;
  trtllm-visual)
    "${V}/bin/python3" -c 'import tensorrt_llm; print("  tensorrt_llm imports")' || exit 1
    ;;
esac

echo "=== pip check ==="
"${V}/bin/pip" check 2>&1 | grep -viE "decord|not supported on this platform" | sed 's/^/  /' || true
echo "=== upgraded ${FW}; RE-MEASURE every cell this framework appears in ==="
