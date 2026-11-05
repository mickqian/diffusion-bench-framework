#!/usr/bin/env bash
# Build what the MiniMax-H3 cookbook cells' lossy arms read from disk. Each
# part is idempotent; run the ones a cell's `requires` names.
#
#   bash scripts/prepare_cookbook_h3.sh sgl-adaln     # sglang AdaLN sidecars, 29 and 27 forwards
#   bash scripts/prepare_cookbook_h3.sh lx-adaln      # LightX2V's AdaLN cache for minimax_h3_tp.json
#   bash scripts/prepare_cookbook_h3.sh sage          # SageAttention 2.2.0 + the SM90 fix, both envs
#
# Paths follow DBF_STATE_DIR (cells name /scratch/bench/...). Needs the
# MiniMax-H3 snapshot in the HF cache, the lightx2v venv, and for sglang
# DBF_SGLANG_PYTHONPATH at the pin. Uses GPU 0 for the cache builds.
set -euo pipefail
STATE="${DBF_STATE_DIR:?DBF_STATE_DIR}"
VENVS="${DBF_VENV_ROOT:-${STATE}/fw-venvs}"
SGPY="${SGLANG_PYTHON:-/opt/sglang/bin/python}"
LXPY="${VENVS}/lightx2v/bin/python"
LIGHTX2V_REF="${LIGHTX2V_REF:-ce0f526ae62c9079bf118dbce320adf7e8a307e6}"
SRC="${STATE}/src"
mkdir -p "$SRC"

h3_snapshot() {
    "$SGPY" -c 'from huggingface_hub import snapshot_download as s; print(s("MiniMaxAI/MiniMax-H3", local_files_only=True))'
}

sgl_adaln() {
    local snap; snap="$(h3_snapshot)"
    mkdir -p "${STATE}/sgl-adaln"
    for steps in 29 27; do
        local out="${STATE}/sgl-adaln/fl2va-t2va-${steps}.safetensors"
        [ -s "$out" ] && { echo "have $out"; continue; }
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${DBF_SGLANG_PYTHONPATH:?DBF_SGLANG_PYTHONPATH}" \
            "$SGPY" -m sglang.multimodal_gen.tools.build_minimax_h3_adaln_cache \
            --transformer-path "${snap}/FL2VA/transformer" --model-variant fl2va --mode t2va \
            --num-inference-steps "$steps" --flow-shift 12 --audio-flow-shift 3 --output "$out"
    done
}

lx_adaln() {
    local snap; snap="$(h3_snapshot)"
    local tree="${SRC}/LightX2V"
    if [ ! -d "${tree}/.git" ]; then
        git clone -q --filter=blob:none --no-checkout https://github.com/ModelTC/LightX2V.git "$tree"
    fi
    git -C "$tree" checkout -q "$LIGHTX2V_REF"
    # The cell's config with the cache directory it reads; the builder keys the
    # cache by variant, steps and shifts.
    local cfg="${STATE}/tmp/minimax_h3_tp_adaln.json"
    mkdir -p "${STATE}/tmp" "${STATE}/adaln-cache"
    "$LXPY" - "${tree}/configs/minimax_h3/minimax_h3_tp.json" "$cfg" "${STATE}/adaln-cache" <<'PY'
import json, sys
src, dst, cache = sys.argv[1:]
cfg = json.load(open(src)); cfg["adaln_cache_dir"] = cache
json.dump(cfg, open(dst, "w"), indent=1)
PY
    (cd "$tree" && CUDA_VISIBLE_DEVICES=0 PLATFORM=cuda "$LXPY" tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
        --model_path "$snap" --config_json "$cfg" --model-variant fl2av)
    ls -la "${STATE}/adaln-cache" | tail -5
}

sage() {
    # thu-ml/SageAttention@d9704247: in PyPI 2.2.0 the register_fake impl shares
    # the real SM90 wrapper's name and replaces it, so sageattn returns
    # uninitialised memory on H200 (rel. L2 error 54.68, 2026-10-30). sglang
    # detects the fix and falls back to FA without it; LightX2V does not.
    local tree="${SRC}/SageAttention"
    [ -d "${tree}/.git" ] || git clone -q --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention "$tree"
    python3 - "${tree}/sageattention/sm90_compile.py" <<'PY'
import re, sys
# A register_fake'd function must not share the real wrapper's name: in 2.2.0
# qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf's fake impl is defined
# under that same name and replaces it. Rename every fake impl not already
# suffixed (the attn_inst_buf one is), which is upstream d9704247's change.
p = sys.argv[1]; s = open(p).read()
pat = re.compile(r"(@torch\.library\.register_fake\([^)]*\)\s*\ndef )(\w+)\(")
bad = [m.group(2) for m in pat.finditer(s) if not m.group(2).endswith("_fake_impl")]
if not bad:
    sys.exit(print("sm90 fix present: every register_fake impl is a *_fake_impl"))
s = pat.sub(lambda m: m.group(0) if m.group(2).endswith("_fake_impl")
            else f"{m.group(1)}{m.group(2)}_fake_impl(", s)
open(p, "w").write(s); print("sm90 fix applied to", bad)
PY
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8"
    (cd "$tree" && "$LXPY" -m pip install -q --no-build-isolation --no-deps --force-reinstall .)
    # The sglang image's torch 2.14 headers need C++20 ("#error C++20 or later
    # compatible compiler is required to use PyTorch"); SageAttention 2.2.0's
    # setup.py pins -std=c++17. Build sglang's copy from a tree with the newer
    # standard -- a compiler flag only; the accuracy check below still decides.
    local sgl_tree="${SRC}/SageAttention-sgl"
    rm -rf "$sgl_tree" && cp -a "$tree" "$sgl_tree"
    if "$SGPY" -c 'import sys, torch; sys.exit(0 if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 14) else 1)'; then
        sed -i 's/-std=c++17/-std=c++20/g' "$sgl_tree/setup.py"
    fi
    (cd "$sgl_tree" && "$SGPY" -m pip install -q --no-build-isolation --no-deps --upgrade \
        --target "${STATE}/sage-sgl-site" .)
    # Accuracy against SDPA, the check that found the bug; > 0.1 means the fix did not take.
    cat > "${STATE}/tmp/sage_check.py" <<'PY'
import torch
from sageattention import sageattn
torch.manual_seed(0)
q, k, v = (torch.randn(1, 8192, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
err = ((sageattn(q, k, v, tensor_layout="NHD").float() - ref.float()).norm() / ref.float().norm()).item()
print(f"sage rel L2 error {err:.4f}"); raise SystemExit(0 if err < 0.1 else 1)
PY
    CUDA_VISIBLE_DEVICES=0 "$LXPY" "${STATE}/tmp/sage_check.py"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${STATE}/sage-sgl-site" "$SGPY" "${STATE}/tmp/sage_check.py"
}

[ "$#" -gt 0 ] || { sed -n '2,12p' "$0"; exit 2; }
for part in "$@"; do
    case "$part" in
        sgl-adaln) sgl_adaln ;;
        lx-adaln) lx_adaln ;;
        sage) sage ;;
        *) echo "unknown part $part" >&2; exit 2 ;;
    esac
done
