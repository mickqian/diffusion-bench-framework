#!/bin/bash
# One cell of the LightX2V README reproduction: one framework, one case, through
# the repo's harness with this directory's config. Modelled on
# scripts/devbox_run_cases.sh (narrow GPU kill, results under /persistent), plus a
# 1 s nvidia-smi sampler so peak memory per GPU is part of the record.
#
#   bash run_cell.sh <tag> <gpu_ids> <port> <framework> <case_id>
#
# DBF_DISABLE_COMPILE=1 (default) runs the deliberate eager comparison;
# DBF_REPEATS overrides the measured repeat count; DBF_LX2V_PROFILE=<level>
# turns on LightX2V's own stage/step timers (they synchronise, so diagnostic
# runs only, never the measured e2e ones).
set -u

TAG="${1:?tag}"; GPU_IDS="${2:?gpu ids}"; readonly PORT="${3:?port}"
FW="${4:?framework}"; CASE="${5:?case id}"

REPO=/persistent/diffusion-bench-framework
CFG=/persistent/lx2v/config.json
OUT=/persistent/logs/cells
mkdir -p "$OUT"

export HF_HOME=/persistent/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/persistent/fw-venvs
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE="${DBF_DISABLE_COMPILE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${DBF_LX2V_PROFILE:-}" ]; then
    export PROFILING_DEBUG_LEVEL="$DBF_LX2V_PROFILE"
fi

kill_own_gpus() {
    local pids attempt
    for attempt in 1 2 3; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_IDS" 2>/dev/null | sort -u | tr '\n' ' ')
        [ -n "${pids// /}" ] || return 0
        kill -9 $pids 2>/dev/null
        sleep 4
    done
}

cfg_used="$CFG"
if [ -n "${DBF_REPEATS:-}" ]; then
    cfg_used="$OUT/${TAG}.config.json"
    python3 - "$CFG" "$cfg_used" "$DBF_REPEATS" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["benchmark_defaults"]["single"]["video_repeats"] = int(sys.argv[3])
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY
fi

kill_own_gpus
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i "$GPU_IDS" -l 1 > "$OUT/${TAG}.smi.csv" 2>/dev/null &
smi_pid=$!

echo "=== CELL $TAG start $(date -u +%FT%TZ) fw=$FW case=$CASE gpus=$GPU_IDS compile_disabled=$DIFFUSION_BENCH_DISABLE_TORCH_COMPILE lx2v_profile=${PROFILING_DEBUG_LEVEL:-0} ==="
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src timeout "${DBF_CASE_TIMEOUT:-9000}" \
    python3 -m diffusion_bench.run_comparison \
    --config "$cfg_used" \
    --frameworks "$FW" --case-ids "$CASE" --modes single_e2e \
    --hardware-profile rtx4090 --port "$PORT" \
    --output "$OUT/${TAG}.json" \
    > "$OUT/${TAG}.runlog" 2>&1
rc=$?

kill "$smi_pid" 2>/dev/null
kill_own_gpus
printf '\nEXIT=%s\nDONE\n' "$rc" >> "$OUT/${TAG}.runlog"
echo "RESULT $TAG rc=$rc $(date -u +%FT%TZ)"
exit "$rc"
