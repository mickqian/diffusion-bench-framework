#!/bin/bash
# Canonical devbox runner for benchmark cases — use THIS instead of hand-rolled
# per-experiment bash. It encodes the ops mistakes we actually made:
#   - `set -u` + a readonly port seed (an ad-hoc kill_gpu() once clobbered the
#     port variable and every run died with an argparse error);
#   - GPU cleanup narrowed to the devices you own (broad pkill on shared nodes
#     kills other people's jobs);
#   - results as greppable `RESULT` lines + all state under /persistent (local
#     pollers die with the session; the devbox log is the source of truth).
#
# Usage (on the devbox):
#   bash scripts/devbox_run_cases.sh <tag> <gpu_ids> <base_port> <modes> <case_id...>
# Example:
#   nohup bash scripts/devbox_run_cases.sh rebench 0,1 37001 "single_e2e throughput" \
#       zimage_turbo_t2i_1024 wan21_t2v_1_3b_480p >/dev/null 2>&1 &
#   tail -f /persistent/logs/run_<tag>.log
set -u

TAG="${1:?tag}"; GPU_IDS="${2:?gpu ids, e.g. 0,1}"; readonly BASE_PORT="${3:?base port}"
MODES="${4:?modes, e.g. 'single_e2e throughput'}"; shift 4
CASES=("$@"); [ "${#CASES[@]}" -gt 0 ] || { echo "no cases" >&2; exit 2; }

LOG="/persistent/logs/run_${TAG}.log"
exec > "$LOG" 2>&1

cd /persistent/diffusion-bench-framework
export HF_TOKEN="$(cat /persistent/.hftoken)"
export HF_HOME=/persistent/hf-cache
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/persistent/fw-venvs
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

kill_own_gpus() {
    # narrow kill: only PIDs on OUR devices; never a broad pkill on shared nodes
    local pids attempt
    for attempt in 1 2 3; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_IDS" 2>/dev/null | sort -u | tr '\n' ' ')
        [ -n "${pids// /}" ] || break
        kill -9 $pids 2>/dev/null
        sleep 4
    done
}

echo "=== RUN $TAG START $(date -u) cases=[${CASES[*]}] modes=[$MODES] gpus=$GPU_IDS ==="
port="$BASE_PORT"
for case_id in "${CASES[@]}"; do
    kill_own_gpus
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src timeout 3000 \
        python3 -m diffusion_bench.run_comparison \
        --config configs/comparison_configs.json \
        --frameworks sglang --case-ids "$case_id" --modes $MODES \
        --hardware-profile h100 --port "$port" \
        --output "/persistent/logs/run_${TAG}_${case_id}.json" \
        > "/persistent/logs/run_${TAG}_${case_id}.runlog" 2>&1
    rc=$?
    kill_own_gpus
    summary=$(grep -hE 'req/s|single_e2e' "/persistent/logs/run_${TAG}_${case_id}.runlog" 2>/dev/null | tail -2 | tr '\n' ' ')
    echo "RESULT $case_id rc=$rc $summary"
    port=$((port + 1))
done
echo "=== RUN $TAG DONE $(date -u) ==="
grep "^RESULT" "$LOG"
