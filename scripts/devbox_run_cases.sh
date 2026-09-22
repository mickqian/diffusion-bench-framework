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
#
# Layout and run scope are env-overridable, because they were hardcoded to one
# devbox's paths, to sglang-only and to h100 — so the first box that mounted
# its state elsewhere (or ran a cross-framework matrix, or was Blackwell) had
# no way to use this script, which is how hand-rolled per-run bash keeps coming
# back. Defaults are the original values, so existing invocations are unchanged:
#   DBF_STATE_DIR=/persistent      DBF_REPO_DIR=$STATE/diffusion-bench-framework
#   DBF_LOG_DIR=$STATE/logs        DBF_VENV_ROOT=$STATE/fw-venvs
#   DBF_HF_HOME=$STATE/hf-cache    DBF_HF_TOKEN_FILE=$STATE/.hftoken
#   DBF_FRAMEWORKS=sglang          DBF_HARDWARE_PROFILE=h100
#   DBF_CASE_TIMEOUT=3000
set -u

TAG="${1:?tag}"; GPU_IDS="${2:?gpu ids, e.g. 0,1}"; readonly BASE_PORT="${3:?base port}"
MODES="${4:?modes, e.g. 'single_e2e throughput'}"; shift 4
CASES=("$@"); [ "${#CASES[@]}" -gt 0 ] || { echo "no cases" >&2; exit 2; }

STATE_DIR="${DBF_STATE_DIR:-/persistent}"
REPO_DIR="${DBF_REPO_DIR:-${STATE_DIR}/diffusion-bench-framework}"
LOG_DIR="${DBF_LOG_DIR:-${STATE_DIR}/logs}"
HW_PROFILE="${DBF_HARDWARE_PROFILE:-h100}"
FRAMEWORKS="${DBF_FRAMEWORKS:-sglang}"
CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-3000}"
TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${STATE_DIR}/.hftoken}"

mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_${TAG}.log"
exec > "$LOG" 2>&1

cd "$REPO_DIR"
# Only override the ambient token when a token file exists: `HF_TOKEN=$(cat
# missing-file)` exports an EMPTY token, which silently replaces a working
# login with an anonymous one and turns gated models into 401s.
[ -r "$TOKEN_FILE" ] && export HF_TOKEN="$(cat "$TOKEN_FILE")"
# HF_HOME alone does not decide where huggingface_hub reads: HF_HUB_CACHE and
# the legacy HUGGINGFACE_HUB_CACHE both outrank it, and several clusters export
# HUGGINGFACE_HUB_CACHE=/cluster-storage/models into every shell. Setting only
# HF_HOME there sends the runtime to a READ-ONLY cache, which surfaces as
# "Could not get model info for '<model>'" -- a message that reads like an
# unsupported model rather than a cache it could not write. Pin all three.
export HF_HOME="${DBF_HF_HOME:-${STATE_DIR}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${DBF_VENV_ROOT:-${STATE_DIR}/fw-venvs}"
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
echo "=== repo=$REPO_DIR hw=$HW_PROFILE frameworks=$FRAMEWORKS venvs=$SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT hf=$HF_HOME ==="
port="$BASE_PORT"
for case_id in "${CASES[@]}"; do
    kill_own_gpus
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=src timeout "$CASE_TIMEOUT" \
        python3 -m diffusion_bench.run_comparison \
        --config configs/comparison_configs.json \
        --frameworks $FRAMEWORKS --case-ids "$case_id" --modes $MODES \
        --hardware-profile "$HW_PROFILE" --port "$port" \
        --output "${LOG_DIR}/run_${TAG}_${case_id}.json" \
        > "${LOG_DIR}/run_${TAG}_${case_id}.runlog" 2>&1
    rc=$?
    kill_own_gpus
    summary=$(grep -hE 'req/s|single_e2e' "${LOG_DIR}/run_${TAG}_${case_id}.runlog" 2>/dev/null | tail -2 | tr '\n' ' ')
    echo "RESULT $case_id rc=$rc $summary"
    port=$((port + 1))
done
echo "=== RUN $TAG DONE $(date -u) ==="
grep "^RESULT" "$LOG"
