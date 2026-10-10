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
#   DBF_CASE_TIMEOUT=3000         DBF_CONFIG=configs/comparison_configs.json
#   DBF_SGLANG_PYTHONPATH=<sglang>/python   run an sglang source tree (e.g. main plus
#     unmerged fixes, committed on a local branch so the result records its sha)
#     instead of the installed one; it goes first on PYTHONPATH for the harness and
#     the servers it starts.
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
# A variant config (e.g. an old competitor profile) A/Bs a command on the same box.
CONFIG="${DBF_CONFIG:-configs/comparison_configs.json}"
HARNESS_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:+${DBF_SGLANG_PYTHONPATH}:}src"

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

own_gpu_pids() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_IDS" 2>/dev/null
        return
    fi
    # ROCm: amd-smi reports HOST pids, which name nothing (or the wrong process) inside
    # a container. Take the container's own processes holding /dev/kfd and keep those
    # launched with exactly our device list, which every server this runner starts inherits.
    local p
    for p in $(find /proc/[0-9]*/fd -lname /dev/kfd 2>/dev/null | cut -d/ -f3 | sort -u); do
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU_IDS" && echo "$p"
    done
}

kill_own_gpus() {
    # narrow kill: only PIDs on OUR devices; never a broad pkill on shared nodes
    local pids attempt
    for attempt in 1 2 3; do
        pids=$(own_gpu_pids | sort -u | tr '\n' ' ')
        [ -n "${pids// /}" ] || break
        kill -9 $pids 2>/dev/null
        sleep 4
    done
}

# Largest VRAM use (MiB) on our devices.
gpu_used_mib() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_IDS" 2>/dev/null | sort -n | tail -1
        return
    fi
    rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, -v ids=",$GPU_IDS," '
        $1 ~ /^card[0-9]+$/ && index(ids, "," substr($1, 5) ",") { mib = int($3 / 1048576); if (mib > max) max = mib }
        END { print max + 0 }'
}

# A case starts on empty cards. Killing is not enough: after a GPU memory fault the
# dying workers held ~250 GB per MI355X for over an hour (2026-10-10), and the cases
# that ran on top of them crawled at 100 s/step or OOMed.
wait_gpus_free() {
    local waited=0 used
    while :; do
        used=$(gpu_used_mib)
        (( ${used:-0} < 2048 )) && return 0
        if (( waited >= 600 )); then
            echo "GPUs $GPU_IDS still hold ${used} MiB after ${waited}s"
            return 1
        fi
        (( waited == 0 )) && echo "waiting for GPUs $GPU_IDS to free (${used} MiB in use)"
        kill_own_gpus
        sleep 15
        waited=$((waited + 15))
    done
}

echo "=== RUN $TAG START $(date -u) cases=[${CASES[*]}] modes=[$MODES] gpus=$GPU_IDS ==="
echo "=== repo=$REPO_DIR config=$CONFIG hw=$HW_PROFILE frameworks=$FRAMEWORKS venvs=$SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT hf=$HF_HOME sglang=${DBF_SGLANG_PYTHONPATH:-installed} ==="
port="$BASE_PORT"
for case_id in "${CASES[@]}"; do
    kill_own_gpus
    if ! wait_gpus_free; then
        echo "RESULT $case_id rc=CONTAMINATED: GPUs $GPU_IDS not free at case start, case not run"
        port=$((port + 1))
        continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$HARNESS_PYTHONPATH" timeout "$CASE_TIMEOUT" \
        python3 -m diffusion_bench.run_comparison \
        --config "$CONFIG" \
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
