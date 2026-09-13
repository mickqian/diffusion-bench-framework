#!/usr/bin/env bash
# Compare sglang serve-arg variants on one case, interleaved, on a devbox.
#
#   scripts/tune_sglang_serve_args.sh <case_id> <rounds> 'tag=<extra args>' ...
#
#   scripts/tune_sglang_serve_args.sh cosmos3_nano_t2i_720p 2 \
#       'bare=' \
#       'cfg=--tp-size 1 --cfg-parallel-size 2' \
#       'cfg_cudnn=--tp-size 1 --cfg-parallel-size 2 --attention-backend torch_cudnn_sdpa'
#
# Extra args APPEND to the selected profile's, so a repeated flag relies on
# argparse last-wins (`--tp-size 2 ... --tp-size 1`). That is why every arm
# echoes the command it actually served with, IN FULL: a truncated echo defeats
# the check it exists for -- a `cut -c1-240` once cut off the two flags under
# test, leaving the arms indistinguishable in the log while the latencies said
# they had clearly taken effect.
#
# Rounds are INTERLEAVED (ABAB, not AABB). Cross-batch noise on these boxes runs
# ~4-5%, so same-batch grouping resolves nothing below that; a claim under 5%
# needs the paired diffs to agree in sign across rounds. See the perf-metric
# rigor rule in the benchmarking SKILL.
#
# Environment (same as the other devbox scripts):
#   DBF_STATE_DIR  /personal/<box>          state + logs live here
#   DBF_REPO_DIR   /scratch/dbf2            a checkout that is NOT being edited
#   DBF_VENV_ROOT  $DBF_STATE_DIR/fw-venvs
#   DBF_HF_HOME    /cluster-storage/models
#   TUNE_GPUS      0,1                      cards to use and to clear between arms
#   TUNE_PORT_BASE 47001
set -u

CASE_ID="${1:?usage: tune_sglang_serve_args.sh <case_id> <rounds> 'tag=args' ...}"
ROUNDS="${2:?rounds}"
shift 2
[ "$#" -ge 1 ] || { echo "need at least one arm" >&2; exit 2; }

export DBF_STATE_DIR="${DBF_STATE_DIR:-/personal/bench0912}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-/scratch/dbf2}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_HF_HOME="${DBF_HF_HOME:-/cluster-storage/models}"
export HF_HOME="${DBF_HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${DBF_HF_HOME}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "${DBF_STATE_DIR}/.hftoken" ] && export HF_TOKEN="$(cat "${DBF_STATE_DIR}/.hftoken")"

GPUS="${TUNE_GPUS:-0,1}"
PORT="${TUNE_PORT_BASE:-47001}"
HW="${DBF_HARDWARE_PROFILE:-blackwell}"
source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
cd "${DBF_REPO_DIR}"

run_arm() {  # run_arm <tag> <extra serve args>
  local tag="$1" extra="$2"
  # Clear the cards first: a server left behind by a previous arm both holds
  # memory and contends for the GPU, which silently taxes the next arm.
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPUS}" 2>/dev/null \
    | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
  DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="$extra" CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONPATH=src \
    timeout "${TUNE_CASE_TIMEOUT:-2400}" python3 -m diffusion_bench.run_comparison \
    --config configs/comparison_configs.json --frameworks sglang --case-ids "${CASE_ID}" \
    --modes single_e2e --hardware-profile "${HW}" --port "${PORT}" \
    --output "${DBF_LOG_DIR}/tune_${tag}.json" \
    > "${DBF_LOG_DIR}/tune_${tag}.runlog" 2>&1
  python3 - "${DBF_LOG_DIR}/tune_${tag}.json" "$tag" <<'PY'
import json, sys
path, tag = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"TUNE {tag}: no result ({type(e).__name__})"); raise SystemExit
r = next((x for x in d.get("results", []) if x.get("framework") == "sglang"), None)
if r is None:
    print(f"TUNE {tag}: no sglang row"); raise SystemExit
m = r.get("metrics") or {}
print(f"TUNE {tag}: median={r.get('latency_s')} samples={m.get('latency_samples_s')} "
      f"spread={m.get('latency_spread_pct')}% server={m.get('server_latency_s')} "
      f"err={str(r.get('error'))[:70]}")
if m.get("native_fallback_components"):
    print(f"     !! this arm fell back to a native implementation: {m['native_fallback_components'][:1]}")
PY
  # In full: the point of echoing it is to check the last-wins assumption.
  echo "     cmd: $(grep -ao 'sglang serve .*' "${DBF_LOG_DIR}/tune_${tag}.runlog" 2>/dev/null | head -1)"
  PORT=$((PORT + 2))
}

echo "=== TUNE_START $(date -Is) case=${CASE_ID} rounds=${ROUNDS} arms=$# ==="
git log --oneline -1 2>/dev/null
for round in $(seq 1 "${ROUNDS}"); do
  for arm in "$@"; do
    run_arm "${arm%%=*}_${round}" "${arm#*=}"
  done
done
echo "=== TUNE_DONE $(date -Is) ==="
