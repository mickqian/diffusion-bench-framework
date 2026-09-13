#!/usr/bin/env bash
# How far is each tuned profile from what the runtime would have picked itself?
#
#   scripts/run_profile_vs_default.sh [rounds] [case_id ...]
#
# Every one of the 15 b200 cases pins at least one flag that says HOW to run
# rather than WHAT to run: parallelism on 8 of them, component residency on 8,
# attention backend on 3. Each pin is a standing claim that we know better than
# the runtime, and each has to be re-validated on every hardware generation or it
# quietly becomes the wrong answer.
#
# It has already been the wrong answer. Cosmos3 T2I pinned `--tp-size 2`, which
# makes `_has_explicit_parallel_policy()` true and therefore SUPPRESSES
# `_enable_cfg_parallel_if_supported()`. The runtime would have chosen CFG
# parallelism for that model on 2 GPUs -- 25% faster than the pin that was
# supposed to be the tuned answer.
#
# Two arms per case, interleaved, on the same box:
#   profile   the config as published
#   default   the same config with sglang's tuning flags stripped
#             (scripts/compare_profile_vs_default.py), so the runtime decides
#             parallelism, residency, attention backend and compile for itself.
# The GPU budget (`num_gpus`) is NOT stripped -- the question is what the runtime
# does with the cards it is given, not whether fewer cards are slower.
#
# Read the result per case:
#   default == profile   the pin earns nothing; drop it and let the runtime adapt
#   default  > profile   sglang's auto-selection has a gap worth fixing there
#   default  < profile   the pin is actively harming, as Cosmos3 T2I was
set -u

ROUNDS="${1:-2}"
shift || true
CASES=("$@")
if [ "${#CASES[@]}" -eq 0 ]; then
  # Default scope: everything that is minutes rather than tens of minutes, so a
  # first answer arrives before the expensive video cases are committed to.
  CASES=(zimage_turbo_t2i_1024 flux1_dev_t2i_1024 flux2_dev_t2i_1024
         qwen_image_2512_t2i_1024 qwen_image_2512_t2i_1024_truecfg
         qwen_image_edit_2511 ideogram4_t2i_1024_2gpu_tp
         cosmos3_nano_t2i_720p ltx2_twostage_t2v ltx2.3_twostage_t2v_2gpus)
fi

export DBF_STATE_DIR="${DBF_STATE_DIR:-/personal/bench0912}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-/scratch/dbf2}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export HF_HOME="${DBF_HF_HOME:-/cluster-storage/models}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1
[ -f "${DBF_STATE_DIR}/.hftoken" ] && export HF_TOKEN="$(cat "${DBF_STATE_DIR}/.hftoken")"

GPUS="${PVD_GPUS:-0,1}"
PORT="${PVD_PORT_BASE:-49001}"
HW="${DBF_HARDWARE_PROFILE:-blackwell}"
BARE="${DBF_STATE_DIR}/config_bare.json"

cd "${DBF_REPO_DIR}"
# Serialise against other GPU jobs on this box. Queueing by "wait until that
# script is gone from ps" failed three ways in one round; see gpu_job_lock.sh.
source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== PROFILE_VS_DEFAULT_START $(date -Is) ==="
git log --oneline -1

echo "--- building the stripped config"
python3 scripts/compare_profile_vs_default.py \
  --config configs/comparison_configs.json --out "${BARE}" || exit 1

run_arm() {  # run_arm <case_id> <arm> <config>
  local case_id="$1" arm="$2" cfg="$3"
  local tag="pvd_${case_id}_${arm}_${ROUND}"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPUS}" 2>/dev/null \
    | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
  CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONPATH=src \
    timeout "${PVD_CASE_TIMEOUT:-3600}" python3 -m diffusion_bench.run_comparison \
    --config "${cfg}" --frameworks sglang --case-ids "${case_id}" \
    --modes single_e2e --hardware-profile "${HW}" --port "${PORT}" \
    --output "${DBF_LOG_DIR}/${tag}.json" \
    > "${DBF_LOG_DIR}/${tag}.runlog" 2>&1
  python3 - "${DBF_LOG_DIR}/${tag}.json" "${case_id}" "${arm}" "${ROUND}" <<'PY'
import json, sys
path, case_id, arm, rnd = sys.argv[1:5]
try:
    d = json.load(open(path))
    r = next(x for x in d.get("results", []) if x.get("framework") == "sglang")
except Exception as e:
    print(f"PVD {case_id} {arm} r{rnd}: no result ({type(e).__name__})"); raise SystemExit
m = r.get("metrics") or {}
print(f"PVD {case_id} {arm} r{rnd}: median={r.get('latency_s')} "
      f"samples={m.get('latency_samples_s')} err={str(r.get('error'))[:60]}")
if m.get("native_fallback_components"):
    print(f"    !! native fallback: {m['native_fallback_components'][:1]}")
PY
  echo "    cmd: $(grep -ao 'sglang serve .*' "${DBF_LOG_DIR}/${tag}.runlog" 2>/dev/null | head -1 | sed 's/--model-path [^ ]* //; s/--host [^ ]* //; s/--port [^ ]* //')"
  PORT=$((PORT + 2))
}

for ROUND in $(seq 1 "${ROUNDS}"); do
  for case_id in "${CASES[@]}"; do
    echo "=== $(date -Is) round ${ROUND} ${case_id} ==="
    run_arm "${case_id}" profile configs/comparison_configs.json
    run_arm "${case_id}" default "${BARE}"
  done
done

echo "=== SUMMARY ==="
python3 - "${DBF_LOG_DIR}" "${ROUNDS}" "${CASES[@]}" <<'PY'
import json, os, statistics, sys
log_dir, rounds = sys.argv[1], int(sys.argv[2])
cases = sys.argv[3:]

def med(case_id, arm):
    vals = []
    for r in range(1, rounds + 1):
        p = os.path.join(log_dir, f"pvd_{case_id}_{arm}_{r}.json")
        try:
            d = json.load(open(p))
            row = next(x for x in d.get("results", []) if x.get("framework") == "sglang")
        except Exception:
            continue
        if row.get("latency_s"):
            vals.append(float(row["latency_s"]))
    return statistics.median(vals) if vals else None, vals

print(f"  {'case':34s} {'profile':>9s} {'default':>9s} {'delta':>9s}  verdict")
for case_id in cases:
    p, pv = med(case_id, "profile")
    d, dv = med(case_id, "default")
    if p is None or d is None:
        print(f"  {case_id:34s} {'--':>9s} {'--':>9s} {'--':>9s}  missing an arm "
              f"(profile {len(pv)}, default {len(dv)} round(s))")
        continue
    delta = (d - p) / p * 100
    if abs(delta) < 3:
        v = "the pin earns nothing -- drop it, let the runtime adapt"
    elif delta > 0:
        v = f"auto-selection is {delta:.0f}% behind the pin -- a real sglang gap"
    else:
        v = f"THE PIN IS HARMING: default is {-delta:.0f}% faster"
    print(f"  {case_id:34s} {p:9.3f} {d:9.3f} {delta:+8.1f}%  {v}")
print()
print("  'delta' is default minus profile: positive = the runtime's own choice is")
print("  slower, negative = our pin is costing us. Under 3% is inside this box's")
print("  cross-batch noise, so it means the pin is not earning its maintenance.")
PY
echo "=== PROFILE_VS_DEFAULT_DONE $(date -Is) ==="
