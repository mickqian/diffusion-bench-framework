#!/usr/bin/env bash
# r1102 supplementary A/B on one 2xB300 devbox: interleaved harness arms, a fresh server per arm,
# one GPU-lock hold per invocation. Derived from run_q10_ab.sh (qwen, 2026-10-30) and run_ab.sh (H3 DP2,
# 2026-10-30); adds a per-arm machine-state snapshot (before/after), a 1 Hz nvidia-smi trace, and MP4 md5s.
#
#   SUPP_PLAN=<file; lines "<arm> <framework> <case> <modes> <gpus> <profile|-> [ENV=VAL ...]">
#   SUPP_TAG=<tag> [SUPP_PORT=<base port>] [DBF_LOG_DIR=...]
#   setsid -f bash run_supp_ab.sh > /workspace/r1102/logs-ab/<tag>.log 2>&1 < /dev/null
#
# modes join with '+' (single_e2e+throughput). A '-' profile means auto-selection for hardware b300.
# sglang arms read SGLANG_BENCH_SGLANG_PROFILE, vllm-omni arms DIFFUSION_BENCH_VLLM_OMNI_PROFILE.
# The HF token is read from the env file the devbox already holds and only ever lives in this
# process's environment (never echoed, never written to a file).
set -u

STATE=/workspace/r1102
export DBF_STATE_DIR="${STATE}"
export DBF_REPO_DIR="${STATE}/dbf"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${STATE}/logs-ab}"
export DBF_VENV_ROOT="${STATE}/fw-venvs" DBF_HF_HOME="${STATE}/hf"
export DBF_HF_TOKEN_FILE=/nonexistent/hftoken   # no token file on purpose: HF_TOKEN below stays the ambient one
export DBF_HARDWARE_PROFILE=b300 DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DBF_SGLANG_PYTHONPATH="${STATE}/sgl/python"
export DIFFUSION_BENCH_KEEP_VIDEO_DIR="${DBF_LOG_DIR}/kept-mp4"
# the specs the venv was installed from, so the result JSON records them (install itself is skipped)
export VLLM_INSTALL_SPEC=vllm==0.32.0
export VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git@d94d8e9809d8b56e4c58e37a39ddf454ab61f2fd
export VLLM_OMNI_VSA_KERNEL_SPEC=fastvideo-kernel==0.3.5

TAG="${SUPP_TAG:?tag}"; PLAN="${SUPP_PLAN:?plan file}"
mkdir -p "${DBF_LOG_DIR}" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}"

HF_TOKEN="$(. ~/.config/radixark/secrets.env >/dev/null 2>&1; printf '%s' "${HF_TOKEN:-}")"
[[ -n "${HF_TOKEN}" ]] || { echo "no HF_TOKEN in ~/.config/radixark/secrets.env"; exit 1; }
export HF_TOKEN

# the sglang tree must be the pinned one, not the image's /sgl-workspace
resolved="$(PYTHONPATH="${DBF_SGLANG_PYTHONPATH}" python3 -c 'import os, sglang; print(os.path.realpath(sglang.__file__))')"
[[ "${resolved}" == "${STATE}/sgl/python/"* ]] || { echo "sglang resolves to ${resolved}"; exit 1; }

# shellcheck source=scripts/gpu_job_lock.sh
source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
SMI=""
trap '[[ -n "${SMI}" ]] && kill "${SMI}" 2>/dev/null; gpu_lock_release_if_mine' EXIT

snap() {  # <file> <label>: host and GPU state that could explain a drift between arms
  {
    echo "--- $2 $(date -u +%FT%TZ)"
    uptime
    echo "loadavg: $(cat /proc/loadavg)"
    top -bn1 | sed -n '3p'
    free -g | sed -n '1,2p'
    nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader
    echo "compute-apps:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    echo "top-cpu:"; ps -eo pid,pcpu,etime,comm --sort=-pcpu | head -6
  } >> "$1" 2>&1
}

echo "=== SUPP START $(date -Is) tag=${TAG} plan=${PLAN} harness $(git -C "${DBF_REPO_DIR}" log --oneline -1 | cut -c1-70) sglang_tree=$(git -C "${STATE}/sgl" rev-parse HEAD)"
port="${SUPP_PORT:-61100}"
while read -r arm fw case_id modes gpus prof envs; do
  [[ -z "${arm}" || "${arm}" == \#* ]] && continue
  modes="${modes//+/ }"
  [[ "${prof}" == - ]] && prof=""
  t="${TAG}_${arm}"
  t0=$(date +%s)
  echo "=== ARM_START ${arm} ${fw} ${case_id} modes=[${modes}] gpus=${gpus} profile=${prof:-auto} port=${port} $(date -u +%FT%TZ)"
  snap "${DBF_LOG_DIR}/state_${t}.txt" PRE
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > "${DBF_LOG_DIR}/smi_${t}.csv" &
  SMI=$!
  if [[ "${fw}" == sglang ]]; then sp="${prof}"; fp=""; else sp=""; fp="${prof}"; fi
  env ${envs} DBF_FRAMEWORKS="${fw}" SGLANG_BENCH_SGLANG_PROFILE="${sp}" DIFFUSION_BENCH_VLLM_OMNI_PROFILE="${fp}" DIFFUSION_BENCH_FRAMEWORK_PROFILE="" \
    bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" "${t}" "${gpus}" "${port}" "${modes}" "${case_id}" < /dev/null
  kill "${SMI}" 2>/dev/null; SMI=""
  snap "${DBF_LOG_DIR}/state_${t}.txt" POST
  out=$(python3 - "${DBF_LOG_DIR}/run_${t}_${case_id}.json" "${fw}" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    print(f"no-result({exc.__class__.__name__})"); sys.exit()
parts = []
sha = ((d.get("sglang_runtime") or {}).get("git_commit") or "")[:10]
for r in d.get("results", []):
    if r.get("framework") == sys.argv[2]:
        m, fm = r.get("metrics") or {}, r.get("framework_metadata") or {}
        parts.append(f"single={r.get('latency_s')} samples={m.get('latency_samples_s')} srv={m.get('server_latency_s')} startup={m.get('server_startup_s')} profile={fm.get('profile')}({fm.get('profile_source')}) sha={sha}"
                     + (f" error={(r.get('error') or '')[:200]!r}" if r.get("error") else ""))
for r in d.get("throughput_results", []):
    if r.get("framework") == sys.argv[2]:
        m, fm = r.get("metrics") or {}, r.get("framework_metadata") or {}
        parts.append(f"tput rps={m.get('throughput_rps')} dur={m.get('duration_s')} p50={m.get('latency_p50_s')} done={m.get('completed_requests')}/{m.get('num_requests')} profile={fm.get('profile')}({fm.get('profile_source')}) sha={sha}"
                     + (f" error={(r.get('error') or '')[:200]!r}" if r.get("error") else ""))
print(" | ".join(parts) or "no-row")
PYEOF
)
  echo "ARM ${arm} ${fw} ${case_id} $(( $(date +%s) - t0 ))s ${out}"
  python3 "${DBF_REPO_DIR}/scripts/probe_mp4_encode.py" --group --since "${t0}" --label "${arm}" \
    "${DBF_REPO_DIR}/outputs" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}" >> "${DBF_LOG_DIR}/mp4_probe_${TAG}.jsonl" 2>/dev/null
  find "${DBF_REPO_DIR}/outputs" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}" -name '*.mp4' -newermt "@${t0}" -exec md5sum {} + 2>/dev/null | sed "s#^#${arm} #" >> "${DBF_LOG_DIR}/md5_${TAG}.txt"
  port=$((port + 10))
done < "${PLAN}"
echo "=== SUPP DONE $(date -Is) tag=${TAG} ==="
