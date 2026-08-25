#!/bin/bash
# Fortnightly apple-to-apple cross-framework benchmark, end to end.
#
#   acquire a fresh H200 devbox -> sglang origin/main + competitors at latest
#   -> run the matrix -> publish to the Pages data -> release the devbox
#
# Run by launchd (see scripts/launchd/com.mick.diffusion-bench-biweekly.plist)
# on the 1st and 15th, and safe to run by hand:
#
#   bash scripts/biweekly_fair_bench.sh
#
# Every exit path releases the devbox: an abandoned 2-GPU box is the expensive
# failure mode, worse than a missed run.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M)"
RUN_ID="h200x2-fair-$(date +%Y%m%d)"
BOX="diffbench-auto-${STAMP}"
LOG_DIR="${REPO}/tmp/biweekly"
LOG="${LOG_DIR}/${STAMP}.log"
REMOTE_REPO="/scratch/diffusion-bench-framework"
mkdir -p "${LOG_DIR}"

# rx's exec transport goes through the local proxy; without it every remote
# call dies on an i/o timeout to a rotating AWS IP.
export https_proxy="${https_proxy:-http://127.0.0.1:7891}"
export http_proxy="${http_proxy:-http://127.0.0.1:7891}"
export all_proxy="${all_proxy:-socks5://127.0.0.1:7891}"

# The publish step imports the harness (requests, packaging); /usr/local/bin's
# python3 has none of it. Override with BENCH_PYTHON if this box differs.
PY="${BENCH_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/bin/python3}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
rxrun() { rx devbox run "${BOX}" -- bash -c "$1"; }

cleanup() {
  local rc=$?
  if [[ -n "${BOX_ACQUIRED:-}" ]]; then
    log "releasing ${BOX}"
    rx devbox release "${BOX}" >>"${LOG}" 2>&1 || log "WARN: release failed — check 'rx devbox list'"
  fi
  log "exit rc=${rc}, log at ${LOG}"
}
trap cleanup EXIT

log "=== biweekly fair bench ${STAMP} ==="

# --- resolve every competitor's latest, so the run is latest-vs-latest -------
VLLM_LATEST="$(curl -s https://pypi.org/pypi/vllm/json | "${PY}" -c 'import json,sys;print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null)"
LX2V_LATEST="$(git ls-remote https://github.com/ModelTC/LightX2V.git refs/heads/main 2>/dev/null | cut -c1-12)"
[[ -n "${VLLM_LATEST}" ]] || { log "FATAL: could not resolve latest vllm"; exit 1; }
[[ -n "${LX2V_LATEST}" ]] || { log "FATAL: could not resolve LightX2V main"; exit 1; }
log "vllm=${VLLM_LATEST} lightx2v=${LX2V_LATEST}"

# --- acquire ----------------------------------------------------------------
log "acquiring 2x H200 as ${BOX}"
# --image skips the interactive image menu; the piped y answers the "you already
# have devboxes, acquire another?" prompt that only appears when some exist.
# TTL is generous on purpose: a box expiring mid-run takes /scratch with it, and
# the whole run is lost (happened 2026-08-25 on a 12h TTL).
printf 'y\n' | rx devbox acquire --gpu h200 --count 2 --image lmsysorg/sglang:latest \
  --name "${BOX}" --ttl 48h >>"${LOG}" 2>&1 || { log "FATAL: acquire failed"; exit 1; }
BOX_ACQUIRED=1

# Probe the capability we actually need -- can we execute on it? -- instead of
# grepping the human-readable status line. `rx devbox run` returns 409 with
# "devbox not running (status=provisioning)" until the box is ready, so a
# successful `true` is the real readiness signal. Parsing status text failed a
# run on 2026-08-25 and left no record of what the status actually said.
READY=""
for _ in $(seq 1 120); do
  if rxrun 'true' >/dev/null 2>&1; then READY=1; break; fi
  sleep 20
done
if [[ -z "${READY}" ]]; then
  log "FATAL: devbox never became executable after 40m. Last status:"
  rx devbox status "${BOX}" 2>&1 | head -5 | tee -a "${LOG}"
  exit 1
fi
log "devbox running"

# --- sglang origin/main -----------------------------------------------------
# The image ships a release branch as a shallow clone whose only refspec is the
# release tag, so origin/main does not exist until the refspec is added.
log "moving sglang to origin/main"
rxrun 'cd /sgl-workspace/sglang \
  && (git config --get-all remote.origin.fetch | grep -q "heads/main" \
      || git config --add remote.origin.fetch "+refs/heads/main:refs/remotes/origin/main") \
  && git fetch --depth 1 -q origin main \
  && git checkout -q -B benchmain origin/main \
  && git log --oneline -1' >>"${LOG}" 2>&1 || { log "FATAL: sglang checkout failed"; exit 1; }
SGL_COMMIT="$(rxrun 'cd /sgl-workspace/sglang && git rev-parse --short=9 HEAD' 2>/dev/null | tr -d "\r\n ")"
log "sglang @ ${SGL_COMMIT}"

# --- harness ----------------------------------------------------------------
log "installing harness"
rxrun "rm -rf ${REMOTE_REPO} \
  && git clone -q --depth 1 https://github.com/mickqian/diffusion-bench-framework.git ${REMOTE_REPO} \
  && cd ${REMOTE_REPO} && pip install -q -e ." >>"${LOG}" 2>&1 || { log "FATAL: harness install failed"; exit 1; }

# --- competitors at latest --------------------------------------------------
# FA3's prebuilt wheel must match the venv's torch+CUDA or LightX2V silently
# falls back to FA2 and gets under-reported; the subdir is picked to match.
read -r -d '' ENVSPEC <<EOS
export HF_HOME=/cluster-storage/models
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export VLLM_INSTALL_SPEC="vllm==${VLLM_LATEST}"
export VLLM_OMNI_INSTALL_SPEC="git+https://github.com/vllm-project/vllm-omni.git@main"
export LIGHTX2V_INSTALL_SPEC="git+https://github.com/ModelTC/LightX2V.git@${LX2V_LATEST}"
export LIGHTX2V_FA3_HF_SUBDIR="build/torch211-cxx11-cu130-x86_64-linux/flash_attention_3"
EOS

for fw in vllm-omni lightx2v trtllm-visual; do
  log "installing ${fw}"
  rxrun "${ENVSPEC}
    cd ${REMOTE_REPO} && bash scripts/install_comparison_frameworks.sh ${fw}" >>"${LOG}" 2>&1
  if rxrun "test -f /tmp/sglang-diffusion-framework-venvs/${fw}/.diffusion-bench-install-stamp" 2>/dev/null; then
    log "  ${fw} OK"
  else
    # A framework that fails to install is reported as a failed cell by the
    # harness, which is the honest outcome — do not abort the whole run.
    log "  WARN: ${fw} has no install stamp; its cells will be classified, not silently dropped"
  fi
done

# --- run --------------------------------------------------------------------
log "running the matrix (this takes hours)"
rxrun "${ENVSPEC}
  cd ${REMOTE_REPO}
  mkdir -p /scratch/results
  setsid bash -c 'diffusion-bench-compare --modes single_e2e throughput \
    --hardware-profile h200 --output /scratch/results/${RUN_ID}.json \
    > /scratch/results/${RUN_ID}.log 2>&1; echo \$? > /scratch/results/${RUN_ID}.done' </dev/null &
  sleep 5" >>"${LOG}" 2>&1

while true; do
  if rxrun "test -f /scratch/results/${RUN_ID}.done" 2>/dev/null; then break; fi
  if ! rxrun 'pgrep -f "diffusion-bench-compar[e]" >/dev/null' 2>/dev/null; then
    log "FATAL: runner vanished without finishing"; rxrun "tail -30 /scratch/results/${RUN_ID}.log" | tee -a "${LOG}"; exit 1
  fi
  sleep 300
done
log "run finished rc=$(rxrun "cat /scratch/results/${RUN_ID}.done" 2>/dev/null | tr -d '\r\n ')"

# --- pull results (chunked: a single base64 of a big file truncates) ---------
log "pulling results"
mkdir -p "${REPO}/tmp/report"
RAW="${REPO}/tmp/report/${RUN_ID}-raw.json"
rxrun "split -b 400k -d /scratch/results/${RUN_ID}.json /scratch/results/part_" >>"${LOG}" 2>&1
: > "${RAW}"
for i in $(rxrun "ls /scratch/results/part_* | sed 's#.*/part_##'" 2>/dev/null | tr -d '\r'); do
  rxrun "base64 -w0 /scratch/results/part_${i}" 2>/dev/null | tr -d '\n\r' | base64 -d >> "${RAW}"
done
REMOTE_MD5="$(rxrun "md5sum /scratch/results/${RUN_ID}.json | cut -d' ' -f1" 2>/dev/null | tr -d '\r\n ')"
LOCAL_MD5="$(md5 -q "${RAW}" 2>/dev/null || md5sum "${RAW}" | cut -d' ' -f1)"
[[ "${REMOTE_MD5}" == "${LOCAL_MD5}" ]] || { log "FATAL: result transfer corrupted (${LOCAL_MD5} != ${REMOTE_MD5})"; exit 1; }
log "results verified (${LOCAL_MD5})"

# --- publish ----------------------------------------------------------------
cd "${REPO}" || exit 1
MERGED="${REPO}/tmp/report/${RUN_ID}-merged.json"
"${PY}" -m diffusion_bench.build_report_artifacts \
  --results "${RAW}" --run-id "${RUN_ID}" \
  --output-json "${MERGED}" \
  --dashboard-md "${REPO}/tmp/report/${RUN_ID}-dashboard.md" \
  --issue-md "${REPO}/tmp/report/${RUN_ID}-issue.md" >>"${LOG}" 2>&1 \
  || { log "FATAL: build_report_artifacts failed"; exit 1; }

"${PY}" scripts/publish_bench_run.py --merged "${MERGED}" --run-id "${RUN_ID}" \
  --label "H200 cross-framework (latest-vs-latest)" \
  --gpu "2x NVIDIA H200 143GB" >>"${LOG}" 2>&1 \
  || { log "FATAL: publish failed"; exit 1; }

git add docs/data docs/index.html
if git diff --cached --quiet; then
  log "no data change to commit"
else
  git commit -q -m "data: cross-framework benchmark ${RUN_ID}

sglang ${SGL_COMMIT} vs vllm ${VLLM_LATEST} / vllm-omni main / lightx2v ${LX2V_LATEST}
on 2x H200, compile-on, no caches or quantization. Generated by
scripts/biweekly_fair_bench.sh." >>"${LOG}" 2>&1
  git -c rebase.autoStash=true pull --rebase -q >>"${LOG}" 2>&1
  git push >>"${LOG}" 2>&1 && log "published — Pages will redeploy" || log "WARN: push failed; commit is local"
fi

log "=== done ==="
