#!/usr/bin/env bash
# A mutex for GPU jobs on a devbox. Source it, then call gpu_lock_acquire.
#
#   source scripts/gpu_job_lock.sh
#   gpu_lock_acquire            # blocks until the cards are free, releases on exit
#
# Queueing by "wait until that other script is not in ps" has now failed three
# different ways in one round:
#
#   1. `rxrun.sh` retries transport failures and re-runs the whole command, so a
#      detached launch inside one started the same probe three times, all parked
#      in the same wait loop and all set to start at once.
#   2. `pgrep -fc <path>` also matched the launching `bash -c`, whose command
#      line contains the path because it wrote the file -- 3 counted, 1 running.
#   3. A waiter matched `bash .../pvd_wait.sh` exactly, but that script ends in
#      `exec bash .../pvd.sh 2`; exec replaced the command line, the match went
#      to zero, and the next job started CONCURRENTLY. Its `nvidia-smi | xargs
#      kill -9` then killed the running job's server, which surfaced as
#      "sglang server exited before health check passed (exit -9)" -- a failure
#      that looks like sglang's and is not.
#
# The name of a process is the wrong thing to synchronise on. `mkdir` is atomic,
# so a lock directory is a real mutex, and a dead owner is detectable.
set -u

GPU_LOCK_DIR="${GPU_LOCK_DIR:-${DBF_STATE_DIR:-/personal/bench0912}/.gpu_lock}"
GPU_LOCK_POLL="${GPU_LOCK_POLL:-60}"

gpu_lock_acquire() {
    local waited=0
    while true; do
        if mkdir "${GPU_LOCK_DIR}" 2>/dev/null; then
            echo "$$" > "${GPU_LOCK_DIR}/pid"
            echo "${0##*/}" > "${GPU_LOCK_DIR}/job"
            # shellcheck disable=SC2064
            trap "rm -rf '${GPU_LOCK_DIR}'" EXIT INT TERM
            [ "${waited}" -gt 0 ] && echo "gpu-lock: acquired after ${waited}s"
            return 0
        fi
        local owner job
        owner="$(cat "${GPU_LOCK_DIR}/pid" 2>/dev/null || true)"
        job="$(cat "${GPU_LOCK_DIR}/job" 2>/dev/null || echo '?')"
        if [ -z "${owner}" ] || ! kill -0 "${owner}" 2>/dev/null; then
            echo "gpu-lock: clearing a stale lock from ${job} (pid ${owner:-none} is gone)"
            rm -rf "${GPU_LOCK_DIR}"
            continue
        fi
        if [ "${waited}" = 0 ]; then
            echo "gpu-lock: waiting for ${job} (pid ${owner})"
        fi
        sleep "${GPU_LOCK_POLL}"
        waited=$((waited + GPU_LOCK_POLL))
    done
}

gpu_lock_release() {
    rm -rf "${GPU_LOCK_DIR}"
    trap - EXIT INT TERM
}
