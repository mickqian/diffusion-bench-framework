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
#
# The same trap survives in the OBSERVABILITY layer even once the lock fixes
# correctness: a monitor counting `pvd_resid.sh` reported the job dead while it
# sat in this very wait loop, because that script ends in `exec bash pvd.sh ...`.
# Count what `ps` shows AFTER any exec -- or better, read the job's log, which
# says `gpu-lock: waiting for <job>` and cannot be confused by a replaced argv.
set -u

GPU_LOCK_DIR="${GPU_LOCK_DIR:-${DBF_STATE_DIR:-/personal/bench0912}/.gpu_lock}"
GPU_LOCK_POLL="${GPU_LOCK_POLL:-60}"

# Release only a lock we still hold. An unconditional `rm -rf` here destroyed
# the lock a DIFFERENT job had since taken: killing a job ran its EXIT trap,
# which deleted the successor's lock directory, and the next starter walked
# straight in. Two jobs then ran on the same GPUs, one of them silently
# contaminating the other's measurements while the lock file named only one.
gpu_lock_release_if_mine() {
    [ "$(cat "${GPU_LOCK_DIR}/pid" 2>/dev/null)" = "$$" ] && rm -rf "${GPU_LOCK_DIR}"
}

# A trapped signal does NOT end the script: bash runs the handler and resumes on
# the next line. With the release handler on INT/TERM and nothing else, `kill`
# made a job give up its lock and KEEP RUNNING -- the worst of both. One did
# exactly that: TERM'd, it released, the next job took the lock and started
# measuring, and the "killed" job walked into its next arm, ran
# `nvidia-smi | xargs kill -9`, and destroyed the running job's server before
# parking its own model on both cards for twenty minutes. So the signal handler
# has to exit; only EXIT may be a bare release.
_gpu_lock_signal_exit() {
    gpu_lock_release_if_mine
    exit 143
}

gpu_lock_acquire() {
    local waited=0
    while true; do
        if mkdir "${GPU_LOCK_DIR}" 2>/dev/null; then
            echo "$$" > "${GPU_LOCK_DIR}/pid"
            echo "${0##*/}" > "${GPU_LOCK_DIR}/job"
            hostname > "${GPU_LOCK_DIR}/host"
            trap gpu_lock_release_if_mine EXIT
            trap _gpu_lock_signal_exit INT TERM
            [ "${waited}" -gt 0 ] && echo "gpu-lock: acquired after ${waited}s"
            return 0
        fi
        local owner job
        owner="$(cat "${GPU_LOCK_DIR}/pid" 2>/dev/null || true)"
        job="$(cat "${GPU_LOCK_DIR}/job" 2>/dev/null || echo '?')"
        # The lock lives under DBF_STATE_DIR, which on these boxes is
        # /personal -- per-developer cluster storage that OUTLIVES the devbox. A
        # lock left by a released box therefore names a pid from a machine that
        # no longer exists, and `kill -0` on this one answers about an unrelated
        # process that happens to share the number. Check the host first.
        local host
        host="$(cat "${GPU_LOCK_DIR}/host" 2>/dev/null || true)"
        if [ -n "${host}" ] && [ "${host}" != "$(hostname)" ]; then
            echo "gpu-lock: clearing a lock from ${job} on ${host}; this is $(hostname)"
            rm -rf "${GPU_LOCK_DIR}"
            continue
        fi
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
    gpu_lock_release_if_mine
    trap - EXIT INT TERM
}
