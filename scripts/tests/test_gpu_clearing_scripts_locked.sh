#!/usr/bin/env bash
# A script that clears the cards must hold the GPU lock first.
#
# Every runner and probe here opens by killing whatever is on the GPUs. That is
# correct in isolation and destructive in company: a second job that starts while
# the first is measuring kills the first one's server, which surfaces as
# "sglang server exited before health check passed (exit -9)" -- a failure that
# reads as sglang's and is not, and both runs then have to be discarded.
#
# The lock landed in scripts/gpu_job_lock.sh, but three scripts that clear the
# cards were written before it and did not take it. This check stops the next one
# from being written the same way.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

fail=0
found=0
for f in scripts/*.sh; do
    # The kill and the nvidia-smi feeding it usually sit on separate lines, so
    # match the kill itself rather than a one-line pipeline.
    grep -qE 'xargs (-r )?kill -9' "$f" || continue
    found=$((found + 1))
    name="${f#scripts/}"
    if grep -q 'gpu_lock_acquire' "$f"; then
        echo "  ok   $name clears the cards and takes the lock"
    else
        echo "  FAIL $name clears the cards without gpu_lock_acquire"
        fail=1
    fi
done

[ "$found" -gt 0 ] || { echo "  FAIL matched no card-clearing script -- the pattern has drifted"; fail=1; }
[ -f scripts/gpu_job_lock.sh ] || { echo "  FAIL scripts/gpu_job_lock.sh is missing"; fail=1; }
exit $fail
