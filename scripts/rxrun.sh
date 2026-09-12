#!/usr/bin/env bash
# Run a command on an rx devbox, retrying ONLY transport failures.
#
# `rx` fails two different ways and they must not be treated alike. A transport
# failure (`websocket: close 1006`, `error listing devboxes: ...: EOF`,
# `rx: dial: EOF`, an i/o timeout) says nothing about the box -- the remote
# command may not have run, or may have run fine with its output lost. Reading
# one as remote state has cost real runs: a `close 1006` was read as "the runner
# vanished" and killed a healthy 40-minute job. A command failure (non-zero exit
# from the remote command itself) is real and must surface immediately -- a
# retry would just re-run a failing command N times and report the last one.
#
# The predicate has also been the bug twice, both times by being too narrow:
# each new rx error shape was patched at one call site while the others kept
# mis-reading it. It lives here now, with tests, so widening it is one edit.
#
#   scripts/rxrun.sh <box> <command...>      # command runs under bash -c there
#   RXRUN_TRIES=8 RXRUN_SLEEP=20 scripts/rxrun.sh ...
#
# Source it for the predicate alone:
#   source scripts/rxrun.sh; rxrun_is_transport_error "$stderr" && ...
set -uo pipefail

# Anchored to rx's own Go error shapes. Deliberately NOT a bare `EOF`: remote
# output legitimately contains "unexpected EOF while parsing" (a Python syntax
# error) and "unexpected EOF" (a truncated heredoc), and retrying those would
# hide a real failure N times over instead of reporting it once.
rxrun_is_transport_error() {
    local text="$1"
    grep -qiE \
        '^rx: |error listing devboxes:|websocket: close|i/o timeout|dial tcp|: dial:|context deadline exceeded|connection reset by peer|: EOF$|no such host|server misbehaving|TLS handshake timeout' \
        <<<"$text"
}

rxrun() {
    local box="$1"; shift
    local cmd="$*"
    local tries="${RXRUN_TRIES:-5}" nap="${RXRUN_SLEEP:-15}"
    local err out rc attempt
    err="$(mktemp)"
    for ((attempt = 1; attempt <= tries; attempt++)); do
        out="$(rx devbox run "$box" -- bash -c "$cmd" 2>"$err")"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            printf '%s\n' "$out"
            rm -f "$err"
            return 0
        fi
        if rxrun_is_transport_error "$(cat "$err")"; then
            echo "rxrun: transport error (attempt $attempt/$tries), retrying: $(head -c 160 "$err")" >&2
            sleep "$nap"
            continue
        fi
        # A real remote failure: surface it as-is, never retry.
        printf '%s\n' "$out"
        cat "$err" >&2
        rm -f "$err"
        return "$rc"
    done
    echo "rxrun: giving up after $tries transport failures" >&2
    rm -f "$err"
    return 75  # EX_TEMPFAIL
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    [[ $# -ge 2 ]] || { echo "usage: rxrun.sh <box> <command...>" >&2; exit 2; }
    rxrun "$@"
fi
