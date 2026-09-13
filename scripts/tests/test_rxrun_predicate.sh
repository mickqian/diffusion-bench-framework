#!/usr/bin/env bash
# The rxrun transport-error predicate, against real strings from real runs.
# Both directions matter: a missed transport error stalls or kills a good run,
# and a false positive silently retries a genuinely failing command.
set -uo pipefail
cd "$(dirname "$0")/.."
source ./rxrun.sh

fail=0
expect() { # expect <yes|no> <description> <text>
    local want="$1" desc="$2" text="$3" got
    if rxrun_is_transport_error "$text"; then got=yes; else got=no; fi
    if [[ "$got" == "$want" ]]; then
        printf 'ok    %-4s %s\n' "$got" "$desc"
    else
        printf 'FAIL  want=%s got=%s  %s\n' "$want" "$got" "$desc"
        fail=1
    fi
}

# --- transport failures: must retry -----------------------------------------
expect yes "rx dial EOF"            'rx: dial: EOF'
expect yes "devbox list EOF"        'error listing devboxes: Get "https://nodes.radixark.ai/api/developer/devbox?include_released=1": EOF'
expect yes "websocket 1006"         'rx: websocket: close 1006 (abnormal closure): unexpected EOF'
expect yes "i/o timeout"            'Get "https://nodes.radixark.ai/api": dial tcp 1.2.3.4:443: i/o timeout'
expect yes "context deadline"       'rpc error: context deadline exceeded'
expect yes "conn reset"             'read tcp 10.0.0.1:5 -> 10.0.0.2:443: connection reset by peer'
expect yes "dns"                    'dial tcp: lookup nodes.radixark.ai: no such host'
expect yes "tls"                    'net/http: TLS handshake timeout'

# --- real remote failures: must NOT retry ------------------------------------
expect no  "python syntax EOF"      'File "<string>", line 4
SyntaxError: unexpected EOF while parsing'
expect no  "truncated heredoc"      'bash: line 12: warning: here-document delimited by end-of-file (wanted `EOF'"'"')'
expect no  "CUDA OOM"               'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB'
expect no  "bad argument"           'run_comparison.py: error: unrecognized arguments: false'
expect no  "missing file"           'cat: /personal/bench0912/nope.json: No such file or directory'
expect no  "nonzero exit"           'RESULT zimage rc=1'
expect no  "empty stderr"           ''

# --- the box is gone: terminal, and must not be mistaken for flaky networking -
# This 409 starts with "rx: ", so the transport predicate matched it and rxrun
# retried a released devbox five times before reporting "giving up after 5
# transport failures" -- which reads like a network problem. It was a TTL
# expiring under a queue of running jobs.
gone() { # gone <yes|no> <description> <text>
    local want="$1" desc="$2" text="$3" got
    if rxrun_is_box_gone "$text"; then got=yes; else got=no; fi
    if [[ "$got" == "$want" ]]; then
        printf 'ok    %-4s %s\n' "$got" "$desc"
    else
        printf 'FAIL  want=%s got=%s  %s\n' "$want" "$got" "$desc"
        fail=1
    fi
}

RELEASED='rx: server rejected run (409): {"error":"devbox not running (status=released) \u2014 released 2026-09-13 14:02:37 UTC; acquire a new one with: rx devbox acquire bench-0912"}'
gone yes "released devbox"          "$RELEASED"
gone yes "releasing devbox"         'rx: server rejected run (409): {"error":"devbox not running (status=releasing)"}'
gone yes "devbox not found"         'rx: devbox not found: bench-0912'
gone no  "a real transport error"   'rx: dial: EOF'
gone no  "a remote command failure" 'cat: /personal/bench0912/nope.json: No such file or directory'
gone no  "empty stderr"             ''

# The two predicates must not both claim it, or the order of the checks decides
# the behaviour by accident.
if rxrun_is_box_gone "$RELEASED" && ! rxrun_is_transport_error "$RELEASED"; then
    printf 'ok    only  a released box is terminal, not transport\n'
else
    printf 'note  both  the released 409 also matches the transport pattern, so\n'
    printf '            rxrun_is_box_gone MUST be checked first\n'
    grep -q 'rxrun_is_box_gone "$(cat "$err")"' ./rxrun.sh || {
        printf 'FAIL        and it is not checked first in rxrun.sh\n'; fail=1; }
fi

exit "$fail"
