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

exit "$fail"
