#!/usr/bin/env bash
# devbox_run_cases.sh refuses a DBF_SGLANG_PYTHONPATH that does not provide sglang,
# before touching a GPU: a missing tree silently ran the installed sglang instead.
set -uo pipefail
cd "$(dirname "$0")/../.."

fail=0
tmp="$(mktemp -d)"
mkdir -p "$tmp/empty-tree"

for tree in "$tmp/no-such-tree" "$tmp/empty-tree"; do
    out="$(DBF_STATE_DIR="$tmp" DBF_SGLANG_PYTHONPATH="$tree" bash scripts/devbox_run_cases.sh t 0 40000 single_e2e case 2>&1)"
    rc=$?
    if [[ $rc -eq 2 && "$out" == *"does not provide sglang"* ]]; then
        echo "ok    refuses $(basename "$tree")"
    else
        echo "FAIL  $(basename "$tree"): rc=$rc out=${out:0:200}"
        fail=1
    fi
done
[[ -d "$tmp/logs" ]] && { echo "FAIL  created a log dir before refusing"; fail=1; } || echo "ok    nothing written before refusing"
exit $fail
