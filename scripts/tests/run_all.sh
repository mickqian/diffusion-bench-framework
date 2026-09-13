#!/usr/bin/env bash
# Run every test in this directory, with an interpreter that can import the harness.
#
# The tests that import `run_comparison` need `requests`, which the system
# python3 does not have. Run them with it and you get a bare ImportError per
# file -- which reads exactly like three tests failing, and cost a cycle of
# chasing a regression that was not there. So: pick the venv, and say plainly
# when the reason for a failure is the interpreter rather than the code.
#
#   scripts/tests/run_all.sh            # repo .venv, else $VIRTUAL_ENV, else python3
#   DBF_TEST_PYTHON=/path/to/python scripts/tests/run_all.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

py="${DBF_TEST_PYTHON:-}"
if [ -z "$py" ]; then
    for cand in .venv/bin/python3 "${VIRTUAL_ENV:-/nonexistent}/bin/python3" "$(command -v python3)"; do
        [ -x "$cand" ] || continue
        if "$cand" -c 'import requests' >/dev/null 2>&1; then py="$cand"; break; fi
        [ -n "$py" ] || py="$cand"   # remember the first one, as a fallback
    done
fi
if ! "$py" -c 'import requests' >/dev/null 2>&1; then
    echo "WARNING: $py cannot import requests -- the tests that import the harness"
    echo "         will fail on the import, not on anything they test."
    echo "         Create the venv (python3 -m venv .venv && .venv/bin/pip install -e .)"
    echo "         or set DBF_TEST_PYTHON."
fi
echo "python: $py"

fail=0 ran=0
for t in scripts/tests/test_*.py scripts/tests/test_*.sh; do
    [ -e "$t" ] || continue
    ran=$((ran + 1))
    name=$(basename "$t")
    if [ "${t##*.}" = "py" ]; then out=$("$py" "$t" 2>&1); rc=$?
    else out=$(bash "$t" 2>&1); rc=$?; fi
    if [ "$rc" -ne 0 ]; then
        fail=$((fail + 1))
        echo "FAIL $name"
        printf '%s\n' "$out" | sed 's/^/     /'
    else
        echo "ok   $name"
    fi
done
echo "---- $((ran - fail))/$ran passed ----"
[ "$fail" -eq 0 ]
