#!/usr/bin/env bash
# A dying job must not free somebody else's lock.
#
# The release was an unconditional `rm -rf "$GPU_LOCK_DIR"` in an EXIT trap.
# Killing a job therefore deleted whatever lock directory existed at that
# moment -- including the one its SUCCESSOR had already taken. The next starter
# found no lock and walked in, and two jobs ran on the same GPUs while the lock
# file named only one of them. One silently contaminated the other's numbers.
#
# Ownership-checked release is the fix, and it is the property worth pinning:
# mutual exclusion that survives a kill is the only kind that is any use on a
# box where jobs get killed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
HELPER="$PWD/scripts/gpu_job_lock.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export GPU_LOCK_DIR="$TMP/lock"
export GPU_LOCK_POLL=1

cat > "$TMP/job.sh" <<EOF
source "$HELPER"
gpu_lock_acquire
echo "\$1 holding"
sleep "\$2"
echo "\$1 done"
EOF

fail=0
check() { # check <name> <condition-result> [detail]
    if [ "$2" = "0" ]; then echo "  ok   $1"; else echo "  FAIL $1${3:+ -- $3}"; fail=1; fi
}

# --- two contenders serialise -------------------------------------------
bash "$TMP/job.sh" A 3 > "$TMP/a.out" 2>&1 &
A=$!
sleep 1
bash "$TMP/job.sh" B 1 > "$TMP/b.out" 2>&1 &
B=$!
wait $A $B 2>/dev/null
grep -q "A holding" "$TMP/a.out"; check "the first holder runs" $?
grep -q "waiting for" "$TMP/b.out"; check "the second waits" $?
grep -q "B done" "$TMP/b.out"; check "the second then runs" $?
[ ! -d "$GPU_LOCK_DIR" ]; check "the lock is released on exit" $?

# --- a killed holder does not take the successor's lock with it ---------
bash "$TMP/job.sh" C 30 > "$TMP/c.out" 2>&1 &
C=$!
sleep 1
bash "$TMP/job.sh" D 4 > "$TMP/d.out" 2>&1 &
D=$!
sleep 1
kill -9 $C 2>/dev/null      # C dies while D is queued behind it
wait $C 2>/dev/null
sleep 3                      # D should now own it
owner="$(cat "$GPU_LOCK_DIR/pid" 2>/dev/null || echo none)"
[ "$owner" = "$D" ]; check "the successor owns the lock after a kill" $? "owner=$owner want=$D"

bash "$TMP/job.sh" E 1 > "$TMP/e.out" 2>&1 &
E=$!
sleep 1
grep -q "waiting for" "$TMP/e.out"; check "a later job still has to wait" $? "$(head -1 "$TMP/e.out" 2>/dev/null)"
wait $D $E 2>/dev/null
[ ! -d "$GPU_LOCK_DIR" ]; check "the lock ends free" $?

# --- a lock whose owner is gone is reclaimed, not waited on forever -----
mkdir -p "$GPU_LOCK_DIR"
echo 999999 > "$GPU_LOCK_DIR/pid"
echo ghost   > "$GPU_LOCK_DIR/job"
# no `timeout` on macOS, so guard the hang with a watchdog instead of relying
# on a coreutils binary that is not there (it silently ran nothing here, and
# both greps failed as if the lock were broken)
bash "$TMP/job.sh" F 1 > "$TMP/f.out" 2>&1 &
F=$!
{ sleep 20; kill -9 $F 2>/dev/null; } 2>/dev/null & WATCHDOG=$!
disown $WATCHDOG 2>/dev/null || true
wait $F 2>/dev/null; rc=$?
kill $WATCHDOG 2>/dev/null
[ "$rc" = 0 ]; check "a job does not hang on a stale lock" $? "exit=$rc"
grep -q "stale" "$TMP/f.out"; check "a stale lock is cleared" $? "$(head -1 "$TMP/f.out" 2>/dev/null)"
grep -q "F done" "$TMP/f.out"; check "and the job proceeds" $?

exit $fail
