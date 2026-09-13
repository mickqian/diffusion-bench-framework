#!/usr/bin/env bash
# Refuse to queue long GPU work onto a devbox that will expire under it.
#
#   source scripts/rx_ttl_guard.sh
#   rx_require_ttl bench-0912 180 "the 15-case profile-vs-default queue"
#
# A 4xB200 box was acquired with a 24h TTL and auto-released a day later, at
# 14:02, with five jobs queued behind a lock:
#
#   release_requested · auto:ttl_expired (util=0.0%, expires_at 14:01:48Z)
#
# Everything still queued died with it. Nothing warned; `rx devbox list` shows an
# EXPIRES column and I never read it. There is also no `rx devbox extend` -- the
# subcommands are list/status/info/acquire/release/reprovision/exec/run/code/
# ssh-config/sync/forward/keepalive/secrets/events/why/diagnose/util, and
# `keepalive` is about the CPU-busy bar on unattended EC2 agent boxes, not TTL.
# So the TTL is fixed at acquire time and a preflight check is the only defence.
set -u

# Remaining minutes on <box>, or nothing when it cannot be determined.
rx_ttl_minutes() {
    local box="$1"
    rx devbox list --json 2>/dev/null | python3 -c '
import datetime, json, sys

name = sys.argv[1]
try:
    boxes = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for b in boxes:
    if b.get("name") != name and b.get("id") != name:
        continue
    expires = b.get("expires_at")
    if not expires:
        sys.exit(0)
    expires = expires.replace("Z", "+00:00")
    try:
        when = datetime.datetime.fromisoformat(expires)
    except ValueError:
        sys.exit(0)
    now = datetime.datetime.now(datetime.timezone.utc)
    print(int((when - now).total_seconds() // 60))
    break
' "$box"
}

# Fail unless <box> has at least <minutes> left.
rx_require_ttl() {
    local box="$1" need="$2" what="${3:-this job}"
    local left
    left="$(rx_ttl_minutes "$box")"
    if [ -z "${left}" ]; then
        echo "rx-ttl: cannot read an expiry for ${box} -- check 'rx devbox list'" >&2
        return 1
    fi
    if [ "${left}" -lt "${need}" ]; then
        echo "rx-ttl: ${box} expires in ${left} min, and ${what} needs ${need}." >&2
        echo "rx-ttl: the TTL cannot be extended, so start this on a box acquired" >&2
        echo "rx-ttl: with a long enough --ttl instead of losing the run midway." >&2
        return 1
    fi
    echo "rx-ttl: ${box} has ${left} min left, ${what} needs ${need} -- ok"
    return 0
}
