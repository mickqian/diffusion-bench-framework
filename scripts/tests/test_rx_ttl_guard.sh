#!/usr/bin/env bash
# Long GPU work must not be queued onto a box that expires under it.
#
# A 4xB200 was acquired with a 24h TTL and auto-released a day later with five
# jobs queued behind the GPU lock; all of them died. `rx devbox list` had been
# showing an EXPIRES column the whole time. There is no `rx devbox extend`, so a
# preflight check is the only defence, and it is worth a test because it will be
# exercised exactly once per box -- at the moment it matters.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ./scripts/rx_ttl_guard.sh

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
check() { # check <name> <condition-result> [detail]
    if [ "$2" = "0" ]; then echo "  ok   $1"; else echo "  FAIL $1${3:+ -- $3}"; fail=1; fi
}

# A stub `rx` that reports whatever expiry the test asks for.
mkfake() { # mkfake <json>
    cat > "$TMP/rx" <<EOF
#!/usr/bin/env bash
cat <<'JSON'
$1
JSON
EOF
    chmod +x "$TMP/rx"
}
export PATH="$TMP:$PATH"

far="$(python3 -c 'import datetime;print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=6)).isoformat().replace("+00:00","Z"))')"
near="$(python3 -c 'import datetime;print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=20)).isoformat().replace("+00:00","Z"))')"

mkfake "[{\"name\": \"bench-0912\", \"expires_at\": \"$far\"}]"
out="$(rx_require_ttl bench-0912 180 "the queue" 2>&1)"; rc=$?
[ "$rc" = 0 ]; check "a box with hours left passes" $? "$out"

mkfake "[{\"name\": \"bench-0912\", \"expires_at\": \"$near\"}]"
out="$(rx_require_ttl bench-0912 180 "the queue" 2>&1)"; rc=$?
[ "$rc" != 0 ]; check "a box expiring under the job is refused" $? "rc=$rc"
grep -q "expires in" <<<"$out"; check "and says how long is left" $? "$out"
grep -q "cannot be extended" <<<"$out"; check "and that waiting will not help" $?

# The real failure mode was silence, so an unreadable expiry must not read as ok.
mkfake '[{"name": "someone-elses-box", "expires_at": "2099-01-01T00:00:00Z"}]'
out="$(rx_require_ttl bench-0912 180 2>&1)"; rc=$?
[ "$rc" != 0 ]; check "an absent box is refused, not assumed fine" $? "rc=$rc"

mkfake '[{"name": "bench-0912"}]'
out="$(rx_require_ttl bench-0912 180 2>&1)"; rc=$?
[ "$rc" != 0 ]; check "a box with no expiry field is refused" $? "rc=$rc"

mkfake 'not json at all'
out="$(rx_require_ttl bench-0912 180 2>&1)"; rc=$?
[ "$rc" != 0 ]; check "unparseable output is refused" $? "rc=$rc"

exit $fail
