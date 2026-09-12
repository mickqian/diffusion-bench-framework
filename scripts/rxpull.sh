#!/usr/bin/env bash
# Pull a file or directory off an rx devbox, and PROVE it arrived intact.
#
# The previous mechanism chunked base64 through `rx devbox run` and produced an
# empty file twice (md5 d41d8cd98f00b204e9800998ecf8427e -- the md5 of nothing);
# worse, the salvage path `base64 -d ... && echo ok` reported success for a
# 0-byte result, because base64 happily decodes an empty stream. A verification
# that can pass on no data is not a verification. So: every file is checked
# against a manifest built on the box, and an empty transfer is a hard failure.
#
# Transport is ssh (via `rx devbox ssh-config`), not `rx devbox run` -- a
# binary-ish stream through the run channel is what kept truncating.
#
#   scripts/rxpull.sh <box> <remote-path> <local-dest-dir> [name-glob]
#
# The optional glob restricts the pull (e.g. '*.json' to skip multi-MB server
# runlogs); without it everything under <remote-path> comes back.
#
# Exit: 0 = every file verified; 1 = mismatch/empty; 2 = usage; 75 = transport.
set -uo pipefail

BOX="${1:?usage: rxpull.sh <box> <remote-path> <local-dest-dir> [name-glob]}"
REMOTE="${2:?remote path}"
DEST="${3:?local destination directory}"
GLOB="${4:-}"
TRIES="${RXPULL_TRIES:-4}"

SSH_CFG="$(mktemp)"
trap 'rm -f "$SSH_CFG"' EXIT
if ! rx devbox ssh-config "$BOX" --dry-run > "$SSH_CFG" 2>/dev/null; then
    echo "rxpull: could not get ssh-config for $BOX" >&2
    exit 75
fi
SSH=(ssh -F "$SSH_CFG" -o ConnectTimeout=60 -o BatchMode=yes "$BOX")

parent="$(dirname "$REMOTE")"
name="$(basename "$REMOTE")"

# Manifest first: if the remote path is missing or empty, fail before transfer
# rather than "succeeding" with nothing.
# Snapshot on the box first. Pulling a directory a run is still writing to
# makes tar fail ("Truncated tar archive" -- the file grew mid-read) and would
# make the md5 manifest disagree with the bytes that arrived even when nothing
# is wrong. Copy once, then read the manifest AND the tar from that copy, so
# the two always describe the same bytes.
snap="/tmp/.rxpull-$$-$(date +%s)"
cleanup_remote() { "${SSH[@]}" "rm -rf '$snap'" >/dev/null 2>&1 || true; }
if ! "${SSH[@]}" "mkdir -p '$snap' && cp -a '$REMOTE' '$snap/'" 2>/dev/null; then
    echo "rxpull: cannot snapshot $REMOTE on $BOX (missing path?)" >&2
    exit 1
fi

find_expr="find '$name' -type f"
[[ -n "$GLOB" ]] && find_expr="$find_expr -name '$GLOB'"

manifest="$(mktemp)"
trap 'rm -f "$SSH_CFG" "$manifest"; cleanup_remote' EXIT
if ! "${SSH[@]}" "cd '$snap' && $find_expr -print0 | xargs -0 -r md5sum" > "$manifest" 2>/dev/null; then
    echo "rxpull: cannot list $REMOTE on $BOX" >&2
    exit 75
fi
want="$(wc -l < "$manifest" | tr -d ' ')"
if [[ "$want" -eq 0 ]]; then
    echo "rxpull: $REMOTE has no matching files on $BOX -- refusing to report success" >&2
    exit 1
fi
echo "rxpull: $want file(s) to fetch from $BOX:$REMOTE${GLOB:+ (glob $GLOB)}"

mkdir -p "$DEST"
# The rx-proxied ssh channel drops on large transfers ("rx: read: unexpected
# EOF" + a broken pipe), and a dropped connection says nothing about the files,
# so compress (these are highly compressible logs) and retry the stream. Extract
# without owner/permission restore: the box runs as root and this end does not.
ok=""
for ((attempt = 1; attempt <= TRIES; attempt++)); do
    if "${SSH[@]}" "cd '$snap' && $find_expr -print0 | tar --null -T - -czf -" \
        | tar -xzf - -C "$DEST" --no-same-owner --no-same-permissions 2>/dev/null; then
        ok=1
        break
    fi
    echo "rxpull: transfer attempt $attempt/$TRIES failed, retrying" >&2
    sleep $((attempt * 5))
done
if [[ -z "$ok" ]]; then
    echo "rxpull: transfer failed after $TRIES attempts" >&2
    exit 75
fi

# Verify every file against the manifest built on the box. Catches truncation
# per file, not just "the tar ran".
ok=0; bad=0
while read -r sum path; do
    local_file="$DEST/$path"
    if [[ ! -f "$local_file" ]]; then
        echo "  MISSING  $path" >&2; bad=$((bad + 1)); continue
    fi
    got="$(md5 -q "$local_file" 2>/dev/null || md5sum "$local_file" | cut -d' ' -f1)"
    if [[ "$got" != "$sum" ]]; then
        echo "  MISMATCH $path (remote $sum, local $got, $(wc -c < "$local_file" | tr -d ' ') bytes)" >&2
        bad=$((bad + 1))
    else
        ok=$((ok + 1))
    fi
done < "$manifest"

echo "rxpull: verified $ok/$want file(s) into $DEST/$name"
if [[ "$bad" -gt 0 ]]; then
    echo "rxpull: $bad file(s) did not verify" >&2
    exit 1
fi
exit 0
