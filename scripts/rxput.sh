#!/usr/bin/env bash
# Copy a local file TO a devbox through the rx exec channel, in chunks.
#
# The obvious one-liner -- `rxrun.sh box "echo <base64> | base64 -d > file"` --
# works until the file grows past roughly 100 KB, and then the box answers
# `exec /usr/bin/bash: argument list too long`. That reads like an rx fault and
# is really ARG_MAX on the remote `bash -c`. It bit a 95 KB patch mid-run.
#
# So: append fixed-size chunks, then compare md5 on both sides. Without the
# checksum a truncated chunk is indistinguishable from a delivered file, and the
# failure surfaces much later as a patch that will not apply.
#
#   scripts/rxput.sh <box> <local-file> <remote-path> [chunk-chars]
#
# Companion to rxpull.sh (the other direction) and rxrun.sh (the transport).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
box="${1:?usage: rxput.sh <box> <local-file> <remote-path> [chunk-chars]}"
src="${2:?local file}"
dst="${3:?remote path}"
chunk="${4:-45000}"

[ -r "$src" ] || { echo "rxput: cannot read $src" >&2; exit 1; }

# md5 on macOS, md5sum on Linux -- this runs from either.
local_md5() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi
}

b64="$(base64 < "$src" | tr -d '\n')"
want="$(local_md5 "$src")"
total=${#b64}

bash "$REPO/scripts/rxrun.sh" "$box" "mkdir -p \"\$(dirname '$dst')\"; : > '$dst.b64'" >/dev/null || exit 1

off=0
parts=0
while [ "$off" -lt "$total" ]; do
    bash "$REPO/scripts/rxrun.sh" "$box" "printf '%s' '${b64:$off:$chunk}' >> '$dst.b64'" >/dev/null || exit 1
    off=$((off + chunk))
    parts=$((parts + 1))
done

got="$(bash "$REPO/scripts/rxrun.sh" "$box" \
    "base64 -d < '$dst.b64' > '$dst' && md5sum '$dst' | cut -d' ' -f1" | tr -d '\r\n ')"

if [ "$got" != "$want" ]; then
    echo "rxput: md5 MISMATCH for $box:$dst -- want $want, got ${got:-<none>}" >&2
    echo "rxput: the staged chunks are left at $dst.b64 for inspection." >&2
    exit 1
fi
# The staging file is left in place on purpose: the next push truncates it with
# `: >`, and deleting things on a box is not this script's business.
echo "  rxput: $box:$dst ok ($parts chunk(s), md5 $want)"
