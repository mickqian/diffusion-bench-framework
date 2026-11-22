#!/bin/bash
# Run arms one after another on an otherwise idle node.
#   chain.sh <name> <arms-file>     arms-file lines: tag engine tree ngpu nreq
set -u
ROOT=${VGX_ROOT:?set VGX_ROOT}
NAME=$1
ARMS=$2
ST=$ROOT/logs/chain_$NAME.status
mkdir -p "$ROOT/logs"
exec 9> "$ROOT/.gpu.lock"
flock 9
echo "$(date -u +%FT%TZ) CHAIN_START $NAME pid=$$" >> "$ST"
while read -r tag engine tree ngpu nreq; do
  case "$tag" in '' | \#*) continue ;; esac
  echo "$(date -u +%FT%TZ) ARM_START $tag" >> "$ST"
  bash "$ROOT/bin/arm.sh" "$tag" "$engine" "$tree" "$ngpu" "$nreq" > "$ROOT/logs/arm_$tag.out" 2>&1
  echo "$(date -u +%FT%TZ) ARM_END $tag rc=$? $(tail -n 1 "$ROOT/logs/client_$tag.log" 2>/dev/null | cut -c1-80)" >> "$ST"
done < "$ARMS"
echo "$(date -u +%FT%TZ) CHAIN_DONE $NAME" >> "$ST"
