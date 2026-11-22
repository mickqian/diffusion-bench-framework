#!/bin/bash
# Download the MiniMax-H3 files VideoGenX keeps on disk (FL2VA partition + root
# metadata) at the pinned revision, and build the vLLM-Omni model view.
set -u
ROOT=${VGX_ROOT:?set VGX_ROOT}
REV=42ed227ee7df40d41602854ae760620d6eb651fe
mkdir -p "$ROOT/hf" "$ROOT/logs"
export HF_HUB_CACHE=$ROOT/hf
python3 -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1
echo "start $(date -u +%FT%TZ)" > "$ROOT/logs/dl.log"
python3 - "$REV" >> "$ROOT/logs/dl.log" 2>&1 << 'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    revision=sys.argv[1],
    allow_patterns=["FL2VA/*", "FL2VA/**", "*.json", "*.md", "LICENSE*", "Ref2VA/model_index.json"],
    max_workers=16,
)
print("DONE", p, flush=True)
PY
RC=$?
SNAP=$(grep '^DONE ' "$ROOT/logs/dl.log" | tail -n 1 | cut -d' ' -f2)
if [ $RC = 0 ] && [ -d "$SNAP/FL2VA" ]; then
  echo "$SNAP" > "$ROOT/SNAPSHOT_DIR"
  rm -rf "$ROOT/model-view"
  mkdir -p "$ROOT/model-view/Ref2VA"
  for name in FL2VA model_index.json LICENSE README.md Ref2VA/model_index.json; do
    [ -e "$SNAP/$name" ] || { echo "missing $name" >> "$ROOT/logs/dl.log"; RC=3; }
    ln -s "$SNAP/$name" "$ROOT/model-view/$name"
  done
fi
echo "exit=$RC $(date -u +%FT%TZ) $(du -sh "$ROOT/hf" 2>/dev/null | cut -f1)" >> "$ROOT/logs/dl.log"
