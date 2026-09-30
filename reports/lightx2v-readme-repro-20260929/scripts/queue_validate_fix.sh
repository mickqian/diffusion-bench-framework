#!/bin/bash
# After the last published cell: re-run the legacy --dit-layerwise-offload probe
# alone (its first run lost a port race to a neighbour), then validate the sglang
# fix end to end with the commands that used to OOM, and put the measured tree
# back where it was.
set -u
C=/persistent/lx2v/run_cell2.sh
H=/persistent/lx2v/config_hint_probe.json
MAIN=/sgl-workspace/sglang
BASE=c7be3e935b5034006cd6ae7977b41e2459b4126c
FIX=${FIX_SHA:?fix sha}
ALL=0,1,2,3,4,5,6,7
while ! grep -q '^QUEUE_BA_DONE' /persistent/logs/queue_1gpu_ba.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"

DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--dit-layerwise-offload true" \
    bash "$C" probe_sgl_1gpu_legacy2 0 31601 sglang wan21_i2v_14b_480p_1gpu

git -C "$MAIN" checkout -q --detach "$FIX" && echo "sglang tree now $(git -C "$MAIN" rev-parse HEAD)"
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" val_sgl_1gpu_default 0 31611 sglang wan21_i2v_14b_480p_1gpu
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" val_sgl_8gpu_default "$ALL" 31621 sglang wan21_i2v_14b_480p_8gpu
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--use-fsdp-inference true" \
    bash "$C" val_sgl_8gpu_fsdp "$ALL" 31631 sglang wan21_i2v_14b_480p_8gpu

git -C "$MAIN" checkout -q --detach "$BASE" && echo "sglang tree restored $(git -C "$MAIN" rev-parse HEAD)"
echo "VALIDATE_DONE $(date -u +%FT%TZ)"
