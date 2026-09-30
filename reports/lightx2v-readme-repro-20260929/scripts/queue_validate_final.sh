#!/bin/bash
# Re-validate the final auto-DiT branch tip end to end after review changes,
# then put the measured tree back where it was.
set -u
C=/persistent/lx2v/run_cell2.sh
H=/persistent/lx2v/config_hint_probe.json
MAIN=/sgl-workspace/sglang
BASE=c7be3e935b5034006cd6ae7977b41e2459b4126c
A=${A_SHA:?}
switch_tree() {
    git -C "$MAIN" checkout -q --detach "$1" || exit 1
    find "$MAIN/python" -name "*.pyc" -delete
    echo "sglang tree now $(git -C "$MAIN" rev-parse --short HEAD):" \
        "auto-dit=$(grep -c _dit_fit_checked "$MAIN/python/sglang/multimodal_gen/runtime/server_args/auto_tune.py")"
}
echo "queue start $(date -u +%FT%TZ)"
switch_tree "$A"
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" final_a_1gpu_default 0 31911 sglang wan21_i2v_14b_480p_1gpu
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" final_a_8gpu_default 0,1,2,3,4,5,6,7 31921 sglang wan21_i2v_14b_480p_8gpu
switch_tree "$BASE"
echo "VALIDATE_FINAL_DONE $(date -u +%FT%TZ)"
