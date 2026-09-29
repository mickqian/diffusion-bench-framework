#!/bin/bash
# Validate the split sglang PRs end to end, each on its own branch tip, with
# the commands that ran out of memory on c7be3e9, then put the measured tree
# back where it was.
set -u
C=/persistent/lx2v/run_cell2.sh
H=/persistent/lx2v/config_hint_probe.json
MAIN=/sgl-workspace/sglang
BASE=c7be3e935b5034006cd6ae7977b41e2459b4126c
A=${A_SHA:?}; B=${B_SHA:?}
ALL=0,1,2,3,4,5,6,7
switch_tree() {
    git -C "$MAIN" checkout -q --detach "$1" || exit 1
    find "$MAIN/python" -name "*.pyc" -delete
    local st="$MAIN/python/sglang/multimodal_gen/runtime/server_args"
    echo "sglang tree now $(git -C "$MAIN" rev-parse --short HEAD):" \
        "auto-dit=$(grep -c _measure_dit_overflow "$st/auto_tune.py")" \
        "fsdp-default=$(grep -c 'FSDP shards only resident components' "$st/server_args.py")"
}
echo "queue start $(date -u +%FT%TZ)"
switch_tree "$A"
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" split_a_1gpu_default 0 31811 sglang wan21_i2v_14b_480p_1gpu
DBF_REPEATS=1 DBF_CONFIG="$H" bash "$C" split_a_8gpu_default "$ALL" 31821 sglang wan21_i2v_14b_480p_8gpu
switch_tree "$B"
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--use-fsdp-inference true" \
    bash "$C" split_b_8gpu_fsdp "$ALL" 31831 sglang wan21_i2v_14b_480p_8gpu
switch_tree "$BASE"
echo "VALIDATE_SPLIT_DONE $(date -u +%FT%TZ)"
