#!/bin/bash
# Try the remedies from sglang's own OOM hint that had not been tried alone,
# each on top of the zero-flag command: --performance-mode memory (1 and 8 GPUs)
# and the legacy --dit-layerwise-offload (1 GPU). Runs after the FSDP probe,
# while the final queue holds its 8-GPU cells for the profile decision.
set -u
C=/persistent/lx2v/run_cell2.sh
H=/persistent/lx2v/config_hint_probe.json
while ! grep -q '^QUEUE3_DONE' /persistent/logs/queue_8gpu_probe_fsdp2.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"

# Two single-card probes side by side: they answer "does it fit", not speed.
# Ports stay apart because an sglang diffusion server also binds port + 1.
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--performance-mode memory" \
    bash "$C" probe_sgl_1gpu_memmode 0 31501 sglang wan21_i2v_14b_480p_1gpu &
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--dit-layerwise-offload true" \
    bash "$C" probe_sgl_1gpu_legacy 1 31511 sglang wan21_i2v_14b_480p_1gpu &
wait

DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--performance-mode memory" \
    bash "$C" probe_sgl_8gpu_memmode 0,1,2,3,4,5,6,7 31503 sglang wan21_i2v_14b_480p_8gpu

echo "QUEUE_HINT_DONE $(date -u +%FT%TZ)"
