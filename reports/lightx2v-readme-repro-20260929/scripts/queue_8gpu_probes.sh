#!/bin/bash
# Probe the 8-GPU cell's candidate commands one after another, after the 1-GPU
# LightX2V probe has released its card. Probes pick the command that goes into
# the profile; they are not the published measurement.
set -u
ALL=0,1,2,3,4,5,6,7
C=/persistent/lx2v/run_cell.sh
while ! grep -q '^EXIT=' /persistent/logs/cells/probe_lx2v_1gpu.runlog 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"

DBF_REPEATS=1 bash "$C" probe_sgl_8gpu_auto "$ALL" 31201 sglang wan21_i2v_14b_480p_8gpu

DBF_REPEATS=1 \
DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--layerwise-offload-components dit,text_encoder,image_encoder,vae" \
    bash "$C" probe_sgl_8gpu_lw "$ALL" 31202 sglang wan21_i2v_14b_480p_8gpu

DBF_REPEATS=1 DBF_LX2V_PROFILE=2 bash "$C" probe_lx2v_8gpu "$ALL" 31301 lightx2v wan21_i2v_14b_480p_8gpu

echo "QUEUE_DONE $(date -u +%FT%TZ)"
