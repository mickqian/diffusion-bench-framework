#!/bin/bash
# Is sglang's longer I2V VAE-encode stage its layerwise placement or its FP32
# VAE? Same command as the published cells with the VAE held resident.
set -u
C=/persistent/lx2v/run_cell2.sh
H=/persistent/lx2v/config_hint_probe.json
ARGS="--layerwise-offload-components dit,text_encoder,image_encoder --component-residency vae=resident"
echo "queue start $(date -u +%FT%TZ)"
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="$ARGS" \
    bash "$C" probe_sgl_1gpu_vae_resident 0 31701 sglang wan21_i2v_14b_480p_1gpu
DBF_REPEATS=1 DBF_CONFIG="$H" DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="$ARGS" \
    bash "$C" probe_sgl_8gpu_vae_resident 0,1,2,3,4,5,6,7 31711 sglang wan21_i2v_14b_480p_8gpu
echo "QUEUE_VAE_DONE $(date -u +%FT%TZ)"
