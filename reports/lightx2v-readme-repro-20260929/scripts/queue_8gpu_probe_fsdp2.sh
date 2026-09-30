#!/bin/bash
# The first FSDP probe kept dit_cpu_offload=true, so the whole DiT was still
# moved onto each card per request and OOMed like the zero-flag default. Probe
# FSDP with component offload off, in the gap after the 1-GPU finals, while the
# final queue holds its 8-GPU cells for the profile decision.
set -u
while ! grep -q '^RESULT final_lx2v_1gpu_B' /persistent/logs/queue_final.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"
DBF_REPEATS=1 DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--use-fsdp-inference true --dit-cpu-offload false" \
    bash /persistent/lx2v/run_cell.sh probe_sgl_8gpu_fsdp2 0,1,2,3,4,5,6,7 31204 sglang wan21_i2v_14b_480p_8gpu
echo "QUEUE3_DONE $(date -u +%FT%TZ)"
