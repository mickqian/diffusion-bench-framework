#!/bin/bash
# SGLang's own OOM hint recommends FSDP for multi-GPU; probe it as the third
# sglang candidate for the 8-GPU cell, after the first probe queue drains.
set -u
while ! grep -q '^QUEUE_DONE' /persistent/logs/queue_8gpu_probes.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"
DBF_REPEATS=1 DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--use-fsdp-inference true" \
    bash /persistent/lx2v/run_cell.sh probe_sgl_8gpu_fsdp 0,1,2,3,4,5,6,7 31203 sglang wan21_i2v_14b_480p_8gpu
echo "QUEUE2_DONE $(date -u +%FT%TZ)"
