#!/bin/bash
# Supplementary LightX2V 8-GPU cells after the published queue: its own timers
# off (pipeline.py defaults PROFILING_DEBUG_LEVEL to 2, and the timers
# synchronise), and its documented compile path. Then the ceiling probe on an
# idle card.
set -u
C=/persistent/lx2v/run_cell2.sh
ALL=0,1,2,3,4,5,6,7
while ! grep -q '^FINAL_DONE' /persistent/logs/queue_final.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"

DBF_REPEATS=3 DBF_LX2V_PROFILE=0 \
    bash "$C" supp_lx2v_8gpu_noprof "$ALL" 31421 lightx2v wan21_i2v_14b_480p_8gpu

DBF_REPEATS=3 DBF_DISABLE_COMPILE=0 DBF_CONFIG=/persistent/lx2v/config_lx2v_compile.json \
    bash "$C" supp_lx2v_8gpu_compile "$ALL" 31422 lightx2v wan21_i2v_14b_480p_8gpu

T=/persistent/logs/cells/tflops_probe.log
{
    echo "== $(date -u +%FT%TZ) GPU 0, sglang env =="
    CUDA_VISIBLE_DEVICES=0 python3 /persistent/lx2v/tflops_probe.py
    echo "== lightx2v venv =="
    CUDA_VISIBLE_DEVICES=0 /persistent/fw-venvs/lightx2v/bin/python /persistent/lx2v/tflops_probe.py
    nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv -i 0
} > "$T" 2>&1

echo "SUPP_DONE $(date -u +%FT%TZ)"
