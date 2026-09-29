#!/bin/bash
# The published measurement: every cell runs through the profile in config.json,
# one at a time on an otherwise idle box. 1 GPU as A then B on the same card;
# 8 GPUs interleaved ABAB. The 8-GPU cells wait for the marker written once the
# probes have settled which sglang command goes into that profile.
set -u
C=/persistent/lx2v/run_cell.sh
ALL=0,1,2,3,4,5,6,7
wait_for_line() { while ! grep -q "^$2" "$1" 2>/dev/null; do sleep 20; done; }

wait_for_line /persistent/logs/queue_8gpu_probe_fsdp.log QUEUE2_DONE
echo "queue start $(date -u +%FT%TZ)"

DBF_REPEATS=2 bash "$C" final_sgl_1gpu_A 0 31401 sglang wan21_i2v_14b_480p_1gpu
DBF_REPEATS=2 bash "$C" final_lx2v_1gpu_B 0 31402 lightx2v wan21_i2v_14b_480p_1gpu

while [ ! -f /persistent/lx2v/8gpu_profile_decided ]; do sleep 20; done
echo "8gpu profile: $(cat /persistent/lx2v/8gpu_profile_decided)"

DBF_REPEATS=3 bash "$C" final_sgl_8gpu_A1 "$ALL" 31411 sglang wan21_i2v_14b_480p_8gpu
DBF_REPEATS=3 bash "$C" final_lx2v_8gpu_B1 "$ALL" 31412 lightx2v wan21_i2v_14b_480p_8gpu
DBF_REPEATS=3 bash "$C" final_sgl_8gpu_A2 "$ALL" 31413 sglang wan21_i2v_14b_480p_8gpu
DBF_REPEATS=3 bash "$C" final_lx2v_8gpu_B2 "$ALL" 31414 lightx2v wan21_i2v_14b_480p_8gpu

echo "FINAL_DONE $(date -u +%FT%TZ)"
