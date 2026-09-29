#!/bin/bash
# The 1-GPU A/B came out within 5%, which this repo's rules do not let a single
# ordering settle: run B then A on the same card to complete an ABBA pair.
set -u
C=/persistent/lx2v/run_cell.sh
while ! grep -q '^SUPP_DONE' /persistent/logs/queue_supp.log 2>/dev/null; do sleep 20; done
echo "queue start $(date -u +%FT%TZ)"
DBF_REPEATS=2 bash "$C" final_lx2v_1gpu_B2 0 31431 lightx2v wan21_i2v_14b_480p_1gpu
DBF_REPEATS=2 bash "$C" final_sgl_1gpu_A2 0 31432 sglang wan21_i2v_14b_480p_1gpu
echo "QUEUE_BA_DONE $(date -u +%FT%TZ)"
