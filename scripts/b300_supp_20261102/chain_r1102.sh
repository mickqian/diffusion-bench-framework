#!/usr/bin/env bash
# r1102 chain: wave 1 (Qwen harness arms, already launched) -> pixel-md5 probe -> wave 2 (H3 throughput).
# Each step takes the GPU lock itself; this script only sequences them and never touches the cards.
set -u
STATE=/workspace/r1102
until grep -q "=== SUPP DONE .* tag=w1q" ${STATE}/logs-ab/w1q.log 2>/dev/null; do sleep 20; done
echo "chain: wave 1 done $(date -Is)"
sleep 65   # lock poll is 60 s in other waiters; nothing else waits here, but keep the gap anyway
bash ${STATE}/scripts/run_px.sh > ${STATE}/logs-ab/px.log 2>&1 < /dev/null
echo "chain: px done $(date -Is)"
sleep 65
SUPP_TAG=w2h SUPP_PLAN=${STATE}/scripts/plan_w2_h3.txt SUPP_PORT=61300 bash ${STATE}/scripts/run_supp_ab.sh > ${STATE}/logs-ab/w2h.log 2>&1 < /dev/null
echo "chain: wave 2 done $(date -Is)"
