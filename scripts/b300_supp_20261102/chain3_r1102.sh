#!/usr/bin/env bash
# r1102 chain 3: after chain 2 (wave 3b) -> vLLM-Omni bisect of the Qwen-Image-2.1 slowdown.
set -u
STATE=/workspace/r1102
until grep -q "chain2: wave 3b done" ${STATE}/logs-ab/chain2.log 2>/dev/null; do sleep 30; done
echo "chain3: chain2 done seen $(date -Is)"
sleep 65
bash ${STATE}/scripts/bisect_vllm_omni.sh > ${STATE}/logs-ab/bisect_driver.log 2>&1 < /dev/null
echo "chain3: bisect done $(date -Is)"
