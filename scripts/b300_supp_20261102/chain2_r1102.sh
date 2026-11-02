#!/usr/bin/env bash
# r1102 chain 2: after wave 2 (H3) -> old-pin vLLM-Omni venv -> wave 3a (Qwen, old pin vs main) -> wave 3b (H3 extras).
set -u
STATE=/workspace/r1102
until grep -q "chain: wave 2 done" ${STATE}/logs-ab/chain.log 2>/dev/null; do sleep 30; done
echo "chain2: wave 2 done seen $(date -Is)"
sleep 65
bash ${STATE}/scripts/install_vllm_omni_oldpin.sh > ${STATE}/logs-ab/install_oldpin.log 2>&1 < /dev/null
echo "chain2: oldpin install done $(date -Is) $(grep -E '^I(_DONE| install)' ${STATE}/logs-ab/install_oldpin.log | tr '\n' ' ')"
grep -q "^I_DONE rc=0" ${STATE}/logs-ab/install_oldpin.log || { echo "chain2: old-pin install failed, skipping wave 3a"; SKIP3A=1; }
sleep 65
if [ -z "${SKIP3A:-}" ]; then
  SUPP_TAG=w3q SUPP_PLAN=${STATE}/scripts/plan_w3q_oldpin.txt SUPP_PORT=61500 bash ${STATE}/scripts/run_supp_ab.sh > ${STATE}/logs-ab/w3q.log 2>&1 < /dev/null
  echo "chain2: wave 3a done $(date -Is)"
  sleep 65
fi
SUPP_TAG=w3h SUPP_PLAN=${STATE}/scripts/plan_w3h_extra.txt SUPP_PORT=61700 bash ${STATE}/scripts/run_supp_ab.sh > ${STATE}/logs-ab/w3h.log 2>&1 < /dev/null
echo "chain2: wave 3b done $(date -Is)"
