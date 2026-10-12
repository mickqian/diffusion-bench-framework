#!/usr/bin/env bash
# LightX2V throughput on minimax_h3_t2va_5s with the torch_sdpa SP2 profile (the published cell ran FA2 TP2).
set -u
W=/scratch/r; cd $W
set -a; . ~/.config/radixark/secrets.env; set +a
export DBF_STATE_DIR=$W DBF_REPO_DIR=$W/dbf DBF_LOG_DIR=$W/logs DBF_VENV_ROOT=$W/fw-venvs DBF_HF_HOME=$W/hf \
       DBF_HARDWARE_PROFILE=gb300 DBF_CASE_TIMEOUT=7200 DBF_FRAMEWORKS=lightx2v
bash $W/dbf/scripts/devbox_run_cases.sh rf4lxtp 0,1 62201 throughput minimax_h3_t2va_5s
python3 - $W/logs/run_rf4lxtp_minimax_h3_t2va_5s.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for x in d.get("throughput_results", []):
    m = x["metrics"]
    print(f"RF4 {x['framework']} qps={m.get('throughput_rps')} done={m.get('completed_requests')}/{m.get('num_requests')} p50={m.get('latency_p50_s')} p95={m.get('latency_p95_s')} err={str(x.get('error'))[:120]}")
PY
echo "=== RF4_DONE $(date -Is)"
