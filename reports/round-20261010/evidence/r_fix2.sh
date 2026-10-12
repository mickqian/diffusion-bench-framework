#!/usr/bin/env bash
# Whole-node 4xGB300, runner cells: sglang H3 calibration, LightX2V torch_sdpa SP cells, sglang FLUX.1 compiled.
set -u
W=/scratch/r; cd $W
set -a; . ~/.config/radixark/secrets.env; set +a
git -C dbf fetch -q origin main && git -C dbf checkout -q --detach origin/main
echo "RF2 dbf $(git -C dbf log --oneline -1 | cut -c1-70)"
export DBF_STATE_DIR=$W DBF_REPO_DIR=$W/dbf DBF_LOG_DIR=$W/logs DBF_VENV_ROOT=$W/fw-venvs DBF_HF_HOME=$W/hf \
       DBF_HARDWARE_PROFILE=gb300 DBF_CASE_TIMEOUT=7200
R=$W/dbf/scripts/devbox_run_cases.sh
summ() {
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    for x in d["results"] + d.get("throughput_results", []):
        m = x.get("metrics") or {}
        print(f"RF2 {sys.argv[2]} {x['case_id']} {x['framework']} {x.get('mode')} median={x.get('latency_s')} qps={m.get('throughput_rps')} samples={m.get('latency_samples_s')} err={str(x.get('error'))[:160]}")
except Exception as e:
    print(f"RF2 {sys.argv[2]} no result {sys.argv[1]} ({type(e).__name__})")
PY
}
DBF_FRAMEWORKS=sglang DBF_SGLANG_PYTHONPATH=$W/sgl-main/python bash $R rf2sglh3 0,1 62001 single_e2e minimax_h3_t2va_5s
summ $W/logs/run_rf2sglh3_minimax_h3_t2va_5s.json calibration
DBF_FRAMEWORKS=lightx2v bash $R rf2lx2 0,1 62011 single_e2e minimax_h3_t2va_5s minimax_h3_ref2va_5s
summ $W/logs/run_rf2lx2_minimax_h3_t2va_5s.json lightx2v
summ $W/logs/run_rf2lx2_minimax_h3_ref2va_5s.json lightx2v
DBF_FRAMEWORKS=lightx2v bash $R rf2lx4 0,1,2,3 62021 single_e2e minimax_h3_t2va_5s_4gpu
summ $W/logs/run_rf2lx4_minimax_h3_t2va_5s_4gpu.json lightx2v
grep -ahoE '"attn_type": "[a-z_0-9]+"|"tensor_p_size": [0-9]|"seq_p_size": [0-9]' /tmp/lightx2v_config_minimax_h3_*.json | sort | uniq -c | sed 's/^/RF2 lightx2v cfg /'
DBF_FRAMEWORKS=sglang DBF_SGLANG_PYTHONPATH=$W/sgl-main/python bash $R rf2sglf1 0,1 62031 "single_e2e throughput" flux1_dev_t2i_1024
summ $W/logs/run_rf2sglf1_flux1_dev_t2i_1024.json flux1
echo "=== RF2_DONE $(date -Is)"
