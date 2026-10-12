#!/usr/bin/env bash
# Whole-node 4xGB300: minimax_h3_t2va_5s_4gpu, sglang main 6576e50 vs LightX2V (torch_sdpa SP4), interleaved twice.
set -u
W=/scratch/r; cd $W
set -a; . ~/.config/radixark/secrets.env; set +a
export DBF_STATE_DIR=$W DBF_REPO_DIR=$W/dbf DBF_LOG_DIR=$W/logs DBF_VENV_ROOT=$W/fw-venvs DBF_HF_HOME=$W/hf \
       DBF_HARDWARE_PROFILE=gb300 DBF_CASE_TIMEOUT=7200
R=$W/dbf/scripts/devbox_run_cases.sh
summ() {
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    for x in json.load(open(sys.argv[1]))["results"]:
        m = x.get("metrics") or {}
        print(f"RF3 {sys.argv[2]} {x['framework']} median={x.get('latency_s')} samples={m.get('latency_samples_s')} err={str(x.get('error'))[:160]}")
except Exception as e:
    print(f"RF3 {sys.argv[2]} no result ({type(e).__name__})")
PY
}
for r in 1 2; do
  DBF_FRAMEWORKS=sglang DBF_SGLANG_PYTHONPATH=$W/sgl-main/python bash $R rf3sgl$r 0,1,2,3 $((62100 + r * 10)) single_e2e minimax_h3_t2va_5s_4gpu
  summ $W/logs/run_rf3sgl${r}_minimax_h3_t2va_5s_4gpu.json r$r
  DBF_FRAMEWORKS=lightx2v bash $R rf3lx$r 0,1,2,3 $((62105 + r * 10)) single_e2e minimax_h3_t2va_5s_4gpu
  summ $W/logs/run_rf3lx${r}_minimax_h3_t2va_5s_4gpu.json r$r
done
echo "=== RF3_DONE $(date -Is)"
