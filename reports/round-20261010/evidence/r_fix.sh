#!/usr/bin/env bash
# Whole-node 4xGB300 (mempolicy permitted, GPU0/1 on node 0, GPU2/3 on node 1), round 2026-10-10 fixes:
#   1. sglang main 6576e50 H3 t2va 2-GPU through the runner, to compare this pod with the published 57.46 s
#   2. LightX2V b6d38283 with the torch_sdpa SP profiles: t2va 2-GPU, ref2va 2-GPU, t2va 4-GPU
#   3. FLUX.1 TP2 sglang eager vs torch.compile, interleaved twice
set -u
W=/scratch/r; mkdir -p $W/logs; cd $W
M=6576e50cd5e5817b10e12245104603eb1b0d8a4f
set -a; . ~/.config/radixark/secrets.env; set +a
[ -n "${HF_TOKEN:-}" ] && echo "RF hf token loaded" || echo "RF no hf token"
export HF_HOME=$W/hf HF_HUB_CACHE=$W/hf/hub HUGGINGFACE_HUB_CACHE=$W/hf/hub
[ -d dbf ] || git clone -q https://github.com/mickqian/diffusion-bench-framework dbf
git -C dbf fetch -q origin main && git -C dbf checkout -q --detach origin/main
echo "RF dbf $(git -C dbf log --oneline -1 | cut -c1-70)"
(cd /sgl-workspace/sglang && { git cat-file -t $M >/dev/null 2>&1 || timeout 900 git fetch -q origin $M; } && { [ -d $W/sgl-main ] || git worktree add -q --detach $W/sgl-main $M; })
PYTHONPATH=$W/sgl-main/python python3 -c "import sglang, sys; sys.exit(0 if sglang.__file__.startswith('$W/sgl-main/') else 1)" || { echo "RF FATAL sglang tree"; exit 1; }

# downloads in the background while LightX2V installs
python3 - > $W/logs/dl_h3.log 2>&1 <<'PY' &
from huggingface_hub import snapshot_download
print(snapshot_download("MiniMaxAI/MiniMax-H3", revision="42ed227ee7df40d41602854ae760620d6eb651fe", max_workers=8))
PY
DL1=$!
python3 - > $W/logs/dl_flux1.log 2>&1 <<'PY' &
from huggingface_hub import snapshot_download
print(snapshot_download("black-forest-labs/FLUX.1-dev", max_workers=8))
PY
DL2=$!
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=$W/fw-venvs
cd $W/dbf
if [ ! -f $W/fw-venvs/lightx2v/.diffusion-bench-install-stamp ]; then
  LIGHTX2V_INSTALL_FA2=0 LIGHTX2V_INSTALL_SPEC=git+https://github.com/ModelTC/LightX2V.git@b6d38283cb8176303f9155e639bcd496553d3402 \
    bash scripts/install_comparison_frameworks.sh lightx2v > $W/logs/install_lightx2v.log 2>&1
  echo "RF lightx2v install rc=$? stamp=$(ls $W/fw-venvs/lightx2v/.diffusion-bench-install-stamp 2>/dev/null)"
fi
$W/fw-venvs/lightx2v/bin/python3 -c "import importlib.metadata as m, torch; print('RF lightx2v', m.version('lightx2v'), 'torch', torch.__version__)"
wait $DL1; echo "RF h3 download rc=$? $(tail -1 $W/logs/dl_h3.log | cut -c1-120)"
wait $DL2; echo "RF flux1 download rc=$? $(tail -1 $W/logs/dl_flux1.log | cut -c1-120)"

export DBF_STATE_DIR=$W DBF_REPO_DIR=$W/dbf DBF_LOG_DIR=$W/logs DBF_VENV_ROOT=$W/fw-venvs DBF_HF_HOME=$W/hf \
       DBF_HARDWARE_PROFILE=gb300 DBF_CASE_TIMEOUT=7200
summ() {  # <json> <label>
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    for x in json.load(open(sys.argv[1]))["results"]:
        m = x.get("metrics") or {}
        print(f"RF {sys.argv[2]} {x['case_id']} {x['framework']} median={x['latency_s']} samples={m.get('latency_samples_s')} err={str(x.get('error'))[:160]}")
except Exception as e:
    print(f"RF {sys.argv[2]} no result ({type(e).__name__})")
PY
}
# 1. calibration
DBF_FRAMEWORKS=sglang DBF_SGLANG_PYTHONPATH=$W/sgl-main/python \
  bash dbf/scripts/devbox_run_cases.sh rfsglh3 0,1 61901 single_e2e minimax_h3_t2va_5s
summ $W/logs/run_rfsglh3_minimax_h3_t2va_5s.json calibration
# 2. LightX2V with the new profiles
DBF_FRAMEWORKS=lightx2v bash dbf/scripts/devbox_run_cases.sh rflx2 0,1 61911 single_e2e minimax_h3_t2va_5s minimax_h3_ref2va_5s
summ $W/logs/run_rflx2_minimax_h3_t2va_5s.json lightx2v
summ $W/logs/run_rflx2_minimax_h3_ref2va_5s.json lightx2v
DBF_FRAMEWORKS=lightx2v bash dbf/scripts/devbox_run_cases.sh rflx4 0,1,2,3 61921 single_e2e minimax_h3_t2va_5s_4gpu
summ $W/logs/run_rflx4_minimax_h3_t2va_5s_4gpu.json lightx2v
grep -ahoE '"attn_type": "[a-z_0-9]+"|"tensor_p_size": [0-9]|"seq_p_size": [0-9]' /tmp/lightx2v_config_minimax_h3_*.json | sort | uniq -c | sed 's/^/RF lightx2v cfg /'
# 3. FLUX.1 eager vs compile
export TUNE_GPUS=0,1 TUNE_CASE_TIMEOUT=3600 TUNE_PORT_BASE=61931
P="ENV:PYTHONPATH=$W/sgl-main/python:src"
cd $W/dbf
bash scripts/tune_sglang_serve_args.sh flux1_dev_t2i_1024 2 "rfeager=$P" "rfcompile=$P --enable-torch-compile true" \
  "rfcompileNoOfc=$P --enable-torch-compile true --offload-during-compile false" 2>&1 | grep -E "^TUNE " | sed 's/^/RF flux1 /'
echo "RF numa: $(grep -ahE 'bound to NUMA|lacks permission' $W/logs/tune_rfeager_1.runlog | sed 's/.*\] //' | sort | uniq -c | tr '\n' ';')"
echo "=== RF_DONE $(date -Is)"
