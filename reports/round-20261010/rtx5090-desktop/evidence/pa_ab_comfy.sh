#!/usr/bin/env bash
# rdxa-5090-pa: Qwen-Image-2.1 t2i, sglang main 6576e50 against ComfyUI, interleaved twice (lead was 4%).
set -u
S=$HOME/h3bench; cd $S
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
# shellcheck disable=SC1091
source $S/base/bin/activate
(
  while sleep 5; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 3072 ]; then
      echo "PA WATCHDOG MemAvailable ${a} MiB at $(date -Is)"
      pkill -TERM -u "$USER" -f "diffusion_bench.run_comparison|sglang serve|ComfyUI/main.py"; sleep 10
      pkill -KILL -u "$USER" -f "diffusion_bench.run_comparison|sglang serve|ComfyUI/main.py"
    fi
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT
export DBF_STATE_DIR=$S DBF_REPO_DIR=$S/dbf DBF_LOG_DIR=$S/logs DBF_VENV_ROOT=$S/fw-venvs DBF_HF_HOME=$S/hf \
       DBF_HARDWARE_PROFILE=rtx5090 DBF_CASE_TIMEOUT=3600 GPUS=0 DIFFUSION_BENCH_COMFYUI_WORKSPACE=$S/comfy-ws
runner=$S/dbf/scripts/run_b200_cross_framework_20260912.sh
port=61401
for r in 1 2; do
  port=$((port + 20))
  DBF_SGLANG_PYTHONPATH=$S/sglang-6576e50/python DBF_FRAMEWORKS=sglang TAG=pa5090ab1011sgl$r BASE_PORT=$port \
    CASES=qwen_image_21_t2i_1024 bash $runner | grep -E "^RESULT" | sed "s/^/PA r$r sglang /" | cut -c1-80
  DBF_FRAMEWORKS=comfyui TAG=pa5090ab1011comfy$r BASE_PORT=$((port + 10)) \
    CASES=qwen_image_21_t2i_1024 bash $runner | grep -E "^RESULT" | sed "s/^/PA r$r comfyui /" | cut -c1-80
done
for f in $S/logs/run_pa5090ab1011*_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json; do
  python3 - "$f" <<'PY'
import json, sys
for x in json.load(open(sys.argv[1]))["results"]:
    print(f"PA {sys.argv[1].split('run_')[1][:24]} {x['framework']} median={x['latency_s']} samples={(x.get('metrics') or {}).get('latency_samples_s')} err={str(x.get('error'))[:100]}")
PY
done
echo "=== PA_AB_DONE $(date -Is)"
