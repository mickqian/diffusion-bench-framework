#!/usr/bin/env bash
# rdxa-5090-pa, 2026-10-11: the sglang column on main 6576e50 (as the GB300/H200 report),
# then vLLM-Omni --enforce-eager for the eager-vs-eager check. A memory watchdog stops this
# run's own processes before the host runs out of RAM: on 10-10 a ComfyUI H3 run exhausted
# the 60 GB and the OOM killer took tailscaled, which took the host off the network.
set -u
S=$HOME/h3bench; cd $S
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1 \
       GIT_CONFIG_KEY_1=http.lowSpeedLimit GIT_CONFIG_VALUE_1=1000
M=6576e50cd5e5817b10e12245104603eb1b0d8a4f
T=$S/sglang-6576e50
if [ ! -f $T/python/pyproject.toml ]; then
  rm -rf $T.tmp; mkdir -p $T.tmp
  curl -fsSL --http1.1 --retry 5 https://codeload.github.com/sgl-project/sglang/tar.gz/$M | tar xz -C $T.tmp --strip-components=1 \
    && mv $T.tmp $T || { echo "PA FATAL sglang tarball"; exit 1; }
fi
# shellcheck disable=SC1091
source $S/base/bin/activate
echo "PA sglang tree: $(PYTHONPATH=$T/python python -c 'import sglang; print(sglang.__file__)')"
git -C $S/dbf fetch -q origin round-20261010 && git -C $S/dbf checkout -q --detach origin/round-20261010
echo "PA dbf $(git -C $S/dbf log --oneline -1 | cut -c1-60)"

(
  while sleep 5; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 3072 ]; then
      echo "PA WATCHDOG MemAvailable ${a} MiB at $(date -Is): stopping this run's servers"
      pkill -TERM -u "$USER" -f "diffusion_bench.run_comparison|sglang serve|vllm serve|ComfyUI/main.py"
      sleep 10
      pkill -KILL -u "$USER" -f "diffusion_bench.run_comparison|sglang serve|vllm serve|ComfyUI/main.py"
    fi
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

export DBF_STATE_DIR=$S DBF_REPO_DIR=$S/dbf DBF_LOG_DIR=$S/logs DBF_VENV_ROOT=$S/fw-venvs DBF_HF_HOME=$S/hf \
       DBF_HARDWARE_PROFILE=rtx5090 DBF_CASE_TIMEOUT=7200 GPUS=0
export DIFFUSION_BENCH_COMFYUI_WORKSPACE=$S/comfy-ws
runner=$S/dbf/scripts/run_b200_cross_framework_20260912.sh

DBF_SGLANG_PYTHONPATH=$T/python DBF_FRAMEWORKS=sglang TAG=pa5090main1011 BASE_PORT=61201 \
  CASES="qwen_image_21_t2i_1024 qwen_image_21_edit_1024 minimax_h3_t2va_480p_1gpu" bash $runner

export HF_HOME=$S/hf HF_HUB_CACHE=$S/hf/hub HUGGINGFACE_HUB_CACHE=$S/hf/hub
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=$S/fw-venvs SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
cd $S/dbf
port=61301
for c in qwen_image_21_t2i_1024 qwen_image_21_edit_1024; do
  port=$((port + 10)); out=$S/logs/veager1011_$c.json
  DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src timeout 3600 python3 -m diffusion_bench.run_comparison \
    --config configs/comparison_configs.json --frameworks vllm-omni --case-ids $c --modes single_e2e \
    --hardware-profile rtx5090 --port $port --output $out > ${out%.json}.runlog 2>&1
  python3 - "$out" "$c" <<'PY'
import json, sys
try:
    x = next(r for r in json.load(open(sys.argv[1]))["results"] if r["framework"] == "vllm-omni")
    print(f"PA {sys.argv[2]} VEAGER median={x['latency_s']} samples={(x.get('metrics') or {}).get('latency_samples_s')} eager={'--enforce-eager' in (x.get('server_command') or '')} err={str(x.get('error'))[:150]}")
except Exception as e:
    print(f"PA {sys.argv[2]} VEAGER no result ({type(e).__name__})")
PY
done
grep -h "^RESULT" $S/logs/run_pa5090main1011_*.log
echo "=== PA_DONE $(date -Is)"
