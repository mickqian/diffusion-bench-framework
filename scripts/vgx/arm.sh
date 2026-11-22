#!/bin/bash
# One arm: a fresh server on GPUs 0..N-1 with the VideoGenX recipe flags, the VGX
# client (1 warmup + NREQ cases), then stop the server.
#   arm.sh <tag> <engine: sglang|omni> <tree|-> <ngpu> [nreq=20]
set -u
ROOT=${VGX_ROOT:?set VGX_ROOT}
TAG=$1
ENGINE=$2
TREE=$3
N=$4
NREQ=${5:-20}
PORT=61283
LOG=$ROOT/logs
RES=$ROOT/res
RUN=$ROOT/run/$TAG
STATUS=$LOG/arm_$TAG.status
SLOG=$LOG/server_$TAG.log
SNAP=$(cat "$ROOT/SNAPSHOT_DIR")
DEVS=$(seq -s, 0 $((N - 1)))
mkdir -p "$LOG" "$RES" "$RUN"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

if [ -s "$RES/$TAG.jsonl" ]; then
  say "SKIP result file exists"
  exit 0
fi
if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/health"; then
  say "FAILED port $PORT already answers"
  exit 1
fi

say "STARTING engine=$ENGINE tree=$TREE gpus=$DEVS nreq=$NREQ"
cd "$RUN"
if [ "$ENGINE" = sglang ]; then
  # tree "-" = the sglang the image ships
  if [ "$TREE" != - ]; then
    export PYTHONPATH=$ROOT/src/$TREE/python${VGX_PYDEPS:+:$VGX_PYDEPS}
  fi
  {
    echo "tree=$TREE $(grep "^$TREE " "$ROOT/src/TREE_IDS")"
    python3 -c "import sglang, torch; print('sglang', sglang.__file__); print('torch', torch.__version__)"
    python3 -c "import aiter, os; print('aiter', os.path.dirname(aiter.__file__))"
    git -C "$(python3 -c 'import aiter, os; print(os.path.dirname(os.path.dirname(aiter.__file__)))')" rev-parse HEAD 2>/dev/null | sed 's/^/aiter_commit /'
  } > "$LOG/ident_$TAG.txt" 2>&1
  ROCR_VISIBLE_DEVICES=$DEVS \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$ROOT/hf \
    SGLANG_USE_AITER=1 ROCM_QUICK_REDUCE_QUANTIZATION=NONE PYTHONUNBUFFERED=1 \
    setsid nohup python3 -c "import sys; from sglang.cli.main import main; sys.argv[0] = 'sglang'; main()" \
    serve --model-type diffusion --model-path "$SNAP" --model-variant fl2va \
    --num-gpus "$N" --ulysses-degree "$N" --tp-size 1 --attention-backend aiter \
    --encoder-parallel auto --warmup-resolutions 1344x768 --disable-conditioning-cache \
    --performance-mode speed --host 127.0.0.1 --port $PORT \
    --enable-torch-compile false --dit-cpu-offload false \
    ${VGX_SGL_EXTRA:-} > "$SLOG" 2>&1 < /dev/null &
else
  {
    python3 -c "import vllm, vllm_omni, torch; print('vllm', vllm.__version__, 'vllm_omni', vllm_omni.__version__, 'torch', torch.__version__)"
  } > "$LOG/ident_$TAG.txt" 2>&1
  ROCR_VISIBLE_DEVICES=$DEVS HIP_VISIBLE_DEVICES=$DEVS \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=$ROOT/hf PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0 \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 VLLM_OMNI_SERVER_STORAGE__PATH=$RUN/storage \
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
    setsid nohup vllm serve "$ROOT/model-view" --served-model-name MiniMaxAI/MiniMax-H3 --omni \
    --host 127.0.0.1 --port $PORT --trust-remote-code --task-type fl2va \
    --num-gpus "$N" --tensor-parallel-size 1 --usp "$N" --ring 1 \
    --text-encoder-tp-size "$N" --vae-patch-parallel-size "$N" --vae-parallel-mode tile \
    --vae-use-tiling --diffusion-attention-backend FLASH_ATTN \
    --cache-config '{"minimax_h3_adaln_cache": false}' \
    ${VGX_OMNI_EXTRA:-} > "$SLOG" 2>&1 < /dev/null &
fi
SPID=$!
echo "$SPID" > "$RUN/server.pid"
say "SERVER_PID $SPID"

stop_server() {
  kill -TERM -- -"$SPID" 2>/dev/null
  for _ in $(seq 1 30); do
    # zombies stay in the table (PID 1 is sleep infinity): only count live ones
    alive=$(ps -o stat= -g "$SPID" 2>/dev/null | grep -vc '^Z')
    [ "$alive" = 0 ] && break
    sleep 2
  done
  kill -KILL -- -"$SPID" 2>/dev/null
  sleep 5
}

ready=0
T0=$(date +%s)
for _ in $(seq 1 720); do
  if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; then
    ready=1
    break
  fi
  if [ "$(ps -o stat= -p "$SPID" 2>/dev/null | grep -vc '^Z')" = 0 ]; then
    break
  fi
  sleep 5
done
if [ $ready != 1 ]; then
  say "FAILED server not ready after $(($(date +%s) - T0)) s"
  tail -n 40 "$SLOG" > "$LOG/fail_$TAG.tail"
  stop_server
  exit 1
fi
say "SERVING startup=$(($(date +%s) - T0))s"

python3 "$ROOT/bin/vgx_client.py" --backend "$ENGINE" --base-url "http://127.0.0.1:$PORT" \
  --cases "$ROOT/bin/vgx_cases.json" --n "$NREQ" --warmup 1 --out "$RES/$TAG.jsonl.part" \
  > "$LOG/client_$TAG.log" 2>&1
RC=$?
if [ $RC = 0 ]; then
  mv "$RES/$TAG.jsonl.part" "$RES/$TAG.jsonl"
  say "CLIENT_DONE $(tail -n 1 "$LOG/client_$TAG.log")"
else
  say "FAILED client rc=$RC $(tail -n 1 "$LOG/client_$TAG.log" | cut -c1-200)"
  tail -n 60 "$SLOG" > "$LOG/fail_$TAG.tail"
fi
stop_server
say "STOPPED"
exit $RC
