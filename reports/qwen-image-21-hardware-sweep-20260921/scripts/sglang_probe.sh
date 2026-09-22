#!/usr/bin/env bash
# Measure Qwen-Image-2.1 on one GPU, using the cookbook's own recipe for that card.
#
# Recipes are transcribed from docs/src/snippets/configs/Qwen/qwen-image-2.1.jsx,
# not invented. Each card's placement/attention pair is the one the picker marks
# `recommendedWhen` for that hardware at 1 GPU, tp=ulysses=ring=1, native
# precision, eager, batching off -- the same point in the space the 5090 run
# already validated.
#
#   h200     resident                     (platform attention = FA, no flag)
#   b200     resident + --attention-backend fa   (the picker emits it only here)
#   rtx5090  DiT layerwise offload        (sm120 -> runtime picks SDPA itself)
#   rtx4090  DiT+VAE resident, encoder layerwise
#
# The 4090 branch is the picker's *narrow* recipe, valid only at fa/native/eager/
# text/n=1/no-batching; outside that it degrades to plain DiT offload. This
# workload sits exactly inside it, so the narrow one is what runs.
set -u
HW="${1:?usage: qi21_bench.sh <h200|b200|rtx5090|rtx4090>}"
STEPS="${2:-40}"
REPS="${3:-3}"

# The 4090 pool has no /scratch; fall back to the largest writable path.
for c in /scratch /personal /workspace /root; do
  [ -d "$c" ] && [ -w "$c" ] && { BASE="$c"; break; }
done
BASE="${BASE:-/tmp}"
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub"
for t in /personal/bench0912/.hftoken /cluster-storage/models/token; do
  [ -f "$t" ] && { export HF_TOKEN="$(cat "$t")"; break; }
done
STATE="$BASE/state"; mkdir -p "$STATE" "$HF_HUB_CACHE"
PORT=57711
LOG="$STATE/qi21_${HW}_server.log"

echo "=== QI21_BENCH_START $(date -Is) hw=$HW host=$(hostname) base=$BASE ==="
python3 -c "import torch;print('  gpu',torch.cuda.get_device_name(0),'cap',torch.cuda.get_device_capability(),'vram',round(torch.cuda.get_device_properties(0).total_memory/2**30,1),'GiB')"

# --- 1. the runtime must actually know this model -------------------------
need_update() {
  python3 - <<'PY'
import pathlib, sglang, sys
root = pathlib.Path(sglang.__file__).parent
sys.exit(0 if list(root.rglob("qwen_image21.py")) else 1)
PY
}
if ! need_update; then
  echo "--- sglang predates qwen-image-2.1; updating /sgl-workspace/sglang ---"
  d=/sgl-workspace/sglang
  if [ -n "$(git -C $d status --porcelain)" ]; then
    echo "  REFUSING: the tree has local changes, not mine to discard"; exit 1
  fi
  # The lmsysorg/sglang:dev image bakes a GitHub Actions token into
  # `http.https://github.com/.extraheader`. It has expired, so git sends a dead
  # credential to a PUBLIC repo, gets 401, and reports "could not read Username
  # for 'https://github.com'" -- which reads like a network fault. Blank the
  # header for this command only; do not rewrite the box's git config.
  if ! git -C $d -c http.https://github.com/.extraheader= fetch --depth=50 origin main; then
    echo "  FATAL: fetch failed (see the error above)"; exit 1
  fi
  git -C $d checkout -q FETCH_HEAD || { echo "  FATAL: checkout FETCH_HEAD failed"; exit 1; }
  echo "  now at $(git -C $d rev-parse --short HEAD) $(git -C $d log -1 --format=%cd --date=short)"
fi
python3 - <<'PY'
import pathlib, sglang
root = pathlib.Path(sglang.__file__).parent
print("  sglang at", root)
print("  qwen-image-2.1 files:", len(list(root.rglob("*qwen_image21*"))))
reg = root / "multimodal_gen" / "registry.py"
print("  detector present:", reg.exists() and "qwen-image-2.1" in reg.read_text())
PY
need_update || { echo "  FATAL: still no qwen-image-2.1 support"; echo "=== QI21_BENCH_DONE $(date -Is) ==="; exit 1; }

# --- 2. weights -----------------------------------------------------------
echo "--- weights ---"
python3 -c "
import os
from huggingface_hub import snapshot_download
print('  at', snapshot_download('Qwen/Qwen-Image-2.1', token=os.environ.get('HF_TOKEN')))
" 2>&1 | tail -2

# --- 3. the card's recipe -------------------------------------------------
case "$HW" in
  h200)    RECIPE="--performance-mode speed" ;;
  b200)    RECIPE="--performance-mode speed --attention-backend fa" ;;
  rtx5090) RECIPE="--performance-mode manual --dit-layerwise-offload true" ;;
  rtx4090) RECIPE="--performance-mode manual --component-residency dit=resident text_encoder=layerwise-offload vae=resident --warmup-resolutions 1024x1024" ;;
  *) echo "  FATAL: no cookbook recipe for '$HW'"; exit 1 ;;
esac
echo "--- serving: $RECIPE ---"
sglang serve --model-path Qwen/Qwen-Image-2.1 --port $PORT --host 127.0.0.1 \
  --backend sglang --num-gpus 1 --model-type diffusion $RECIPE \
  > "$LOG" 2>&1 &
SRV=$!
up=0
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SRV 2>/dev/null || break
  sleep 5
done
if [ "$up" != 1 ]; then
  echo "  server never became healthy"
  grep -aoE "[A-Za-z]*Error: .{0,140}|Traceback|out of memory.{0,60}" "$LOG" | sort -u | head -6 | sed 's/^/    /'
  kill -9 $SRV 2>/dev/null
  echo "=== QI21_BENCH_DONE $(date -Is) ==="; exit 1
fi
echo "  healthy"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | sed 's/^/  vram after load: /'

# --- 4. measure -----------------------------------------------------------
python3 - "$PORT" "$HW" "$STEPS" "$REPS" <<'PY'
import json, statistics, sys, time, urllib.request
port, hw, steps, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
url = f"http://127.0.0.1:{port}/v1/images/generations"
def fire():
    # the cookbook's own generation payload. output_format=png is not optional:
    # this checkpoint emits RGBA and the default JPEG encoder dies with
    # "cannot write mode RGBA as JPEG", surfacing as a bare HTTP 500.
    payload = {"model": "Qwen/Qwen-Image-2.1",
               "prompt": "A warehouse robot folds a blue cloth on a clean workbench.",
               "n": 1, "size": "1024x1024", "num_inference_steps": steps,
               "guidance_scale": 1, "seed": 42, "generator_device": "cpu",
               "output_format": "png", "response_format": "b64_json",
               "background": "auto", "enable_cache_dit": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        n = len(r.read())
    return time.time() - t0, n
try:
    _, nbytes = fire()                       # warm the steady state, discard
    lats = [fire()[0] for _ in range(reps)]
    print(f"  RESULT {hw} p50={statistics.median(lats):.3f}s "
          f"{[round(x, 2) for x in lats]} bytes={nbytes}")
except Exception as exc:
    print(f"  REQUEST FAILED: {type(exc).__name__}: {str(exc)[:200]}")
PY
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | sed 's/^/  vram peak-ish: /'
grep -aoE "average time per step: [0-9.]+ seconds" "$LOG" | tail -1 | sed 's/^/  /'
kill $SRV 2>/dev/null; sleep 10; kill -9 $SRV 2>/dev/null
echo "=== QI21_BENCH_DONE $(date -Is) ==="
