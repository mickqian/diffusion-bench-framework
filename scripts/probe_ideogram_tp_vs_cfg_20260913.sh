#!/usr/bin/env bash
# Why is ideogram4-fp8 68% slower under CFG parallelism than under TP?
#
# It is the one case where the runtime's own choice loses badly, and it breaks
# the rule the other cases suggested ("CFG model -> CFG parallelism wins"):
#
#   pinned  --tp-size 2            3.423s / 3.430s   (two rounds)
#   default cfg_parallel_degree=2  5.764s / 5.765s
#
# Both arms are stable to three decimals, so this is structural, not noise.
# Three explanations were tested against the existing logs and all three failed:
#   * offload difference -- identical, text_encoder only, same lines in both
#   * single-branch degeneration -- the case carries no guidance override, so
#     the model's CFG default applies and both branches really run
#   * fp8 dequant path -- identical, both log the same "once at first use" and
#     FP8-resident lines
#
# So ask the server where the time goes. The perf dump carries per-stage
# `steps[].duration_ms` and per-step `denoise_steps_ms`, which separates
# "denoise got slower" from "something around denoise got slower".
#
# Written for bench-0912 (4xB200).
set -u
PORT_BASE="${PORT_BASE:-66001}"
REQUESTS="${REQUESTS:-6}"
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== IDEOGRAM_TP_VS_CFG_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

MODEL=$(python3 -c "
import json
d=json.load(open('configs/comparison_configs.json'))
print(next(c for c in d['cases'] if c['id']=='ideogram4_t2i_1024_2gpu_tp')['model'])")
echo "  model: $MODEL"

i=0
for ARM in "tp:--tp-size 2" "cfgpar:--tp-size 1 --cfg-parallel-size 2"; do
  tag="${ARM%%:*}"; extra="${ARM#*:}"
  i=$((i+1)); PORT=$((PORT_BASE + i*2))
  echo "--- arm $tag ($extra) on port $PORT"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
  # shellcheck disable=SC2086
  sglang serve --model-path "$MODEL" --port $PORT --host 127.0.0.1 \
    --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server $extra \
    > "/personal/bench0912/ideogram_${tag}.log" 2>&1 &
  SERVER_PID=$!
  # Poll health only. A `kill -0 $SERVER_PID` early-break was here and reported
  # "did not come up" on a server that WAS coming up -- it reached "fired up and
  # ready to roll" 3.5 minutes in, inside a 900s budget. Treating "the pid I
  # captured looks gone" as "the service failed" is a false negative the health
  # poll already covers: a server that truly died just times out.
  up=0; waited=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 5; waited=$((waited + 5))
  done
  if [ "$up" != 1 ]; then
    echo "    never became healthy after ${waited}s"
    tail -5 "/personal/bench0912/ideogram_${tag}.log"
    kill "$SERVER_PID" 2>/dev/null; sleep 5; kill -9 "$SERVER_PID" 2>/dev/null
    continue
  fi
  echo "    healthy after ${waited}s"

  python3 - "$PORT" "$tag" "$REQUESTS" "$MODEL" <<'PY'
import json, os, statistics, sys, time, urllib.request
port, tag, n, model = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
url = f"http://127.0.0.1:{port}/v1/images/generations"
DUMP = f"/personal/bench0912/ideogram_perf_{tag}.json"

def fire(with_dump):
    if with_dump and os.path.exists(DUMP):
        os.remove(DUMP)
    payload = {"model": model, "prompt": "A warehouse robot folds a blue cloth on a clean workbench.",
               "size": "1024x1024", "n": 1, "seed": 0, "response_format": "b64_json"}
    if with_dump:
        payload["perf_dump_path"] = DUMP
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.time() - t0

for _ in range(3):
    fire(False)
lats = [fire(False) for _ in range(n)]
fire(True)
dump = {}
deadline = time.time() + 20
while time.time() < deadline:
    try:
        dump = json.load(open(DUMP)); break
    except Exception:
        time.sleep(0.2)
print(f"    {tag}: client p50={statistics.median(lats):.3f} ({[round(x,3) for x in lats]})")
if dump:
    print(f"    {tag}: server total={dump.get('total_duration_ms', 0)/1000:.3f}s")
    for key in ("steps", "denoise_steps_ms"):
        items = dump.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        durs = [x.get("duration_ms", 0) for x in items if isinstance(x, dict)]
        if not durs:
            continue
        if key == "steps":
            print(f"    {tag}: stage ms = {[round(d,1) for d in durs]}")
        else:
            print(f"    {tag}: {len(durs)} denoise steps, median {statistics.median(durs):.2f} ms, "
                  f"total {sum(durs)/1000:.3f}s")
else:
    print(f"    {tag}: no perf dump")
PY
  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 12); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 5; done
  kill -9 "$SERVER_PID" 2>/dev/null
  sleep 5
done
echo "  Read it as: if the denoise total moves with the client time, the strategy"
echo "  changes the per-step cost; if denoise is flat and the stage list is not,"
echo "  the cost is around denoise (encode, decode, or the combine)."
echo "=== IDEOGRAM_TP_VS_CFG_DONE $(date -Is) ==="
