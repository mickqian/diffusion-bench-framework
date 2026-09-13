#!/usr/bin/env bash
# How often does a fresh sglang server land in zimage's slow mode?
#
# The mode is locked per server instance, not per request: one instance measured
# 0.695, 0.478, 0.474, 0.472, 0.473 and another, 45 minutes later on the same box
# with an identical command, measured 0.680, 0.474, 0.678, 0.680, 0.673. The
# nightly series shows the same shape on 4xH100 -- zimage sat at ~0.95 for seven
# consecutive runs, then went back to ~0.77.
#
# That makes the published median a draw, not a measurement: 0.474 beats
# vLLM-Omni's stable 0.497, 0.678 loses to it by 27%. Five more repeats inside
# one instance cannot fix that, because the instance has already picked its mode.
# So sample the thing that actually varies -- restart the server.
#
# The companion probe (probe_zimage_bimodality_20260913.sh) asks WHERE the two
# modes come from, by recording client and server time per request. This one asks
# HOW OFTEN each occurs, which is what decides whether the cell can carry a
# single number at all.
#
# Written for bench-0912 (4xB200). Run on an idle box.
set -u
INSTANCES=8
REQUESTS=8
PORT_BASE=63001
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

while pgrep -f "/personal/bench0912/probe_zimage_bimodal.sh" >/dev/null 2>&1; do sleep 120; done
source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== ZIMAGE_INSTANCES_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

RESULTS=/personal/bench0912/zimage_instances.jsonl
: > "$RESULTS"

for i in $(seq 1 $INSTANCES); do
  PORT=$((PORT_BASE + i * 2))
  echo "--- instance $i/$INSTANCES on port $PORT"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
  sglang serve --model-path Tongyi-MAI/Z-Image-Turbo --port $PORT --host 127.0.0.1 \
    --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server --tp-size 2 \
    > "/personal/bench0912/zimage_inst_${i}.log" 2>&1 &
  SERVER_PID=$!
  up=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SERVER_PID 2>/dev/null || break
    sleep 5
  done
  if [ "$up" != "1" ]; then
    echo "    server did not come up; skipping instance $i"
    kill $SERVER_PID 2>/dev/null
    continue
  fi
  python3 - "$PORT" "$i" "$REQUESTS" "$RESULTS" <<'PY'
import json, statistics, sys, time, urllib.request
port, inst, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
url = f"http://127.0.0.1:{port}/v1/images/generations"

def fire():
    payload = {"model": "Tongyi-MAI/Z-Image-Turbo",
               "prompt": "A serene mountain landscape at sunset with a crystal clear lake",
               "size": "1024x1024", "n": 1, "seed": 42}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return time.time() - t0

for _ in range(4):        # warm this instance before measuring it
    fire()
lats = [fire() for _ in range(n)]
row = {"instance": inst, "latencies": [round(x, 4) for x in lats],
       "median": round(statistics.median(lats), 4),
       "min": round(min(lats), 4), "max": round(max(lats), 4)}
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"    instance {inst}: median={row['median']} min={row['min']} max={row['max']}")
PY
  kill $SERVER_PID 2>/dev/null
  sleep 5
done

echo "=== RESULT ==="
python3 - "$RESULTS" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  no instances measured"); raise SystemExit
meds = sorted(r["median"] for r in rows)
print("  per-instance medians:", meds)
# split at the widest gap between consecutive instance medians
gaps = [(meds[i + 1] - meds[i], i) for i in range(len(meds) - 1)]
width, idx = max(gaps) if gaps else (0, 0)
lo, hi = meds[: idx + 1], meds[idx + 1 :]
if width > 0.08 * meds[0] and len(lo) and len(hi):
    print(f"  two modes: {len(lo)} instance(s) at ~{statistics.median(lo):.3f}s, "
          f"{len(hi)} at ~{statistics.median(hi):.3f}s  (+{(statistics.median(hi)/statistics.median(lo)-1)*100:.0f}%)")
    print(f"  slow-mode rate: {len(hi)}/{len(rows)}")
else:
    print(f"  no clear split; medians span {meds[0]:.3f}-{meds[-1]:.3f}s")
print("  vLLM-Omni on this case is stable at ~0.497s, so a slow-mode instance loses")
print("  and a fast-mode one wins -- which is why the cell cannot carry one number")
print("  until this rate is known.")
for r in rows:
    print(f"    inst {r['instance']}: {r['latencies']}")
PY
echo "=== ZIMAGE_INSTANCES_DONE $(date -Is) ==="
