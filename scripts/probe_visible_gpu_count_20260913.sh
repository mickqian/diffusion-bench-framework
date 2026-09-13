#!/usr/bin/env bash
# Does the number of VISIBLE GPUs change a 2-GPU case's latency?
#
# Every probe in this round ran at CUDA_VISIBLE_DEVICES=0,1 while the real runs
# expose all four and let `--num-gpus 2 --tp-size 2` use two. That is a
# systematic difference between the instrument and the thing measured, and it
# has to be ruled in or out before probe results are carried over to the
# benchmark.
#
# What made it worth asking (and it is only suggestive -- n=2 on one side):
#
#   variance_check  visible 0,1      zimage sglang 0.474
#   b200main        visible ?        zimage sglang 0.482
#   r2 round        visible 0,1,2,3  zimage sglang 0.514
#   final run       visible 0,1,2,3  zimage sglang 0.678
#
# and 14 fresh probe instances at visible 0,1 all landed at ~0.47, never at the
# 0.678 the final run published. If visibility matters, the benchmark should pin
# it rather than inherit whatever the launcher happened to pass; if it does not,
# the discrepancy is somewhere else and this rules out an entire explanation.
#
# Same command both arms, interleaved, several fresh servers each.
set -u
INSTANCES="${INSTANCES:-4}"
REQUESTS="${REQUESTS:-10}"
PORT_BASE="${PORT_BASE:-65001}"
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/personal/bench0912/visible_gpu_rows.jsonl
: > "$OUT"

source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== VISIBLE_GPU_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

i=0
for round in $(seq 1 "$INSTANCES"); do
  for VIS in 0,1 0,1,2,3; do
    i=$((i + 1))
    PORT=$((PORT_BASE + i * 2))
    echo "--- round $round, visible=$VIS, port $PORT"
    # Only this probe's own leftovers: an unfiltered kill took out a
    # concurrently-running job's server and surfaced as an sglang health-check
    # failure that had nothing to do with sglang.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1,2,3 2>/dev/null \
      | sort -u | xargs -r kill -9 2>/dev/null
    sleep 5
    CUDA_VISIBLE_DEVICES="$VIS" sglang serve --model-path Tongyi-MAI/Z-Image-Turbo \
      --port $PORT --host 127.0.0.1 --backend sglang --num-gpus 2 \
      --model-type diffusion --warmup-mode server --tp-size 2 \
      > "/personal/bench0912/visgpu_${i}.log" 2>&1 &
    SERVER_PID=$!
    up=0
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
      kill -0 $SERVER_PID 2>/dev/null || break
      sleep 5
    done
    [ "$up" = 1 ] || { echo "    did not come up"; kill $SERVER_PID 2>/dev/null; continue; }
    python3 - "$PORT" "$VIS" "$REQUESTS" "$OUT" <<'PY'
import json, statistics, sys, time, urllib.request
port, vis, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
url = f"http://127.0.0.1:{port}/v1/images/generations"

def fire():
    payload = {"model": "Tongyi-MAI/Z-Image-Turbo",
               "prompt": "A serene mountain landscape at sunset with a crystal clear lake",
               "size": "1024x1024", "n": 1, "seed": 42,
               "response_format": "b64_json"}   # what the harness asks for
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return time.time() - t0

for _ in range(5):
    fire()
lats = [fire() for _ in range(n)]
row = {"visible": vis, "median": round(statistics.median(lats), 4),
       "latencies": [round(x, 4) for x in lats]}
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"    visible={vis}: median={row['median']} min={min(lats):.4f} max={max(lats):.4f}")
PY
    kill $SERVER_PID 2>/dev/null
    sleep 5
  done
done

echo "=== RESULT ==="
python3 - "$OUT" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["visible"], []).append(r["median"])
for vis in sorted(by, key=len):
    v = by[vis]
    print(f"  visible={vis:8s} n={len(v)} medians={v} -> p50 {statistics.median(v):.4f}")
if len(by) == 2:
    a, b = sorted(by, key=lambda k: len(k.split(",")))
    ma, mb = statistics.median(by[a]), statistics.median(by[b])
    delta = (mb - ma) / ma * 100
    print(f"  {b} vs {a}: {delta:+.1f}%")
    if abs(delta) < 3:
        print("  VERDICT: visible GPU count does not explain it -- the probes at 0,1")
        print("           are measuring the same thing the benchmark does.")
    else:
        print(f"  VERDICT: visible GPU count MATTERS ({delta:+.1f}%). The benchmark")
        print("           should pin CUDA_VISIBLE_DEVICES rather than inherit whatever")
        print("           the launcher passed, and every cross-run comparison has to")
        print("           hold it constant.")
    allv = [x for v in by.values() for x in v]
    if max(allv) / min(allv) < 1.25:
        print("  NOTE: the ~0.678 mode did not appear in either arm; it is not")
        print("        explained by this variable, whatever this variable does.")
PY
echo "=== VISIBLE_GPU_DONE $(date -Is) ==="
