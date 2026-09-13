#!/usr/bin/env bash
# Does asking sglang for a perf dump cost anything, and where does the ~0.69s
# outlier on zimage actually come from?
#
# The 5-sample harness runs cannot answer this. zimage sglang measured
# 0.695, 0.478, 0.474, 0.472, 0.473 with the dump on -- which looked like the
# dump's one-time cost landing on request 1. But vLLM-Omni's UNINSTRUMENTED
# full-shape warmups on the same case, minutes apart, were 0.485, 0.69, 0.489,
# 0.484: the same outlier, no dump, a different framework. Five samples cannot
# separate "the dump costs 0.22s once" from "this box produces a ~0.69s request
# every so often and sglang happened to catch one first".
#
# So: one server, many requests, the dump toggled request-by-request in an
# interleaved A/B. Everything except that one payload field is held constant --
# same process, same weights, no restart, no scheduling gap between arms.
#
# Written for bench-0912 (4xB200); the paths and the serve command are that
# box's, taken verbatim from the harness runlog for this case.
set -u
PORT=61001
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

while pgrep -f "/personal/bench0912/variance_check.sh" >/dev/null 2>&1; do sleep 120; done
echo "=== PERFDUMP_DIST_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

# The command the harness actually ran for this case (from its runlog), verbatim.
sglang serve --model-path Tongyi-MAI/Z-Image-Turbo --port $PORT --host 127.0.0.1 \
  --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server --tp-size 2 \
  > /personal/bench0912/perfdump_dist_server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "--- waiting for health"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  up after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /personal/bench0912/perfdump_dist_server.log; exit 1; }
  sleep 10
done

python3 - "$PORT" <<'PY'
import json, statistics, sys, time, urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/images/generations"
DUMP = "/personal/bench0912/perfdump_dist_probe.json"

def fire(with_dump):
    payload = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A serene mountain landscape at sunset with a crystal clear lake",
        "size": "1024x1024",
        "n": 1,
        "seed": 42,
    }
    if with_dump:
        payload["perf_dump_path"] = DUMP
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()          # drain, as the harness does
    return time.time() - t0

print("--- warming (20 requests, no dump) until the series is flat")
warm = [fire(False) for _ in range(20)]
print("    last five: " + ", ".join(f"{x:.3f}" for x in warm[-5:]))

N = 40
print(f"--- interleaved: {N} with dump, {N} without, alternating")
on, off = [], []
for i in range(N):
    off.append(fire(False))
    on.append(fire(True))

def describe(name, xs):
    xs_sorted = sorted(xs)
    p50 = statistics.median(xs)
    lo, hi = xs_sorted[0], xs_sorted[-1]
    # an "outlier" here = anything 20%+ above this arm's own median
    out = [x for x in xs if x > p50 * 1.2]
    print(f"  {name:18s} n={len(xs)} p50={p50:.3f} min={lo:.3f} max={hi:.3f} "
          f"mean={statistics.fmean(xs):.3f}  outliers>1.2xp50: {len(out)}"
          + (f"  {[round(x,3) for x in out]}" if out else ""))
    return p50, len(out)

print("=== RESULT ===")
p50_off, n_off = describe("no perf dump", off)
p50_on, n_on = describe("with perf dump", on)
print(f"  p50 delta (on - off) = {(p50_on - p50_off) * 1000:+.1f} ms "
      f"({(p50_on / p50_off - 1) * 100:+.2f}%)")
print(f"  outlier rate: off {n_off}/{len(off)}, on {n_on}/{len(on)}")
print("  positions of outliers (off):", [i for i, x in enumerate(off) if x > p50_off * 1.2])
print("  positions of outliers (on) :", [i for i, x in enumerate(on) if x > p50_on * 1.2])
print("  first sample of each arm:", f"off={off[0]:.3f} on={on[0]:.3f}")
print("  VERDICT: the dump costs the p50 delta above; an outlier rate that is")
print("           similar in both arms means the spike is not the dump.")
PY

echo "=== PERFDUMP_DIST_DONE $(date -Is) ==="
