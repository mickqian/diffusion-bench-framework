#!/usr/bin/env bash
# Where does sglang's zimage bimodality live -- in compute, or after the clock stops?
#
# Two clean modes, ~0.474s and ~0.678s. Only the MIX varies, and it is locked
# per server instance, so the published median swings 43% run to run:
#
#   variance_check 03:25  0.695, 0.478, 0.474, 0.472, 0.473   (median 0.474)
#   final run      03:20  0.680, 0.474, 0.678, 0.680, 0.673   (median 0.678)
#
# It is sglang-specific: on the same case vLLM-Omni spreads 5.5%, LightX2V 0.4%.
#
# sglang's `total_duration_ms` covers denoise + encode + VAE and stops BEFORE
# `_materialize_output_transport` and the conditional `empty_cache()`, both of
# which are inside the client's window. So the question is whether the slow mode
# moves `server` or moves the gap.
#
# TWO THINGS THIS PROBE GOT WRONG THE FIRST TIME, both fixed here:
#
#   1. It sent no `response_format`, so the server returned a 406-byte path
#      while the harness asks for `b64_json` and receives ~1.4MB. Those are
#      different post-compute paths, so the gap it measured was not the gap the
#      benchmark sees. The payload below matches the harness.
#   2. It printed its verdict sentence unconditionally. The run landed entirely
#      in the fast mode (52 samples, 0.467-0.489) and it still announced "the
#      bimodality is outside the region sglang times" -- while its own numbers
#      said the opposite, server differing 0.361/0.378 and the gap flat at
#      0.110/0.111. A probe that states a conclusion the data does not support
#      is worse than one that states none. The verdict is now derived.
#
# And one instance is not enough: the mode is locked per instance, so this
# restarts the server until it has seen both, or runs out of tries.
#
# Written for bench-0912 (4xB200). Run when the GPUs are otherwise idle.
set -u
MAX_INSTANCES="${MAX_INSTANCES:-6}"
REQUESTS="${REQUESTS:-30}"
PORT_BASE="${PORT_BASE:-62001}"
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

for job in final_run probe_zimage_inst; do
  while [ "$(ps -e -o cmd= | grep -cx "bash /personal/bench0912/${job}.sh")" -gt 0 ]; do sleep 120; done
done
source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== ZIMAGE_BIMODAL_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

OUT=/personal/bench0912/zimage_bimodal_rows.jsonl
: > "$OUT"

for i in $(seq 1 "$MAX_INSTANCES"); do
  PORT=$((PORT_BASE + i * 2))
  echo "--- instance $i/$MAX_INSTANCES on port $PORT"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5
  sglang serve --model-path Tongyi-MAI/Z-Image-Turbo --port $PORT --host 127.0.0.1 \
    --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server --tp-size 2 \
    > "/personal/bench0912/zimage_bimodal_srv_${i}.log" 2>&1 &
  SERVER_PID=$!
  up=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SERVER_PID 2>/dev/null || break
    sleep 5
  done
  [ "$up" = 1 ] || { echo "    did not come up; skipping"; kill $SERVER_PID 2>/dev/null; continue; }

  python3 - "$PORT" "$i" "$REQUESTS" "$OUT" <<'PY'
import json, os, statistics, sys, time, urllib.request
port, inst, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
url = f"http://127.0.0.1:{port}/v1/images/generations"
DUMP = f"/personal/bench0912/zimage_bimodal_perf_{inst}.json"

def fire():
    if os.path.exists(DUMP):
        os.remove(DUMP)
    payload = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A serene mountain landscape at sunset with a crystal clear lake",
        "size": "1024x1024", "n": 1, "seed": 42,
        # The harness asks for b64_json, so the ~1.4MB encode and transfer are
        # part of what it measures. Matching it is the whole point.
        "response_format": "b64_json",
        "perf_dump_path": DUMP,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    client = time.time() - t0
    server, deadline = None, time.time() + 10
    while time.time() < deadline:
        try:
            v = json.load(open(DUMP)).get("total_duration_ms")
            if v is not None:
                server = v / 1000.0
                break
        except Exception:
            time.sleep(0.05)
    return client, server, len(body)

for _ in range(6):
    fire()
rows = [fire() for _ in range(n)]
kept = [{"instance": inst, "client": c, "server": s, "gap": c - s, "bytes": b}
        for c, s, b in rows if s is not None]
with open(out, "a") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
if kept:
    print(f"    instance {inst}: client p50={statistics.median([r['client'] for r in kept]):.3f} "
          f"server p50={statistics.median([r['server'] for r in kept]):.3f} "
          f"gap p50={statistics.median([r['gap'] for r in kept]):.3f} "
          f"bytes={kept[0]['bytes']} ({len(kept)}/{len(rows)} with a dump)")
PY
  kill $SERVER_PID 2>/dev/null
  sleep 5

  # Stop early once both modes have been observed across instances.
  python3 - "$OUT" <<'PY' && { echo "--- both modes seen; stopping early"; break; }
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by_inst = {}
for r in rows:
    by_inst.setdefault(r["instance"], []).append(r["client"])
meds = sorted(statistics.median(v) for v in by_inst.values())
sys.exit(0 if len(meds) >= 2 and meds[-1] / meds[0] >= 1.25 else 1)
PY
done

echo "=== RESULT ==="
python3 - "$OUT" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  no rows"); raise SystemExit
by_inst = {}
for r in rows:
    by_inst.setdefault(r["instance"], []).append(r)

print(f"  {'inst':>4s} {'n':>3s} {'client':>8s} {'server':>8s} {'gap':>8s}")
summary = []
for inst in sorted(by_inst):
    g = by_inst[inst]
    c = statistics.median([r["client"] for r in g])
    s = statistics.median([r["server"] for r in g])
    gp = statistics.median([r["gap"] for r in g])
    summary.append((inst, c, s, gp))
    print(f"  {inst:4d} {len(g):3d} {c:8.3f} {s:8.3f} {gp:8.3f}")

meds = sorted(x[1] for x in summary)
if len(meds) < 2 or meds[-1] / meds[0] < 1.25:
    print(f"  only one mode observed ({meds[0]:.3f}-{meds[-1]:.3f}s across "
          f"{len(summary)} instance(s)); the slow mode did not occur, so this run")
    print("  does not decide where it lives. Re-run, or raise MAX_INSTANCES.")
    raise SystemExit
cut = (meds[0] + meds[-1]) / 2
fast = [x for x in summary if x[1] <= cut]
slow = [x for x in summary if x[1] > cut]
fs, ss = statistics.median([x[2] for x in fast]), statistics.median([x[2] for x in slow])
fg, sg = statistics.median([x[3] for x in fast]), statistics.median([x[3] for x in slow])
print(f"  fast instances ({len(fast)}): server {fs:.3f}  gap {fg:.3f}")
print(f"  slow instances ({len(slow)}): server {ss:.3f}  gap {sg:.3f}")
d_server, d_gap = ss - fs, sg - fg
print(f"  the slow mode adds {d_server*1000:+.0f} ms of server time and {d_gap*1000:+.0f} ms of gap")
# Derive the verdict from the numbers rather than asserting one.
if abs(d_server) > 2 * abs(d_gap):
    print("  VERDICT: it is COMPUTE -- sglang really is slower in the slow mode,")
    print("           and the fix belongs in the denoise/encode/VAE path.")
elif abs(d_gap) > 2 * abs(d_server):
    print("  VERDICT: it is AFTER the clock stops -- output materialisation,")
    print("           empty_cache or transport, none of which sglang's own number sees.")
else:
    print("  VERDICT: both move; neither dominates. Needs a stage-level breakdown.")
PY
echo "=== ZIMAGE_BIMODAL_DONE $(date -Is) ==="
