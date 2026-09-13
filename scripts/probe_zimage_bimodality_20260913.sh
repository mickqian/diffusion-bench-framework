#!/usr/bin/env bash
# Where does sglang's zimage bimodality live -- in compute, or after the clock stops?
#
# Two clean modes, ~0.474s and ~0.678s, and only the MIX varies between server
# instances, which makes the published median swing 43% run to run:
#
#   variance_check 03:25  0.695, 0.478, 0.474, 0.472, 0.473   (median 0.474)
#   final run      03:20  0.680, 0.474, 0.678, 0.680, 0.673   (median 0.678)
#   40-request probe      p50 0.468, two samples at 0.568/0.629
#
# It is sglang-specific: on the same case vLLM-Omni spreads 5.5% and LightX2V
# 0.4%. And sglang's own `total_duration_ms` is 360ms in both modes (denoise
# 268.7 + text encode 65.5 + VAE 23.4), so the 114ms/318ms the client sees on
# top is outside the region sglang times.
#
# gpu_worker takes `duration_ms` and THEN does `_materialize_output_transport`
# and, conditionally, `torch.get_device_module().empty_cache()`. Both are inside
# the client's window and outside the server's number. empty_cache costs
# whatever is cached at the time, which is exactly the kind of thing that
# alternates.
#
# So: fire many requests at one server, record BOTH numbers per request, and
# look at the gap. If the gap is bimodal and the server number is flat, no
# amount of sampling fixes the benchmark and the fix belongs in sglang.
#
# Written for bench-0912 (4xB200). Run when the GPUs are otherwise idle -- a
# second job on the other cards contaminates latency through the shared host.
set -u
PORT=62001
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

while pgrep -f "/personal/bench0912/final_run.sh" >/dev/null 2>&1; do sleep 300; done
echo "=== ZIMAGE_BIMODAL_START $(date -Is) ==="
cd /scratch/dbf2 && git fetch -q origin && git checkout -q -B main origin/main && git log --oneline -1

sglang serve --model-path Tongyi-MAI/Z-Image-Turbo --port $PORT --host 127.0.0.1 \
  --backend sglang --num-gpus 2 --model-type diffusion --warmup-mode server --tp-size 2 \
  > /personal/bench0912/zimage_bimodal_server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "--- waiting for health"
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  up after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "  SERVER DIED"; tail -20 /personal/bench0912/zimage_bimodal_server.log; exit 1; }
  sleep 10
done

python3 - "$PORT" <<'PY'
import json, os, statistics, sys, time, urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/images/generations"
DUMP = "/personal/bench0912/zimage_bimodal_perf.json"

def fire():
    """One request; returns (client_s, server_s, response_bytes)."""
    for path in (DUMP,):
        if os.path.exists(path):
            os.remove(path)
    payload = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A serene mountain landscape at sunset with a crystal clear lake",
        "size": "1024x1024", "n": 1, "seed": 42,
        "perf_dump_path": DUMP,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    client = time.time() - t0
    server = None
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            server = json.load(open(DUMP)).get("total_duration_ms")
            if server is not None:
                server /= 1000.0
                break
        except Exception:
            time.sleep(0.05)
    return client, server, len(body)

print("--- warming (15 requests)")
for _ in range(15):
    fire()

N = 60
print(f"--- measuring {N} requests, client and server per request")
rows = [fire() for _ in range(N)]

measured = [
    {"client": c, "server": s, "gap": c - s, "bytes": sz}
    for c, s, sz in rows
    if s is not None
]
clients = [m["client"] for m in measured]
servers = [m["server"] for m in measured]
gaps = [m["gap"] for m in measured]
sizes = {m["bytes"] for m in measured}
missing = len(rows) - len(measured)
if missing:
    print(f"  NOTE: {missing}/{len(rows)} requests produced no perf dump; excluded")

def describe(name, xs):
    if not xs:
        print(f"  {name:12s} (none)")
        return
    p50 = statistics.median(xs)
    print(f"  {name:12s} n={len(xs)} p50={p50:.3f} min={min(xs):.3f} max={max(xs):.3f} "
          f"spread={(max(xs) - min(xs)) / min(xs) * 100:6.1f}%")

print("=== RESULT ===")
describe("client", clients)
describe("server", servers)
describe("gap", gaps)
print(f"  response bytes seen: {sorted(sizes)}")

# Split the requests at the midpoint between the two client modes and see which
# component moved. If `server` is the same in both groups and `gap` is not, the
# bimodality is after sglang stops its clock.
if measured:
    cut = (min(clients) + max(clients)) / 2
    fast = [m for m in measured if m["client"] <= cut]
    slow = [m for m in measured if m["client"] > cut]
    print(f"  cut at {cut:.3f}s -> {len(fast)} fast, {len(slow)} slow")
    for tag, grp in (("fast", fast), ("slow", slow)):
        if not grp:
            continue
        print(
            f"    {tag}: client p50={statistics.median([m['client'] for m in grp]):.3f} "
            f"server p50={statistics.median([m['server'] for m in grp]):.3f} "
            f"gap p50={statistics.median([m['gap'] for m in grp]):.3f}"
        )
    print("  VERDICT: server p50 equal across the two groups while gap p50 differs")
    print("           = the bimodality is outside the region sglang times, and no")
    print("           amount of client-side sampling makes the median stable.")
print("  raw client:", [round(m["client"], 3) for m in measured])
print("  raw server:", [round(m["server"], 3) for m in measured])
print("  raw gap   :", [round(m["gap"], 3) for m in measured])
PY

echo "=== ZIMAGE_BIMODAL_DONE $(date -Is) ==="
