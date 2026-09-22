#!/usr/bin/env bash
# LightX2V's Qwen-Image-2.1 arm, run inside LightX2V's OWN published image.
#
# The repo's install script builds flash-attn 2.8.3 from source, and on a box
# whose torch is 2.14+cu130 that build dies with "#error C++20 or later
# compatible compiler is required to use ATen" -- flash-attn's setup.py pins
# -std=c++17 and PyPI has nothing newer than 2.8.3.post1 to bump to. Rather
# than patch a competitor's build, run it the way upstream documents: their
# docker image plus a source clone ("it can run directly without installing").
# That is also the most faithful reading of "give each framework its best
# supported path".
#
# Config and request are upstream's own: configs/qwen_image_21/qwen_image_21*.json
# and scripts/qwen_image_21/server/post_t2i.py.
set -u
HW="${1:?usage: qi21_lx2v_image.sh <h200|rtx5090> [reps]}"
REPS="${2:-3}"

for c in /scratch /personal /workspace /root; do
  [ -d "$c" ] && [ -w "$c" ] && { BASE="$c"; break; }
done
BASE="${BASE:-/tmp}"
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub" HUGGINGFACE_HUB_CACHE="$BASE/hf/hub"
STATE="$BASE/state"; SRC="$BASE/LightX2V"; mkdir -p "$STATE" "$HF_HUB_CACHE"
PORT=58000

echo "=== QI21_LX2VIMG_START $(date -Is) hw=$HW host=$(hostname) base=$BASE ==="
python3 -c "
import torch
print('  gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability())
print('  torch', torch.__version__, 'cuda', torch.version.cuda)
for m in ('flash_attn', 'flash_attn_interface', 'flashinfer', 'sageattention'):
    try:
        __import__(m); print(f'  {m}: present')
    except Exception as e:
        print(f'  {m}: MISSING ({type(e).__name__})')
"

if [ ! -d "$SRC/.git" ]; then
  echo "--- cloning LightX2V main ---"
  git -c http.https://github.com/.extraheader= clone -q --depth 1 \
    https://github.com/ModelTC/LightX2V.git "$SRC" || { echo "  FATAL: clone failed"; exit 1; }
fi
echo "  LightX2V at $(git -C "$SRC" rev-parse --short=12 HEAD) $(git -C "$SRC" log -1 --format=%cd --date=short)"

CFG="$SRC/configs/qwen_image_21/qwen_image_21.json"
[ "$HW" = "rtx5090" ] && CFG="$SRC/configs/qwen_image_21/qwen_image_21_5090.json"
echo "--- config: $(basename "$CFG") ---"
python3 -c "
import json; d=json.load(open('$CFG'))
print('  ', json.dumps({k: d[k] for k in ('infer_steps','resolution','attn_type','enable_cfg','sample_guide_scale')}))
" || { echo "  FATAL: config missing"; exit 1; }

echo "--- weights ---"
MODEL=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen-Image-2.1'))" 2>/dev/null | tail -1)
echo "  at $MODEL"
[ -d "$MODEL" ] || { echo "  FATAL: no weights"; exit 1; }

echo "--- serving (upstream's own launch) ---"
cd "$SRC"
# base.sh is not optional: it puts the source clone on PYTHONPATH (upstream says
# the repo "can run directly without installing the package") and sets the
# allocator config their recipes assume. It reads two variables by name and
# exits if either is empty, so export them first -- and relax `set -u` while it
# runs, because it also touches variables it does not define.
export lightx2v_path="$SRC" model_path="$MODEL"
set +u
. scripts/base/base.sh || { echo "  FATAL: base.sh refused"; exit 1; }
set -u
python3 -m lightx2v.server --model_cls qwen_image_21 --model_path "$MODEL" \
  --config_json "$CFG" --host 127.0.0.1 --port $PORT > "$STATE/lx2vimg_server.log" 2>&1 &
SRV=$!
up=0
for _ in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/v1/service/status" >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SRV 2>/dev/null || break
  sleep 5
done
if [ "$up" != 1 ]; then
  echo "  server never became ready"
  grep -aoE "[A-Za-z]*Error: .{0,150}|out of memory.{0,60}|Traceback|ModuleNotFound.{0,80}" \
    "$STATE/lx2vimg_server.log" | sort -u | head -6 | sed 's/^/    /'
  kill -9 $SRV 2>/dev/null
  echo "=== QI21_LX2VIMG_DONE $(date -Is) ==="; exit 1
fi
echo "  ready"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | sed 's/^/  vram after load: /'

python3 - "$PORT" "$HW" "$REPS" <<'PY'
import json, statistics, sys, time, urllib.request
port, hw, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://127.0.0.1:{port}/v1/tasks/image/sync"
# upstream's own client payload (scripts/qwen_image_21/server/post_t2i.py),
# with this benchmark's prompt. Steps/resolution/CFG come from the launch
# config, which is how LightX2V is meant to be driven.
payload = {"task": "t2i", "seed": 42, "size": [1024, 1024],
           "prompt": "A warehouse robot folds a blue cloth on a clean workbench."}
def fire():
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        n = len(r.read())
    return time.time() - t0, n
try:
    _, nbytes = fire()
    lats = [fire()[0] for _ in range(reps)]
    print(f"  RESULT lightx2v {hw} p50={statistics.median(lats):.3f}s "
          f"{[round(x,2) for x in lats]} bytes={nbytes}")
except Exception as exc:
    body = ""
    if hasattr(exc, "read"):
        try: body = exc.read().decode()[:220]
        except Exception: pass
    print(f"  REQUEST FAILED: {type(exc).__name__}: {str(exc)[:140]} {body}")
PY
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | sed 's/^/  vram peak-ish: /'
grep -aoE "infer_steps[^,}]*|step [0-9]+/[0-9]+|[0-9.]+ *s/it|it/s" "$STATE/lx2vimg_server.log" | tail -3 | sed 's/^/  /'
kill $SRV 2>/dev/null; sleep 10; kill -9 $SRV 2>/dev/null
echo "=== QI21_LX2VIMG_DONE $(date -Is) ==="
