#!/usr/bin/env bash
# Can a default 2-GPU sglang server answer a request that turns CFG off?
#
# `sglang serve --model-path Qwen/Qwen-Image-2512 --num-gpus 2 --model-type
# diffusion` passes no parallelism flags, so the runtime AUTO-enables CFG
# parallelism from the model's default sampling params. Every guidance_scale=1.0
# request was then rejected -- by a guard whose message blamed a flag the user
# never passed. The profile-vs-default study hit this on two cells
# (qwen_image_2512_t2i_1024, qwen_image_edit_2511): "default FAILS".
#
# Unit tests can show the validation stage stops raising. They cannot show the
# request completes, which is the claim. So run both arms against a real server:
#
#   base    an untouched copy of the box's sglang  -> expect HTTP 400
#   patched the same copy plus the fix             -> expect an image
#
# Both arms are the same checkout, the same weights, the same request; the only
# difference is the patch. Written for bench-0912 (4xB200).
set -u
PORT_BASE="${PORT_BASE:-45001}"
STEPS="${STEPS:-4}"
MODEL="${MODEL:-Qwen/Qwen-Image-2512}"
BASE_TREE="${BASE_TREE:-/tmp/sgl-cfgbase}"
PATCHED_TREE="${PATCHED_TREE:-/tmp/sgl-cfgtest}"
export HF_HOME=/cluster-storage/models HUGGINGFACE_HUB_CACHE=/cluster-storage/models
export HF_TOKEN="$(cat /personal/bench0912/.hftoken)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

source "${DBF_REPO_DIR:-/scratch/dbf2}/scripts/gpu_job_lock.sh"
gpu_lock_acquire
echo "=== CFG_SINGLE_BRANCH_START $(date -Is) ==="

i=0
for ARM in "base:${BASE_TREE}" "patched:${PATCHED_TREE}"; do
  tag="${ARM%%:*}"; tree="${ARM#*:}"
  i=$((i+1)); PORT=$((PORT_BASE + i*2))
  [ "$PORT" -gt 65535 ] && { echo "  port $PORT out of range"; exit 1; }
  LOG="/personal/bench0912/cfgsb_${tag}.log"
  echo "--- arm ${tag}: tree=${tree} port=${PORT}"
  if [ ! -d "${tree}/python/sglang" ]; then
    echo "    missing ${tree}/python/sglang -- stage it first"; continue
  fi
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1 2>/dev/null \
    | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5

  # The arm is only meaningful if the server imports the tree we think it does.
  echo "    imports: $(PYTHONPATH="${tree}/python" /opt/sglang/bin/python3 \
      -c 'import sglang; print(sglang.__file__)' 2>/dev/null | tail -1)"

  # The exact command from the bug report; PYTHONPATH decides which tree it
  # imports, and the line above prints which one that turned out to be.
  PYTHONPATH="${tree}/python" sglang serve \
    --model-path "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --model-type diffusion --num-gpus 2 --warmup-mode server \
    > "$LOG" 2>&1 &
  SERVER_PID=$!

  up=0; waited=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    # A server that started somewhere this poll cannot reach looks identical to
    # one that never started, for the whole budget. Its own readiness line is
    # the signal that waiting longer is pointless.
    if grep -q "fired up and ready to roll" "$LOG" 2>/dev/null; then
      echo "    ready but 127.0.0.1:$PORT does not answer; it bound:"
      (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -i sgl | head -3 | sed 's/^/      /'
      break
    fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "$up" != 1 ]; then
    echo "    never became healthy after ${waited}s"; tail -5 "$LOG" | sed 's/^/      /'
    kill -9 "$SERVER_PID" 2>/dev/null; sleep 5; continue
  fi
  echo "    healthy after ${waited}s"
  echo "    auto-selected: $(grep -hoE '"(tp_size|enable_cfg_parallel|cfg_parallel_degree)": [^,}]*' "$LOG" | sort -u | tr '\n' ' ')"

  python3 - "$PORT" "$tag" "$MODEL" "$STEPS" <<'PY'
import json, sys, time, urllib.error, urllib.request

port, tag, model, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
url = f"http://127.0.0.1:{port}/v1/images/generations"


def fire(label, **extra):
    payload = {
        "model": model,
        "prompt": "A warehouse robot folds a blue cloth on a clean workbench.",
        "size": "1024x1024", "n": 1, "seed": 0,
        "response_format": "b64_json", "num_inference_steps": steps,
    }
    payload.update(extra)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            body = r.read()
        print(f"    {tag}/{label}: HTTP {r.status}, {len(body)} bytes, {time.time()-t0:.2f}s")
        return "ok"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        print(f"    {tag}/{label}: HTTP {e.code} after {time.time()-t0:.2f}s :: {detail[:220]}")
        return f"http{e.code}"
    except Exception as e:  # noqa: BLE001 - a timeout here is a result, not a crash
        print(f"    {tag}/{label}: {type(e).__name__} after {time.time()-t0:.2f}s :: {str(e)[:160]}")
        return type(e).__name__


# The request the study saw rejected, and a CFG request to show the fix does not
# cost the case cfg-parallel exists for.
single = fire("guidance_scale=1.0", guidance_scale=1.0)
cfg = fire("true_cfg_scale=4.0", guidance_scale=1.0, true_cfg_scale=4.0,
           negative_prompt="blurry, low quality")
print(f"    {tag}: RESULT single_branch={single} cfg={cfg}")
PY

  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 12); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 5; done
  kill -9 "$SERVER_PID" 2>/dev/null
  sleep 5
done
echo "  Read it as: base single_branch=http400 and patched single_branch=ok is the"
echo "  fix. Both arms must keep cfg=ok, or the fix cost the CFG case."
echo "=== CFG_SINGLE_BRANCH_DONE $(date -Is) ==="
