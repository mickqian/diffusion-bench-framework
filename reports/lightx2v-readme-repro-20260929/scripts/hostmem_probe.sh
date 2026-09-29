#!/bin/bash
# Host memory the --performance-mode memory recipe needs: MemAvailable on the
# idle box, then its minimum from launch until 60 s after the server is ready
# (warmup included), for 1 and 8 GPUs.
set -u
export HF_HOME=/persistent/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
avail_mib() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
probe() {
    local tag=$1 port=$2; shift 2
    local log=/persistent/logs/hostmem_${tag}.log
    local idle; idle=$(avail_mib)
    setsid nohup sglang serve --model-path "$MODEL" --port "$port" \
        --performance-mode memory "$@" > "$log" 2>&1 < /dev/null &
    local pid=$! low=$idle ready_at=""
    while kill -0 "$pid" 2>/dev/null; do
        local now; now=$(avail_mib); [ "$now" -lt "$low" ] && low=$now
        if [ -z "$ready_at" ] && curl -sf "http://127.0.0.1:$port/health" > /dev/null; then
            ready_at=$(date +%s)
        fi
        if [ -n "$ready_at" ] && [ $(( $(date +%s) - ready_at )) -ge 60 ]; then break; fi
        sleep 2
    done
    kill -TERM -- -"$pid" 2>/dev/null; sleep 20; kill -KILL -- -"$pid" 2>/dev/null
    echo "HOSTMEM $tag idle_avail_mib=$idle min_avail_mib=$low used_gib=$(( (idle - low) / 1024 )) ready=${ready_at:+yes}"
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 5; done
    sleep 30
}
echo "tree $(git -C /sgl-workspace/sglang rev-parse --short HEAD)"
probe 1gpu 32001
probe 8gpu 32011 --num-gpus 8 --enable-cfg-parallel --ulysses-degree 4
echo "HOSTMEM_DONE $(date -u +%FT%TZ)"
