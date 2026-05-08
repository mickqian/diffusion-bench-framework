#!/usr/bin/env bash
set -euo pipefail

PATTERN="sglang serve|sglang-diffusionWorker|vllm serve|lightx2v.server|torchrun.*lightx2v.server"

pkill -TERM -f "${PATTERN}" 2>/dev/null || true
sleep 2
pkill -KILL -f "${PATTERN}" 2>/dev/null || true
