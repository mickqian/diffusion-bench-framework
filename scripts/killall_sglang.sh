#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f "sglang|sgl_diffusion|vllm serve|lightx2v.server" 2>/dev/null || true
sleep 2
pkill -KILL -f "sglang|sgl_diffusion|vllm serve|lightx2v.server" 2>/dev/null || true

