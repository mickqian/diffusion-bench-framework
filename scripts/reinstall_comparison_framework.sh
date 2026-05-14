#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK="${1:?usage: reinstall_comparison_framework.sh <vllm-omni|lightx2v>}"
export FORCE_FRAMEWORK_REINSTALL=1
exec "$(dirname "$0")/install_comparison_frameworks.sh" "${FRAMEWORK}"
