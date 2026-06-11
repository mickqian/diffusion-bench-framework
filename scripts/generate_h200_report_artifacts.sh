#!/usr/bin/env bash
set -euo pipefail

# Merge selected H200 result JSONs and regenerate the formal report artifacts.

CONFIG="${CONFIG:-configs/comparison_configs.json}"
OUTPUT_JSON="${OUTPUT_JSON:-tmp/report/h200-framework-comparison-merged.json}"
DASHBOARD_MD="${DASHBOARD_MD:-${OUTPUT_JSON%.json}.dashboard.md}"
ISSUE_MD="${ISSUE_MD:-${OUTPUT_JSON%.json}.issue.md}"
IMAGE_PNG="${IMAGE_PNG:-${OUTPUT_JSON%.json}.png}"
IMAGE_SVG="${IMAGE_SVG:-${OUTPUT_JSON%.json}.svg}"
RUN_ID="${RUN_ID:-$(basename "${OUTPUT_JSON%.json}")}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${RESULTS:-}" ]]; then
  read -r -a RESULT_FILES <<< "${RESULTS}"
else
  RESULT_FILES=(
    tmp/report/combined-h200-20260510-single-e2e-be2e4d2-resfix.json
    tmp/report/h200-wan-vllm-omni-20260514-c7916aa-ion7.json
    tmp/report/h200-throughput-common-20260514-d13d74e.json
    tmp/report/h200-ltx-lightx2v-20260513-6ee98c7.json
    tmp/report/h200-ltx23-lightx2v-2gpu-tp-20260513-6e6ae7e.json
    reports/h200-video-parallelism-refresh-20260611/raw/h200-video-parallelism-refresh-20260611-vllm-omni-h200-2gpu-cfg.json
    reports/h200-video-parallelism-refresh-20260611/raw/h200-video-parallelism-refresh-20260611-lightx2v-h200-fa3.json
    reports/h200-video-parallelism-refresh-20260611/raw/h200-video-parallelism-refresh-20260611-sglang-h200-2gpu-cfg-bf16-vae-resident.json
  )
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

"${PYTHON_BIN}" -m diffusion_bench.build_report_artifacts \
  --config "${CONFIG}" \
  --results "${RESULT_FILES[@]}" \
  --output-json "${OUTPUT_JSON}" \
  --dashboard-md "${DASHBOARD_MD}" \
  --issue-md "${ISSUE_MD}" \
  --image-png "${IMAGE_PNG}" \
  --image-svg "${IMAGE_SVG}" \
  --run-id "${RUN_ID}"
