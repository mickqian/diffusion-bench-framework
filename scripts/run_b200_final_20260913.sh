#!/usr/bin/env bash
# The publishable 4xB200 run: every cell, one harness revision, one set of
# framework versions.
#
# This wraps scripts/run_b200_cross_framework_20260912.sh with the settings the
# published run used, so `--reproduce` points at something complete rather than
# at a case list whose environment lived only in a shell on the box.
#
# Why a full re-run rather than repairing cells: the matrix assembled over the
# previous day mixed four harness revisions and two LightX2V dependency stacks,
# and three of the changes move the measured number --
#   * the HTTP client read responses 10 KiB at a time (qwen's sglang spread:
#     61% -> 9%)
#   * warmup converged on a reduced-step shape nobody measures, so the measured
#     window opened cold
#   * measured requests carried a perf dump only sglang was asked for: +233 ms
#     on the first and +1.35% on every one after, charged to sglang alone
# and LightX2V moved transformers 4.57 -> 5.17 / huggingface_hub 0.36 -> 1.31
# (which is what un-broke its FLUX.2 cell). A published artifact carries ONE
# framework_runtime block; it cannot be true of cells measured on two stacks.
#
# Prerequisites on the box:
#   * /personal/bench0912/.hftoken  -- the gated models abort without it
#   * /personal/bench0912/fw-venvs  -- per-framework venvs, already installed
#   * /scratch/dbf2                 -- a checkout that is NOT the one being edited
#
#   scripts/run_b200_final_20260913.sh
#
# Then merge and publish (the --reproduce path is existence-checked):
#   python3 -m diffusion_bench.build_report_artifacts \
#       --results <DBF_LOG_DIR>/run_b200final0913_*.json \
#       --config configs/comparison_configs.json \
#       --output-json tmp/report/merged.json \
#       --dashboard-md tmp/report/dashboard.md --issue-md tmp/report/issue.md \
#       --run-id b200x4-final-20260913
#   python3 scripts/publish_bench_run.py --merged tmp/report/merged.json \
#       --run-id b200x4-final-20260913 \
#       --label "4xB200 cross-framework (latest-vs-latest, all cells re-measured on one harness)" \
#       --gpu "4x NVIDIA B200 183GB" \
#       --reproduce scripts/run_b200_final_20260913.sh \
#       --note harness="..." --note jitter="..."
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/personal/bench0912}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-/scratch/dbf2}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-/cluster-storage/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${DBF_HF_HOME}}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-blackwell}"
# wan22 needs the headroom, and the harness checkpoints after each framework, so
# a case that does run long can no longer erase the frameworks that finished.
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-14400}"
export TAG="${TAG:-b200final0913}"
export GPUS="${GPUS:-0,1,2,3}"
export BASE_PORT="${BASE_PORT:-64001}"

exec bash "${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"
