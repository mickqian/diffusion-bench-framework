#!/usr/bin/env bash
# The 2026-09-25 4xB200 round: every case, five frameworks, one harness
# revision, latest-vs-latest.
#
# Wraps scripts/run_b200_cross_framework_20260912.sh (the case list and the
# per-case runner) with this round's settings, as scripts/run_b200_final_20260913.sh
# did for the last published round. What differs from that round:
#   * ComfyUI is a fifth framework: workflows under configs/benchmark/comfyui/,
#     each run once at 2 steps by scripts/smoke_comfyui_workflows.py before this
#   * qwen_image_21_t2i_1024 joins the image block
#   * framework venvs live in a FRESH root. `venv --clear` over an existing tree
#     on /personal failed in the 09-15 biweekly run ("Directory not empty:
#     'torch'") and took every vLLM-Omni cell with it
#   * LightX2V's venv holds torch 2.11.0 -- the version LightX2V's own image
#     ships -- through PIP_CONSTRAINT, because flash-attn 2.8.3 does not compile
#     against torch 2.14's headers (std::strong_ordering needs C++20)
#   * the HF cache is a writable overlay on /scratch: /cluster-storage/models is
#     read-only on this cluster with empty snapshots/, so every shared blob is
#     symlinked in and only the files it lacks are downloaded
#
# Installed for this round with scripts/install_comparison_frameworks.sh,
# versions resolved 2026-09-25 ~07:15 UTC the way scripts/biweekly_fair_bench.sh
# resolves them and pinned:
#   sglang origin/main 434c2e3a | vllm 0.30.0 + vllm-omni main 69de153f
#   LightX2V main a4b8ce30 | tensorrt-llm 1.3.0rc28 | ComfyUI master 88ab4a06
#
# Prerequisites on the box:
#   * /personal/bench0912/.hftoken  -- the gated models abort without it
#   * ${DBF_VENV_ROOT}              -- the five framework venvs, installed
#   * ${DBF_REPO_DIR}               -- a checkout that is NOT the one being edited
#   * ${DBF_HF_HOME}/hub            -- the cache overlay (and ComfyUI's files)
#
#   bash scripts/run_b200_full_20260925.sh
#
# Then merge and publish -- dry run first, --live only once the numbers are reviewed:
#   scripts/merge_and_publish_run.sh "<DBF_LOG_DIR>/run_b200full0925_*.json" \
#       b200x4-full-20260925 "4xB200 cross-framework, ComfyUI added (latest-vs-latest)" \
#       "4x NVIDIA B200 183GB" scripts/run_b200_full_20260925.sh
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/personal/bench0925}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-/scratch/dbf-0925}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-/scratch/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-/personal/bench0912/.hftoken}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual comfyui}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-b200}"
# wan22 already needed 4h for three frameworks; ComfyUI adds a fourth. The
# harness checkpoints after every framework, so a timeout loses one cell, not
# the case.
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-21600}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-/scratch/comfy-ws}"
export TAG="${TAG:-b200full0925}"
export GPUS="${GPUS:-0,1,2,3}"
export BASE_PORT="${BASE_PORT:-65001}"
export CASES="${CASES:-
zimage_turbo_t2i_1024
flux1_dev_t2i_1024
flux2_dev_t2i_1024
qwen_image_2512_t2i_1024
qwen_image_2512_t2i_1024_truecfg
qwen_image_edit_2511
qwen_image_21_t2i_1024
ideogram4_t2i_1024_2gpu_tp
cosmos3_nano_t2i_720p
minimax_h3_t2va_5s
minimax_h3_ref2va_5s
wan22_t2v_a14b_720p
ltx2_twostage_t2v
ltx2.3_twostage_t2v_2gpus
cosmos3_nano_t2v_720p_189f
cosmos3_nano_i2v_720p_189f
}"

cd "${DBF_REPO_DIR}" || exit 1
# The runner clears the cards between cases; nothing else may hold them.
source scripts/gpu_job_lock.sh
gpu_lock_acquire
echo "=== $(git -C "${DBF_REPO_DIR}" log --oneline -1) | sglang $(git -C /sgl-workspace/sglang log --oneline -1) ==="
bash "${DBF_REPO_DIR}/scripts/run_b200_cross_framework_20260912.sh"
