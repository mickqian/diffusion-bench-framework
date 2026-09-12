#!/usr/bin/env bash
# Reproduce the 2026-09-12 Blackwell cross-framework run (run id
# b200x4-fair-20260912) on a 4x B200 rx devbox.
#
# This is the recipe, not a transcript: the published numbers were gathered over
# several passes while framework breakage was diagnosed and fixed, but every
# published cell is what THIS script produces against the config at the run's
# commit. Diagnosis (attention-backend probes, vLLM-Omni CLI migration, the
# LightX2V request-schema change) is recorded in the case profiles and the
# `diffusion-framework-benchmarking` skill, not here.
#
# Prerequisites on the box:
#   - sglang importable from its own checkout (origin/main); competitors
#     installed into isolated venvs by scripts/install_comparison_frameworks.sh
#     with every *_INSTALL_SPEC overridden to latest (latest-vs-latest policy)
#   - an HF token whose account has accepted the gated licences for
#     black-forest-labs/FLUX.1-dev, FLUX.2-dev and ideogram-ai/ideogram-4-fp8,
#     written to $DBF_STATE_DIR/.hftoken -- without it those three cases abort
#     at sglang startup with "Could not get model info" (a 403, not a bug)
#   - HF_HOME *and* HUGGINGFACE_HUB_CACHE both pointing at the model cache
#
#   bash scripts/run_b200_cross_framework_20260912.sh
#
# Results land in $DBF_LOG_DIR as run_<tag>_<case>.json; publish with
# build_report_artifacts + publish_bench_run.py --reproduce <this file>.
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/personal/bench0912}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-/scratch/dbf2}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-/cluster-storage/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${DBF_HF_HOME}}"
export DBF_FRAMEWORKS="${DBF_FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual}"
export DBF_HARDWARE_PROFILE="${DBF_HARDWARE_PROFILE:-blackwell}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-5400}"

# All four GPUs: several video cases declare num_gpus 4, and handing the runner
# fewer produces `CUDA error: invalid device ordinal` rather than a clear
# message.
GPUS="${GPUS:-0,1,2,3}"
TAG="${TAG:-b200fair0912}"

# Image cases first: they are minutes rather than tens of minutes, so a broken
# framework surfaces early instead of after the video block.
CASES="${CASES:-
zimage_turbo_t2i_1024
flux1_dev_t2i_1024
flux2_dev_t2i_1024
qwen_image_2512_t2i_1024
qwen_image_2512_t2i_1024_truecfg
qwen_image_edit_2511
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

echo "=== B200 cross-framework run START $(date -Is) ==="
echo "=== repo=$DBF_REPO_DIR frameworks=[$DBF_FRAMEWORKS] hw=$DBF_HARDWARE_PROFILE gpus=$GPUS ==="
port="${BASE_PORT:-44001}"
for case_id in $CASES; do
    echo "=== $(date -Is) case $case_id ==="
    # Throughput is opt-in per case (configs/benchmark/workloads.json); the
    # runner drops the mode for cases that have not opted in, so passing both
    # modes here runs single_e2e everywhere and throughput only on the
    # representative pair.
    bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" \
        "${TAG}_${case_id}" "$GPUS" "$port" "single_e2e throughput" "$case_id"
    grep -h "^RESULT" "${DBF_LOG_DIR}/run_${TAG}_${case_id}.log" 2>/dev/null
    port=$((port + 4))
done
echo "=== B200 cross-framework run DONE $(date -Is) ==="
