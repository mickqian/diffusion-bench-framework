#!/usr/bin/env bash
# Competitor A/B pass of the 2026-10-22 round (RunPod 4xB300), before the matrix:
# each arm is a candidate command from the round's recipe audit (vLLM-Omni,
# LightX2V, ComfyUI, FastH3 V2, TRT-LLM's attention backend), measured against the
# profile B300 currently selects ("b300"). Winners are promoted into the case's
# Blackwell profile, then scripts/run_b300_full_20261022.sh measures every cell
# through its profile.
#
#   bash scripts/run_b300_ab_20261022.sh
#   AB_CASES='cosmos3.* trtllm-visual|ltx2.3.* vllm-omni| sglang$' bash scripts/run_b300_ab_20261022.sh  # second wave
#   WAVE=4 bash scripts/run_b300_ab_20261022.sh          # only the lines after `wave 4`
#   AB_DRY_RUN=1 WAVE=4 bash scripts/run_b300_ab_20261022.sh  # list them, run nothing
#
# WAVE selects a section; AB_CASES (a regex on "<case> <framework>") narrows it
# further. A regex cannot tell two waves' lines for one cell apart (the third and
# fourth waves both have `minimax_h3_ref2va_5s trtllm-visual`), so select waves
# with WAVE. Every line keeps its port whatever is selected.
# Arms print as `ARM <case> <framework> <arm> r<k> RESULT ...` lines.
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/r1022}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs-ab}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_SGLANG_PYTHONPATH="${DBF_SGLANG_PYTHONPATH:-${DBF_STATE_DIR}/sgl/python}"
export DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"

if [[ -z "${AB_DRY_RUN:-}" ]]; then
  # shellcheck source=scripts/gpu_job_lock.sh
  source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
  gpu_lock_acquire
fi

wave=1
wave() { wave=$1; }  # the section the following arms lines belong to

arms() {  # <case> <framework> <rounds> <arms...>; WAVE=<n> / AB_CASES=<regex> select a subset
  local case_id=$1 fw=$2 rounds=$3
  shift 3
  if [[ -n "${WAVE:-}" && "${wave}" != "${WAVE}" ]] \
    || [[ -n "${AB_CASES:-}" && ! "${case_id} ${fw}" =~ ${AB_CASES} ]]; then
    port=$((port + 100))
    return
  fi
  if [[ -n "${AB_DRY_RUN:-}" ]]; then
    echo "PLAN wave=${wave} port=${port} ${case_id} ${fw} rounds=${rounds} arms=$*"
  else
    ARMS_CASE="${case_id}" ARMS_FW="${fw}" ARMS="$*" ROUNDS="${rounds}" \
      bash "${DBF_REPO_DIR}/scripts/run_competitor_arms.sh" "${TAG}_${fw}" "${port}"
  fi
  port=$((port + 100))
}

export ARMS_HW=b300 GPUS=0,1,2,3 TAG=ab1022b300
port=40000  # 100 ports per arms line; 63000 overflowed 65535 at the 27th line
echo "=== AB START $(date -Is) | $(git -C "${DBF_REPO_DIR}" log --oneline -1 2>/dev/null) | WAVE=${WAVE:-all} ==="
wave 1
# images, two rounds each
arms zimage_turbo_t2i_1024 vllm-omni 2 b300 cand-usp2 cand-usp2-permute
arms zimage_turbo_t2i_1024 lightx2v 2 b300 cand-fusion
arms cosmos3_nano_t2i_720p vllm-omni 2 b300 cand-dynamic
arms qwen_image_21_t2i_1024 vllm-omni 2 b300 cand-static
arms qwen_image_21_t2i_1024 lightx2v 2 b300 cand-vae-compile
arms qwen_image_21_t2i_1024 comfyui 2 b300 cand-compile-gpuonly
arms qwen_image_21_t2i_1024_2gpu vllm-omni 2 b300 cand-static
arms qwen_image_21_edit_1024 vllm-omni 2 b300 cand-static
arms ideogram4_t2i_1024_2gpu_tp comfyui 2 b300 cand-compile-gpuonly
# videos, one round each
arms ltx2_twostage_t2v trtllm-visual 1 b300 default
arms minimax_h3_t2va_5s trtllm-visual 1 b300 default
arms minimax_h3_t2va_5s vllm-omni 1 b300 cand-static
arms minimax_h3_fasth3_v2_t2va_5s vllm-omni 1 b300 cudnn-vsa-candidate
arms cosmos3_nano_t2v_720p_189f vllm-omni 1 b300 cand-permute
arms wan22_t2v_a14b_720p vllm-omni 1 b300 cand-hsdp cand-permute cand-devpost
arms wan22_t2v_a14b_720p lightx2v 1 b300 cand-tuned
arms wan22_t2v_a14b_720p trtllm-visual 1 b300 default
wave 2
# second wave, after the matrix: NVIDIA's Blackwell Cosmos3 config runs FA4, the profiles VANILLA
arms cosmos3_nano_t2i_720p trtllm-visual 2 b300 cand-fa4 cand-fa4-graph cand-graph
arms cosmos3_nano_t2v_720p_189f trtllm-visual 1 b300 cand-fa4
arms cosmos3_nano_i2v_720p_189f trtllm-visual 1 b300 cand-fa4
# the new vLLM-Omni LTX-2.3 two-stage cell, validated once before the matrix reruns it
arms ltx2.3_twostage_t2v_2gpus vllm-omni 1 b300
# sglang: 7d2e355 moved four B300 cells onto GB300 profiles, measured behind a Grace host. LTX-2
# under them read 9.9 s on B300 against 4.4 s server-side eager on x86 B200 (2026-09-25), so each
# GB300 profile is A/B'd here against the profile B300 selected before.
arms ltx2_twostage_t2v sglang 2 gb300-cfg-parallel-compile default cand-bcg cand-usp2 cand-usp2-bcg
arms ltx2.3_twostage_t2v_2gpus sglang 2 default cand-bcg cand-usp2 cand-usp2-bcg
arms cosmos3_nano_t2i_720p sglang 2 gb300-2gpu-cfg-parallel-compile blackwell-2gpu-cfg-parallel
arms qwen_image_21_t2i_1024_2gpu sglang 2 gb300-2gpu-tp2 b200-2gpu-ulysses-fa
arms cosmos3_nano_t2v_720p_189f sglang 1 gb-4gpu-cfg-ulysses-speed default
wave 3
# third wave: TRT-LLM H3 Ref2VA pads its odd sequence under Ulysses 2 and VANILLA falls to masked SDPA
arms minimax_h3_ref2va_5s trtllm-visual 1 b300 cand-fa4
wave 4
# fourth wave (2026-10-23): what three workload-equivalence audits of the matrix flagged, each
# against the profile B300 selects. The H3 cells now send sglang 49 forwards, and every request
# tags its negative prompts and asks for PNG, so every arm here runs on the fixed harness.
# Images, two rounds each.
arms qwen_image_21_t2i_1024 vllm-omni 2 b300 cand-trtllm-attn
arms qwen_image_21_t2i_1024_2gpu vllm-omni 2 b300 cand-tp2 cand-usp2-permute
arms cosmos3_nano_t2i_720p trtllm-visual 2 b300 cand-cudnn cand-cutedsl
arms qwen_image_21_t2i_1024 lightx2v 2 b300 cand-sdpa
arms zimage_turbo_t2i_1024 lightx2v 2 b300 cand-sdpa
arms cosmos3_nano_t2i_720p lightx2v 2 b300
arms cosmos3_nano_t2i_720p sglang 2 b300 cand-autoattn
arms zimage_turbo_t2i_1024 sglang 2 b300 cand-speed
arms ideogram4_t2i_1024_2gpu_tp sglang 2 b300 cand-speed
arms qwen_image_21_t2i_1024 sglang 2 b300 cand-bcg
arms qwen_image_21_t2i_1024_2gpu sglang 2 b300 cand-bcg
arms qwen_image_21_edit_1024 sglang 2 b300 cand-bcg
# videos, one round each
arms wan22_t2v_a14b_720p vllm-omni 1 b300 cand-trtllm-attn
arms minimax_h3_t2va_5s vllm-omni 1 b300 cand-usp2
arms minimax_h3_ref2va_5s vllm-omni 1 b300 cand-usp2
arms ltx2_twostage_t2v vllm-omni 1 b300 cand-vaepp
arms ltx2.3_twostage_t2v_2gpus vllm-omni 1 b300 cand-vaepp
arms minimax_h3_fasth3_v2_t2va_5s vllm-omni 1 b300 cand-cudnn-vsa
arms wan22_t2v_a14b_720p trtllm-visual 1 b300 cand-cudnn cand-trtllm cand-cutedsl
arms cosmos3_nano_t2v_720p_189f trtllm-visual 1 b300 cand-cudnn
arms minimax_h3_t2va_5s trtllm-visual 1 b300 cand-omp8
arms minimax_h3_t2va_5s_4gpu trtllm-visual 1 b300 cand-fa4 cand-fa4-omp8
# the third wave's line (it died on the port overflow) folded in, with FA4 + OMP on top
arms minimax_h3_ref2va_5s trtllm-visual 1 b300 cand-fa4 cand-fa4-omp8
arms wan22_t2v_a14b_720p lightx2v 1 b300 cand-sdpa
arms cosmos3_nano_t2v_720p_189f lightx2v 1 b300
arms cosmos3_nano_i2v_720p_189f lightx2v 1 b300
arms minimax_h3_t2va_5s fastvideo 1 b300 cand-eager
arms minimax_h3_t2va_5s_4gpu fastvideo 1 b300 cand-eager
arms wan22_t2v_a14b_720p sglang 1 b300 cand-noexp cand-kvg cand-kvg-noexp cand-fa4attn
arms cosmos3_nano_t2v_720p_189f sglang 1 b300 cand-noexp cand-kvg cand-kvg-noexp cand-autoattn
arms cosmos3_nano_i2v_720p_189f sglang 1 b300 cand-noexp cand-kvg cand-kvg-noexp cand-autoattn
arms minimax_h3_t2va_5s sglang 1 b300 cand-noexp cand-fa4attn
arms minimax_h3_ref2va_5s sglang 1 b300 cand-noexp cand-fa4attn
arms minimax_h3_t2va_5s_4gpu sglang 1 b300 cand-noexp
echo "=== AB DONE $(date -Is) ==="
