#!/usr/bin/env bash
# MP4-encode pass of the 2026-10-22 round (RunPod 4xB300, 2026-10-24). The encode rule (benchmarking
# SKILL, fairness checklist) changed what some frameworks are asked to encode in six video cells; this
# re-measures each of them through the harness: single_e2e, a fresh server per arm, arms interleaved
# over rounds with the order reversed in even rounds.
#
# Per cell, sglang runs twice with the aligned encode -- on the cell's preview stack (main plus the
# round's open branches for that model) and on sglang main -- and the preview stack once more on the
# case config from before the rule (BEFORE_CONFIG), which isolates what the encode itself costs. The
# closest competitor runs its selected B300 profile, and so does every other configurable competitor
# whose encode the rule changed (vLLM-Omni in Wan2.2 and Cosmos3 T2V, LightX2V in the H3 cells,
# ComfyUI in Wan2.2). Every MP4 an arm writes is probed afterwards (scripts/probe_mp4_encode.py), so
# each row's preset and CRF are read from the stream, not assumed.
#
#   bash scripts/run_b300_encode_align_20261024.sh                      # every cell
#   ENC_CELLS="wan cos_t2v" bash scripts/run_b300_encode_align_20261024.sh
#   ENC_DRY_RUN=1 bash scripts/run_b300_encode_align_20261024.sh        # list the arms, run nothing
#
# Trees (python dirs): SGL_MAIN (sglang main 327c9067), SGL_WAN (mick/wan-b300-stack3-w7 4be217383f),
# SGL_COS (mick/cos-b300-final2 e52f28491c), SGL_H3 (preview/open-perf-prs-20261024 3042cea8, wave 7's).
# BEFORE_CONFIG: configs/comparison_configs.json from dbf 20b4753 (the commit before the rule).
# Results: $DBF_LOG_DIR/run_enc1024_<cell>_<arm>_r<k>_<case>.json; `ARM` lines here; MP4 probes in
# $DBF_LOG_DIR/enc_probe.jsonl.
set -u

export DBF_STATE_DIR="${DBF_STATE_DIR:-/workspace/r1022}"
export DBF_REPO_DIR="${DBF_REPO_DIR:-${DBF_STATE_DIR}/dbf-enc}"
export DBF_LOG_DIR="${DBF_LOG_DIR:-${DBF_STATE_DIR}/logs-enc}"
export DBF_VENV_ROOT="${DBF_VENV_ROOT:-${DBF_STATE_DIR}/fw-venvs}"
export DBF_HF_HOME="${DBF_HF_HOME:-${DBF_STATE_DIR}/hf}"
export DBF_HF_TOKEN_FILE="${DBF_HF_TOKEN_FILE:-${DBF_STATE_DIR}/.hftoken}"
export DBF_HARDWARE_PROFILE=b300 DBF_CASE_TIMEOUT="${DBF_CASE_TIMEOUT:-7200}"
export DIFFUSION_BENCH_COMFYUI_WORKSPACE="${DIFFUSION_BENCH_COMFYUI_WORKSPACE:-${DBF_STATE_DIR}/comfy-ws}"
export DIFFUSION_BENCH_KEEP_VIDEO_DIR="${DBF_LOG_DIR}/kept-mp4"
SGL_MAIN="${SGL_MAIN:-${DBF_STATE_DIR}/sgl-ltx25/python}"
SGL_WAN="${SGL_WAN:-${DBF_STATE_DIR}/sgl-wan-w10/python}"
SGL_COS="${SGL_COS:-${DBF_STATE_DIR}/sgl-cos-fin2/python}"
SGL_H3="${SGL_H3:-${DBF_STATE_DIR}/sgl-preview/python}"
BEFORE_CONFIG="${BEFORE_CONFIG:-${DBF_STATE_DIR}/enc/comparison_configs.before.json}"
CELLS="${ENC_CELLS:-wan cos_t2v cos_i2v h3_4gpu h3 ref2va comfy_wan}"

declare -A CASE_OF=(
  [wan]=wan22_t2v_a14b_720p [cos_t2v]=cosmos3_nano_t2v_720p_189f [cos_i2v]=cosmos3_nano_i2v_720p_189f
  [h3]=minimax_h3_t2va_5s [h3_4gpu]=minimax_h3_t2va_5s_4gpu [ref2va]=minimax_h3_ref2va_5s
  [comfy_wan]=wan22_t2v_a14b_720p
)
declare -A ROUNDS_OF=([wan]=2 [cos_t2v]=2 [cos_i2v]=2 [h3_4gpu]=2 [h3]=1 [ref2va]=1 [comfy_wan]=1)

# "<arm> <framework> <sglang tree or -> <config: new|before>" per line, for <cell> <round>
cell_arms() {
  local cell=$1 round=$2 tree
  case "${cell}" in
    wan) tree=${SGL_WAN} ;; cos_t2v | cos_i2v) tree=${SGL_COS} ;; h3 | h3_4gpu | ref2va) tree=${SGL_H3} ;;
    comfy_wan) echo "comfyui comfyui - new"; return ;;
  esac
  echo "sgl-stack sglang ${tree} new"
  case "${cell}" in
    wan | cos_t2v) echo "trtllm trtllm-visual - new" ;;
    *) echo "vllm-omni vllm-omni - new" ;;
  esac
  echo "sgl-main sglang ${SGL_MAIN} new"
  echo "sgl-stack-before sglang ${tree} before"
  case "${cell}:${round}" in
    wan:1 | cos_t2v:1) echo "vllm-omni vllm-omni - new" ;;
    h3:* | h3_4gpu:* | ref2va:*) echo "lightx2v lightx2v - new" ;;
  esac
}

if [[ -z "${ENC_DRY_RUN:-}" ]]; then
  mkdir -p "${DBF_LOG_DIR}"
  # shellcheck source=scripts/gpu_job_lock.sh
  source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
  gpu_lock_acquire
fi
echo "=== ENC START $(date -Is) | $(git -C "${DBF_REPO_DIR}" log --oneline -1) | cells=[${CELLS}] ==="
for t in "${SGL_MAIN}" "${SGL_WAN}" "${SGL_COS}" "${SGL_H3}"; do
  echo "=== tree ${t} $(git -C "${t%/python}" log --oneline -1 | cut -c1-80)"
done

port=61500  # above the ephemeral range (32768-60999); 10 per arm
for cell in ${CELLS}; do
  case_id=${CASE_OF[${cell}]}
  for round in $(seq 1 "${ROUNDS_OF[${cell}]}"); do
    mapfile -t arms < <(cell_arms "${cell}" "${round}")
    if (( round % 2 == 0 )); then
      reversed=()
      for ((i = ${#arms[@]} - 1; i >= 0; i--)); do reversed+=("${arms[i]}"); done
      arms=("${reversed[@]}")
    fi
    for line in "${arms[@]}"; do
      read -r arm fw tree cfg <<<"${line}"
      tag="enc1024_${cell}_${arm}_r${round}"
      config=configs/comparison_configs.json
      [[ "${cfg}" == before ]] && config="${BEFORE_CONFIG}"
      [[ "${tree}" == - ]] && tree=""
      if [[ -n "${ENC_DRY_RUN:-}" ]]; then
        echo "PLAN ${cell} r${round} ${arm} ${fw} case=${case_id} config=${config} sglang=${tree:-n/a} port=${port}"
        port=$((port + 10))
        continue
      fi
      t0=$(date +%s)
      DBF_FRAMEWORKS="${fw}" DBF_SGLANG_PYTHONPATH="${tree}" DBF_CONFIG="${config}" \
        bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" "${tag}" 0,1,2,3 "${port}" single_e2e "${case_id}"
      latency=$(python3 - "${DBF_LOG_DIR}/run_${tag}_${case_id}.json" "${fw}" <<'EOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))["results"]
except (OSError, ValueError, KeyError) as exc:
    print(f"no-result ({exc.__class__.__name__})")
    sys.exit()
for r in rows:
    if r.get("framework") == sys.argv[2] and r.get("mode", "single_e2e") == "single_e2e":
        samples = (r.get("metrics") or {}).get("latency_samples_s")
        print(r.get("latency_s"), samples or "", (r.get("error") or "")[:160])
EOF
)
      echo "ARM ${cell} ${arm} r${round} ${fw} ${case_id} $(( $(date +%s) - t0 ))s latency ${latency}"
      python3 "${DBF_REPO_DIR}/scripts/probe_mp4_encode.py" --group --since "${t0}" --label "${cell} ${arm} r${round}" \
        "${DBF_REPO_DIR}/outputs" "${DBF_REPO_DIR}/trtllm_generated" "${DIFFUSION_BENCH_KEEP_VIDEO_DIR}" \
        "${DIFFUSION_BENCH_COMFYUI_WORKSPACE}/${case_id}" /tmp/lightx2v_"${case_id}"_*.mp4 \
        >> "${DBF_LOG_DIR}/enc_probe.jsonl" 2>/dev/null
      port=$((port + 10))
    done
  done
done
echo "=== ENC DONE $(date -Is) ==="
