#!/usr/bin/env bash
# A/B one framework's command profiles on one case: a fresh server per arm, arms
# interleaved over rounds, through scripts/devbox_run_cases.sh. An arm is a
# profile name, selected explicitly (DIFFUSION_BENCH_FRAMEWORK_PROFILE), or the
# hardware class itself (ARMS_HW, default h200) for the profile that class
# auto-selects. `--hardware-profile` alone cannot pick a candidate: run_comparison
# keeps only known hardware tokens from it.
#
#   ARMS_CASE=zimage_turbo_t2i_1024 ARMS_FW=vllm-omni ARMS="h200 cand-usp2" ROUNDS=2 GPUS=0,1 \
#     bash scripts/run_competitor_arms.sh <tag> <base_port>
#
# Results: $DBF_LOG_DIR/run_<tag>_<arm>_r<k>_<case>.json, one ARM line per run.
# A winner must be promoted into the case's hardware profile and the cell run
# again through it before it is published (merge_and_publish_run.sh checks drift).
set -u

TAG="${1:?tag}"; port="${2:?base port}"
: "${ARMS_CASE:?ARMS_CASE}" "${ARMS_FW:?ARMS_FW}" "${ARMS:?ARMS}" "${DBF_REPO_DIR:?DBF_REPO_DIR}" "${DBF_LOG_DIR:?DBF_LOG_DIR}"
for round in $(seq 1 "${ROUNDS:-1}"); do
  for arm in ${ARMS}; do
    run_tag="${TAG}_${arm}_r${round}"
    explicit="${arm}"
    [[ "${arm}" == "${ARMS_HW:-h200}" ]] && explicit=""
    DBF_FRAMEWORKS="${ARMS_FW}" DBF_HARDWARE_PROFILE="${ARMS_HW:-h200}" DIFFUSION_BENCH_FRAMEWORK_PROFILE="${explicit}" \
      bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" "${run_tag}" "${GPUS:-0,1}" "${port}" single_e2e "${ARMS_CASE}"
    result="$(grep -h '^RESULT' "${DBF_LOG_DIR}/run_${run_tag}.log" 2>/dev/null | tail -1)"
    echo "ARM ${ARMS_CASE} ${ARMS_FW} ${arm} r${round} ${result}"
    port=$((port + 4))
  done
done
