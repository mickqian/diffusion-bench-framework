#!/usr/bin/env bash
# Run the arms of one cookbook cell (configs/cookbook_configs.json, built by
# scripts/build_cookbook_config.py): one devbox_run_cases.sh run per arm, a
# fresh server each, the arm selected by name. Every arm is explicit -- a
# cookbook arm never comes from hardware auto-selection.
#
#   CB_SGLANG_COMMIT=3a01d4cc DBF_SGLANG_PYTHONPATH=/scratch/bench/sglang/python \
#     bash scripts/run_cookbook_cells.sh <tag> <gpu_ids> <base_port> <cell_id> [fw:arm ...]
#
# With no fw:arm list, every arm of the cell runs, in the cell's order.
# Results: $DBF_LOG_DIR/run_<tag>_<fw>_<arm>_<cell>.json, one ARM line per arm.
#
# Why each guard is here (2026-10-29/30 rounds):
#   * sglang arms refuse to start unless DBF_SGLANG_PYTHONPATH is a git tree at
#     CB_SGLANG_COMMIT. The image's /sgl-workspace/sglang was the pin on the
#     2xH200 box and baa9befa on the 2xB200 one; only checking tells them apart.
#   * Each arm takes the GPU lock (scripts/gpu_job_lock.sh) for its own run only.
#   * The run stops when the state disk has less than CB_MIN_FREE_GB free (120).
#     That guards a full disk, NOT the pod's ephemeral-storage limit: two B200
#     boxes and the 2xH200 box were evicted for exceeding it (250Gi on the H200,
#     2026-10-30 10:50 UTC) with terabytes free on `df`. Keep weights and logs
#     on a volume that does not count toward that limit.
set -u

TAG="${1:?tag}"; GPUS="${2:?gpu ids, e.g. 0,1}"; port="${3:?base port}"; CELL="${4:?cell id}"; shift 4
: "${DBF_REPO_DIR:?DBF_REPO_DIR}" "${DBF_LOG_DIR:?DBF_LOG_DIR}"
CONFIG="${DBF_REPO_DIR}/configs/cookbook_configs.json"
STATE="${DBF_STATE_DIR:-${DBF_LOG_DIR}}"

arms=("$@")
listed="$(python3 - "$CONFIG" "$CELL" <<'PY'
import json, sys
cases = {c["id"]: c for c in json.load(open(sys.argv[1]))["cases"]}
if sys.argv[2] not in cases:
    sys.exit(f"no cookbook cell {sys.argv[2]!r} in {sys.argv[1]}")
for fw, body in cases[sys.argv[2]]["frameworks"].items():
    for name in body["command_profiles"]:
        print(f"{fw}:{name}")
PY
)" || exit 2
if [ "${#arms[@]}" -eq 0 ]; then
    while IFS= read -r line; do arms+=("$line"); done <<< "$listed"
fi
for arm in "${arms[@]}"; do
    # A typo here would otherwise surface only after the GPU lock and a server start.
    grep -qx -- "$arm" <<< "$listed" || { echo "cell $CELL has no arm $arm (has: $(echo $listed))" >&2; exit 2; }
done

for arm in "${arms[@]}"; do
    if [ "${arm%%:*}" = "sglang" ]; then
        : "${CB_SGLANG_COMMIT:?sglang arms need CB_SGLANG_COMMIT, the sglang pin this round runs}"
        : "${DBF_SGLANG_PYTHONPATH:?sglang arms need DBF_SGLANG_PYTHONPATH, a source tree at CB_SGLANG_COMMIT}"
        head="$(git -C "$DBF_SGLANG_PYTHONPATH" rev-parse HEAD 2>/dev/null)"
        if [[ -z "$head" || "$head" != "$CB_SGLANG_COMMIT"* ]]; then
            echo "DBF_SGLANG_PYTHONPATH=$DBF_SGLANG_PYTHONPATH is at ${head:-no git tree}, not CB_SGLANG_COMMIT=$CB_SGLANG_COMMIT" >&2
            exit 2
        fi
        break
    fi
done

for arm in "${arms[@]}"; do
    fw="${arm%%:*}"; name="${arm#*:}"
    free_kb="$(df -Pk "$STATE" | awk 'NR==2 {print $4}')"
    if [ "${free_kb:-0}" -lt $(( ${CB_MIN_FREE_GB:-120} * 1024 * 1024 )) ]; then
        echo "STOP $CELL: $STATE has $((free_kb / 1024 / 1024)) GB free, below CB_MIN_FREE_GB=${CB_MIN_FREE_GB:-120}"
        exit 1
    fi
    run_tag="${TAG}_${fw}_${name}"
    env_key="DIFFUSION_BENCH_$(echo "$fw" | tr 'a-z-' 'A-Z_')_PROFILE"
    [ "$fw" = "sglang" ] && env_key="SGLANG_BENCH_SGLANG_PROFILE"
    (
        export DBF_CONFIG="$CONFIG" DBF_FRAMEWORKS="$fw" "$env_key=$name"
        # shellcheck disable=SC1091
        source "${DBF_REPO_DIR}/scripts/gpu_job_lock.sh"
        gpu_lock_acquire
        bash "${DBF_REPO_DIR}/scripts/devbox_run_cases.sh" "$run_tag" "$GPUS" "$port" single_e2e "$CELL"
    )
    rc=$?
    result="$(grep -h '^RESULT' "${DBF_LOG_DIR}/run_${run_tag}.log" 2>/dev/null | tail -1)"
    echo "ARM $CELL $fw $name rc=$rc ${result}"
    port=$((port + 4))
done
