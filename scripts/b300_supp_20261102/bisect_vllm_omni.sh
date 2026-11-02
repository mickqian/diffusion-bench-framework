#!/usr/bin/env bash
# r1102 wave 4: which vLLM-Omni commit between 0da1d831 (2.345 s, last round's pin) and d94d8e98 (main, ~2.49 s) slowed
# Qwen-Image-2.1 T2I single_e2e on B300? The old-pin venv (fw-venvs-0da1) is an editable install of a full clone,
# so a step is `git checkout` + one harness arm (single_e2e, fresh server), no reinstall. Both venvs resolve
# identical package versions (only torch-einops-utils 0.2.1 vs 0.2.2), so the delta is vLLM-Omni's own source.
# slow = steady-state median > 2.42 s (midpoint of the two known ends). Confirms the boundary with P C P C at the end.
set -u
STATE=/workspace/r1102
SRC=${STATE}/fw-venvs-0da1/vllm-omni/src/vllm-omni
GOOD=0da1d831e07fadccc3ab51ef27b7c21623b693c2
BAD=d94d8e9809d8b56e4c58e37a39ddf454ab61f2fd
LOGDIR=${STATE}/logs-ab
THRESH=2.42
OUT=${LOGDIR}/bisect_vllm_omni.log
echo "BISECT start $(date -Is) good=${GOOD:0:8} bad=${BAD:0:8}" | tee -a "${OUT}"

test_commit() {  # <sha> -> prints latency (or NA) on the last line, logs the rest
  local sha=$1 short=${1:0:8}
  step=$(( $(cat "${LOGDIR}/.bisstep" 2>/dev/null || echo 0) + 1 )); echo "${step}" > "${LOGDIR}/.bisstep"   # a counter file: this runs in a $(...) subshell
  git -C "${SRC}" checkout -q "${sha}" || { echo NA; return; }
  local plan="${STATE}/scripts/plan_bis${step}.txt" tag="bis${step}"
  printf 'b%s vllm-omni qwen_image_21_t2i_1024 single_e2e 0 - DBF_VENV_ROOT=%s/fw-venvs-0da1 VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git@%s\n' "${short}" "${STATE}" "${sha}" > "${plan}"
  SUPP_TAG="${tag}" SUPP_PLAN="${plan}" SUPP_PORT=$((62000 + step * 10)) bash "${STATE}/scripts/run_supp_ab.sh" > "${LOGDIR}/${tag}.log" 2>&1 < /dev/null
  python3 - "${LOGDIR}/run_${tag}_b${short}_qwen_image_21_t2i_1024.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    r = d["results"][0]
    print(r["latency_s"] if not r.get("error") else "NA")
except Exception:
    print("NA")
PY
}

mapfile -t commits < <(git -C "${SRC}" rev-list --reverse "${GOOD}..${BAD}")
n=${#commits[@]}
echo "BISECT ${n} commits in (good, bad]" | tee -a "${OUT}"
lo=-1; hi=$((n - 1))
v=$(test_commit "${commits[hi]}"); echo "BISECT bad end ${commits[hi]:0:8} -> ${v}" | tee -a "${OUT}"
python3 -c "import sys; sys.exit(0 if '${v}' != 'NA' and float('${v}') > ${THRESH} else 1)" || { echo "BISECT bad end is not slow in this venv; stop" | tee -a "${OUT}"; git -C "${SRC}" checkout -q "${GOOD}"; exit 1; }
while [ $((hi - lo)) -gt 1 ]; do
  mid=$(((lo + hi) / 2))
  v=$(test_commit "${commits[mid]}")
  subj=$(git -C "${SRC}" log -1 --format=%s "${commits[mid]}")
  echo "BISECT idx=${mid}/${n} ${commits[mid]:0:8} -> ${v}  ${subj}" | tee -a "${OUT}"
  if [ "${v}" = NA ]; then echo "BISECT step failed; stop (manual)" | tee -a "${OUT}"; git -C "${SRC}" checkout -q "${GOOD}"; exit 2; fi
  if python3 -c "import sys; sys.exit(0 if float('${v}') > ${THRESH} else 1)"; then hi=${mid}; else lo=${mid}; fi
done
first_bad=${commits[hi]}
if [ "${lo}" -ge 0 ]; then parent=${commits[lo]}; else parent=${GOOD}; fi
echo "BISECT first slow: ${first_bad:0:12} $(git -C "${SRC}" log -1 --format='%s (%ad)' --date=short "${first_bad}")" | tee -a "${OUT}"
echo "BISECT last fast : ${parent:0:12} $(git -C "${SRC}" log -1 --format='%s (%ad)' --date=short "${parent}")" | tee -a "${OUT}"
for sha in "${parent}" "${first_bad}" "${parent}" "${first_bad}"; do
  v=$(test_commit "${sha}"); echo "BISECT confirm ${sha:0:8} -> ${v}" | tee -a "${OUT}"
done
git -C "${SRC}" checkout -q "${GOOD}"
echo "BISECT done $(date -Is); src restored to ${GOOD:0:8}" | tee -a "${OUT}"
