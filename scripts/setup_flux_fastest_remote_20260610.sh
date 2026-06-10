#!/usr/bin/env bash
set -euo pipefail

# Run inside an rx devbox after copying BENCH_BUNDLE to the same machine.

WORKDIR="${WORKDIR:-/scratch/flux_fastest_20260610}"
BENCH_BUNDLE="${BENCH_BUNDLE:-/scratch/diffusion-bench.bundle}"
BENCH_COMMIT="${BENCH_COMMIT:-HEAD}"
SGLANG_CURRENT_COMMIT="${SGLANG_CURRENT_COMMIT:-165331a2004fbda1531091341ed081d6a39d2162}"
SGLANG_BASELINE_COMMIT="${SGLANG_BASELINE_COMMIT:-8227187d472da41a9c56ab6a0d1ba11efc574dd5}"
SGLANG_REMOTE="${SGLANG_REMOTE:-https://github.com/sgl-project/sglang.git}"

mkdir -p "${WORKDIR}"

BENCH_DIR="${WORKDIR}/diffusion-bench-framework"
if [[ ! -d "${BENCH_DIR}/.git" ]]; then
  git clone "${BENCH_BUNDLE}" "${BENCH_DIR}"
fi
git -C "${BENCH_DIR}" fetch "${BENCH_BUNDLE}"
git -C "${BENCH_DIR}" checkout "${BENCH_COMMIT}"
python3 -m pip install -e "${BENCH_DIR}"

clone_sglang() {
  local name="$1"
  local commit="$2"
  local dir="${WORKDIR}/${name}"

  if [[ ! -d "${dir}/.git" ]]; then
    git clone "${SGLANG_REMOTE}" "${dir}"
  fi
  git -C "${dir}" fetch origin main
  git -C "${dir}" fetch origin "${commit}"
  git -C "${dir}" checkout "${commit}"
  git -C "${dir}" rev-parse HEAD
}

clone_sglang sglang-current "${SGLANG_CURRENT_COMMIT}"
clone_sglang sglang-baseline "${SGLANG_BASELINE_COMMIT}"

echo "${BENCH_DIR}"
echo "${WORKDIR}/sglang-current"
echo "${WORKDIR}/sglang-baseline"
