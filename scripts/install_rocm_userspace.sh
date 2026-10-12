#!/usr/bin/env bash
# Unpack one ROCm release's userspace into a private prefix, for a competitor
# venv whose torch was built against a different ROCm than the host image's.
#
# vLLM's ROCm wheels (vllm 0.31.0+rocm723, torch 2.13.0+git733fca1) link torch
# against the system ROCm. On sglang's ROCm images /opt/rocm is ROCm 10 (its pip
# SDK), so that torch ran on a newer HIP runtime and hipBLASLt than it was built
# for and faulted the GPU on the first H3 request (MI355X, 2026-10-10:
# HSA_STATUS_ERROR_MEMORY_FAULT in a Cijk_..._Bias hipBLASLt kernel).
#
# Uses a private apt state and extracts the debs; nothing is installed, and the
# image's /opt/rocm is left alone (an apt install would re-point it, and sglang
# runs on it). run_comparison._isolate_rocm_userspace points a venv at
# <venv>/rocm when it exists.
#
#   bash scripts/install_rocm_userspace.sh 7.2.3 "$DBF_VENV_ROOT/vllm-omni/rocm"
set -euo pipefail

RELEASE="${1:?ROCm release, e.g. 7.2.3}"
PREFIX="${2:?prefix to unpack into, e.g. <venv>/rocm}"
WORK="${PREFIX}.work"
CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
ROOTS="rocm-hip-runtime rocm-hip-libraries miopen-hip rccl roctracer rocprofiler-sdk
amd-smi-lib rocm-smi-lib hsa-amd-aqlprofile hipcc hip-dev"

mkdir -p "${WORK}"/lists/partial "${WORK}"/archives/partial "${WORK}"/debs
echo "deb [arch=amd64 trusted=yes] https://repo.radeon.com/rocm/apt/${RELEASE} ${CODENAME} main" > "${WORK}/sources.list"
APT=(-o Dir::Etc::sourcelist="${WORK}/sources.list" -o Dir::Etc::sourceparts=-
     -o Dir::State::Lists="${WORK}/lists" -o Dir::Cache::archives="${WORK}/archives"
     -o APT::Get::List-Cleanup=0 -o Debug::NoLocking=1)
apt-get "${APT[@]}" update

# The closure, kept to what the ROCm repo provides: the image already has the
# distribution libraries these depend on.
apt-cache "${APT[@]}" depends --recurse --no-recommends --no-suggests --no-conflicts \
  --no-breaks --no-replaces --no-enhances ${ROOTS} | grep -E '^[a-z0-9]' | sort -u > "${WORK}/closure.txt"
: > "${WORK}/packages.txt"
while read -r pkg; do
  if apt-cache "${APT[@]}" policy "${pkg}" | grep -q repo.radeon.com; then
    echo "${pkg}" >> "${WORK}/packages.txt"
  fi
done < "${WORK}/closure.txt"
(cd "${WORK}/debs" && xargs -a "${WORK}/packages.txt" apt-get "${APT[@]}" download)
for deb in "${WORK}"/debs/*.deb; do dpkg-deb -x "${deb}" "${WORK}/root"; done

if [[ -e "${PREFIX}" ]]; then
  echo "${PREFIX} already exists; move it aside first" >&2
  exit 1
fi
mv "${WORK}/root/opt/rocm-${RELEASE}" "${PREFIX}"
# vLLM's extension modules also link librocm-openblas.so.0, a CPU-only OpenBLAS
# that is not in the release's apt repo. The image's SDK ships one.
blas="$(compgen -G '/opt/venv/lib/python3*/site-packages/_rocm_sdk_core/lib/host-math/lib/librocm-openblas.so.0' | head -1 || true)"
if [[ -n "${blas}" ]]; then
  cp -L "${blas}" "${PREFIX}/lib/"
fi
echo "ROCm ${RELEASE}: $(wc -l < "${WORK}/packages.txt") packages unpacked into ${PREFIX} ($(du -sh "${PREFIX}" | cut -f1))"
