#!/usr/bin/env bash
set -uo pipefail

# B200 cross-framework run (latest-vs-latest), 2026-07-01.
#
# Hardware: 4x B200 (b200-verda-k8s rx devbox), --hardware-profile b200 so flux1/flux2
# pick the `blackwell-*` command_profiles; other cases fall back to `default`.
# Storage: HF cache on cluster-storage (49T shared) — NOT the 250Gi ephemeral layer.
# Cosmos3 is benchmarked as text-to-video (cosmos3_nano_t2v_720p_189f), not t2i.
#
# Per-framework case scope (only cases each framework actually supports in the config):
#   sglang        img: zimage,flux1,flux2,qwen-image      vid: wan21-t2v-1.3b,wan22-ti2v-5b,ltx2.3,cosmos3-t2v
#   vllm-omni     img: zimage,flux1,flux2,qwen-image      vid: wan21-t2v-1.3b,wan22-ti2v-5b,cosmos3-t2v
#   lightx2v      img: zimage,flux2                       vid: wan21-t2v-1.3b,wan22-ti2v-5b,ltx2.3
#   trtllm-visual img: flux1,flux2,qwen-image             (image-only; VisualGen has no t2v path)
#
# Compile stays disabled (harness sets TORCH_COMPILE_DISABLE=1). single_e2e + throughput
# (image 32 req / conc 4, video 8 req / conc 2) are written to separate result JSONs.

RUN_ID="${RUN_ID:-b200-cross-framework-$(date -u +%Y%m%d-%H%M%S)}"
PORT="${PORT:-30420}"
PROFILE="${HARDWARE_PROFILE:-b200}"
CONFIG="${CONFIG:-configs/comparison_configs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IMAGE_NUM_REQUESTS="${IMAGE_NUM_REQUESTS:-32}"
IMAGE_MAX_CONCURRENCY="${IMAGE_MAX_CONCURRENCY:-4}"
VIDEO_NUM_REQUESTS="${VIDEO_NUM_REQUESTS:-8}"
VIDEO_MAX_CONCURRENCY="${VIDEO_MAX_CONCURRENCY:-2}"
THROUGHPUT_REQUEST_RATE="${THROUGHPUT_REQUEST_RATE:-inf}"

# Storage / cache (B200 cluster-storage; override for other clusters).
export HF_HOME="${HF_HOME:-/cluster-storage/models}"
export DIFFUSION_BENCH_HF_CACHE_DIR="${DIFFUSION_BENCH_HF_CACHE_DIR:-${HF_HOME}/hub}"
export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/scratch/sglang-diffusion-framework-venvs}"
export SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL="${SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
: "${HF_TOKEN:?set HF_TOKEN for gated repos (flux/cosmos3/ltx)}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

# latest-vs-latest competitor specs (record resolved versions in result artifacts).
export VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm}"
export VLLM_OMNI_INSTALL_SPEC="${VLLM_OMNI_INSTALL_SPEC:-git+https://github.com/vllm-project/vllm-omni.git}"
export LIGHTX2V_INSTALL_SPEC="${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

# Per-framework case scope.
sglang_IMG="zimage_turbo_t2i_1024 flux1_dev_t2i_1024 flux2_dev_t2i_1024 qwen_image_2512_t2i_1024"
sglang_VID="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p ltx2.3_twostage_t2v_2gpus cosmos3_nano_t2v_720p_189f"
vllm_omni_IMG="zimage_turbo_t2i_1024 flux1_dev_t2i_1024 flux2_dev_t2i_1024 qwen_image_2512_t2i_1024"
vllm_omni_VID="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p cosmos3_nano_t2v_720p_189f"
lightx2v_IMG="zimage_turbo_t2i_1024 flux2_dev_t2i_1024"
lightx2v_VID="wan21_t2v_1_3b_480p wan22_ti2v_5b_704p ltx2.3_twostage_t2v_2gpus"
trtllm_visual_IMG="flux1_dev_t2i_1024 flux2_dev_t2i_1024 qwen_image_2512_t2i_1024"
trtllm_visual_VID=""

FRAMEWORKS="${FRAMEWORKS:-sglang vllm-omni lightx2v trtllm-visual}"

mkdir -p "${OUTPUT_DIR}"

run_one() {
  local fw="$1" img="$2" vid="$3" key="$4"
  echo "########## ${fw} single_e2e $(date -u +%H:%M:%S) ##########"
  diffusion-bench-compare --config "${CONFIG}" --case-ids ${img} ${vid} --frameworks "${fw}" \
    --modes single_e2e --hardware-profile "${PROFILE}" \
    --run-id "${RUN_ID}-${key}-single" --port "${PORT}" \
    --output "${OUTPUT_DIR}/${RUN_ID}-${key}-single.json" || echo "${fw} single rc=$?"

  if [ "${fw}" = "sglang" ]; then export DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--batching-max-size ${IMAGE_MAX_CONCURRENCY} --batching-delay-ms 0"; fi
  echo "########## ${fw} throughput image $(date -u +%H:%M:%S) ##########"
  diffusion-bench-compare --config "${CONFIG}" --case-ids ${img} --frameworks "${fw}" \
    --modes throughput --hardware-profile "${PROFILE}" \
    --throughput-num-requests "${IMAGE_NUM_REQUESTS}" --throughput-max-concurrency "${IMAGE_MAX_CONCURRENCY}" \
    --throughput-request-rate "${THROUGHPUT_REQUEST_RATE}" \
    --run-id "${RUN_ID}-${key}-tput-image" --port "${PORT}" \
    --output "${OUTPUT_DIR}/${RUN_ID}-${key}-tput-image.json" || echo "${fw} tput-image rc=$?"

  if [ -n "${vid}" ]; then
    if [ "${fw}" = "sglang" ]; then export DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--batching-max-size ${VIDEO_MAX_CONCURRENCY} --batching-delay-ms 0"; fi
    echo "########## ${fw} throughput video $(date -u +%H:%M:%S) ##########"
    diffusion-bench-compare --config "${CONFIG}" --case-ids ${vid} --frameworks "${fw}" \
      --modes throughput --hardware-profile "${PROFILE}" \
      --throughput-num-requests "${VIDEO_NUM_REQUESTS}" --throughput-max-concurrency "${VIDEO_MAX_CONCURRENCY}" \
      --throughput-request-rate "${THROUGHPUT_REQUEST_RATE}" \
      --run-id "${RUN_ID}-${key}-tput-video" --port "${PORT}" \
      --output "${OUTPUT_DIR}/${RUN_ID}-${key}-tput-video.json" || echo "${fw} tput-video rc=$?"
  fi
}

for fw in ${FRAMEWORKS}; do
  case "${fw}" in
    sglang)        run_one sglang "${sglang_IMG}" "${sglang_VID}" sglang ;;
    vllm-omni)     run_one vllm-omni "${vllm_omni_IMG}" "${vllm_omni_VID}" vllm-omni ;;
    lightx2v)      run_one lightx2v "${lightx2v_IMG}" "${lightx2v_VID}" lightx2v ;;
    trtllm-visual) run_one trtllm-visual "${trtllm_visual_IMG}" "${trtllm_visual_VID}" trtllm-visual ;;
    *) echo "unknown framework ${fw}" ;;
  esac
done

echo "########## RUN_ALL DONE $(date -u +%H:%M:%S) ##########"
