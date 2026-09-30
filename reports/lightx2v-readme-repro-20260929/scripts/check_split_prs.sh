#!/bin/bash
# Unit tests and a real Hub sizing check for each split sglang PR branch, in a
# scratch worktree so the measured tree is untouched.
set -u
W=/persistent/sgl-fix
BUNDLE=/persistent/lx2v/sgl-prs.bundle
A=${A_SHA:?}; B=${B_SHA:?}; C=${C_SHA:?}
export HF_HOME=/persistent/hf-cache HF_HUB_CACHE=/persistent/hf-cache/hub
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$W/python
git -C /sgl-workspace/sglang fetch -q "$BUNDLE" \
    "+refs/heads/*:refs/remotes/split/*" || exit 1
run_tests() {
    git -C "$W" checkout -q --detach "$1" || exit 1
    find "$W/python" -name "*.pyc" -delete
    echo "== $2 at $(git -C "$W" rev-parse --short HEAD)"
    (cd "$W/python" && python3 -m pytest -q -p no:cacheprovider "${@:3}" 2>&1 | tail -2)
}
run_tests "$A" "A auto-DiT" \
    sglang/multimodal_gen/test/unit/test_server_args.py \
    sglang/multimodal_gen/test/unit/test_weight_utils.py
(cd "$W/python" && python3 - <<'PY'
import time
from sglang.multimodal_gen.runtime.loader.utils import dit_parameter_count
for repo in (
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",  # in the local HF cache
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",  # not downloaded: Hub headers only
    "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
):
    t = time.time()
    n = dit_parameter_count(repo, subfolder=None, revision=None)
    print(f"size {repo}: {n} params, {time.time() - t:.1f} s")
import os
repo = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
snapshots = os.path.join(
    os.environ["HF_HUB_CACHE"], "models--" + repo.replace("/", "--"), "snapshots"
)
pinned = sorted(os.listdir(snapshots))[0]  # the snapshot the published cells ran
t = time.time()
n = dit_parameter_count(repo, subfolder=None, revision=pinned)
print(f"size {repo}@{pinned}: {n} params, {time.time() - t:.1f} s")
PY
)
run_tests "$B" "B FSDP" sglang/multimodal_gen/test/unit/test_server_args.py
git -C "$W" checkout -q --detach "$C" || exit 1
echo "== C at $(git -C "$W" rev-parse --short HEAD)"
(cd "$W/python" && python3 -c "
from sglang.multimodal_gen.runtime.managers.gpu_worker import OOM_MSG
print(OOM_MSG)")
echo "CHECK_DONE"
