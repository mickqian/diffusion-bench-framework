#!/usr/bin/env python3
"""Install LightX2V's FlashAttention-3 from the prebuilt kernels repo.

The repo ships one build per (torch minor, CUDA minor, arch) and the .so is
linked against that exact libtorch, so a hardcoded subdir only works on the
box it was written for: a torch28-cu128 artifact on a torch 2.11+cu130 box
imports as `undefined symbol`. Derive the variant from the interpreter that
will actually import it, and verify the extension loads before declaring
success.
"""
import os
import platform
import shutil
import site
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

repo = os.environ.get("LIGHTX2V_FA3_HF_REPO", "varunneal/flash-attention-3")
revision = os.environ.get(
    "LIGHTX2V_FA3_HF_REVISION", "de87b9b5af06dd9984df595bef90b2eba44b181a"
)


def _variant() -> str:
    tmaj, tmin = torch.__version__.split(".")[:2]
    cuda = torch.version.cuda
    if not cuda:
        raise SystemExit("FA3: torch has no CUDA build; refusing to guess a variant")
    cmaj, cmin = cuda.split(".")[:2]
    arch = "aarch64" if platform.machine() in ("aarch64", "arm64") else "x86_64"
    return f"torch{tmaj}{tmin}-cxx11-cu{cmaj}{cmin}-{arch}-linux"


subdir = os.environ.get("LIGHTX2V_FA3_HF_SUBDIR")
if not subdir:
    subdir = f"build/{_variant()}/flash_attention_3"
print(f"FA3: torch {torch.__version__} cuda {torch.version.cuda} -> {subdir}")

# A private download dir, not the ambient HF cache: the artifact is copied into
# site-packages below, and on clusters whose shared cache is mounted read-only
# (b200-verda-k8s exports HF_HOME=/cluster-storage/models) fetching into it
# failed with "Read-only file system" after a 25-minute flash-attn build.
cache_dir = os.environ.get("LIGHTX2V_FA3_CACHE_DIR") or tempfile.mkdtemp(prefix="lightx2v-fa3-")
try:
    snapshot = Path(
        snapshot_download(repo, revision=revision, allow_patterns=[subdir + "/*"], cache_dir=cache_dir)
    )
except Exception as exc:  # noqa: BLE001 - report which variant is missing
    raise SystemExit(f"FA3: cannot fetch {subdir} from {repo}@{revision}: {exc}")
src = snapshot / subdir
if not src.is_dir():
    raise SystemExit(
        f"FA3: {repo}@{revision} has no {subdir} — this torch/CUDA combination is "
        f"not prebuilt. Set LIGHTX2V_FA3_HF_SUBDIR to a published variant, or "
        f"install a torch whose variant exists."
    )

site_dir = Path(site.getsitepackages()[0])
dst = site_dir / "flash_attention_3"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=False)
(site_dir / "flash_attn_interface.py").write_text(
    "from flash_attention_3.flash_attn_interface import *\n"
)

# A mismatched artifact copies fine and only fails when something imports it,
# which is far from here. Fail now, with the variant named.
try:
    import flash_attn_interface  # noqa: F401
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"FA3: installed {subdir} but it does not import: {exc}")
if not hasattr(flash_attn_interface, "flash_attn_func"):
    raise SystemExit(f"FA3: {subdir} imported but exposes no flash_attn_func")
print(f"FA3: {subdir} installed and imports OK", file=sys.stderr)
