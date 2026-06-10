#!/usr/bin/env python3
import os
import shutil
import site
from pathlib import Path

from huggingface_hub import snapshot_download


repo = os.environ.get("LIGHTX2V_FA3_HF_REPO", "varunneal/flash-attention-3")
revision = os.environ.get(
    "LIGHTX2V_FA3_HF_REVISION", "de87b9b5af06dd9984df595bef90b2eba44b181a"
)
subdir = os.environ.get(
    "LIGHTX2V_FA3_HF_SUBDIR",
    "build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3",
)

snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=[subdir + "/*"]))
site_dir = Path(site.getsitepackages()[0])
dst = site_dir / "flash_attention_3"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(snapshot / subdir, dst, symlinks=False)
(site_dir / "flash_attn_interface.py").write_text(
    "from flash_attention_3.flash_attn_interface import *\n"
)
