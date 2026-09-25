#!/usr/bin/env python3
"""The runtime block must name the dependencies a cell actually depended on.

A published artifact carries ONE `framework_runtime` block, and readers treat it
as the answer to "what was this measured against". It recorded each framework's
own package and its kernels -- and not transformers, diffusers or
huggingface_hub, which is where the interesting failures live.

Concretely: LightX2V's FLUX.2 cell was published as `failed` for weeks because
the harness pinned transformers below 5, which caps huggingface_hub below 1.0,
which breaks every `diffusers.pipelines.*` import, which left LightX2V's flux2
scheduler as None. Three packages decided whether that cell ran at all and the
artifact named none of them. torch belongs there for the same reason: flash-attn
is compiled against it, so the row means something different on another torch.

The three frameworks do NOT agree on these versions -- measured on 4xB200
2026-09-13: vLLM-Omni on torch 2.13.0 / transformers 5.14.1, LightX2V on 2.11.0
/ 5.17.0, trtllm-visual on 2.12.0 / 5.5.4 -- which is exactly why one shared
sentence in a report cannot stand in for recording them.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

src = (ROOT / "src" / "diffusion_bench" / "run_comparison.py").read_text()

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


block = src[src.index("SHARED_STACK = [") : src.index("for framework, packages in")]

for pkg in ("torch", "transformers", "diffusers", "huggingface_hub"):
    check(f"the shared stack includes {pkg}", f'"{pkg}"' in block)

# and each framework must still record its own
for fw, own in (
    ("vllm-omni", "vllm-omni"),
    ("lightx2v", "lightx2v"),
    ("trtllm-visual", "tensorrt-llm"),
):
    check(f"{fw} still records {own}", f'"{own}"' in block)

# One spread per installable framework: a framework added to the harness
# without the shared stack would publish a row whose torch is unrecorded.
framework_keys = re.findall(r'^\s{8}"([a-z0-9-]+)": \[', block, re.M)
check(
    "the stack is shared, not copy-pasted per framework",
    block.count("*SHARED_STACK") == len(framework_keys) >= 4,
    f"one list, spread into each of {framework_keys} (found {block.count('*SHARED_STACK')})",
)
check("comfyui records the shared stack too", '"comfyui"' in block)

# pip show tolerates a name that is not installed (it warns and returns the
# rest), which the existing flash-attn-3 entry already relies on -- so adding
# names cannot break a framework whose venv lacks one.
check("relies on pip show's tolerance, as flash-attn-3 already does", '"flash-attn-3"' in block)

sys.exit(fail)
