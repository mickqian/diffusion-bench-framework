#!/usr/bin/env python3
"""Idempotent post-install patches for the `trtllm-visual` (TensorRT-LLM
VisualGen) isolated venv. Run with that venv's python (the installer does this
right after `pip install`).

Patch: eager-mode LayerNorm dtype mismatch
------------------------------------------
`tensorrt_llm/_torch/modules/layer_norm.py::forward` upcasts the activations to
fp32 for numerical stability but leaves `self.weight` / `self.bias` in the
module dtype (bf16). `nn.functional.layer_norm(fp32_input, weight=bf16_weight)`
raises `RuntimeError: expected scalar type Float but found BFloat16` in eager
mode. `torch.compile` (`@maybe_compile`) promotes the dtypes so it only bites
when compilation is disabled (`TORCH_COMPILE_DISABLE=1`, which the benchmark's
cache-free policy sets). We cast weight/bias to the (fp32) input dtype, matching
the upcast intent and working in both eager and compiled paths.

This lets `trtllm-visual` be benchmarked compile-OFF for a same-policy
comparison against the other frameworks. Re-apply after every (re)install.

RESOLVED UPSTREAM in 1.3.0rc24. Checking the published tags, rc18-rc23 still
pass `weight=self.weight`; rc24, rc25 and rc26 cast both to fp32 with the same
rationale we used ("Eager torch.layer_norm needs weight/bias in the fp32
compute dtype"). So on any version we would run today this patch is a no-op and
reports itself as one; it stays only for reproducing a report pinned to
rc18-rc23. Delete it once no published run pins one of those.
"""
from __future__ import annotations

import importlib.util
import pathlib


def patch_layer_norm_eager_dtype() -> None:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.origin:
        print("[patch] tensorrt_llm not importable; skipping")
        return
    target = pathlib.Path(spec.origin).parent / "_torch" / "modules" / "layer_norm.py"
    if not target.exists():
        print(f"[patch] {target} not found; skipping (version drift?)")
        return
    src = target.read_text()
    if "self.weight.to(hidden_states.dtype)" in src:
        print("[patch] layer_norm.py already patched")
        return
    if "weight=self.weight.to(" in src or "self.weight.to(" in src:
        print("[patch] layer_norm.py casts weight upstream (1.3.0rc24+); nothing to do")
        return
    if "weight=self.weight," not in src or "bias=self.bias," not in src:
        # Loud on purpose. This printed quietly for three consecutive runs while
        # the patch did nothing: it only matters compile-off, and the benchmark
        # runs competitors compile-on, so nothing failed and nobody looked.
        print(
            "[patch] WARNING: layer_norm.py matches neither the rc18-rc23 patch "
            "target nor the rc24+ upstream fix -- it has moved again. The "
            "eager-mode dtype fix is NOT applied -- trtllm-visual will crash if "
            "run with TORCH_COMPILE_DISABLE=1. Re-target or retire this patch: "
            f"{target}"
        )
        return
    src = src.replace(
        "weight=self.weight,", "weight=self.weight.to(hidden_states.dtype),"
    ).replace("bias=self.bias,", "bias=self.bias.to(hidden_states.dtype),")
    target.write_text(src)
    print(f"[patch] applied eager-dtype fix to {target}")


if __name__ == "__main__":
    patch_layer_norm_eager_dtype()
