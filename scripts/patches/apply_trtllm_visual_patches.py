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
comparison against the other frameworks. Kept local (not upstreamed); re-apply
after every (re)install. Verified against tensorrt-llm 1.3.0rc18.
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
    if "weight=self.weight," not in src or "bias=self.bias," not in src:
        print("[patch] layer_norm.py: expected pattern not found; skipping (version drift?)")
        return
    src = src.replace(
        "weight=self.weight,", "weight=self.weight.to(hidden_states.dtype),"
    ).replace("bias=self.bias,", "bias=self.bias.to(hidden_states.dtype),")
    target.write_text(src)
    print(f"[patch] applied eager-dtype fix to {target}")


if __name__ == "__main__":
    patch_layer_norm_eager_dtype()
