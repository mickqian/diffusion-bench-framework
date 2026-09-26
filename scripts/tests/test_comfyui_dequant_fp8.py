#!/usr/bin/env python3
"""A `dequant-fp8:` ComfyUI model holds exactly the weights sglang computes with.

ComfyUI runs an FP8 checkpoint as W8A8 on sm90+ (activations quantized to FP8),
a lossy class the other rows do not use. Ideogram-4's only release is FP8, so the
ComfyUI row loads that release dequantized to BF16 instead -- and that is only a
like-for-like comparison if the BF16 values are the ones sglang's weight-only FP8
linears compute: `weight.to(bf16) * weight_scale.to(bf16)`, per row.
"""
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import huggingface_hub  # noqa: E402
import huggingface_hub.constants  # noqa: E402
from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    snapshot = tmp / "hub" / "models--org--model" / "snapshots" / "abc123"
    (snapshot / "transformer").mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    row_w = (torch.randn(4, 8, generator=g) * 50).to(torch.float8_e4m3fn)
    row_s = torch.rand(4, generator=g) + 0.01
    tensor_w = (torch.randn(3, 8, generator=g) * 50).to(torch.float8_e4m3fn)
    tensor_s = torch.tensor(0.02)
    norm = torch.randn(8, generator=g).to(torch.bfloat16)
    save_file({"a.weight": row_w, "a.weight_scale": row_s, "a.bias": norm}, str(snapshot / "transformer" / "one.safetensors"))
    save_file({"b.weight": tensor_w, "b.weight_scale": tensor_s}, str(snapshot / "transformer" / "two.safetensors"))

    huggingface_hub.snapshot_download = lambda repo_id, allow_patterns: str(snapshot)
    huggingface_hub.constants.HF_HUB_CACHE = str(tmp / "hub")

    path = rc._resolve_comfyui_model("dequant-fp8:org/model:transformer")
    out = load_file(path)
    check("scales are dropped", sorted(out) == ["a.bias", "a.weight", "b.weight"], f"{sorted(out)}")
    check(
        "row scales match sglang's weight-only FP8 math bit for bit",
        torch.equal(out["a.weight"], row_w.to(torch.bfloat16) * row_s.to(torch.bfloat16).unsqueeze(1)),
    )
    check(
        "a per-tensor scale applies to the whole matrix",
        torch.equal(out["b.weight"], tensor_w.to(torch.bfloat16) * tensor_s.to(torch.bfloat16)),
    )
    check("tensors that are not FP8 pass through", torch.equal(out["a.bias"], norm))
    check("the file is cached under the snapshot revision", "abc123" in Path(path).name, path)

    stamp = Path(path).stat().st_mtime_ns
    check(
        "a second resolve reuses the cached file",
        rc._resolve_comfyui_model("dequant-fp8:org/model:transformer") == path
        and Path(path).stat().st_mtime_ns == stamp,
    )

sys.exit(fail)
