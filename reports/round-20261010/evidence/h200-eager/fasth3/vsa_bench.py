"""VSA-H3 block-sparse attention kernel timing on one GPU, FastH3 V2 shapes at Ulysses 4.

python vsa_bench.py sgl [warps stages]...   sglang's Triton kernel (sgl-main tree on PYTHONPATH)
python vsa_bench.py fv                      fastvideo-kernel block_sparse_attn (vLLM-Omni venv)
Both build the same inputs from one seed and save the output for a cross-check.
"""

import math
import sys

import torch

HEADS, D, BLK = 14, 128, 64
PREFIX_TILES, VIDEO_TILES = 20, 8 * 6 * 11
KEEP = math.ceil(0.2 * VIDEO_TILES)
NT = PREFIX_TILES + VIDEO_TILES
S = NT * BLK


def inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, HEADS, S, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    sizes = torch.full((NT,), BLK, dtype=torch.int32, device="cuda")
    sizes[PREFIX_TILES - 1] = 40  # a ragged prefix chunk
    sizes[-1] = 32  # a ragged video tile
    for t in (q, k, v):  # pad rows zero
        t.view(1, HEADS, NT, BLK, D)[:, :, PREFIX_TILES - 1, 40:] = 0
        t.view(1, HEADS, NT, BLK, D)[:, :, -1, 32:] = 0
    scores = torch.rand(1, HEADS, NT, VIDEO_TILES, device="cuda", generator=g)
    picked = scores.topk(KEEP, dim=-1).indices + PREFIX_TILES
    block_map = torch.zeros(1, HEADS, NT, NT, dtype=torch.bool, device="cuda")
    block_map[..., :PREFIX_TILES] = True
    block_map.scatter_(-1, picked, True)
    return q, k, v, sizes, block_map


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    mode = sys.argv[1]
    q, k, v, sizes, block_map = inputs()
    flops = 4 * HEADS * NT * BLK * BLK * D * int(block_map[0, 0].sum(-1).float().mean().item())
    if mode == "sgl":
        from sglang.multimodal_gen.runtime.layers.attention.backends import vsa_h3_kernels as kmod

        num = block_map.sum(-1).to(torch.int32).contiguous()
        idx = torch.sort(block_map.to(torch.int8), dim=-1, descending=True, stable=True).indices
        idx = idx[..., : int(num.max())].to(torch.int32).contiguous()
        out = torch.empty_like(q)
        configs = [(int(a), int(b)) for a, b in zip(sys.argv[2::2], sys.argv[3::2])] or [(4, 5)]
        for warps, stages in configs:
            kmod._ATTN_NUM_WARPS, kmod._ATTN_NUM_STAGES = warps, stages
            ms = timeit(lambda: kmod.vsa_h3_block_sparse_attn_forward(q, k, v, idx, num, sizes, out=out))
            print(f"VSA sgl-triton warps={warps} stages={stages}: {ms:.3f} ms  {flops / ms / 1e9:.0f} TFLOP/s")
            if (warps, stages) == configs[0]:
                torch.save(out.cpu(), "/scratch/h200eager/vsa_out_sgl.pt")
    else:
        from fastvideo_kernel.block_sparse_attn import block_sparse_attn

        out = None

        def run():
            nonlocal out
            out, _ = block_sparse_attn(q, k, v, block_map, sizes)

        ms = timeit(run)
        print(f"VSA fastvideo ({sys.argv[2] if len(sys.argv) > 2 else 'default route'}): {ms:.3f} ms  {flops / ms / 1e9:.0f} TFLOP/s")
        torch.save(out.cpu(), "/scratch/h200eager/vsa_out_fv.pt")
        try:
            ref = torch.load("/scratch/h200eager/vsa_out_sgl.pt").float()
            valid = torch.ones(S, dtype=torch.bool)
            valid.view(NT, BLK)[PREFIX_TILES - 1, 40:] = False
            valid.view(NT, BLK)[-1, 32:] = False
            diff = (out.cpu().float() - ref)[:, :, valid].abs()
            print(f"VSA fastvideo vs sgl: max {diff.max().item():.4g} mean {diff.mean().item():.4g}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
