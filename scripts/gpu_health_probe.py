"""One-minute box health check before a round: per-GPU GEMM, then a 2-GPU all-reduce.

A box can be healthy one GPU at a time and still useless for the matrix. On the
2026-10-10 MI355X node every card ran a bf16 GEMM at ~1465 TFLOP/s alone, while
a 2-rank RCCL all-reduce of 128 MB took 624 ms (~0.2 GB/s over XGMI links that
all reported Up) and each rank's GEMM fell to ~380 TFLOP/s. Hours of H3 cells
crawled at 100+ s/step before anyone looked at the transport.

    python3 scripts/gpu_health_probe.py                  # every visible GPU, then GPUs 0,1
    CUDA_VISIBLE_DEVICES=4,5 python3 scripts/gpu_health_probe.py

Prints one line per check. Compare GEMM across cards (a slow card stands out) and
the all-reduce bus bandwidth against the interconnect (hundreds of GB/s over
NVLink/XGMI; single-digit GB/s means the round will not measure the framework).
"""
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist

N = 8192
ALLREDUCE_ELEMS = 64 * 1024 * 1024  # 128 MB of bf16


def gemm_tflops(device: torch.device) -> float:
    a = torch.randn(N, N, device=device, dtype=torch.bfloat16)
    b = torch.randn(N, N, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(device)
    iters = 50
    start = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(device)
    return 2 * N**3 * iters / (time.time() - start) / 1e12


def worker() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    x = torch.randn(ALLREDUCE_ELEMS, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    iters = 10
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    seconds = (time.time() - start) / iters
    busbw = 2 * (world - 1) / world * x.numel() * x.element_size() / seconds / 1e9
    if rank == 0:
        print(f"all_reduce 128MB over {world} GPUs: {seconds * 1e3:.1f} ms, bus bandwidth {busbw:.1f} GB/s", flush=True)
    dist.destroy_process_group()


def main() -> int:
    if "RANK" in os.environ:
        worker()
        return 0
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        print(f"GPU {index} ({name}): bf16 GEMM {gemm_tflops(torch.device('cuda', index)):.0f} TFLOP/s", flush=True)
    if torch.cuda.device_count() < 2:
        return 0
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-port", "29653", __file__]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
