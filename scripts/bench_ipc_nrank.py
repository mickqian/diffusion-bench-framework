"""NCCL vs direct-IPC all-to-all, swept over rank count and message size.

Answers whether the 2-rank IPC win survives at R=4, and doubles as a clean
N-rank reference: peer access is enabled for EVERY peer (the shipped 2-rank code
hardcodes `1 - dev`, which would make a kernel dereference of a third device's
mapping an illegal access) and each rank owns one flag per peer.
"""

import ctypes
import os
import time

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void bump_kernel(int* seq, volatile int* const* peer_flags, int n) {
    int v = *seq + 1;
    *seq = v;
    __threadfence_system();
    for (int i = 0; i < n; ++i) *(peer_flags[i]) = v;
}
__global__ void wait_kernel(volatile int* flags, const int* target, int world,
                            int self, long long deadline) {
    // peer p publishes into flags[p], so wait on every slot but my own
    int t = *target;
    long long start = clock64();
    for (int i = 0; i < world; ++i) {
        if (i == self) continue;
        while (flags[i] < t) {
            if (clock64() - start > deadline) return;
        }
    }
    __threadfence_system();
}
void bump(torch::Tensor seq, torch::Tensor peer_flag_ptrs, int64_t n) {
    bump_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        seq.data_ptr<int>(), (volatile int* const*)peer_flag_ptrs.data_ptr<int64_t>(), (int)n);
}
void wait(torch::Tensor flags, torch::Tensor target, int64_t world, int64_t self,
          int64_t deadline) {
    wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        (volatile int*)flags.data_ptr<int>(), target.data_ptr<int>(), (int)world,
        (int)self, (long long)deadline);
}
"""
_DECL = (
    "void bump(torch::Tensor seq, torch::Tensor peer_flag_ptrs, int64_t n);\n"
    "void wait(torch::Tensor flags, torch::Tensor target, int64_t world, int64_t self, int64_t deadline);"
)


def share(t, group, src_rank, world):
    """Broadcast one rank's tensor by IPC handle; every other rank reopens it
    in its own device context."""
    from torch.multiprocessing.reductions import reduce_tensor

    payload = [reduce_tensor(t)] if dist.get_rank() == src_rank else [None]
    dist.broadcast_object_list(payload, src=src_rank, group=group)
    fn, args = payload[0]
    if dist.get_rank() == src_rank:
        return t
    args = list(args)
    dev = torch.cuda.current_device()
    for i, v in enumerate(args):
        if isinstance(v, torch.device):
            args[i] = torch.device(f"cuda:{dev}")
        elif isinstance(v, int) and i == 6:
            args[i] = dev
    return fn(*args)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.cuda.current_device()
    cudart = ctypes.CDLL("libcudart.so")
    for peer in range(world):
        if peer != dev:
            cudart.cudaDeviceEnablePeerAccess(peer, 0)

    build_dir = f"/tmp/ipc_nrank_build_{dev}"
    os.makedirs(build_dir, exist_ok=True)
    ops = load_inline(
        name=f"ipc_nrank_{dev}",
        cpp_sources=_DECL,
        cuda_sources=_SRC,
        functions=["bump", "wait"],
        extra_cuda_cflags=["-O3"],
        build_directory=build_dir,
        verbose=False,
    )

    # one flag per peer that writes into me; one seq counter of my own
    flags = torch.zeros(world, dtype=torch.int32, device="cuda")
    seq = torch.zeros(1, dtype=torch.int32, device="cuda")
    peer_flags = [share(flags, dist.group.WORLD, r, world) for r in range(world)]
    ptrs = torch.tensor(
        [peer_flags[r][rank].data_ptr() for r in range(world) if r != rank],
        dtype=torch.int64,
        device="cuda",
    )

    deadline = int(2000 * torch.cuda.clock_rate(dev) * 1000)  # 2s guard
    print(f"rank{rank} peer-access + handles ready", flush=True)
    results = []
    for mb in (1, 4, 16):
        n = mb * 1024 * 1024 // 2  # bf16 elements per exchange
        chunk = n // world
        send = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        recv = torch.empty(n, dtype=torch.bfloat16, device="cuda")

        def timed(fn, iters=30, warmup=10):
            """per-call latency: barrier + sync on both sides of one exchange, so
            the number is one exchange and not the launch queue's throughput"""
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iters):
                dist.barrier()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            samples.sort()
            return samples[len(samples) // 2]

        nccl_ms = timed(lambda: dist.all_to_all_single(recv, send))
        expected = recv.clone()

        local = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        peers = [share(local, dist.group.WORLD, r, world) for r in range(world)]

        def ipc_exchange():
            for p in range(world):
                src_chunk = send.narrow(0, p * chunk, chunk)
                dst = local if p == rank else peers[p]
                dst.narrow(0, rank * chunk, chunk).copy_(src_chunk, non_blocking=True)
            ops.bump(seq, ptrs, world - 1)
            ops.wait(flags, seq, world, rank, deadline)

        ipc_ms = timed(ipc_exchange)
        # a wrong result would make the latency meaningless, so check it
        ok = torch.equal(local, expected)
        oks = torch.tensor([1 if ok else 0], device="cuda")
        dist.all_reduce(oks)
        if rank == 0:
            print(
                f"R={world} {mb:2d}MB  nccl={nccl_ms:.3f}ms  ipc={ipc_ms:.3f}ms  "
                f"speedup={nccl_ms / ipc_ms:.2f}x  correct={int(oks.item())}/{world}",
                flush=True,
            )
        del peers, local
        dist.barrier()

    if rank == 0:
        print("BENCH_IPC_DONE", flush=True)
    dist.destroy_process_group()


main()
