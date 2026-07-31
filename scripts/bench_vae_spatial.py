"""Is the (default-on) spatial-shard VAE decode a win at video activation sizes?

The image measurement said no -- qwen 1024x1024 decode was *slower* split than
replicated -- but its activations are tiny. Video is where decode actually costs
seconds, so measure there. Random weights: convolution cost does not depend on
their values, so no checkpoint is needed.

Reports, per shape, the decode latency split across ranks vs replicated on each
rank, plus peak memory, which is the other reason to split.
"""

import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.parallel_conv import (
        disable_spatial_parallel_decode,
    )
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )

    config = WanVAEConfig()
    vae = AutoencoderKLWan(config).to("cuda", torch.bfloat16).eval()
    z_dim = config.arch_config.z_dim
    scale = config.arch_config.scale_factor_spatial

    # (label, width, height, frames) -- the published benchmark shapes
    shapes = [
        ("480p_81f", 832, 480, 81),
        ("720p_81f", 1280, 720, 81),
    ]

    def timed(fn, iters=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        return samples[len(samples) // 2], torch.cuda.max_memory_allocated() / 2**30

    for label, w, h, frames in shapes:
        t_lat = (frames - 1) // config.arch_config.scale_factor_temporal + 1
        z = torch.randn(
            1, z_dim, t_lat, h // scale, w // scale,
            dtype=torch.bfloat16, device="cuda",
        )

        with torch.no_grad():
            def split():
                return vae.decode(z)

            def replicated():
                with disable_spatial_parallel_decode():
                    return vae.decode(z)

            try:
                t_split, m_split = timed(split)
                t_repl, m_repl = timed(replicated)
            except torch.OutOfMemoryError:
                if rank == 0:
                    print(f"{label} OOM", flush=True)
                continue

        if rank == 0:
            print(
                f"{label} latents={tuple(z.shape)}  "
                f"split={t_split:.1f}ms/{m_split:.1f}GiB  "
                f"replicated={t_repl:.1f}ms/{m_repl:.1f}GiB  "
                f"speedup={t_repl / t_split:.2f}x  mem={m_repl / m_split:.2f}x",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        print("BENCH_VAE_SPATIAL_DONE", flush=True)
    dist.destroy_process_group()


main()
