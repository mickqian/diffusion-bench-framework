# Is the default-on spatial-shard VAE decode right? (measure before writing)

**Date**: 2026-07-31 · **Machine**: `native-diffusion-b200` devbox, 2×B200, NVLink5
**Script**: `scripts/bench_vae_spatial.py` (WanVAE, random weights —
convolution cost does not depend on weight values, so no checkpoint is needed)
**Verdict**: the current default is correct. There is no regression to fix, and
the batch-data-parallel variant is not justified by any measured case.

## What prompted this

An earlier image-side measurement appeared to show spatial-shard decode being a
**32% pessimisation** on small activations (qwen 1024²: 0.062 → 0.091 s) while
being on by default — which would have been a live regression worth an immediate
fix. Both halves of that claim are false.

**It is not on by default for image VAEs.** `VAEConfig.auto_parallel_decode_prefers_spatial_shard()`
returns `False` in the base class, and only three configs override it — `wanvae`,
`hunyuanvae`, `ltx_video`, all **video** VAEs. Under `auto`, an image VAE
(flux/qwen/zimage `AutoencoderKL`) never enters the spatial path, so toggling
`--vae-config.use-parallel-decode` on those cases changed nothing. A second gate
(`auto_parallel_decode_min_latent_elements_per_rank = 4096`) additionally keeps a
video VAE replicated when its latent is too small to divide usefully.

**The 32% was noise.** Both arms were three samples and their ranges overlapped:
0.091/0.061/0.106 s against 0.093/0.061/0.062 s. Three samples cannot support a
percentage claim.

## The measurement that was missing

Video is where decode actually costs seconds, so that is where the default has to
be judged. Median of 5, 2 warmup, `torch.cuda.synchronize()` on both sides, peak
memory from `max_memory_allocated`:

| shape | latents | spatial split | replicated | speedup | peak memory |
|---|---|---|---|---|---|
| 480p 81f | (1,16,21,60,104) | **706.7 ms / 2.8 GiB** | 1022.4 ms / 4.8 GiB | **1.45x** | **1.73x lower** |
| 720p 81f | (1,16,21,90,160) | **1380.1 ms / 6.1 GiB** | 2322.4 ms / 10.7 GiB | **1.68x** | **1.77x lower** |

Spatial split wins on **both** axes at video sizes, and the margin grows with
resolution. Defaulting it on for video VAEs is right, and the 4096-element gate
is the correct shape of protection for the small-latent tail.

## Consequence for a batch-DP decode PR

The proposed follow-up was to data-parallel the decode across ranks by splitting
the *batch*, mirroring encoder dp. Measurement says do not write it:

- **Video, batch 1** — the common case — is already served by the spatial split
  at 1.45–1.68x. Batch-DP does nothing at batch 1.
- **Video, batch ≥ 2** is rare, and batch-DP would *raise* peak memory (each rank
  decodes a whole sample) exactly where the spatial split currently *lowers* it
  by 1.7x. That is the wrong direction on the axis that decides whether a 720p
  video decode fits at all.
- **Image decode** is ~60 ms for 4 images, about 1.5% of a request. Even a perfect
  2x there is inside the run-to-run noise of the e2e cell.

No measured configuration is left where batch-DP decode wins, so the honest
outcome of "measure first" is that this one does not get written.
