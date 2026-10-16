# FLUX.3 Image Klein Base on H200: four frameworks

2026-10-16 · rx devbox flux3x-h200 (2× H200 141 GB, driver 595.71.05) · `black-forest-labs/flux-3-image-klein-base`
· 1024×1024, 50 steps, CFG 3.5, seed 42, 1 output · native BF16, no cache · dbf `36e76e3` + the case fixes in this commit

Every framework encodes the same 303-word prompt: the case's short request, expanded
once by BFL's prompt upsampler and frozen into the case (md5 `b0e44fd9c061`). The
upsampler is not timed. Latency is the client wall clock, median of 5 requests after
warmup.

## Two GPUs

| framework | version | layout | p50 | min–max | vs sglang |
|---|---|---|---|---|---|
| **sglang** | main `e5274752a8` | CFG parallel, eager | **4.035 s** | 4.033–4.048 | 1.00× |
| vLLM-Omni ⚠ pre-merge | PR #9007 `389a02a149ce`, vllm 0.31.0 | CFG parallel 2, regional compile | 4.534 s | 4.523–4.553 | 1.12× |
| ComfyUI | master `5a665890` | MultiGPU CFG split, eager | 5.364 s | 5.321–5.527 | 1.33× |

## One GPU

| framework | version | mode | p50 | min–max | vs sglang |
|---|---|---|---|---|---|
| **sglang** | main `e5274752a8` | eager | **7.170 s** | 7.168–7.178 | 1.00× |
| ComfyUI | master `5a665890` | TorchCompileModel | 9.046 s | 9.034–9.352 | 1.26× |
| vLLM-Omni ⚠ pre-merge | PR #9007 `389a02a149ce`, vllm 0.31.0 | regional compile | 9.101 s | 9.068–9.109 | 1.27× |
| diffusers (reference) | main `d42ebf9c` (0.41.0.dev0) | eager, ModularPipeline | 10.715 s | 10.681–10.732 | 1.49× |

sglang is fastest in both tables. vLLM-Omni has no FLUX.3 on main yet. Its rows
measure the open, approved PR, so they stay out of the committed matrix. That cell
remains `unsupported` until the PR merges.

sglang stages (server timers, one instrumented request):

| | text encode | denoise (50 steps) | decode |
|---|---|---|---|
| 1 GPU | 0.042 s | 6.771 s (0.135 s/step) | 0.285 s |
| 2 GPUs | 0.041 s | 3.736 s (0.075 s/step) | 0.241 s |

## What each row needed

- **sglang**: the case's `h200-2gpu-cfg-parallel` profile
  (`--enable-cfg-parallel --performance-mode speed`). The one-GPU row turns CFG parallelism
  off. Both are eager, as the cookbook recommends.
- **ComfyUI**:
  - **Weights.** The official template's weights come from
    `black-forest-labs/flux-3-image-klein-comfy`, which is gated separately and returned
    403 for this account. Its BF16 DiT and VAE have the same LFS sha256 as the Base repo's
    `model.safetensors` and `ae.safetensors`. Its Qwen3-VL text encoder is the Base repo's
    two `text_encoder` shards joined into one file (`merge:` source, names and values
    unchanged). The cell therefore loads the Base repo.
  - **CFG split.** The committed case lacked `params.cfg_gpus = 2`, so a first "2-GPU"
    run was a one-GPU run (9.05 s). Fixed here.
  - **Compile.**
    - Without `--gpu-only`, ComfyUI's auto-enabled dynamic VRAM records a malloc graph
      inside the dynamo trace. KSampler then fails with
      `'torch.Stream' object has no attribute 'cuda_stream'`.
    - With `--gpu-only`, compile works on one GPU: 9.046 s against 9.734 s eager.
    - Compile still breaks the CFG split (`Tensor device mismatch`, cuda:1 vs cuda:0),
      so the two-GPU cell runs eager. This is recorded as a `policy_exception`.
- **vLLM-Omni (pre-merge)**:
  - **vLLM version.** Use the version from the head's own `docker/Dockerfile.ci`. The PR
    moved from a v0.29.0 base to v0.31.0 that day, and pairing it with the wrong vLLM fails
    at import or worker init.
  - **Model path.** The FLUX.3 loader reads only a local package directory, so
    `model_path` is the snapshot dir.
  - **Model name.** `--served-model-name` must equal the case model, or every request is
    refused as a model mismatch.
  - These rows ran from a variant config, and `manifest.json` records how it was derived.
- **diffusers**:
  - The FLUX.3 ModularPipeline runs the prompt upsampler (Qwen3.5-9B) before every call
    by default. A first run measured 22.94 s with the 303-word caption being rewritten
    again, which is not comparable.
  - `diffusers_ref.py` drops that block before the pipeline is built
    (`Flux3AutoBlocks().sub_blocks.pop("prompt_upsampler")`; `pipe.blocks` is a deepcopy,
    so popping from it after the build does nothing).
  - Guidance comes from the guider default, `ClassifierFreeGuidance(guidance_scale=3.5)`.
  - It is a correctness baseline, run eager.

## Checks

- **Images.** One image per framework was inspected. All four follow the caption: the
  "NIGHT MARKET" and "SYNTHETIC DISTRICT" signs, the autonomous taxi, the hooded figure
  under an umbrella, and the drain grate. diffusers and ComfyUI draw near-identical
  frames from the same seed. `images/` holds 512² copies.
- **Model revision.** The model repo published two revisions during the round
  (`6d332ea7`, then `3f7fa5ee` at about 04:58 UTC). sglang, ComfyUI and vLLM-Omni ran on
  `6d332ea7`; diffusers ran on `3f7fa5ee`. Every weight blob is identical across both
  revisions and the earlier `f66af34f`: DiT, VAE, text encoder shards, tokenizer and chat
  template. Only `modular_model_index.json` changed.
- **Snapshot hazard.** A snapshot that one framework filled partially can break another.
  `3f7fa5ee` has no `config.json`, because diffusers never fetched it, and vLLM-Omni pointed
  at it fails with `Flux3ModularPipeline not found in diffusion model registry`.

## CFG-parallel scaling, measured

sglang scales 1.78× from one GPU to two, and vLLM-Omni 2.01×. A follow-up checked
whether sglang's two-GPU path has overhead to recover. It does not.

Setup: CUDA events around each rank's forward and the gather, in a measurement-only patch
(`cfg_parallel_breakdown/measurement.patch`, never upstreamed). Fresh 2×H200 box, sglang
main `e1d2057`, median of steps 5–49, three requests per arm after two warmups.

| arm | uncond rank forward | its wait + gather | cond rank forward | step | e2e p50 |
|---|---|---|---|---|---|
| 2 GPUs, empty negative (80 vs 480 text tokens) | 64.87 ms | 12.93 ms | 77.88 ms | 77.94 ms | 4.040 s |
| 2 GPUs, negative = caption (480 vs 480) | 76.70 ms | 0.88 ms | 77.41 ms | 77.44 ms | 4.008 s |
| 1 GPU, empty negative (packed, one forward) | | | | 135.5 ms | 7.182 s |
| 1 GPU, negative = caption (packed) | | | | 144.0 ms | 7.599 s |

- **The CFG combine costs under 1 ms a step.** The two-GPU step is the cond branch's
  forward plus about 0.05 ms.
- **Branch imbalance.** With the empty negative, the uncond rank idles about 13 ms (17%) of
  every step waiting for the cond rank. This is inherent to splitting two unequal branches.
- **Packing on one GPU.** sglang packs both branches into one varlen forward on a single
  GPU. That is 5–7% faster than two separate forwards (135.5 ms against 64.87 + 77.88 =
  142.75 ms), so the one-GPU baseline is faster.
- **Where the gap to 2× comes from.** Together, the imbalance and the packing explain it.
  Even with equal branches, scaling is 1.90× end to end.
- **vLLM-Omni.** Its one-GPU step (about 176 ms) is about twice its two-GPU step, which is
  consistent with running both branches at equal cost. Its higher ratio comes from a slower
  baseline. sglang is faster in absolute terms at both GPU counts.

## Not run

Throughput mode (FLUX.3 is not in the representative throughput pair), other hardware,
TensorRT-LLM (draft PR #20430), LightX2V and FastVideo (no FLUX.3).

The first-day sglang-only report, `flux3-klein-base-h200-20261016`, used the short prompt
(3.631 s on two GPUs). Its numbers are not comparable with these.
