# LightX2V's README table, re-run on RTX 4090 (2026-09-29)

LightX2V's README carries a cross-framework table "Updated on 2025.12.01" for
Wan2.1-I2V-14B-480P (40 steps, 81 frames) on H100 and RTX 4090D, 1 and 8 GPUs,
reporting step time only. Its SGL-Diffusion rows name no version, precision or
launch command, and read OOM on the 4090D. SGLang-Diffusion first shipped on
2025-11-07 (v0.5.5), so those rows describe its first weeks. This run measures
the same workload on current versions of both frameworks, on the consumer class
of that table: RTX 4090 (not 4090D, which has 114 SMs against 128).

## Results

Client wall clock for one 40-step request, the cell's median over its measured
repeats after warmup. 1 GPU ran on the same card for both frameworks in the
order A B B A; 8 GPUs ran A B A B. BF16 DiT, exact attention, CFG on, no cache
or quantization, compile off on both sides; the encoders and the VAE run at each
framework's default precision (see Setup).

| GPUs | SGLang-Diffusion | LightX2V | LightX2V / SGLang |
|---|---:|---:|---:|
| 1 x RTX 4090 | 840.1 s, 841.1 s | 871.4 s, 871.4 s | 1.04x |
| 8 x RTX 4090 | 198.8 s, 200.0 s | 214.3 s, 205.5 s | 1.08x, 1.03x (paired) |

Every repeat sits within 0.3% of its cell's median. The 1-GPU pairs agree to
0.1%; LightX2V's two 8-GPU cells differ by 4%, so the 8-GPU gap is given per
pair rather than as one ratio.

Step time, the README's metric, is the denoise stage over 40 steps:

| GPUs | SGLang-Diffusion | LightX2V |
|---|---:|---:|
| 1 | 20.7 s/it | 21.6 s/it |
| 8 | 4.84 s/it | 5.1-5.3 s/it |

LightX2V's `Run DiT` timer includes VAE decoding (4.2 s on one GPU), which is
subtracted here. Peak memory per GPU: 13.3 GB against 22.9 GB on 1 GPU, 12.6 GB
against 17.4 GB on 8.

Outside the denoise loop, SGLang's image VAE encode is the slow stage: 6.9 s
against LightX2V's 2.6 s on one GPU, 2.2 s against 0.4 s on eight. That is
precision, not placement. SGLang encodes with Wan's VAE in FP32 (its T5 and CLIP
also run in FP32), LightX2V in BF16. Holding SGLang's VAE resident instead of
streaming it changes nothing (`raw/probes/probe_sgl_*_vae_resident`, same
command plus `--component-residency vae=resident`):

| | VAE streamed (published cells) | VAE resident |
|---|---:|---:|
| 1 GPU: VAE encode / decode | 6.87 s / 3.16 s | 6.84 s / 3.13 s |
| 1 GPU: e2e | 840.1 s, 841.1 s | 841.3 s |
| 8 GPUs: VAE encode / decode | 2.24 s / 1.37 s | 2.16 s / 1.32 s |
| 8 GPUs: e2e | 198.8 s, 200.0 s | 200.4 s |

Supplementary LightX2V cells on 8 GPUs, same box, after the ABAB block:

| variant | e2e |
|---|---:|
| its timers off (`PROFILING_DEBUG_LEVEL=0`; `pipeline.py` defaults it to 2) | 207.2 s |
| its compile path (`use_compile` and `warmup`, as `configs/wan/wan_i2v_compile.json`) | 202.2 s |

The timers cost nothing measurable. Compile is LightX2V's best configuration
here, 1.4% behind SGLang's eager mean of 199.4 s.

### Against the README

- **SGL-Diffusion OOM rows.** On `c7be3e9` with no placement flags, still true on
  1 and 8 GPUs (below). With one flag, `--layerwise-offload-components dit,...` or
  `--performance-mode memory`, it runs at the numbers above. With the sglang fixes
  described below, no flag is needed.
- **LightX2V 1 x 4090D, 20.26 s/it.** LightX2V measures 21.6 s/it here at BF16
  with exact attention, on a card with 12% more SMs. At 3.3 PFLOP per CFG step
  and the rates in the ceiling check, a BF16 step with exact attention cannot go
  much below 20.6 s on a 4090, so that row is unlikely to be the same precision.
  LightX2V's own 24 GB-class configs default to FP8 and SageAttention.
- **8 GPUs.** The README's 4090D rows (LightX2V 4.75 s/it) come from an unstated
  topology. This box has no P2P between any pair of GPUs, which is the worst
  case for sequence-parallel all-to-alls.

## Setup

| item | value |
|---|---|
| box | rx devbox `lx2v-repro-0929`, cluster 4090-atlas-k8s: 8x RTX 4090 24 GB, PCIe, `SYS` between every GPU pair (no P2P), each GPU on its own NUMA node, 128 cores, 1 TB RAM |
| SGLang-Diffusion | main `c7be3e935b50` (2026-09-29), torch 2.13.0+cu130, image `lmsysorg/sglang:dev` |
| LightX2V | main `d46ab933b76e` (2026-09-28), 0.5.0, its own venv with torch 2.11.0+cu130 (the version its image ships), flash-attn 2.8.3 built from source |
| weights | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` @ `b184e23a` (SGLang), `Wan-AI/Wan2.1-I2V-14B-480P` @ `6b73f84e` (LightX2V, original layout, same weights) |
| workload | image-to-video, 480x832, 81 frames, 40 steps, guidance 5.0 (CFG on), flow shift 3.0, seed 42, reference `cat.png` from the diffusers documentation images |
| precision | DiT in BF16 on both. Encoders and VAE at each framework's default: SGLang runs the T5 text encoder, CLIP image encoder and VAE encoder in FP32 and the VAE decoder in BF16; LightX2V runs all of them in BF16 (its `DTYPE` default). Exact attention (SGLang `fa`, LightX2V `flash_attn2`; FA3 is Hopper-only); no cache, no quantization, no distillation |
| compile | off on both sides (`DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=1`, LightX2V `use_compile: false`) except the supplementary compile cell |
| harness | this repo's `run_comparison` at `270599b` via `scripts/run_cell.sh`, client-side wall clock, one cell at a time on an otherwise idle box |

Commands are in [config.json](config.json); the probes that chose them are in
`raw/probes/`.

## Why SGLang needs a placement flag on `c7be3e9`

With no placement flags the first request fails with CUDA OOM, on 1 GPU and on 8
(`probe_sgl_1gpu`, `probe_sgl_8gpu_auto`). The auto policy streams the text
encoder, image encoder and VAE but leaves the 30.5 GiB DiT on component offload,
which moves it onto the card whole per request. `WanT2V480PConfig`, inherited by
every Wan2.1 variant, lists only `memory` in `dit_layerwise_offload_modes`, and
the runtime warns about it at startup and proceeds.

SGLang's own OOM message offers eight remedies. Measured against this case:

| remedy in the message | result |
|---|---|
| check free memory on every GPU | diagnostic; all eight were idle (23.1 GiB free each) |
| single GPU: component CPU offload | this is the default, and it is what OOMs |
| single GPU: `--dit-layerwise-offload` | OOM in server warmup (`probe_sgl_1gpu_legacy2`): the legacy flag streams only the DiT and leaves the FP32 text encoder (~22 GB) on component offload |
| single GPU / runtime: `--performance-mode memory` | runs, 840.4 s; same placement as the explicit flags (`probe_sgl_1gpu_memmode`) |
| multi-GPU: keep `auto` | OOM on every rank |
| multi-GPU: `--use-fsdp-inference true` | OOM (`probe_sgl_8gpu_fsdp`): with component offload on, the DiT loaded as a plain module and FSDP was never applied; adding `--dit-cpu-offload false` runs, 347.1 s (`probe_sgl_8gpu_fsdp2`), all-gathers crossing the host |
| runtime: reduce resolution, frames or batch | changes the workload; the weights are what do not fit |
| runtime: SP / Ulysses / Ring | already CFG 2 x Ulysses 4; sequence parallelism does not shard weights |
| runtime: lower-memory attention or quantization | attention is not the problem; quantization is lossy |

The published SGLang cells use `--layerwise-offload-components dit,text_encoder,image_encoder,vae`,
the set auto already picks plus the DiT. On 8 GPUs that beats FSDP by 1.75x on
this topology (198.9 s against 347.1 s).

## The sglang fixes

Three pull requests against sglang main, each validated on this box on its own
branch tip with the commands that used to fail (`raw/sglang-fix-validation/`:
`final_a_*` for #41721, `split_b_8gpu_fsdp` for #41722):

- [sgl-project/sglang#41721](https://github.com/sgl-project/sglang/pull/41721): in auto mode, a model whose DiT layerwise offload is validated in
  memory mode streams the DiT when its weights exceed 80% of the free memory on
  the least-free selected GPU. The size comes from the transformer's safetensors
  headers: the local files or cache when present, otherwise only the headers
  from the Hub (under 0.1 s for this checkpoint from the local cache, about 6 s
  for the T2V-14B's 14 shards from the Hub, about 1 s for the 1.3B Fun-InP).
  Counted here: 16,395,083,584 parameters, 30.5 GiB in BF16, matching the
  loader's own report. A DiT that cannot be sized, is stored quantized, is
  replaced or quantized by flags, or is served by ModelScope keeps the declared
  placement, as do the 1.3B models and cards of 48 GB and up.
  No flags: 834.1 s (20.73 s/step) on 1 GPU, 196.5 s (4.81 s/step) on 8.
- [#41722](https://github.com/sgl-project/sglang/pull/41722): `--use-fsdp-inference true` on several GPUs keeps the DiT off
  component offload, so FSDP actually shards it. That flag alone on 8 GPUs:
  346.7 s (8.50 s/step).
- [#41723](https://github.com/sgl-project/sglang/pull/41723): the OOM message recommends the placements that ran, and the Wan2.1
  cookbook page gains a 24 GB section with the commands above and their host
  memory: each GPU's worker keeps its own copy of the streamed weights, 59 GiB
  on 1 GPU and 483 GiB on 8 (`raw/probes/hostmem_probe.log`).

The branch cells ran on a newer main (`98fce73`) than the published ones. Its
conditioning cache ([#40470](https://github.com/sgl-project/sglang/pull/40470))
serves the harness's repeated prompt and image, so the measured request skips
text and image VAE encoding (0.003 s and 0.5 s here, instead of 0.8 s and
6.9 s): compare those cells per step. Later rounds should vary the prompt per
request or pass `--disable-conditioning-cache` to sglang.

## LightX2V's configuration

LightX2V's own 24 GB-class Wan2.1 I2V configs quantize DiT, T5 and CLIP to FP8
(`configs/offload/block/wan_i2v_block.json`), and the phase-offload variant adds
SageAttention2 (int8 QK). Both are dropped here: the cells keep its block offload
at BF16 with FlashAttention 2. The 8-GPU cell uses its `ulysses-4090`
sequence-parallel variant (round-robin all-to-all for hosts without full P2P),
CFG 2 x Ulysses 4, the shape of `configs/dist_infer/wan_i2v_dist_cfg_ulysses.json`.
Its harness warmup is one full request, because LightX2V takes the step count
from its launch config only.

## A ceiling check

One denoise step is two DiT forwards (CFG) over 32,760 tokens: about 0.78 PFLOP
of GEMMs and 0.88 PFLOP of attention per forward, 3.3 PFLOP per step. Measured
on an idle card (`raw/supplementary/tflops_probe.log`): BF16 GEMMs at the DiT's
shapes 158-161 TFLOPS, exact attention 160 (SDPA) and 164 (FlashAttention 2)
TFLOPS. That puts one BF16 step with CFG at about 20.6 s; SGLang's 20.7 s is
99.5% of it and LightX2V's 21.6 s 95%. Under the full workload the cards sit at
the 450 W cap around 2.6 GHz.

## Harness issues found on the way

- `kill_server` gave a server 10 s to be reaped after SIGKILL and raised past
  that; an 8-rank LightX2V server with block offload unpins tens of GiB of host
  memory per rank at exit and outlasts it, so a completed 8-GPU cell crashed the
  harness before it wrote its JSON. Fixed in `270599b`, which every published
  cell ran with ([#5](https://github.com/mickqian/diffusion-bench-framework/pull/5)).
- The harness turns LightX2V compile on by writing `"compile": true` (and
  `compile_shapes` to turn it off). LightX2V main reads neither key; its switch is
  `use_compile` (`lightx2v/common/transformer_infer/transformer_infer.py`,
  verified in the installed `d46ab933` package). Every published LightX2V cell
  therefore ran eager while the stated policy says competitors run compile-on.
  [#6](https://github.com/mickqian/diffusion-bench-framework/pull/6) writes `use_compile`; it turns compile on for every LightX2V
  cell at once and needs its own validation round. On this case compile gives
  202.2 s against 205.5 s and 214.3 s eager.
- `REQUEST_TIMEOUT` is 1200 s, against 870-950 s single-GPU requests here.

## Files

- `config.json`: the two cases and the commands every published number ran.
  `config_hint_probe.json` starts sglang from no flags for the remedy probes;
  `config_lx2v_compile.json` is the supplementary compile cell.
- `scripts/`: `run_cell.sh` / `run_cell2.sh` (one cell), the probe, final,
  supplementary, VAE-placement and validation queues, `check_split_prs.sh` (unit
  tests and a Hub sizing check per sglang branch), `hostmem_probe.sh` (host
  memory of the 24 GB recipe), `summarize_cells.py` (per-cell numbers from the
  JSON, runlog and nvidia-smi samples), `tflops_probe.py`.
- `raw/final/`, `raw/supplementary/`, `raw/probes/`, `raw/sglang-fix-validation/`:
  harness JSON, runlogs and 1 s nvidia-smi samples for every cell;
  `raw/probes/hostmem_probe.log` for host memory.
