# Qwen-Image-2.1 across datacenter and consumer GPUs

2026-09-21 · sglang `6ad78f22` · `Qwen/Qwen-Image-2.1` rev `b3179ad3`
· 1024×1024, 40 steps, guidance 1.0 (no CFG), seed 42, 1 output, PNG
· eager, native BF16/FP32, no cache

First measurement of Qwen-Image-2.1 in this repo. Support landed in sglang on
2026-09-17 (#39983), so every box was moved to `origin/main` HEAD before
measuring; all four rows below share one sglang commit and one checkpoint
revision.

Single framework for now. vLLM-Omni and TensorRT-LLM VisualGen cannot serve
this model on their main lines; LightX2V can and is not yet profiled here
(see *Framework coverage*).

## Single GPU

`scripts/devbox_run_cases.sh` → `run_comparison --modes single_e2e`, median of 5
measured requests after warmup, client-side wall clock. One rx devbox per card.

| GPU | VRAM | p50 | min–max | s/step | vs RTX 4090 | selected profile |
|---|---|---|---|---|---|---|
| B200 | 183 GB | **2.467 s** | 2.463–2.472 | 0.0539 | 7.84× | `b200-1gpu-resident-fa` |
| H200 | 143 GB | **4.535 s** | 4.529–4.538 | 0.1034 | 4.27× | `h200-1gpu-resident` |
| RTX 5090 | 32 GB | **14.122 s** | 14.076–14.134 | 0.3173 | 1.37× | `rtx5090-1gpu-encoder-layerwise-offload` |
| RTX 4090 | 24 GB | **19.343 s** | 19.293–19.352 | 0.4361 | 1.00× | `rtx4090-1gpu-encoder-layerwise-offload` |

Every row repeats to under 0.4%. The denoise loop is essentially the whole
request: 40 × s/step accounts for 2.16 s of B200's 2.47 s, 4.14 s of H200's
4.54 s, 12.69 s of the 5090's 14.12 s and 17.44 s of the 4090's 19.34 s.

Placement differs by necessity, not by tuning. The checkpoint is 30.9 GiB, so
the cookbook disables resident placement on both consumer cards — "the full
resident pipeline exceeds one consumer GPU's memory". H200 and B200 run fully
resident; RTX 5090 and 4090 keep the DiT and VAE on the card and stream the
Qwen3-VL encoder's layers. Attention is each card's platform kernel:
FlashAttention on Hopper/Blackwell/sm89, Torch SDPA on sm120 (the runtime picks
it, and asking for FA there falls back to SDPA anyway).

`--performance-mode speed` does **not** turn on torch.compile here — unlike
qwen-image-2512, where it has to be paired with `--enable-torch-compile false`.
The runtime refuses compile for this model on its own (`torch.compile disabled
for this model`, resolved `enable_torch_compile: false`), so all four rows are
eager, which is what the best-lossless policy wants anyway.

## The RTX 5090's published recipe is the wrong one

The cookbook's `verifiedRecipes` table marks exactly one entry `unverified:
true` — `rtx5090-1-offload` — and spells out why: *"This RTX 5090 recipe has not
been retested with the updated checkpoint."* It works. It is also not the
fastest thing that card can run, by a wide margin.

A/B on one box (`bench-5090-qi21`), same card, same Torch SDPA, same 40 steps /
seed 42 / CFG 1 / PNG, same checkpoint snapshot; only component placement
differs, which does not change numerics:

| RTX 5090 recipe | p50 | s/step | steady VRAM | peak at load |
|---|---|---|---|---|
| cookbook: `--dit-layerwise-offload true` | 19.946 s | 0.4674 | 19.3 GB | 19.3 GB |
| 4090's: `--component-residency dit=resident text_encoder=layerwise-offload vae=resident` | **14.082 s** | **0.3166** | **17.0 GB** | 24.9 GB |

**1.42× faster on less steady memory.** Each arm repeats to 0.1% and the two do
not overlap. Both return a PNG of identical byte length (2 432 136), consistent
with placement leaving the output alone.

The cause is structural: the cookbook's 5090 branch streams the DiT through
every one of the 40 steps, while the 4090's keeps the DiT resident and streams
only the encoder, which runs once per request. Upstream gates the DiT-resident
branch on `s.hw === "rtx4090"`; on this evidence it should cover the RTX 5090.
`proposed-sglang-cookbook-rtx5090.patch` makes that change. It applies to sglang
`6ad78f22`, and the patched config was *executed* to confirm it moves only the
one cell: RTX 5090 on its platform kernel (or on an `fa` request, which falls
back to SDPA there) now resolves to the DiT-resident recipe, while RTX 5090 with
SageAttention, every RTX 4090 selection and every RTX PRO 6000 selection resolve
exactly as before. It also drops `unverified: true` from `rtx5090-1-offload`,
which this run earns. **The patch has not been submitted.**

This is also why the 24 GB card lands close behind the 32 GB one: the 4090 was
already running the better recipe.

## Two GPUs: TP, Ulysses and CFG-parallel

One 2×B200 box, sglang `176dbcb8` (same day, a few commits past the single-GPU
runs — its own 1-GPU no-CFG arm reads 2.497 s against the harness's 2.467 s, so
the cross-commit difference is inside the noise). Probe form, median of 3 after
warmup; the tracked case stays 1 GPU.

| workload | topology | p50 | vs 1 GPU | VRAM/card |
|---|---|---|---|---|
| no-CFG | 1 GPU | 2.497 s | — | 39.7 GB |
| no-CFG | `--tp-size 2` | **1.626 s** | 1.54× | 26.6 GB |
| no-CFG | `--ulysses-degree 2` | 1.703 s | 1.47× | 37.4 GB |
| true-CFG | 1 GPU | 4.827 s | — | 39.7 GB |
| true-CFG | `--tp-size 2` | 3.106 s | 1.55× | 26.6 GB |
| true-CFG | `--cfg-parallel-size 2` | **2.601 s** | **1.86×** | 35.7 GB |

**CFG-parallel is the multi-GPU answer whenever the request actually uses CFG.**
1.86× is near-linear, and it beats TP on the same workload by 1.19×, because
each GPU runs one whole branch and the two combine once per step instead of
all-reducing inside every block. It matches what this repo already found for
Qwen-Image-2512: CFG parallelism is the reliable Qwen multi-GPU speedup, TP is
not the latency path. Worth noting how far it goes — true-CFG on two GPUs
(2.601 s) costs barely more than no-CFG on one (2.497 s).

CFG-parallel cannot be measured on the tracked no-CFG workload at all. With one
branch, rank 1 has no work and the runtime rejects the request rather than hang
on a gloo timeout, so the true-CFG arms use this repo's existing Qwen
convention: `true_cfg_scale: 4.0` plus the standard negative prompt, two
transformer forwards per step. They are a different amount of work and are only
compared with each other.

TP also shards the weights where Ulysses replicates them (26.6 GB vs 37.4 GB per
card), which is the lever that matters if a bigger checkpoint is next.

### TP vs Ulysses, interleaved

The sweep above ran the arms in sequence, and its 4.7% gap sits under this
repo's 5% bar, where A-then-B is not evidence — a box drifts, and whichever arm
ran second wears the drift. So the two were re-measured ABAB, restarting both
servers every round because what is being compared is a launch configuration.

| round | `--tp-size 2` | `--ulysses-degree 2` | faster |
|---|---|---|---|
| 1 | 1.6217 s | 1.7566 s | TP by 7.7% |
| 2 | 1.6246 s | 1.7407 s | TP by 6.7% |
| 3 | 1.6171 s | 1.9309 s | TP by 16.3% |

**3/3 rounds agree in sign**, median 1.6217 s against 1.7566 s — TP ahead by 7.7%.

Read the median, not round 3. TP repeats to 0.5% across the three rounds
(1.6171–1.6246) while Ulysses' third round comes in 11% above its own first
two, so 16.3% is that round's noise rather than the effect. The finding is the
sign agreement plus the ~7% the two clean rounds and the medians all give.

Per step the same ordering shows up directly: 0.0343 s TP against 0.0375 s
Ulysses, versus 0.0547 s on one GPU. A plausible reason the ordering differs
from Qwen-Image-2512 — where Ulysses won — is block count: this DiT has 32
blocks against 2512's 60, so TP pays roughly half the per-block all-reduces for
the same GEMM saving. This run did not profile the kernels, so that is a
hypothesis, not a measurement.

Note what is and is not being said about the cookbook. It carries verified
recipes for both (`b200-2-tp` and `b200-2-ulysses`), so neither is unsupported;
what its `autoTopology` does is pick Ulysses whenever the user asks for more
than one GPU (`{tp_size: 1, ulysses_degree: gpus_per_node, ring_degree: 1}`).
At two B200s that is the slower of the two. Four GPUs were not measured, and the
balance can move with degree, so this says nothing about `autoTopology` at 4.

## Against LightX2V

LightX2V is the one competitor whose main line serves this model, so it is the
one that produces a number. It ran in **upstream's own published image** with
**upstream's own config** — `lightx2v/lightx2v:26062001-cu130` (and
`...-cu130-5090-fix-260921` for the consumer card), LightX2V main `4d709c43`,
`configs/qwen_image_21/qwen_image_21.json` on H200 and `..._5090.json` on the
RTX 5090. Its `base.sh` banner confirms BF16 with no sensitive-layer override,
so this is the lossless path: FlashAttention3 (FA2 on sm120), FlashInfer RoPE,
Triton LayerNorm and modulation, fused QK RMSNorm, fused block ops, CFG off.
Explicitly **not** the FP8 scripts in the same directory, which quantize the DiT
linears and switch to dense SageAttention2 — upstream's published RTX 5090
figure (5.930 s) comes from those and is not comparable with anything here.

Same workload throughout: 1024×1024, 40 steps, seed 42, no CFG, one image.

| GPU | | sglang | LightX2V | |
|---|---|---|---|---|
| H200 | client p50 | 4.522 s | 4.505 s | not separable (see below) |
| | server-side | 4.342 s | 4.443–4.477 s | sglang ~2.6% ahead |
| | VRAM after load | 39.6 GB | 36.5 GB | |
| RTX 5090 | client p50 | **14.087 s** | 14.513 s | **sglang 3.0% ahead**, 3/3 rounds |
| | server-side | 13.879 s | 14.10–14.20 s | sglang ~1.9% ahead |
| | VRAM | **17.0 GB** steady | 30.7 GB | **sglang on 55% of the memory** |

Both client rows are medians of three interleaved rounds of five requests. Two
of these figures survive the measurement: the RTX 5090 speed gap, whose rounds
agree in sign, and the RTX 5090 memory gap, which is not close to noise. The
H200 speed rows do not — see below.

### H200, interleaved: no resolvable speed difference

The arms were re-measured ABAB on the same node — three rounds, five measured
requests each, both servers restarted every round, and each measurement gated on
the neighbouring devbox's GPU being idle (the two boxes share a host, and a busy
neighbour contaminates latency through it).

| round | sglang | LightX2V | faster | LightX2V's five samples |
|---|---|---|---|---|
| 1 | 4.515 s | 4.505 s | LightX2V by 0.2% | 4.50, 4.50, 4.50, 4.50, 5.00 |
| 2 | 4.524 s | 5.004 s | sglang by 9.6% | 4.51, 4.50, 5.00, 5.01, 5.01 |
| 3 | 4.522 s | 4.505 s | LightX2V by 0.4% | 4.50, 4.50, 4.50, 4.51, 5.00 |

**The paired diffs disagree in sign, so there is no resolvable difference.** An
earlier 7-repeat run had read 5.004 s and I reported sglang 9.4% ahead on that
basis; interleaving shows that was one round of a coin flip, and the claim is
withdrawn.

What the rounds do resolve is the *shape*. sglang's fifteen samples span
4.51–4.57 s. LightX2V's fifteen are ten at ~4.50 and five at ~5.00, with
nothing in between — a 0.5 s tail on a third of requests. Which median a round
reports is decided by how many landed in each mode.

### RTX 5090, interleaved: sglang 3.0% ahead, and it holds

Same protocol, same node, three rounds:

| round | sglang | LightX2V | faster |
|---|---|---|---|
| 1 | 14.087 s | 14.514 s | sglang by 2.9% |
| 2 | 14.096 s | 14.513 s | sglang by 2.9% |
| 3 | 14.080 s | 14.512 s | sglang by 3.0% |

**3/3 rounds agree in sign** and the round-to-round spread is 0.1% on both
sides, so unlike the H200 this one resolves. LightX2V is unimodal here — thirteen
of its fifteen samples read 14.51 s and the other two 14.52 s — because its
compute sits safely below the 14.5 s tick boundary instead of straddling it.

About a third of the 3.0% is that tick: server-side the two are 13.879 s against
14.10–14.20 s, a ~1.9% gap. Both numbers are worth having. 1.9% is what the
frameworks cost; 3.0% is what a client waits.


### Where the 0.5 s comes from

Not from compute. The LightX2V server log puts `RUN pipeline` at
**4.443–4.477 s** across eight requests — a 0.8% spread, no bimodality at all,
with the DiT itself at 4.27–4.32 s. The two client modes are the same compute
plus different waiting.

The waiting is quantised. **Every client latency measured against LightX2V lands
within 0.010 s of a multiple of 0.5 s** — across 15 H200 samples and 10 on the
RTX 5090 — while its own server-side times (4.44–4.48, 14.10–14.20) are nowhere
near one. Its sync endpoint appears to release responses on a ~0.5 s tick, so a
client sees the true cost rounded up to the next half second. On H200 the
compute sits right at the 4.5 s boundary and jitters across it, which is the
bimodality; on the RTX 5090 it sits safely below 14.5 and pays a flat ~0.36 s,
seven times out of seven.

That has a consequence for the whole comparison worth stating plainly: **the
tick is coarser than the difference being measured.** 0.5 s is 11% of the H200
workload and 3.4% of the RTX 5090 one, against framework gaps of a couple of
percent. On the client metric — which is this repo's headline, and rightly so,
being the only framework-agnostic number — these two frameworks are simply not
separable on this model. Reporting either as faster would be reporting the
quantiser.

The server-side numbers are the ones that can separate them, and they are
diagnostics, not the headline:

| | sglang | LightX2V | overhead sglang / LightX2V |
|---|---|---|---|
| H200 | 4.342 s | 4.443–4.477 s | 0.19 s / 0.04 or 0.54 s |
| RTX 5090 | 13.879 s | 14.10–14.20 s | 0.24 s / ~0.36 s |

sglang is ~2.6% and ~1.9% ahead there, and its serving overhead is constant
where LightX2V's is quantised. Both gaps are small enough that they are worth
stating as "sglang is not behind", not as a win.

The two produce the same image in the same format — checked, not assumed. Both
return a 1024×1024 PNG at `color type 6 (RGBA)`, and LightX2V's alpha is real
data rather than padding (range 253–255). The apparent size gap between them is
an artifact of the response envelope, not of the image: sglang's 2.43 MB is
base64-wrapped JSON around a ~1.83 MB PNG, while LightX2V's sync endpoint
returns 1.88 MB of raw PNG. So there is no encode-cost asymmetry here, only
~0.55 MB more on the wire for sglang, which over loopback is milliseconds.

(An earlier draft of this report inferred from those two numbers that LightX2V
was emitting three channels and that sglang was therefore paying for a fourth.
Decoding the PNG header showed that was wrong. The inference is removed rather
than softened.)


## Against vLLM-Omni, on a branch

vLLM-Omni cannot serve this model on main or in any release; support sits on the
open PR [#7759](https://github.com/vllm-project/vllm-omni/pull/7759), opened
2026-09-18 and still open. That makes "cannot serve it" the answer this repo's
version policy gives, and it is why the tracked config still classifies
vllm-omni `unsupported` for this case. The number below is a labelled
pre-release data point and is deliberately outside the matrix.

Installed and served exactly as recipes.vllm.ai documents: PR branch
`qwen-image-2.1` at `3ea9e605011f`, `vllm==0.29.0`, torch 2.13.0+cu132,
`vllm serve Qwen/Qwen-Image-2.1 --omni --step-execution --max-num-seqs 8`.
Same B200 box and same workload as the sglang row.

| B200, 1024×1024, 40 steps, no CFG | p50 | VRAM after load | returned |
|---|---|---|---|
| sglang | **2.467 s** | 39.7 GB | 1024² RGBA PNG, 1.83 MB |
| vLLM-Omni (PR #7759) | 3.421 s | 34.0 GB | 1024² RGBA PNG, 4.19 MB |

**sglang is 1.39× faster**, and that is a floor rather than a headline, because
the two asymmetries here both favour vLLM-Omni:

* **Its PNG is not compressed.** 4,196,804 bytes against 1024×1024×4 = 4,194,304
  bytes of raw pixels — 2.5 kB of overhead, i.e. stored deflate blocks. sglang
  compresses the same image to 1.83 MB and pays the encode for it. Over loopback
  the extra 2.3 MB of transport costs milliseconds; skipping deflate on a 4 MB
  RGBA buffer saves considerably more than that.
* **CUDA graphs are on by default** in that serve command, where sglang runs
  eager — the cookbook marks breakable CUDA graph `soft`/unverified for this
  model ("tested requests fell back to eager"), so it was not enabled. If
  vLLM-Omni had come out ahead, the first conclusion would have been that sglang
  should make BCG work here rather than that its kernels are slower.

Both sides produce the same thing at the same size — the output was decoded and
checked (`1024x1024, bit depth 8, color type 6 (RGBA)`) before this row was
written, because a 2.3× difference in response size is exactly the shape of a
comparison that is secretly measuring two different images.

Upstream's own published figure for this PR is 4.5 s on a GB300 at 34.0 GB peak;
the peak memory matches what was measured here.


## Framework coverage

Checked against each project's main line on 2026-09-21, then actually run where
a main line supports the model. "Cannot serve it" is a result too, but it is a
different claim from "is slower", and this section keeps them apart.

| framework | status | evidence |
|---|---|---|
| sglang | measured | all rows above |
| LightX2V | **measured** | main line serves it; run in upstream's image with upstream's lossless config — see *Against LightX2V* |
| vLLM-Omni | unsupported on main; **measured on the PR branch** | `supported_models.md` lists Qwen-Image, -2512, -Edit{,-2509,-2511} and -Layered, not 2.1; a code search for `QwenImage21` / `Qwen-Image-2.1` over the repo returns nothing. Support is on the open PR [vllm-project/vllm-omni#7759](https://github.com/vllm-project/vllm-omni/pull/7759) (opened 2026-09-18, still OPEN); recipes.vllm.ai documents `vllm serve Qwen/Qwen-Image-2.1 --omni` against that checkout and says support is not in a tagged release |
| TensorRT-LLM VisualGen | unsupported | has `_torch/visual_gen/models/qwen_image/`, but no `QwenImage21` anywhere and nothing for 2.1 in `pipeline_registry.py`. 2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder) |
| LightX2V (details) | measured | ships `scripts/qwen_image_21/` including `server/start_server.sh` + `server/post_t2i.py`, so it is a real serving path, unlike the Qwen-Image-2512 case. **Its published RTX 5090 figure (T2I 1024², 5.930 s) is not comparable to this report**: that script quantizes the DiT linears to FP8 with FP16 accumulation and uses dense SageAttention2, both lossy under this repo's policy. The comparable command is the general config (FlashAttention3, FlashInfer RoPE, Triton LayerNorm, fused QK RMSNorm, CFG off — all lossless), and FA3 is Hopper-only, so a consumer-card profile still needs a same-precision-class attention pick |

Re-classify vLLM-Omni as soon as #7759 merges.

## Reproducing

```bash
python3 scripts/build_benchmark_config.py
# on a devbox, one GPU, with the case's hardware profile:
DBF_HARDWARE_PROFILE=b200 DBF_FRAMEWORKS=sglang \
  bash scripts/devbox_run_cases.sh qi21 0 47001 single_e2e qwen_image_21_t2i_1024
```

Three things will stop a fresh `lmsysorg/sglang:dev` box, all of them fixed in
this change and written up in the `diffusion-framework-benchmarking` skill:

- the image bakes an **expired** GitHub Actions token into
  `/sgl-workspace/sglang`'s git config, so `git fetch` on a public repo 401s and
  reports `could not read Username for 'https://github.com'`;
- several clusters export a read-only `HUGGINGFACE_HUB_CACHE`, which outranks
  `HF_HOME` and surfaces as `Could not get model info for 'Qwen/Qwen-Image-2.1'`
  — a message that reads as "unsupported model";
- the checkpoint emits **RGBA**, so a request without `output_format: png` dies
  in the JPEG encoder and reaches the client as a bare HTTP 500.

## Files

- `raw/harness_*.json` — `run_comparison` output, one per GPU class. These are
  the numbers the single-GPU table publishes.
- `raw/probe/*.log` — the probes that settled each card's recipe, including both
  arms of the RTX 5090 A/B.
- `raw/multigpu_2xb200.log`, `raw/abab_tp_vs_ulysses.log` — the 2-GPU sweep and
  its interleaved re-measurement.
- `raw/competitors/lightx2v_*.log` — LightX2V in upstream's own image, including
  the 7-repeat runs that exposed the bimodality, plus its server-side timings.
- `raw/competitors/abab_h200_sglang_vs_lightx2v.log` — the interleaved
  cross-framework rounds on one H200 node.
- `scripts/` — the probes, the cross-framework ABAB orchestrator and the
  paired-verdict analyser. They live with the report rather than in the repo's
  `scripts/`, because they drive files staged on specific devboxes.
- `proposed-sglang-cookbook-rtx5090.patch` — the upstream docs change this run
  justifies. Verified to apply to sglang `6ad78f22`; not submitted.
