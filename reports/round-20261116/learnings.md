# Round 2026-11-16: what the other frameworks' changes teach, and where sglang stood

Follow-up to the published round `b300x4-20261116`. Everything here was measured on one RunPod 4xB300 box on 2026-11-18/19
(arms through `scripts/run_plan_ab.sh`, a fresh server per arm, mirrored rounds) unless a line says otherwise. Numbers are
single-request end-to-end medians in seconds. Pins: sglang `25c1c43942` (round pin) and main `9fbab1f6ed`; vLLM-Omni
`24e5472849` on vllm 0.33.0; TRT-LLM 1.4.0rc3; LightX2V `395e2ac49f`. Raw results: `results/r1118-b300/` (untracked).

Status: eleven sglang PRs, ten merged (sections 1, 2 and 6; #51563 is a draft). Every current cell has been measured against the sglang tree its own previous-round row ran (section 2). Section 5's confirmation rounds are in and the faster competitor commands are the B300-selected profiles; the published round predates them.

## 1. One cell where sglang was behind: FLUX.3 Klein Base (2 GPUs)

vLLM-Omni has served FLUX.3 since vllm-omni #9523 (2026-11-10), inside the round's pin. The case still classified it
`unsupported`, so the published round has no vLLM-Omni FLUX.3 row. Measured now (1024x1024, 50 steps, CFG 3.5, CFG parallel 2):

| framework, command | s |
|---|---:|
| sglang, round pin / main | 2.050 / 2.044-2.053 |
| vLLM-Omni `default` (CFG parallel 2, cuDNN attention) | 1.909 |
| vLLM-Omni + `--no-diffusion-compile-dynamic` | 1.901 / 1.903 / 1.905 |
| vLLM-Omni + FlashInfer attention (now its selected B300 profile) | 1.892 / 1.885 / 1.885 |
| vLLM-Omni TP2 / Ulysses 2 (batched CFG) | 2.254 / 4.851 |
| sglang main + the five PRs below | 1.897 / 1.894 (four of them, without the text cache: 1.902-1.906) |

sglang main with synchronized stage timers: text 0.044, denoise 1.916 (38.3 ms/step), decode 0.066, outside the stages 0.030.

What it was not: host launch cost (the breakable CUDA graph cut the host loop to 1.617 s and the request stayed at 2.051 s),
per-step host syncs (none in the trace), the output path (PNG 3.5 ms + file 0.8 ms on the worker), the attention backend
(`fa` 2.010 / 2.012 against 2.009 / 2.004), `--encoder-parallel dp` (2.046), `--async-output-save` (2.049), a channels_last
decoder (1.946 / 1.949 against 1.902 / 1.906: slower).

What it was, and the fixes (each measured alone against main, mirrored):

| fix | sglang PR | alone | image against main |
|---|---|---|---|
| Under CFG parallel every step recomputed the static text stream: the per-step slice dropped `StaticStreams` and the gate excluded CFG parallel | sgl-project/sglang#51380 | 2.046 / 2.048 -> 2.006 / 2.001 (-2.1%) | identical bytes |
| The native SwiGLU called `sgl_kernel.silu_and_mul` (101 us per call at these widths); the shared `SiluAndMul` layer's in-tree kernel takes 38 us | sgl-project/sglang#51387 | 2.050 / 2.049 -> 1.975 / 1.973 (-3.7%) | PSNR 39.4-39.9 dB (differs only for gates below -15) |
| The image VAE sharded its decode across the CFG ranks although `vae_sp` is false; the sharded decode is slower than the serial one from 512x512 to 2560x2560 (1024x1024: 0.069 against 0.050 s) | sgl-project/sglang#51406 | 2.044 / 2.047 -> 2.027 / 2.029 (-0.9%) | identical bytes |
| Image responses through pybase64 and orjson | sgl-project/sglang#51318 | 5.3 ms per 1024x1024 response (host component) | identical bytes |
| The text stage encoded the (constant, empty) negative prompt on every request; it now goes through the conditioning cache like the other text stages | sgl-project/sglang#51411 | 2.041 / 2.048 -> 2.037 / 2.039 (-0.3%); text stage 0.043 -> 0.036 s | identical bytes |
| all five | | 2.047 -> 1.894-1.897 (-7.4%); synchronized: denoise 1.792, decode 0.052, text 0.036, outside 0.028 | |

The SwiGLU fix came from laying the two frameworks' torch-profiler tables side by side: GEMMs and cuDNN SDPA were the same
kernels with the same call counts; vLLM-Omni's compiled SwiGLU ran 33 times for 1.75 ms per step, sglang's 38 times for 3.74 ms.

Where it stands (2026-11-19, a second 2xB300 box, the PR branches as pushed and merged onto main `5e32149fd6`, mirrored against
vLLM-Omni's selected FlashInfer command in one window):

| | s |
|---|---:|
| sglang main | 2.043 / 2.043 |
| sglang, the five open PRs of the table above and #51529 (fp32 VAE, the checkpoint's) | 1.901 / 1.901 |
| the same with sgl-project/sglang#51562 (the text encoder's forward replayed as a CUDA graph) | 1.882 / 1.881 |
| the same with `--component-precisions.vae bf16` (sgl-project/sglang#51529 makes the flag work) | 1.873 / 1.875 |
| vLLM-Omni `b300-cfg2-static-flashinfer` (bf16 VAE, no switch) | 1.887 / 1.890 |

With all seven changes sglang leads this cell by 0.3-0.5% while decoding at the checkpoint's fp32, and by 0.8% at vLLM-Omni's VAE
precision. The gap that was left after the first five PRs was never in the denoising loop (sglang 1.795 s; vLLM-Omni about
1.82 s by its progress bar) but in what surrounds it:

- Text encode, 0.036 s: an eager `transformers` forward of a 32-layer language model over a prompt padded to a multiple of 80
  tokens, almost all host time. Replayed as a CUDA graph it takes 0.018 s with the same bytes (#51562; Qwen-Image-2.1's stage
  already did this, and the class is now shared).
- Decode, 0.050 s in fp32 against 0.040 s in bf16. `--component-precisions.vae` was ignored for FLUX.3 because both VAE stages
  pinned the package's dtype (#51529); images against the fp32 decode read PSNR 40.0-41.2 dB.

Checked and ruled out: the attention kernel (FlashInfer's `cute-dsl`, which vLLM-Omni's FlashInfer backend selects on Blackwell,
is slower than cuDNN at this shape: section 6); the attention input layout (cuDNN SDPA takes transposed views, a strided QKV
projection and contiguous BHLD inputs in the same 0.14-0.16 ms); the batch hand-off between the CFG ranks (a 248 KB pickle,
0.05 ms); the time outside the stages (12 ms with a plain client, as for Z-Image under TP2). In the VAE decode (41.5 ms alone in
fp32, 32.8 in bf16) the mid attention over 16384 tokens takes 10 ms on fp32 SIMT GEMMs and 2.2 ms with TF32 matmul; TF32 is a
`high`-tier approximation by sglang's numerics policy, so it is not a change for the default tier. `cudnn.benchmark` does
nothing there and channels_last doubles the time.

The cell should still not be published before the PRs land and it is measured again.

## 2. Did sglang regress between rounds? One cell: LTX-2.5 two-stage

Across the two published rounds (different boxes) sglang's own rows read slower in seven cells: Cosmos3 T2I +4.6%, LTX-2.5
+2.7%, Qwen-Image-2.1 T2I +1.6%, Edit +1.5%, FLUX.3 +1.0%, Ideogram 4 +1.0%, Wan2.2 +0.4%. The previous round's rows did not
all run one sglang commit (the per-row commit is `sglang_runtime.git_commit` in each raw result; `merged.json`'s `commit_sha`
is the harness): `1ab008f5b0` (2026-10-29) for most, `ca819c11cd` (2026-11-02) for the two single-GPU Qwen-Image-2.1 rows,
`82f7249ce6` (2026-10-21) for FastH3, and for LTX-2.5 the head of sglang PR #47411 (`4e1a4bb5c3`, 2026-10-30). Each cell
against the tree its own row ran, one box per table, mirrored arms, medians. Image cells and LTX-2.5 on 2xB300 against main
`5e32149fd6`:

| cell | previous row's tree | that tree | main |
|---|---|---:|---:|
| Qwen-Image-2.1 T2I | `ca819c11cd` | 2.134 | 2.134 / 2.134 |
| Qwen-Image-2.1 Edit | `ca819c11cd` | 2.564 | 2.567 / 2.562 |
| Cosmos3 T2I | `1ab008f5b0` | 0.513 / 0.511 | 0.515 / 0.519 (0.506 / 0.515 an hour earlier) |
| FLUX.3 Klein Base | `1ab008f5b0` | 2.049 / 2.047 | 2.045 / 2.047 |
| Ideogram 4 (2 GPUs) | `1ab008f5b0` | 1.916 / 1.915 | 1.904 / 1.904 |
| Z-Image Turbo | `1ab008f5b0` | 0.342 / 0.343 | 0.311 / 0.314 |
| **LTX-2.5 two-stage** | `4e1a4bb5c3` | **5.091 / 5.118** | **5.188 / 5.172** |

The 4-GPU video cells on 4xB300 against main `2bf882aefc`:

| cell | previous row's tree | that tree | main |
|---|---|---:|---:|
| Cosmos3 T2V 189f | `1ab008f5b0` | 22.687 / 22.769 | 22.693 / 22.720 |
| Cosmos3 I2V 189f | `1ab008f5b0` | 22.839 | 22.701 |
| Wan2.2 T2V A14B | `1ab008f5b0` | 88.677 / 88.713 | 88.652 |

Six of the seven slower-reading cells are cross-box noise. LTX-2.5 is a real regression of 1.3-1.6%, and an earlier version
of this section missed it: its "old" tree was `ca819c11cd` for every cell, three days newer than the LTX-2.5 row's, with the
regression in between.

Bisected by the two denoise stages' own times (they repeat within 3 ms; the decode stage absorbs the asynchronous tail and
moves by 30 ms, which sent an end-to-end bisect to the wrong commit):

| tree | stage 1 | stage 2 |
|---|---:|---:|
| `4e1a4bb5c3`, and eight commits on main up to `e93f817325` (#47301's parent) | 1.017-1.021 | 1.534-1.542 |
| `504f9cf720` (sglang #47301), `0f92ddaefb`, `ca819c11cd`, main | 1.042-1.047 | 1.559-1.565 |

sglang #47301 moved Blackwell attention with one side under 128 tokens (LTX's 126 audio tokens against the video or the
text) from FA4 to cuDNN for `lossless` and `high` requests "in eager execution", because FA4's host dispatch loses there. Its
measurement was plain eager. The rule also caught the eager breaks of a graph replay, which run outside the capture scope:
LTX-2.5 under the breakable graph is 289 segments with 288 attention breaks per forward, and at its token counts (8160 video
tokens in stage 1, four times that in stage 2) cuDNN's fallback kernel for these shapes costs more than FA4's dispatch. The
fix keeps FA4 in a replay's breaks and leaves plain eager alone (sgl-project/sglang#51611). Same box, main `2bf882aefc` against the same tree with the fix, mirrored:

| arm | stage 1 | stage 2 | request |
|---|---:|---:|---:|
| main | 1.043 / 1.042 | 1.561 / 1.562 | 5.193 / 5.163 |
| fix | 1.020 / 1.021 | 1.539 / 1.544 | 5.135 / 5.132 |

FLUX.3, Ideogram 4 and Z-Image give byte-identical images on the two trees (no such attention shape), so the change reaches
only models with a short cross-attention side. The retired LTX-2.3 two-stage cell (768x512, CFG parallel, breakable graph;
1536 video tokens in stage 1) is such a model and does not move: stage 1 / stage 2 2.049 / 0.329 and 2.052 / 0.329 s on main,
2.050 / 0.327 and 2.049 / 0.328 s with the fix, the request 3.858 / 3.871 against 3.854 / 3.873 s. Its previous-round row
(3.905 s at `3042cea843`, another box) is not a regression either.

The fix's LTX-2.5 videos are 24.9-27.7 dB (per-frame PSNR, averaged per video) from main's. That is the band any two
numerically different runs of this model sit in: against `quality=exact` videos of the same requests main's lossless videos
are 23.9-28.3 dB and the fix's 25.0-26.7 dB, and two exact servers that differ only in the sequence-parallel exchange are
24.3-28.7 dB apart. The exact videos are identical with and without the fix.

Getting that reference found a second defect: on this server command (two GPUs, the runtime's own K/V-gather degree) every
`quality=exact` LTX-2.5 request failed with a shape error. `exact` keeps the reference masked path, the video-to-audio
attention passes the video's mask as a key mask, and the masked K/V-gather path read it as the query's. With
sgl-project/sglang#51620 the call takes the Ulysses exchange instead and the requests complete (6.757 / 6.751 s).

The other five cells on a third 4xB300 box (2026-11-21), each row's tree against the round's pin `25c1c43942`, mirrored;
request, then the denoise stage's own time:

| cell | previous row's tree | that tree: request / denoise | pin: request / denoise |
|---|---|---:|---:|
| FastH3 V2 | `82f7249ce6` | 5.096 / 5.089; 4.044 / 4.040 | 5.067 / 5.059; 4.029 / 4.027 |
| Qwen-Image-2.1 T2I 2 GPUs | `1ab008f5b0` | 1.384 / 1.386; 1.217 / 1.218 | 1.319 / 1.323; 1.155 / 1.156 |
| MiniMax-H3 T2VA 2 GPUs | `1ab008f5b0` | 62.890 / 62.882; 60.572 / 60.588 | 62.480; 60.145 |
| MiniMax-H3 Ref2VA | `1ab008f5b0` | 96.684 / 96.695; 92.482 / 92.542 | 96.072; 92.033 |
| **MiniMax-H3 T2VA 4 GPUs** | `1ab008f5b0` | 31.969 / 32.025; **30.535 / 30.536** | 31.596 / 31.615; **30.623 / 30.632** |

Every request is faster on the pin. One stage is not: H3 on 4 GPUs denoises 88-96 ms slower (+0.3%), repeatable within 9 ms,
under a request that is 1.2% faster. Bisected by that stage's time, the step is sglang #47565 (`67bd749ada`, 2026-10-31,
Ulysses attention pipelined over 7 head groups instead of 2 at this shape): denoise 30.547-30.560 -> 30.848 s, but the time
outside the stages 0.630-0.661 -> 0.200 s and the request 31.976-32.007 -> 31.840 s, the -0.16 s that PR claimed at 1344x768.
The asynchronous stage timer charged work to the denoise stage that it used to charge outside it. Not a regression.

So: one real regression between the rounds (LTX-2.5), fixed; every other cell flat or faster against the tree its own row ran.

The "Cosmos3 T2I got 5% slower" reading of the published tables was cross-box noise: five trees on one box agree within 1% by
the fastest sample, and sglang #49162 (the suspect) measures 0.512 before and 0.511 after. No other cell's row read slower
across the two rounds, and the H3 dp2 throughput p50 is 118.290 s (2026-11-02) and 118.278 s (this round): the
narrower throughput lead is all vLLM-Omni's own 2%.

## 3. Why the others got faster

| change | effect | does sglang need it |
|---|---|---|
| vLLM-Omni's stack moved to torch 2.14 / cuDNN 9.24 | cuDNN attention -3% per step in every `CUDNN_ATTN` video cell (cuDNN 9.20 -> 9.24, independent of the torch version) | has had it since 2026-10-04 |
| vllm-omni #4989 (FlashInfer attention as one custom op under `torch.compile`) | Z-Image -81 ms | compile-only |
| vllm-omni #7597 (one-launch QK-norm + RoPE) | Z-Image +18 ms under compile on B300 | sglang upper bound measured with an in-place fused kernel: -0.7%; not pursued |
| vllm-omni #10120 (stored PNG, pybase64, orjson) | outside-engine time 59 -> 16 ms | sgl-project/sglang#51318 (pybase64 + orjson): 5.3 ms per 1024x1024 response on the B300 host, not visible end to end |
| vllm-omni #9523 (FLUX.3) | a new cell | section 1 |
| vllm-omni `4b63581cf..24e547284` (2026-11-11 to 11-16; #9487 bf16 tensors over shared memory as raw bits, #10120, #10171 shared-memory wakeup, #9553 Wan-family VAE fast path) | same box, old and new stacks bucketed: FastH3 8.19-8.25 -> 7.70-7.71 (-6.2%), Qwen-Image-2.1 2 GPUs 1.498 -> 1.425-1.427 (-4.8%), Cosmos3 T2I 0.635 -> 0.586-0.603 (-6.5%); the stack move alone: Qwen -2.6%, Cosmos3 T2I -1.9%, FastH3 none | its own worker-to-API and response overhead; sglang's time outside the stages is 9-30 ms on the image cells and 4-45 ms on H3 (section 4) |
| TRT-LLM rc1 -> rc3 | per-request fixed cost about -1.1 s on H3; per-step unchanged | sglang's non-denoise time is already lower (section 4) |
| LightX2V `19c19614..395e2ac4` | Qwen-Image-2.1 2.560 -> 2.478 (-3.2%), Z-Image 0.831 -> 0.852 (+2.5%), the rest within 1.3% | not attributed upstream-side (its only Qwen hot-path change casts a Triton norm kernel's `eps` to fp32; its Z-Image FA2 calls now take the varlen entry); neither is a path sglang runs |

From the outcome side (2026-11-21): every competitor cell that moved by 1.5% or more between the two published rounds,
and what explains it.

| framework, cell | move | explained by |
|---|---:|---|
| vLLM-Omni Z-Image | -23.2% | #4989 (-81 ms under compile), #10120 (-43 ms outside the engine), #7597 (+18 ms) |
| vLLM-Omni Cosmos3 T2I, Qwen-Image-2.1 2 GPUs, FastH3 | -8.1%, -7.1%, -6.2% | the 11-11..11-16 window, bucketed on one box (-6.5%, -4.8%, -6.2%) |
| vLLM-Omni Cosmos3 I2V / T2V, LTX-2.5, Ref2VA, H3 2 GPUs, Wan2.2 | -4.0% to -2.1% | cuDNN 9.24 attention, -3% per step |
| vLLM-Omni Qwen-Image-2.1 Edit (and T2I, -1.5%) | -3.7% | #10268 restored the decode graph that #9272 dropped per request; the previous round's Qwen rows (2026-11-02) carried that regression |
| TensorRT-LLM H3 4 GPUs / 2 GPUs | -3.4% / -1.8% | about 1.1 s less fixed cost per request, per step unchanged (two-point fit) |
| LightX2V Qwen-Image-2.1 / Z-Image | -3.2% / +2.5% | not attributed: different boxes, and no hot-path change of note in its window; sglang leads both cells by 14% and 2.7x |
| ComfyUI FLUX.3 | -2.5% | #17196 fused Flux2 VAE; different boxes |

Nothing in this list is a change sglang lacks: each is already in sglang (cuDNN 9.24, response encoding, a persistent decode
graph), compile-only, or the competitor catching up on its own overhead.

Checked and closed without a change: the Qwen-Image-2.1 VAE padding fold that B300 rejects costs at most 5 ms (forced on:
2.126 / 2.130 against 2.135 / 2.132); LightX2V's fused anti-aliased SnakeBeta for the H3 vocoder (#1721) targets an audio
decode that takes 0.080-0.085 s of a request in sglang (1.6% of FastH3, 0.25% of H3 on four GPUs); the `SiluAndMul` layer
Z-Image and Cosmos3 use is already on the fast kernel.

## 4. sglang's real stage split (synchronized timers)

| cell | denoise | decode | text / other | outside stages |
|---|---:|---:|---:|---:|
| Wan2.2 T2V (4) | 87.727 | 0.761 | 0.100 | 0.143 |
| MiniMax-H3 T2VA (2) | 60.211 | 1.169 | 0.080 | 0.015 |
| MiniMax-H3 Ref2VA (2) | 92.047 | 1.166 | 0.609 text + 0.482 visual | 0.073 |
| MiniMax-H3 T2VA (4) | 30.707 | 0.696 | 0.084 | 0.004 |
| FastH3 V2 (4) | 4.222 | 0.685 | 0.084 | 0.045 |
| Cosmos3 T2V (4) | 21.444 | 1.272 | 0.002 | 0.017 |
| LTX-2.5 two-stage (2) | 3.310 | 1.766 | 0.107 | 0 |
| Z-Image (2) | 0.253 | 0.022 | 0.032 | 0.012 |
| Cosmos3 T2I (2) | | | | 0.023 |
| Qwen-Image-2.1 T2I (2) | | | | 0.012 |

The default stage timers are asynchronous: a stage's GPU work is charged to whichever later stage first waits on it, so
"decode" and the client-minus-stages tail absorb the denoise tail (Wan's decode read 2.95 s that way; it is 0.761 s). In every
video cell sglang's time outside denoising is below vLLM-Omni's (derived from its logs) and TRT-LLM's (two-point fit).
sglang has no first-step penalty (H3 on four GPUs: 30.707 s against 49 x 0.625 s).

## 5. Commands we were not giving the competitors

| framework, cell | option | result |
|---|---|---|
| TRT-LLM, MiniMax-H3 T2VA 2 / 4 GPUs | `parallel_config.async_ulysses` | -1.72% / -1.65% and -1.59% / -1.50%: promoted (`b300-omp8-async`, `b300-fa4-omp8-async`); the published TRT-LLM H3 rows predate it |
| vLLM-Omni, FLUX.3 | the whole cell | section 1; B300 now selects `b300-cfg2-static-flashinfer` (1.885-1.892) |
| vLLM-Omni | `OMP_NUM_THREADS=8`, FA4, FastH3 static compile | not gaps (2026-11-18 first box) |
| vLLM-Omni | `--ulysses-a2a-permute` (copy-engine Ulysses exchange, #9922) | two rounds: Qwen-Image-2.1 2 GPUs 1.435 / 1.426 -> 1.385 / 1.395, FastH3 7.785 / 7.683 -> 7.338 / 7.305, MiniMax-H3 4 GPUs 36.613 / 36.582 -> 34.779 / 34.764, Wan2.2 97.710 / 97.656 -> 94.642 / 94.561; one round: Cosmos3 T2V 26.037 -> 25.746, I2V 25.621 -> 25.347, LTX-2.5 5.797 -> 5.697: all promoted as B300 profiles (`b300-*-a2a`); sglang on the same box 1.311 / 5.040 / 31.530 / 88.967 / 22.750 / (I2V not run) / 5.172 |
| vLLM-Omni, MiniMax-H3 2 GPUs | Ulysses 2 with the copy-engine exchange in place of the selected TP2 | one round each: T2VA 67.114 -> 64.942 (-3.2%), Ref2VA 102.267 -> 99.703 (-2.5%): promoted (`b300-2gpu-usp2-static-a2a`); sglang on the same box 61.487 / 94.778, so the lead on these two cells is now 1.056x / 1.052x |
| vLLM-Omni | denoise-step CUDA graph | no gain (Z-Image 0.449 -> 0.447, Wan2.2 97.710 -> 97.783) |
| vLLM-Omni | text-encoder TP (Qwen 2 GPUs), Wan Ulysses 4 + batched CFG | 1.424 -> 1.434 (none); 98.464 (slower) |
| vLLM-Omni, Wan2.2 | `preencode_mp4` on every request | alone 97.517 -> 97.141 (-0.39%, one round); on top of the copy-engine exchange, two mirrored rounds: 94.526 / 94.495 -> 94.246 / 94.018 (-0.3% / -0.5%): promoted (`b300-4gpu-cfg2-usp2-a2a-preencode`); sglang 88.967, TRT-LLM 93.673 |
| TRT-LLM, MiniMax-H3 Ref2VA | `async_ulysses` | -0.44% on the first box, +0.36% on the second (104.074 against 103.704): not promoted |
| TRT-LLM | CUDA graphs | Wan2.2 93.673 -> 93.655 (none); on Cosmos3 the server exits before the health check |
| TRT-LLM | token-sharded TP (Wan2.2), TP2 x Ulysses 2 (MiniMax-H3 4 GPUs), text-encoder TP (MiniMax-H3) | 99.685 against 93.673 (slower); 38.330 against 35.749 (slower); server exits |
| LightX2V | `PROFILING_DEBUG_LEVEL=0` on the non-H3 cells (it defaults to 2: a device synchronize and a log line around every step; only our H3 entries set 0) | Qwen-Image-2.1 2.483 / 2.480 -> 2.449 / 2.446 (-1.4%): promoted (`b300-sdpa-noprof`); Z-Image, Cosmos3 T2V, Wan2.2: no difference |
| LightX2V, MiniMax-H3 | VAE compile, packed Ulysses exchange (`seq_p_tensor_fusion`, Triton pre/post) | T2VA 2 GPUs, two rounds: 71.286 / 71.338 -> VAE compile 70.281 / 70.211, packed exchange 70.788 / 70.627, both 69.644 / 69.632 (-2.4%); 4 GPUs, one round: 36.799 -> 36.262 (-1.5%); Ref2VA, one round: 112.33 -> 110.13 (-2.0%): promoted (`b300-sp2-sdpa-fusion-vaec`, `b300-sp4-sdpa-fusion-vaec`); sglang on the same box 61.487 / 31.530 / 94.778 |
| LightX2V, MiniMax-H3 T2VA | head-parallel Ulysses, fused QKV, traceable RMSNorm | 84.761, 73.879, 72.244: slower |
| LightX2V, Wan2.2 | VAE compile (#1646) | two rounds: 99.904 / 99.274 -> 99.529 / 99.004 (-0.38% / -0.27%): promoted (`b300-sdpa-vae-compile`); sglang 88.967 |

Fairness note for the FLUX.3 cell: vLLM-Omni decodes with a bf16 VAE where the checkpoint is fp32.

## 6. What sglang takes from the copy-engine exchange

vLLM-Omni's gain made the question concrete: sglang has had an N-rank copy-engine all-to-all since October, used by the
pipelined dense attention (MiniMax-H3, plain `USPAttention`) and off for the plain exchange
(`SGLANG_DIFFUSION_IPC_A2A_MULTI`). Same box, the switch forced on, MP4 md5s identical between arms wherever a video is saved:

| cell | layout | default | switch on |
|---|---|---:|---:|
| FastH3 V2 | Ulysses 4, sparse attention (selected) | 5.040 / 5.029 | 4.952 / 4.951 (-1.6%) |
| MiniMax-H3 T2VA, 4 GPUs | Ulysses 4 (selected) | 31.520 | 31.504 |
| Qwen-Image-2.1 | Ulysses 4 on 4 GPUs (candidate) | 1.756 / 1.823 | 1.909 / 1.889 (+8.7% / +3.6%) |
| Cosmos3 T2V | Ulysses 4, CFG in sequence (candidate) | 23.352 | 23.326 |
| Wan2.2 | Ulysses 4, CFG in sequence (candidate) | 90.563 | 90.458 |
| Qwen-Image-2.1 | Ulysses 2 on 2 GPUs (candidate) | 1.734 | 1.738 |
| LTX-2.5 | selected | 5.173 | 5.164 |
| Cosmos3 T2V | CFG 2 x Ulysses 2 (selected) | 22.665 | 22.661 |
| MiniMax-H3 T2VA, 2 GPUs | Ulysses 2 (selected) | 61.477 | 61.446 |
| Wan2.2 | CFG 2 x Ulysses 2 (selected) | 88.546 | 88.581 |

It pays where the plain exchange carries a large payload over more than two ranks (FastH3) and costs where the payload is
small (Qwen-Image at Ulysses 4, about 20 MB per exchange); the pairs and the pipelined cells do not move. The exchange alone, by
payload (`bench_a2a_sweep.py`, bf16, mean ms per exchange, one 4xB300 box):

| MB sent per rank | 4 ranks: NCCL | copy engine | 2 ranks: NCCL | copy engine |
|---:|---:|---:|---:|---:|
| 1 | 0.032 | 0.089 | 0.030 | 0.042 |
| 16 | 0.045 | 0.090 | 0.047 | 0.042 |
| 64 | 0.126 | 0.130 | 0.124 | 0.077 |
| 128 | 0.225 | 0.226 | 0.220 | 0.134 |
| 256 | 0.423 | 0.359 | 0.414 | 0.243 |
| 512 | 0.814 | 0.638 | 0.792 | 0.455 |
| 1024 | 1.481 | 1.196 | 1.227 | 0.882 |

So the default cannot simply flip. sgl-project/sglang#51554 leaves the choice to the transport when the switch is unset: on
Blackwell, three- and four-rank groups take the copy engine from 48 / 128 MiB sent per rank (the sizes at which it measured no
slower than NCCL alone). FastH3 sends 387 MiB (packed QKV) and 129 MiB (output) per exchange, Qwen-Image at Ulysses 4 sends 8 MiB.
Main b6d4389a77 against the change, mirrored: FastH3 5.022 / 5.027 -> 4.965 / 4.962 (-1.1% / -1.3%; forced on in an earlier
window 4.961 / 4.954 against 5.033 / 5.021), Qwen-Image at Ulysses 4 1.780 against 1.798 (NCCL on both; the cell ranged
1.748-1.831 over the day on either tree), MiniMax-H3 on four GPUs 31.493 against 31.475, MP4 md5s identical.

Checked, no gain (draft PR sgl-project/sglang#51563 for consistency only): Cosmos3's decode stage runs on every rank under CFG parallel (the generic decode stage runs on the main CFG
rank only), so the other rank decodes, converts and safety-checks an output nobody reads. Restricting it to the main CFG rank
changed nothing measurable: T2I 0.528 / 0.514 against 0.519 / 0.513, T2V 22.654 against 22.721, I2V 22.719 against 22.706, peak
memory per GPU and output bytes identical.

Also settled here: vLLM-Omni's FlashInfer attention on Blackwell is FlashInfer's `cute-dsl` kernel, which at FLUX.3's shape
(24 heads x 128, bf16) is slower than cuDNN in isolation (L=4352: 0.146-0.149 against 0.138-0.143 ms; L=4608: 0.195 against
0.153; `fa2` 0.62-0.67). Its end-to-end edge over its own cuDNN command is not a faster kernel, and sglang already runs cuDNN.

## 7. Method notes worth keeping

- Short cells (0.3-0.5 s) differ by 2-5% between boxes and between arms on a shared host; judge a regression only by old and
  new trees interleaved on one box, by the fastest sample.
- A torch profiler that makes the host the bottleneck drops GPU load, and B300 kernels run faster at low load: the same GEMM
  read 0.167 ms in sglang's trace (70% busy) and 0.192 ms in vLLM-Omni's (96% busy). Compare shares and call counts.
- "They compile, we do not" was not the cause of the FLUX.3 gap; one slow kernel call was. Lay the kernel tables side by side
  before concluding.

## Appendix: every change examined, and its verdict

The full list behind sections 1-6: each competitor change inside the round's version window (and each lead that came up on
the way), what it does to a cell of ours, and where it ended. "have" = sglang already does it; "PR" = an sglang change;
"profile" = the competitor's command in this repository changed; "none" = measured or read, nothing to take.

Coverage was checked mechanically on 2026-11-19: every commit of each window was listed with the files it touches
(`git log --name-only`), and each commit that touches code one of our cells runs is either a row below or falls in a row's
category. The counts are under each heading.

A second pass from the other side (2026-11-21): every window commit whose title has a speed word (perf, fast, fuse,
graph, cache, compile, kernel, latency, overlap...) and is not named here: vLLM-Omni 147 of 175, TensorRT-LLM 122 of
126, LightX2V 5 of 14, ComfyUI 4 of 10, FastVideo 16 of 32. Read one by one, they are other models (TTS and omni
models, Hy-Nomas, HunyuanVideo-1.5, MiniCPM-o, PAN2), the LLM side of TensorRT-LLM (its QKV preprocessing, Qwen3-VL
encoder graphs and NCCL scope changes live in LLM attention and VLM encoders, not VisualGen), Ascend, ROCm, XPU, MUSA,
MLX and DGX Spark paths, lossy caches, CI and docs. None reaches a cell of ours.

### vLLM-Omni (`0da1d831e` -> `24e547284`, vllm 0.32.0 -> 0.33.0)

| change | reaches our cells | verdict |
|---|---|---|
| stack: torch 2.14.0, cuDNN 9.24, NCCL 2.30.7, Triton 3.8 | every `CUDNN_ATTN` video cell -3% per step | have (torch 2.14.1 since 2026-10-04); library versions are now in every result JSON |
| #9523 FLUX.3 | a new cell, and sglang was behind in it | PR x7 (section 1): 2.043 -> 1.881 s against its 1.887-1.890 |
| #10120 stored PNG, pybase64, orjson | every image cell, outside the engine | PR sgl-project/sglang#51318 (merged); stored PNG: have |
| #9922 copy-engine Ulysses exchange (`--ulysses-a2a-permute`, default off) | every Ulysses cell | profile (nine cells, -1.1% to -5.7%); PR sgl-project/sglang#51554 for sglang's plain exchange |
| #9487 bf16 tensors over shared memory as raw bits, #10171 shared-memory wakeup | its worker-to-API hand-off (FastH3 -6.2%, Qwen 2 GPUs -4.8%, Cosmos3 T2I -6.5% with #10120) | none: sglang has no tensor hand-off there; its time outside the stages is 9-30 ms |
| #9194 OpenMP threads per worker = cores / GPUs | 2- and 4-GPU cells | have (stricter); `OMP_NUM_THREADS=8` for it measured: not a gap |
| #4989 FlashInfer attention as one custom op under `torch.compile` | Z-Image -81 ms | none: compile-only; FlashInfer's `cute-dsl` kernel is slower than cuDNN alone at FLUX.3's shape |
| #7597 one-launch QK-norm + RoPE | Z-Image | none: sglang upper bound with a fused in-place kernel -0.7% |
| #9998 FlashInfer autotune (SM100+) | FP8 GEMMs only; first request +1.4 s | none (bf16 cells) |
| #9553 Wan-family VAE fast-path installer | not installed for spatially sharded decodes | none |
| #9424 Cosmos3 / Wan VAE peak memory | no speed claim | none |
| #9074 MP4 encode overlapped with the decode | H3 outside denoise -0.3 s | have (`encode_while_decoding`) |
| #9438 text-encoder TP, #9797 CFG batching, #9306 / #9597 denoise graph replay (all default off) | Qwen 2 GPUs, Wan, Z-Image | measured as its candidates: none faster |
| `preencode_mp4` on Wan2.2 (documented, never sent) | Wan2.2 | profile (-0.3% / -0.5% on top of the exchange) |
| FA4 extra, `--no-diffusion-compile-dynamic` on FastH3 | Blackwell cells | measured: not gaps |
| #8072 reference-KV refresh, #9245 LiteVAE | lossy | out of scope |
| #9669, #9248, #9440, #6711, #10283, #9088, #9455, #9804, #6662 | other backends, models or flags | none |
| #9272 (dropped the decode graph per request, fixed by #10268) | its own Qwen 1-GPU cells +6% in the 11-02 rows | a caution: keep graphs across requests, bounded by shape |
| #8714, #8718, #8746 attention schedules (approximate attention on the later steps or below a noise level; H3, Wan2.2) | opt-in | lossy; out of scope |
| #4637 Cosmos3-Nano pipeline parallel, #9926 / #10122 Noisy pipeline parallel (CausVid) | a memory-split layout, off by default; another model | none |
| #6374 chunked weight transport for distributed layerwise offload, #9259 contiguous host layers for H3 reference-KV offload | offload paths; its cells run resident | none |
| #9977 serializer writes into a caller's buffer, #10117 | its stage-to-stage shared-memory hand-off | none: same family as #9487 |
| #9333, #9850 one copy of the weights for colocated AR and DiT stages | HunyuanImage-3.0 memory | none |
| #10077 Wan2.2 S2V audio injection compiled with the blocks | S2V only | none |

Window: 626 commits. 245 touch no package code (docs, tests, CI, examples). 147 touch only other code (omni, TTS and speech
models 55, other diffusion models 26, deployment YAML, NPU). 66 touch the engine, scheduler or serializer and no diffusion
path: omni duplex and AR work except #10171 (above) and #9977 / #10117. 168 touch a path one of our cells runs; the 130 of
them not in this table by number are fixes, other accelerators (Ascend, XPU, ROCm, MUSA), speech front ends, LoRA,
quantized checkpoints and the lossy caches (Cache-DiT, TeaCache, SeaCache, TaylorSeer), read by title.

### TensorRT-LLM VisualGen (1.4.0rc1 -> rc3)

| change | reaches our cells | verdict |
|---|---|---|
| fixed cost per request -1.1 s on the three H3 cells (cause not found upstream) | H3 | none: sglang's time outside denoising is already lower (section 4) |
| #20333 `async_ulysses` (default off) | H3 2 / 4 GPUs | profile (-1.5% to -1.7%); sglang pipelines its exchange already; Ref2VA: not promoted |
| #17147 Q/K/V in one all-to-all | H3, Cosmos3 | have (packed QKV) |
| #19645 fused QK-norm + RoPE | H3 | have |
| #19773 SwiGLU in the GEMM epilogue | H3 | part have; changes bytes; not pursued |
| `tp_layout: token_sharded`, CUDA graphs, TP2 x Ulysses 2, text-encoder TP | Wan2.2, H3 | measured as its candidates: slower, no gain, or the server exits |
| SageAttention, NVFP4 / FP8 attention, static FP8 Cosmos3, Sol-Attn, TeaCache | lossy | out of scope |
| #21396 the fused per-token AdaLN was gated on exactly SM100, so B300 ran the unfused path | Wan paths with per-token timesteps (TI2V-5B); the A14B T2V cell has one timestep per sample | none for the cell. Checked sglang's diffusion runtime for the same mistake: no fast path is gated on exactly `(10, 0)`; the SubBlock sparse backend refuses SM103 loudly |
| #20176 H3 DiT tensor parallelism, #19807 H3 text-encoder TP | H3 | the PRs behind the TP2 x Ulysses 2 and text-encoder TP candidates above |
| #21301 H3 CPU offload, #19401 video encode always on the thread pool (it already was by default) | memory; no change | none |

Window: 510 commits, 27 of them touching VisualGen code. Besides the rows above: a paged K/V cache for causal models, LTX-2.3,
FLUX.1 NVFP4, MXFP8 refit, kernel-tree moves and example fixes.

### LightX2V (`19c19614` -> `395e2ac4`)

| change | reaches our cells | verdict |
|---|---|---|
| `PROFILING_DEBUG_LEVEL` defaults to 2 (a synchronize and a log line around every step) | every non-H3 cell | profile for Qwen-Image-2.1 (-1.4%); no difference elsewhere |
| #1646 VAE compile | Wan2.2, H3 | profile (Wan -0.3%, H3 -1.4%) |
| packed Ulysses exchange (`seq_p_tensor_fusion`, Triton pre/post) | H3 | profile with the VAE compile (-1.5% to -2.4%); sglang: have (packed QKV) |
| #1721 fused anti-aliased SnakeBeta in the H3 vocoder | H3 audio decode | none: 0.080-0.085 s of a request in sglang |
| head-parallel Ulysses, fused QKV, traceable RMSNorm | H3 | measured as its candidates: slower |
| #1656, #1698, #1697 / #1699, #1706, #1720, #1654, #1655, #1700, #1711, #1673 | no hot-path change for our cells | none |
| #1650, #1657, #1692 causal (streaming) H3: Hopper FP8 GEMM, fused QKV cache updates | another model variant, Hopper, FP8 | none |
| #1666 release the resident text encoder's host copy | host memory | none |

Window: 43 commits. The 26 not named here are the ComfyUI integration, H3 Image, RelayVSR, InfiniteTalk, Depth Anything 3,
Seko, MPS, training and docs.

### ComfyUI (`1a82e6d0` -> `a308f3f1`, 148 commits)

Every cell within 1% except FLUX.3 (3.340 -> 3.255, -2.5%, different boxes). The commits that touch a model path of ours:

| change | reaches our cells | verdict |
|---|---|---|
| #17196 Flux2 VAE fused and stripped through comfy-kitchen | FLUX.3 decode | have (fused GroupNorm + SiLU in the FLUX.3 VAE) |
| #17273, #17361 Qwen 3.x: adaptive-tree MTP, combined parallel linear weights; #17625 its fix | Qwen3 text encoders (Z-Image, FLUX.3) | none: sglang replays these encoders as CUDA graphs (Z-Image on one GPU already, FLUX.3 with #51562) |
| #16941 `fp8_e4m3fn_fast` no longer falls back to bf16 compute under dynamic VRAM | Ideogram 4 fp8 (3.696 -> 3.685) | none |
| #17413 skip the concat of a single cond, #17439 no `torch.equal` on the same tensor | host micro-costs | none |
| #17274 dynamic-VRAM KV caches, #17290 NTK RoPE scaling, #17450 VAE compute dtype node, #17487, #17741 | options and fixes | none |
| the other 130-odd commits | partner API nodes, front end, new models (MiniMax-H3 image, PAN2, WeVisOne, RelayVSR, MOSS) | none |

Window: 148 commits. 49 touch code a ComfyUI cell of ours runs (`comfy/` core, its text encoders, the node and execution
layers). Besides the rows above they are new models (PAN2, MiniMax-H3 Image and its VAE and upscaler, UltraTex, WeVisOne,
RelayVSR, MOSS-Image), Qwen 3.x text generation and video input (not the text-encoding path), AMD and DirectML, video save
formats, feature flags, and correctness fixes to offload and dynamic VRAM (#17297, #16491); read by
title, none on a cell's hot path.

ComfyUI is 1.4x to 5.8x behind sglang in every cell it runs.

### FastVideo (`fa2d1324` -> `c938a8aa`, 76 commits)

Its four cells moved by -0.5% to +1.1% (FastH3 9.196 -> 9.294). What the window holds for them:

| change | reaches our cells | verdict |
|---|---|---|
| #2040 eight VAE tiles per decode call, #2041 fused QK RMSNorm + partial RoPE in the H3 VAE decoder | H3 decode | on by default and its cells did not move; sglang's H3 decode is 0.7-1.2 s of 31-96 s |
| #2042 opt-in faster kernel tier for the DiT fused ops | H3 | default off; not measured (FastVideo trails sglang by 1.2x to 1.8x in these cells) |
| #2046 FP4 attention padding, uint8 VAE pixel path, cuDNN FP4 GEMMs; #2052, #2051, #1989 NVFP4 / FP4 attention | NVFP4 paths | lossy; out of scope |
| #2004 training-free token reduction | H3 | lossy; out of scope |
| #2022 text conditioning reused across seed sweeps, #2058 free the negative-prompt encoder | H3 | have (conditioning cache); none |
| #2102-#2105, #2107, #2093 DGX Spark work | GB10 | none |
| the rest | new models (PAN2, Qwen-Image-2.1, Pika H3 Flex, Open-H3-Regenerate-2K), serving, LoRA, CI, docs | none |
| #2043 FA4 varlen keywords only where the installed kernel takes them, #2080 masked FA keeps its mask and scale, #2049 | crash and correctness fixes in its FA4 wrappers | none: sglang vendors its FA4 and passes `max_seqlen` (the omission #2043 warns over-sizes the launch) |

Window: 76 commits, 46 of them on its runtime paths. Besides the rows above: the model catalog server, Ray health, MPS and NPU
fixes, Wan compile and causal-mask fixes, OpenAI API fields, LoRA lifecycle fixes.

### sglang itself

| lead | verdict |
|---|---|
| regression between the rounds | one: LTX-2.5 two-stage +1.3-1.6% since sglang #47301 (a replay's attention breaks moved to cuDNN's slower sub-tile kernel); PR sgl-project/sglang#51611, stages back to 1.020 / 1.540 s. Every other cell is flat or faster against the tree its own row ran; H3 on 4 GPUs' slower denoise stage is timer attribution from sglang #47565, whose request is 0.16 s faster (section 2) |
| FLUX.3: static text stream dropped under CFG parallel, slow SwiGLU kernel, sharded VAE decode, negative prompt re-encoded, `--component-precisions.vae` ignored, host-bound text encode | PR sgl-project/sglang#51380, #51387, #51406, #51411, #51529, #51562 |
| Wan2.2 decode "2.95 s" | a timer artifact: 0.761 s with synchronized timers |
| Qwen-Image-2.1 VAE padding fold rejected on B300 | at most 5 ms |
| Cosmos3 decodes on every rank under CFG parallel | no measurable effect; draft PR sgl-project/sglang#51563 for consistency with the generic stage |
| Z-Image under TP2: the text encoder (0.032 s of 0.31 s) runs eagerly because the native Qwen3 encoder's graphs step aside under tensor parallelism | a lead, not measured further: 25.9-27.8 ms eager against 11.5 ms replayed for the same encoder alone; filed as a follow-up |
| time outside the stages on the CFG-parallel image cells | not a hand-off cost (0.05 ms); 12 ms with a plain client, as under TP2 |
| `quality=exact` on LTX-2.5 two-stage with the auto-assigned K/V-gather degree | every request failed (a cross-attention key mask read as the query's); PR sgl-project/sglang#51620, requests complete |
| a fast path gated on exactly SM100 that B300 (SM103) would silently miss (TRT-LLM's #21396 mistake) | none in the diffusion runtime: every `(10, 0)` test either admits `(10, 3)` beside it or belongs to the SubBlock sparse backend, which refuses SM103 with an error |
