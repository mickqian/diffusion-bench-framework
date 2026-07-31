# Does #30211 (`--encoder-parallel auto`) make these sglang numbers stale?

**Date**: 2026-07-31 · **Verdict**: no rerun needed, the published cells stand.

## Why the question

This report was measured before sgl-project/sglang#30211 merged (`db3da62`).
That PR unified the encoder layout under one knob, `--encoder-parallel
auto|fold|dp|replicate`, **defaulting to `auto`** in `ServerArgs` — with no
per-entrypoint override, so `serve` gets `auto` too. Under `auto` at 2 ranks, a
text encoder whose hidden size is ≥ `FOLD_MIN_HIDDEN_SIZE` (4096) and whose dims
divide is now **tensor-parallel folded across the two ranks** where it was
previously **replicated**. That is a strategy change on the exact hardware
profile this report uses (2 GPUs), so the published sglang column could in
principle be understating current main.

Affected cases (encoder wide enough to fold at 2 ranks):

| encoder | hidden | cases | folds now? |
|---|---|---|---|
| T5-XXL | 4096 | FLUX.1-dev | yes |
| UMT5-XXL | 4096 | Wan2.1 / Wan2.2 family | yes |
| Mistral-Small (FLUX.2) | 5120 | FLUX.2-dev | yes |
| Qwen2.5-VL-7B | 3584 | Qwen-Image / Qwen-Image-Edit | no (below 4096) |
| Qwen3 / CLIP-L | ≤ 2560 | Z-Image, flux CLIP branch | no |

## Why it does not move the numbers

#30211's own end-to-end measurement covers this directly — FLUX.1-dev, 2×H100
sp=2, 28 steps, 1024×1024, one server per policy under identical load. At
batch 1, which is what this report's latency section measures:

| policy | `TextEncodingStage` @ b1 |
|---|---|
| replicate | 25–27 ms |
| **auto (T5 folded)** | **30–34 ms** |

Folding is *slightly slower* in wall clock at batch 1, not faster: its −24% GPU
time (CUDA events) is offset by the per-layer `all_reduce` launch overhead
(~4 ms of CPU for 24 layers). e2e came out indistinguishable across policies
(~3.1 s), the encode delta being **under 0.15% of e2e** — an order of magnitude
below this harness's cross-batch noise (~4–5%). For the video cases the encode is
a far smaller fraction of a 30–200 s request, so the margin only widens.

The other new layout, `dp`, cannot engage here at all: it requires a merged batch
> 1, and `batching_max_size` defaults to 1, so every request in this report
encodes at batch 1.

## Conclusion

The strategy did change for the wide-encoder cases, but its measured e2e effect
(<0.15%, and in the *slower* direction at batch 1) is far inside noise. Rerunning
would spend hours to reproduce the same cells. The knob matters for
encode-heavy workloads — few-step models, long prompts, batched serving — none of
which this report's latency section is.

Worth revisiting if a future run enables dynamic batching (`--batching-max-size
> 1`), where `dp` engages and was measured at **−31%** on the encode stage.
