# `--layerwise-resident-layers` does nothing for a once-per-request component

2026-09-21 · sglang `0f6761b5` · Qwen/Qwen-Image-2.1 · RTX 5090 and RTX 4090

## What the flag promises

> resident layers are transferred **once (not re-streamed every step)**, so this
> trades VRAM for lower denoise latency when memory is available.
> — `--dit-layerwise-resident-layers` help text

## What it does

Nothing, unless the component runs a denoise loop. Measured on one RTX 5090,
1024×1024 / 40 steps / seed 42, five requests per arm:

| arm | p50 | s/step | steady VRAM | manager reports |
|---|---|---|---|---|
| baseline | 14.136 s | 0.3173 | 17047 MiB | `resident=0/66` |
| `--layerwise-resident-layers text_encoder=0.8` | 14.168 s | 0.3173 | 17047 MiB | `resident=53/66` |

The flag is accepted, the manager logs `resident=53/66`, and neither memory nor
latency moves. `Layerwise offload summary` is byte-identical in both arms
(`vram: 1.83 GB, host pinned: 13.93 GB`).

## Why

`LayerwiseOffloadStrategy.finish_use` calls `manager.release_all()`, whose
docstring states the scope plainly:

> Release every layer, including the resident ones: **this ends the denoise
> stage that the resident set is scoped to.**

For the DiT one "use" spans all 40 denoise steps, so the set survives the loop
and pays for itself. For the text encoder one use is a single forward pass: the
set is prefetched at `prepare_for_use` and dropped again at `finish_use`, every
request. Instrumenting the decision shows `finish_use` running twice per request
for this component, the first time with 9.08 GiB held.

## Why it is worth fixing

In Qwen-Image-2.1 the text encoder is the **largest** component — 16.33 GiB
against the DiT's 13.25 GiB — and it is the one being re-streamed. Of it,
13.93 GB sits in pinned host memory and crosses PCIe on every request, on a card
that is otherwise idle: the 5090 holds 17.0 GB of its 31.4 GiB.

The cost shows up as the non-DiT remainder of each request: 1.19 s on the
RTX 5090 (8.4% of e2e) and 1.47 s on the RTX 4090 (7.6%).

## A mechanism that works, and a policy that does not

Patching `release_all` to keep the resident set (`proposed-retain-residents.patch`)
does work on the RTX 5090:

| | p50 | steady VRAM | load peak |
|---|---|---|---|
| baseline | 14.136 s | 17047 MiB | 24891 MiB |
| retained | **13.737 s** | 28593 MiB | 31291 MiB (96% of the card) |

−2.8%, and numerically an FP-reorder equivalence: max deviation **1/255**, MAE
0.011/255, and **zero** pixels differing by more than one LSB across 1024².

The same patch **prevents the RTX 4090 from starting at all**:

```
OutOfMemoryError: Tried to allocate 370.00 MiB.
GPU 0 has 23.52 GiB of which 121.81 MiB is free.
out of memory; retrying warmup at server warmup req (720x720, 2/40 steps)
```

That is not a threshold that needs tuning. The two cards differ structurally:

| | capacity | free once DiT+VAE are resident | the resident set needs |
|---|---|---|---|
| RTX 5090 | 31.4 GiB | ~14.4 GiB | 9.08 GiB — fits |
| RTX 4090 | 23.5 GiB | ~6.9 GiB | 9.08 GiB — cannot fit |

On the 4090 the correct answer is simply "do not retain". A decision taken at
`finish_use` cannot reach it: what it can see is how much is free *now*, and
what it needs to know is how much a *later* stage will want — the OOM lands on a
720×720 warmup, a shape the decision point never saw. Any fixed fraction of free
or total memory is a guess about future allocations.

That knowledge is what `#37917 Plan component residency from calibrated warmup
records` produces. The retain decision belongs there, not in a constant here.

## What is reusable

* `release_all(keep_resident=...)` — one line, since `release_layer` already
  skips the resident set and only `force=True` overrode it.
* `LayerwiseOffloadManager.retained_parameter_bytes()` — what a manager's
  resident set costs.
* Four unit tests, including one pinning that the default path is unchanged.
* The measurement above, as an input to whatever policy decides.

Two mistakes worth recording, because both were caught by validation rather than
review: the decision helper first landed on the mixin while using manager state
(caught by the new unit tests — it would have raised at runtime), and the first
headroom test measured free memory *during* the use, when the use's own
transients are still allocated (caught on GPU — the patch changed nothing).
