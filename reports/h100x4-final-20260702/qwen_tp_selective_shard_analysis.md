# Qwen-Image TP Selective-Sharding: Resolution-Dependent Crossover

**Hardware:** 2×H100, tp=2, torch.compile on, no-CFG (guidance-scale=1.0), bf16.
**Base:** sglang origin/main + PR #29774 (`Shard QwenImage DiT across TP ranks`).
**Metric:** median denoise step time (ms), 20 steps, isolated from text-encode/VAE via `--perf-dump-path` (`denoise_steps_ms`). This isolates the DiT, which is the only part affected by sharding.

## Problem

PR #29774 shards *every* projection in the QwenImage DiT (12 Column + 3 Row per block × 60 blocks).
Profiling the sharded run shows it becomes **communication-bound**: RowParallel all-reduce is
~70% of DiT kernel time. Each block has **4 all-reduce points**: `to_out` (image attn out),
`to_add_out` (text attn out), `img_mlp` down, `txt_mlp` down.

Two selective-sharding levers were swept, both env-driven for the experiment
(`QWEN_REPL_MLP`, `QWEN_REPL_ATTN` = `none|txt|img|txt_img`):

1. **MLP replication** — keep an MLP `ReplicatedLinear` (full weight on each rank, no all-reduce)
   instead of Column-up/Row-down. Trades a redundant GEMM for a saved all-reduce.
2. **Attention-out replication** — replace the RowParallel `to_out`/`to_add_out` all-reduce with an
   **all-gather** of the head-sharded attention output (dim=-1) followed by a `ReplicatedLinear`.
   all-gather moves ~half the traffic of an all-reduce, and the out-proj GEMM is small (inner_dim→dim,
   no 4× MLP expansion), so it is far cheaper to replicate than an MLP.

## Correctness

Attention-out replication is mathematically identical to the sharded path (all-gather + full matmul
== sharded matmul + all-reduce). Validated end-to-end at 512², seed-fixed, comparing output images:

```
CORRECTNESS ATTN=txt    vs none: MAE=0.648/255  MAX=35/255  OK
CORRECTNESS ATTN=txt_img vs none: MAE=0.648/255  MAX=33/255  OK
```

MAE 0.648/255 is pure bf16 reduction-order noise; a head-ordering bug would produce MAE > 50.

## Result: the optimal config is token-count (resolution) dependent

| Resolution (img tokens) | best MLP | best ATTN | best step (ms) | #29774 vanilla (all-shard) | speedup |
|---|---|---|---|---|---|
| **512²** (1024) | txt_img | txt_img | **77.12** | 100.44 | **−23.2%** |
| **1024²** (4096) | txt | txt_img | **91.51** | 111.50 | **−17.9%** |
| **1536²** (9216) | none | none | **197.32** | 197.32 | 0% (shard wins) |

- **512² small** → replicate *everything* (both MLPs + both attn-outs). GEMMs are tiny, all-reduce dominates.
- **1024² medium** → replicate txt_mlp + both attn-outs, but **keep img_mlp sharded** (its 4× GEMM is too big to duplicate). This is the standard benchmark resolution.
- **1536² large** → **full shard wins**; every replication is a net loss (GEMM/all-gather cost > all-reduce saved).

> ⚠️ **These step-time numbers are the CLI `sglang generate` `denoise_steps_ms` median — a fast
> *ranking* proxy, not the production metric.** Only the **1024² txt_mlp** win is confirmed on
> real-server e2e (see e2e section). The CLI over-states comm-reduction wins (the server pipelines
> steps / captures graphs, partly hiding the all-reduce). Treat the 512²/1536² rows and the attn-out
> deltas as *directional hypotheses* until e2e-with-interleaved-repeats confirms them.

## Full sweep data (median step ms)

### MLP replication × resolution
| RES | MLP=none | MLP=txt | MLP=txt_img |
|---|---|---|---|
| 512² | 100.44 | 89.81 | **80.70** |
| 1024² | 111.50 | **97.60** | 106.43 |
| 1536² | **197.32** | 199.14 | 251.95 |

### Attention-out replication (on best MLP for that RES)
| RES (MLP) | ATTN=none | ATTN=txt | ATTN=img | ATTN=txt_img |
|---|---|---|---|---|
| 512² (txt_img) | 79.79 | 85.26 | 82.12 | **77.12** |
| 1024² (txt) | 100.01 | 97.82 | 102.90 | **91.51** |
| 1536² (none) | 198.30 | — | 204.22 | 204.34 |

## Findings

1. **txt_mlp replication**: helps ≤1024² (−10 to −12%), neutral at 1536². Text is ~1024 tokens
   regardless of image size, so its saved all-reduce is constant — it just becomes negligible vs a
   huge image step at 1536². Safe to always enable.
2. **img_mlp replication**: helps **only at 512²**. At ≥1024² the duplicated 4×-expansion GEMM
   costs more than the all-reduce it removes.
3. **Attention-out replication must be done as a pair** (`txt_img`): replicating only one side
   (`img` or `txt`) lands within run-to-run noise or slightly worse, but replicating **both** is a
   clean win ≤1024² (1024²: 91.51 vs 100.0, −8.5%, reproduced across round-2 and round-3).
   At 1536² it flips to a ~3% loss (big-image all-gather + duplicated out-proj > all-reduce).
4. **Cumulative** at the benchmark 1024²: #29774 vanilla 111.5 → +txt_mlp 97.6 → +attn-out pair
   **91.5** ms/step (**−17.9%** on the DiT).

## Recommendation (proposed #29774 follow-up)

**Ship one thing: `txt_mlp` → `ReplicatedLinear`.** It is the only change that produces a
reproducible e2e win (full-shard 5.58 → ~5.12, −8%, confirmed across every batch and against vLLM).
It is also trivially safe (one small always-full-input MLP; no comm-pattern change).

**Do not ship the attn-out replication.** Despite a clean CLI `denoise_steps_ms` signal and one
lucky low-noise e2e batch, the drift-cancelled interleaved test puts it at +0.8% with mixed-sign
paired diffs — i.e. within noise. The extra all-gather + full/`colgather` out-proj is not worth the
complexity for a null e2e effect. `colgather` specifically is measurably *worse*.

The attn-out all-gather code is mathematically exact (validated MAE 0.65–0.68/255) and remains in
`qwen_image_selective_shard.py` behind `QWEN_REPL_ATTN` / `QWEN_ATTN_SCHEME` env flags as a
**research artifact**, in case a future regime (much larger TP, different interconnect, no CUDA
graphs) changes the comm/compute balance — but on 2×H100 with compile it is not a win.

The 512²/1536² MLP-crossover rows above are **CLI-level hypotheses only** (not e2e-validated, and the
CLI over-states comm wins). If a deployment is pinned to a very different resolution, re-run the e2e
interleaved test at that resolution before changing the sharding.

## e2e validation (benchmark 1024², single_e2e, 50 steps, vs vLLM-Omni 5.17s)

Measured through the real server path (HTTP → scheduler → denoise). All e2e measurements across
every batch, per config:

| config | single_e2e (s), all runs | verdict |
|---|---|---|
| #29774 full-shard (MLP=none, ATTN=none) | 5.584 | **worst — reproducible** |
| + txt_mlp replicated (MLP=txt, ATTN=none) *(deployed)* | 5.017, 5.131, 5.19, 5.138 | **robust −8% win** |
| + attn-out pair, **repl** (MLP=txt, ATTN=txt_img) | 5.086, 4.942, 4.887, 5.15, 5.061 | within noise of row |
| + attn-out pair, **colgather** (MLP=txt, ATTN=txt_img) | 5.25, 5.33 | **worst attn-out variant — rejected** |

**txt_mlp replication is the one robust, reproducible e2e win** (full-shard 5.58 → ~5.12, −8%; every
batch agrees, no overlap). It takes qwen-image no-CFG from **losing to vLLM (5.58)** to **tie/edge
(≈5.12 vs 5.17)**.

**attn-out replication (repl) is within cross-batch noise.** One batch read it low (4.89–4.94, which
looked like a clean −5% win), but another read it 5.06–5.15 (≈ row). Its range (4.89–5.15) overlaps
row's (5.02–5.19), and the effect (~2–3%) is smaller than cross-batch noise (~4–5%). An interleaved
row/repl test (drift-cancelling) is required to resolve it — see below.

**colgather (ColumnParallel+all-gather) is confirmed worse** (5.25–5.33, consistently the slowest
attn-out variant), matching theory: the out-proj's contraction dim (heads) is sharded, so a
ColumnParallel formulation needs an input all-gather *and* an output all-gather (≈ one all-reduce of
traffic) plus an extra kernel/sync — strictly dominated by plain RowParallel. Numerically correct
(MAE 0.682/255), just slower.

### Interleaved row-vs-repl (definitive attn-out test)

Row and repl alternated ×3 (ABAB, drift-cancelled), paired differences:

```
row (none)     [5.195, 5.092, 5.160]  mean 5.149
repl (txt_img) [5.145, 5.124, 5.057]  mean 5.109
delta = +0.8%   paired diffs (row-repl) = [+0.050, -0.032, +0.103]
```

**Verdict: attn-out replication is within noise.** Mean delta +0.8% (below per-run noise) and the
paired differences are **mixed-sign** (one pair had row faster). The earlier "−5% win" was a
low-noise outlier batch. Attn-out replication (repl) does **not** provide a reliable e2e win — not
worth the extra code; `colgather` is worse still.

### Metric note — trust e2e-with-interleaved-repeats, not single runs or CLI steps

An early *single* e2e pair (none=5.017, txt_img=5.086) suggested attn-out *hurt*; a later *same-batch*
pair (4.89–4.94 vs 5.13–5.19) suggested a clean −5% *win*. Both were misleading: cross-batch e2e
noise (~4–5%, from per-run server restarts / thermal / clock drift) exceeds the effect. Same-batch
back-to-back repeats are correlated and can drift together. **To resolve a <5% effect you must
interleave the two configs (ABAB) and use paired differences.** The CLI `sglang generate`
`denoise_steps_ms` median is a fast *ranking* proxy but its absolute deltas over-state the server win
(the server pipelines steps / captures graphs, partly hiding the all-reduce the CLI exposes).

## Follow-up validation matrix (resolution × TP degree × model, e2e ABAB ×3)

All real-server single-request e2e on H100, ABAB-interleaved pairs, paired diffs same-sign unless
noted. One first-cold-start run per stream failed on a startup transient (`EOFError` before health
check) and was excluded; every completed pair is consistent.

| experiment | full-shard | txt_mlp replicated | paired verdict |
|---|---|---|---|
| qwen 1024², tp=2 (baseline) | 5.58 | **5.12 (−8%)** | robust win |
| qwen 512², tp=2 | ~5.04 | **4.54 (−9%)** | win — crossover holds at small res |
| qwen 512², tp=2, **+img_mlp** | ~5.04 | **4.11 (−18%)** | small-res-only win (regresses ≥1024²) |
| qwen 1536², tp=2 | **10.23** | 10.28 (+0.4%) | tiny consistent cost — acceptable for the default |
| qwen 1024², **tp=4** | **6.04** | 6.11 (+1.1%) | **regression, all pairs same-sign** → gate on tp≤2 (PR ec69bdd) |
| flux 1024², tp=2 (`ff_context`) | **4.41** | 4.44 (+0.8%) | **does NOT transfer** |

Key takeaways:

1. **The TP-degree crossover is real**: at tp=4 the duplicated GEMM (full FFN per rank vs 1/4)
   outgrows the 4-rank all-reduce it saves. The PR now gates replication on
   `get_tp_world_size() <= 2`. Context: qwen tp=4 e2e (6.04) is *slower* than tp=2 (5.12) anyway —
   TP is not the multi-GPU latency path for this model; CFG-parallel is.
2. **The resolution crossover survives e2e at 512²** (unlike the attn-out CLI signal): txt −9%,
   txt+img_mlp −18%. This is the first e2e-validated basis for a `--expected-shape`-driven
   "aggressive" sharding plan at pinned small resolutions.
3. **The rule does not blanket-transfer across MMDiTs**: FLUX.1-dev's `ff_context` (identical
   Column→Row structure) is a consistent small loss when replicated — FLUX has only 19 dual-stream
   blocks vs qwen's 60, so 3× fewer all-reduces saved for the same duplicated-GEMM cost. Any
   generalization must be a per-layer cost comparison (blocks × all-reduce cost vs duplicated GEMM
   at the branch's token count), i.e. the sharding-planner design — not a per-model copy-paste.
