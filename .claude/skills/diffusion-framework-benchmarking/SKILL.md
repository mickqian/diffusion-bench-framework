---
name: diffusion-framework-benchmarking
description: Use when planning, running, extending, or interpreting diffusion framework benchmarks across SGLang-Diffusion, vLLM-Omni, LightX2V, diffusers, or similar systems, especially when comparing single-request latency, throughput, command profiles, framework versions, model support, fairness, failure reasons, or report/image generation in diffusion-bench-framework.
---

# Diffusion Framework Benchmarking

## Goal

Produce reproducible, fair, and actionable performance data for open-source diffusion serving frameworks. The benchmark is a guardrail for SGLang-Diffusion, so invalid comparisons are worse than missing data.

## Default Run Configuration (Complete Cross-Framework Run)

The concrete run matrix — which frameworks, which models/cases, which workloads, and each framework's best-known command per case/hardware — is **explicit config, not prose**: it lives in `configs/benchmark/` (per-case files under `cases/`, plus `frameworks.json` and `workloads.json`). Regenerate `configs/comparison_configs.json` with `scripts/build_benchmark_config.py`, which **fails if any case leaves a framework unclassified**. See `configs/benchmark/README.md` and the auto-generated `configs/benchmark/MATRIX.md`. This section is the *policy* those files must satisfy.

Unless the user asks otherwise, a complete cross-framework run defaults to:

- **Version policy: latest-vs-latest.** SGLang runs `origin/main` HEAD; every competitor runs its newest line — `vllm` newest, `vllm-omni` main HEAD, `LightX2V` main HEAD, newest `trtllm` release candidate. Override every competitor `*_INSTALL_SPEC` to latest and record each resolved version/commit in the result artifact. Pinned snapshots (the version sets in the Install reference and in `manifests/`) are **only** for reproducing a specific dated historical report — do not silently inherit an old pin for a fresh run.
- **Framework scope: all frameworks.** Include `sglang`, `vllm-omni`, `lightx2v`, and `trtllm-visual` (image cases). `diffusers` is an optional correctness/baseline reference, labeled as such. A complete run includes every framework in scope; cells a framework can't run are classified (`unsupported` / `no_profile` / `failed` / `not_run` / `invalid`), never silently dropped.
- **Execution environment: a freshly-acquired `rx devbox` H200.** Acquire a fresh H200 via the `rx` CLI (see the `rx-devbox` / `remote-development` skills) as the default machine; 2-GPU is the standard profile budget. Keep conflicting competitor frameworks in isolated virtualenvs. The local MacBook cannot run these benchmarks.

When the user explicitly narrows any of these (a single framework, a pinned ref to reproduce, an existing machine), follow that instead — these are defaults, not overrides of an explicit request.

## Non-Negotiables

- **Goal = best performance under no precision loss.** Split optimizations by whether they change the output:
  - **Lossy — keep OFF** (they change the output, so they are not part of a no-precision-loss comparison, unless a case explicitly compares those semantics): response cache, Cache-DiT / TeaCache / feature-caching (approximate step caching), precomputed outputs, quantized checkpoints/weights (int8/fp8), reduced steps, distilled weights, and int8/approximate attention (SageAttention, sparse/block-sparse).
  - **Lossless — turn ON for every framework** (same dtype + same algorithm → same output; disabling them undersells a framework's real capability): torch.compile / kernel fusion, TensorRT engines (same dtype, no quantization), the fastest exact-precision attention, TP / CFG / sequence parallelism, continuous batching. Run with `DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0` (the harness then drops vLLM's `--enforce-eager`, does not set `TORCH_COMPILE_DISABLE`, and defaults LightX2V `compile=true`).
- **CUDA graph is production-realistic only when captured at the served concurrency.** It needs extra VRAM, and a concurrency-1 capture can make a *single-request* number unrealistically perfect in a way that does not survive real traffic — so the headline metric is throughput under real concurrency (see below), never a single-request figure. If a framework/model's compile path breaks, OOMs, or would need a precision-changing setting (e.g. forced TF32), fall back to eager for that cell and note it — do not fold a broken or precision-changed cell into the main comparison silently.
- Keep external frameworks in isolated virtualenvs when dependencies conflict with SGLang.
- Never treat startup failure, import failure, OOM, timeout, compile stall, or NaN as latency data.
- Compare single-request latency with single-request latency, and throughput with throughput. Do not mix the two in one conclusion.
- A complete cross-framework run must include both single-request e2e AND a multi-request server throughput workload (several concurrent requests), at minimum for the image cases — a single-request-only run is incomplete. Report the two sections separately.
- Track actual framework version/ref, selected command profile, hardware, GPU count, sampling params, dimensions, frames, concurrency, and actual server command in result artifacts.

## Fairness Checklist

Before accepting a number, verify:

- same model weights or an explicitly documented equivalent model path
- same task type, prompt, seed, resolution, `num_frames`, FPS, steps, guidance/CFG, scheduler semantics, dtype, and VAE path
- same output count and response format expectations
- intended GPU class and GPU count are recorded; H100/H200 may need different profiles
- warmup is comparable and not dominated by cold model download or compile
- framework-specific fast paths are fair: fast attention is fine; hidden caches or lower-quality models are not
- attention backend: give **each framework its fastest exact-precision kernel** for the hardware. They need NOT be identical across frameworks — sglang FA4, LightX2V FA3, vLLM FlashInfer are all fair against each other because they are the same *precision class* (exact softmax), and "best lossless perf" means each framework runs its fastest exact kernel, not a lowest-common-denominator shared one. Exact/full-precision attention (FlashAttention 2/3/4, FlashInfer, Torch SDPA) is that fair-swappable family — pick the fastest available member per framework. Quantized or approximate attention (SageAttention int8 QK, block-sparse/sparse attention, distilled attention) is a *different numeric class*. Do not silently substitute a quantized/approximate backend for a competitor when the reference (e.g. SGLang's backend for that case) runs full precision — that lets the competitor trade away accuracy the reference isn't trading away, which inflates its speed unfairly. If the only working backend on the target hardware is a different numeric class, either keep hunting for a same-class option, mark the cell `no_profile`/`failed` with the root cause, or run and report it as an explicitly labeled different-precision data point — never fold it into the main comparison unlabeled.
- latency metric is unified across frameworks on **client-side wall clock** — it is the only framework-agnostic, consistently-defined number. Server-side timers (e.g. SGLang's perf-dump `total_duration_ms`) are framework-specific and not emitted by every framework, so keep them only as per-framework diagnostic annotations, never as the cross-framework headline.
- single-request latency is a **steady-state median of several back-to-back requests after warmup**, not one isolated request. A single shot can absorb a cold/idle-path artifact — e.g. vLLM-Omni 0.24 stalls ~55s on spaced concurrency-1 requests while its engine runs in ~1s. Image cases repeat cheaply (≈5); video is expensive, so warm harder (≈3) and repeat fewer (≈2). If the repeats do not converge — or the single-request median exceeds the steady-state throughput p50 for the same case — the number is untrustworthy: mark the cell an anomaly and withhold the value (do not publish a stalled number as if it were latency). throughput records `num_requests`, concurrency, p50, p95, p99, and QPS; continuous load does not hit the isolated-request stall.
- **A 5-repeat median converging tightly (low variance) is NOT proof the number is representative — sglang H100 image cases showed a cross-process bimodal latency pattern that this check alone would miss.** Discovered 2026-07-29 (full 18-case rerun): `qwen_image_2512_t2i_1024` (no-CFG, Ulysses profile) measured 9.15s in the original run with a *tight* 9.0–9.19s spread across its 5 repeats (2% variance — looks perfectly converged by the rule above), but two independent re-runs of the exact same command landed at 5.78s and 5.74s (each also tightly converged, 1–2% variance). `flux1_dev_t2i_1024` (TP=2 profile) showed the same shape: 6.92s original vs 4.32s on independent re-run. The server process appears to lock into one of two stable performance regimes at startup/warmup and stay there for its whole lifetime — every request in that process's 5-repeat window agrees with each other, but a fresh process can land in the other regime entirely. Root cause unconfirmed (suspected torch.compile cache reuse or NCCL topology-selection non-determinism across process restarts on a devbox that's been running many hours of mixed cases); tracked as a standalone sglang investigation, not fixed here. **Practical implication: when a single_e2e number looks surprising (a framework that should be fast reads slow, or vice versa), re-run that one case as an independent fresh process 1–2 more times before trusting either number — don't just trust a tight within-run spread.** Not every case is affected (5 other sglang image cases re-checked independently in the same run were stable within 2%), so this isn't a blanket "always re-run everything," but a surprising result is a specific trigger to re-verify with a fresh process, not just a fresh set of repeats. Evidence: `reports/h100x4-full-20260728/bimodal-evidence-*.json` and `other-case-recheck/`.
- framework version parity: when sglang runs `origin/main` (latest), competitors must also run latest (git main HEAD / newest release), not a stale pinned ref — latest-sglang vs an old competitor pin is not a fair comparison

## Command Profiles

Every framework entry should carry `command_profiles`, not just inline args.

- Use `sglang_ref` or `framework_ref` to pin the release, commit, package version, or meaningful upstream line.
- Split profiles by hardware when best commands differ, for example `h100-80gb-2gpu` vs `h200-2gpu`.
- Split profiles by framework version when the best command changes across releases.
- Temporary extra serve args are acceptable for probes, but formal report data must promote the command into `command_profiles` and rerun through the profile before updating raw results.
- For SGLang failures, fix the backend or add a stable hardware-specific profile before using the comparison.
- For non-SGLang frameworks, seek the fastest fair command too; do not leave a slow default if upstream has a documented faster path.
- When upstream support changes, update install specs and profiles before claiming unsupported. Example: latest LightX2V supports LTX-2/LTX-2.3 even if an older pinned commit did not.
- When a case's default attention backend fails only on specific hardware (e.g. a Hopper-only precompiled kernel on Blackwell), the hardware-specific profile should first try another backend from the *same precision class* (another full-precision kernel: FA2/FA3/FA4/FlashInfer/SDPA). Only reach for a cross-class substitution (full-precision -> quantized/approximate) as a last resort, and label it explicitly in the profile `description`/`notes` and in the report — it is not a drop-in "fastest working" pick.

## Framework-Specific Pitfalls

- SGLang-Diffusion should not have failed cells in the final matrix. Treat SGLang import errors, NaNs, wrong scheduler behavior, OOMs, or request failures as bugs or bad profiles to fix before publishing.
- vLLM-Omni diffusion paths are usually single-GPU. Do not make SGLang multi-GPU only because another framework is slower; compare same-GPU-count data when possible and label intentional differences.
- LightX2V often needs model-specific attention, parallelism, and offload settings. Check upstream cookbook/configs before marking a case unsupported. Prefer another full-precision attention type (FA2/FA3/FA4/SDPA) over SageAttention when a same-precision-class option works — SageAttention (`sage_attn2`/`sage_attn3`) is a legitimate LightX2V fast path but is int8 QK-quantized, a different numeric class from full-precision FA3/FA4; it is not an automatic "fair" swap and must be labeled as a distinct precision path if used (see the Fairness Checklist attention-backend rule).
- diffusers is useful as a correctness and baseline reference, but it is not always a serving-optimized framework. Label it clearly and still use its fastest fair path (compile included, lossy tricks excluded).
- trtllm-visual (TensorRT-LLM VisualGen): (1) diffusion runs **only** the `_torch` (PyTorch) backend — there is no TensorRT-engine build for diffusion, so `--backend tensorrt` is not the path; `--backend pytorch` is correct. (2) VisualGen **requires torch.compile** (eager warmup crashes with a layer_norm dtype error) — it always runs compile-on; any profile note claiming "no torch compile" is stale. (3) Attention backend is set via `--extra_visual_gen_options <yml>` (`attention_config.backend`: VANILLA/TRTLLM/FA4/CUTEDSL). Default VANILLA = torch SDPA, which **already dispatches to FlashAttention on Hopper (fast exact)**. Measured on FLUX.1 / H100: VANILLA 7.25s ≈ FA4 7.28s, both *faster* than the TRTLLM kernel 7.78s. FA4 is sm100/Blackwell-tuned. So do not assume the "TRTLLM/TensorRT" backend is fastest — measure per hardware: VANILLA (default) is the fastest exact on Hopper (no `--extra_visual_gen_options` needed), FA4 on Blackwell. TRTLLM was slowest on Hopper here. `quant_attention_config`/`sparse_attention_config` are lossy — leave unset.
- SGLang Qwen-Image DiT bf16 was historically **not tensor-parallel-sharded** (`qwen_image.py` used `ReplicatedLinear` for all projections except under Nunchaku quant; RowParallel = 0), so `--tp-size >1` replicated the whole DiT and wasted the extra GPU (tp=1 ≈ tp=2). **PR #29774 (`Shard QwenImage DiT across TP ranks`) fixes this** — after it, qwen_image.py has ColumnParallel(12)/RowParallel(3). Measured effect on H100 (compile+resident, tp=2): no-CFG 6.35→5.66s, true-CFG 12.9→11.14s. BUT profiling the sharded run shows the bottleneck **shifts to communication**: RowParallel all-reduce becomes ~70% of kernel time (nccl AllReduce + sglang cross_device_reduce), because sharding every projection (incl. small ones and the small text-branch add_* on ~1024 tokens) adds ~3 all-reduces/block ×60 blocks. **Small layers are often better left ReplicatedLinear**: shard only the big Column→Row pairs (image QKV→attn-out, MLP-up→down) where GEMM-saved > all-reduce-cost; a Replicated middle layer would break the Column→Row chain (needs a gather), so keep small *standalone* projections (img_in/txt_in/proj_out, modulation) replicated rather than splitting a fused pair.
  - **For multi-branch (true-CFG), use CFG parallelism, NOT TP** — it sidesteps per-layer all-reduce entirely (each GPU runs one full CFG branch, DiT replicated, combine once/step). Measured true-CFG H100: `--enable-cfg-parallel` **6.49s** vs TP+#29774 11.14s vs vLLM 10.14s — CFG-parallel is **1.56× faster than vLLM** (the 2 branches run in parallel, so 2-pass CFG costs ~1 pass wall-clock). This matches the cookbook ("CFG parallelism is the most reliable multi-GPU Qwen/Wan speedup; TP is not the latency path").
  - no-CFG (single branch) can't use CFG-parallel, so selective sharding is the win: on top of #29774, keep the **text-branch MLP (txt_mlp) ReplicatedLinear** (its input is already full after the attention-out all-reduce; ~1024 tokens, so redundant compute < the all-reduce it saves ×60 blocks). Measured H100 no-CFG: full-shard #29774 5.65s → **txt_mlp-replicated 5.076s**, which now **beats vLLM 5.17s** (was losing). So the rule "shard big Column→Row pairs, replicate the small text branch" empirically closes AND flips the gap. flux/wan/ltx/cosmos DiTs were already Column/Row parallel.
  - **What actually survives e2e validation: only `txt_mlp` → `ReplicatedLinear`.** A big CLI sweep (per-step `denoise_steps_ms`) suggested a rich resolution crossover — also replicating `img_mlp` (small res) and the attention out-projs (`to_out`/`to_add_out` via all-gather+Replicated, "attn-out pair") — with apparent −18–23% wins. **Most of that did NOT hold on real-server e2e.** Verified on 2×H100 (1024², no-CFG, 50 steps, single_e2e): full-shard #29774 **5.58s** → +txt_mlp **~5.12s (−8%, robust, every batch agrees)** → +attn-out pair **~5.11s (+0.8%, mixed-sign interleaved paired diffs = within noise)**. So ship **txt_mlp-replicated only**; attn-out replication is a null e2e effect (not worth the code), and **`colgather` (ColumnParallel+all-gather for the out-proj) is measurably *worse* 5.25–5.33s** — the out-proj's contraction dim (heads) is sharded, so a Column formulation needs an input *and* output all-gather (≈ one all-reduce) plus an extra kernel, strictly dominated by RowParallel. All attn-out code is exact (MAE 0.65/255) but stays a research artifact behind `QWEN_REPL_ATTN`/`QWEN_ATTN_SCHEME` env flags. **Metric lesson (critical): the CLI `denoise_steps_ms` over-states comm-reduction wins** (server pipelines steps / captures graphs, hiding the all-reduce the CLI exposes), and cross-batch e2e noise (~4–5%) exceeds sub-5% effects — **resolve small latency claims only with e2e ABAB-interleaved paired repeats, never single runs, same-batch AABB, or CLI steps.** Full data + reference impl in `reports/h100x4-final-20260702/qwen_tp_selective_shard_analysis.md` + `qwen_image_selective_shard.py`.
  - **Validation matrix (e2e ABAB ×3, H100) — where txt_mlp replication applies and where it flips**: qwen 1024² tp=2 **−8%**; 512² tp=2 **−9%** (adding img_mlp → **−18%**, small-res-only, regresses ≥1024²); 1536² tp=2 +0.4% tiny cost; **tp=4 +1.1% consistent regression** (duplicated GEMM outgrows the 4-rank all-reduce → PR #29774 gates on `get_tp_world_size() <= 2`; qwen tp=4 e2e 6.04 is slower than tp=2 5.12 anyway — TP is not the latency path); **FLUX `ff_context` +0.8% = does NOT transfer** (19 dual-stream blocks vs qwen's 60 → 3× fewer all-reduces saved for the same duplicated-GEMM cost). Generalizing to another model / TP degree / resolution needs a per-layer cost comparison (blocks × all-reduce cost vs duplicated GEMM at the branch's token count) confirmed by e2e ABAB — never copy the qwen rule blindly. **Productized in sglang PR #30004**: `--dit-tp-plan {auto,full,aggressive,<plan.json>}` + `--dit-tp-plan-workload WxH` (per-layer planner, measured-rules registry, offline ABAB tuner `tools/tune_dit_tp_plan.py`) — prefer these flags over hand-patching qwen_image.py once merged.
- **Throughput mode measures 1/latency for every framework today — batching is default-off everywhere and vLLM-Omni's cannot even be forced on.** sglang default `batching_max_size=1`; vLLM-Omni diffusion default `max_num_seqs=1`. Measured (H100, qwen 1024² + zimage, conc=2/4 req, ABAB ×2): vllm-omni with explicit `--max-num-seqs 2` (flag confirmed received in server log) produces **identical qps AND an unchanged serial latency shape** (p50≈2×single, duration≈4×single) — no batch ever forms, because its request-granular scheduler admits the first arrival immediately (no batching window; `StepScheduler` mid-flight join is a placeholder) and the ms-later second request waits for the whole first request. Implications: (1) published throughput numbers are fair — vllm gains nothing from its batching flag; (2) a batch-window (sglang's `batching_delay_ms`) or step-level join is the piece that actually makes diffusion batching real under live arrivals — validate sglang's own dynamic batching forms batches before claiming throughput wins.
- **Parallelism STRATEGY (TP vs Ulysses/SP vs CFG) is a per-model choice, measured e2e.** qwen-image no-CFG 1024²: Ulysses (`--tp-size 1 --ulysses-degree 2 --enable-cfg-parallel false`) beats best-TP (incl. #29774 selective shard) by −4.7% (idle-node ABAB×3, 3/3; 4.89 vs 5.14; vLLM 1.04×) — big comm-heavy image → SP all-to-all < TP all-reduce. But flux1/zimage LOSE on Ulysses (+4.7%/+8%) — smaller/less-comm-bound → TP wins. multi-branch (true-CFG) → CFG-parallel. Lossless check that mattered: TP-vs-Ulysses is same-precision-class (converged MAE 0.74/255 eager, 2.27 with compile), and it's NOT a bug — MAE DECREASES with steps (3.41→0.74 over 1→30, TP-vs-TP floor=0.0) = convergent FP-reorder from the all-to-all. Method note: run parallelism A/B on an IDLE node — a second GPU job on other cards of the same node contaminates latency via shared host/PCIe (it silently wrecked the first qwen Ulysses run). This strategy axis belongs in the #30004 planner's measured registry (currently it only tunes layer layout WITHIN tp).
- **Command-error guards are in the system — use them, don't re-derive**: ① `configs/benchmark/SELECTED.md` (auto-generated by the build) shows the profile the harness will ACTUALLY select per case on the policy hardware — edit THAT profile, never guess; ambiguous multi-matches are flagged there and warned at build AND harness runtime. ② `build_report_artifacts` cross-checks every published sglang row against the current config selection (profile name + serve_args token subset) and prints COMMAND DRIFT warnings; `DIFFUSION_BENCH_STRICT_COMMANDS=1` turns them fatal — a config edited without re-running its cell can no longer publish silently. ③ Selection semantics live in `src/diffusion_bench/config_guard.py`, MIRRORING `run_comparison._select_command_profile` (candidate substring of profile NAME or `hardware` values; first non-default match wins) — change both in the same commit. ④ Run devbox cases with `scripts/devbox_run_cases.sh` (set -u, readonly port, narrow per-device GPU kill, greppable RESULT lines) instead of hand-rolled bash. ⑤ `policy_exception` must cite dated evidence (lint-enforced).
- **The build now machine-enforces "best lossless" per case** (`build_benchmark_config.py` lint): the sglang profile actually SELECTED on the published hardware must have compile on and offloads off, or carry a `policy_exception` with measured evidence. Origin: the 07-02 published run silently left 5 cases on conservative auto/offload defaults. Measured-fastest H100 rules it encoded (A/B 2026-07-03): wan21/wan22/cosmos want `--performance-mode speed --enable-torch-compile` (-5.7%/-11.4%/-2.7%); **Z-Image wants speed WITHOUT compile** (0.70s eager vs 0.99s compiled vs 0.79s auto — compile loses on 9-step turbo models); **LTX-2.3 speed+compile OOMs on 2x80GB** (snapshot mode stays). sglang PR #30016 makes performance_mode=speed auto-enable compile, so eager exceptions must say `--enable-torch-compile false` explicitly.
- For Wan, LTX, Flux, Qwen-Image, and Z-Image, verify official model-family semantics first: scheduler, frame count, guidance fields, VAE dtype/offload, and whether the upstream implementation is one-stage or two-stage.
- **LightX2V main HEAD's `flash_attn.cute` submodule can be unusable regardless of install order, because no single `nvidia-cutlass-dsl` release satisfies it.** `flash-attn==2.8.3`'s `cute/utils.py` references `cutlass.cute.core.ThrMma` at import time; `flashinfer-python==0.6.11`'s unpinned `nvidia-cutlass-dsl>=4.5.0` constraint resolves to the newest release, and `ThrMma` was removed in 4.6.0 (confirmed present-but-broken at 4.6.0/4.6.1). Pinning back to the last pre-removal 4.5.x (4.5.3) doesn't fix it either — that release is itself missing `cutlass.utils.ampere_helpers`, which the same cute code also needs, and `nvidia-cutlass-dsl-libs-core` (a hard dependency for that module) was never published for 4.5.x at all (PyPI only lists 4.6.0/4.6.1 for it). LightX2V's own attn modules (`flash_attn.py`, `dynamic_sparse_attn.py`, `sparse_operator.py`) already wrap the cute import in `except ImportError`, so the fix is to delete the unusable `flash_attn/cute/` directory post-install (`scripts/install_comparison_frameworks.sh`, lightx2v branch) rather than chase a cutlass-dsl version — the missing directory then raises a plain `ModuleNotFoundError` (an `ImportError` subclass) that the existing fallback already handles correctly. This is a moving-target problem under the latest-vs-latest policy: re-check it whenever `flash-attn`/`nvidia-cutlass-dsl` versions drift.
- **A `rope_type` value in a case's `lightx2v_config` is describing a specific `ROPE_REGISTER` backend key, not a semantic/architecture label — don't hand-write one without checking the current registry.** Six Wan case profiles carried `"rope_type": "torch"` (set months earlier, likely valid against an older LightX2V), but current main's registry only has `chunked_rope`/`flashinfer_rope`/`torch_complex_rope`/`torch_real_rope` — `"torch"` alone raises `KeyError: 'torch'`. Upstream's own example configs (`configs/wan/*.json` in the LightX2V repo) don't set `rope_type` at all, so the fix was to delete the stale key and let it default to `flashinfer_rope` (also the fastest option, so this is a pure win, not just a fix). Separately, `_merge_ltx2_single_file_metadata` (`run_comparison.py`) was blindly forwarding a checkpoint's own `rope_type` metadata field (describing rope *layout*, e.g. `"split"`) straight into `lightx2v_config["rope_type"]` as if it were the backend-selection key — two different axes that happen to share a field name in the checkpoint's metadata vs. LightX2V's config schema. Don't assume a same-named field means the same thing across a model's own metadata and a serving framework's config.

## Running Benchmarks

Use repo scripts when available instead of ad hoc one-liners:

- single request: `scripts/run_h200_single_e2e_*.sh`
- throughput: `scripts/run_h200_throughput_*.sh`
- targeted reruns: set `CASES` / `FRAMEWORKS` env vars when the script supports them, or add a dedicated script for repeatable reruns

Before running:

- Use isolated pyvenvs for conflicting frameworks, and record the install spec plus observed version/commit.
- Keep H100 and H200 data separate unless a report explicitly compares hardware.
- Confirm generated commands before expensive runs, especially `num_frames`, resolution, steps, dtype, attention backend, GPU count, and selected command profile.
- Warm up long enough to remove first-request artifacts, but do not hide compile, cache, or model-download effects inside reported latency.

For throughput, use enough warmup to avoid first-request artifacts, then record at least:

- `num_requests`
- max concurrency
- p50 latency
- p99 latency
- QPS
- failure count

Also record p95 latency when available. For very fast image cases, run multiple requests at realistic concurrency; for long video cases, a small number of measured requests (with extra warmup) is enough — but single-request latency is still the steady-state median of those repeats, never one isolated shot (see the Fairness Checklist).
For high-pressure cross-framework throughput reports, prefer cases supported by every framework in scope. It is fine to include one or two video cases, but use a smaller request/concurrency budget than fast image cases and keep the image/video budgets explicit in the reproduce script.

Use `py-spy` only for diagnosis, not as part of the benchmark timing path.

## Failure Classification

Classify every failed or missing cell:

- `unsupported`: upstream does not support this model/task
- `no_profile`: upstream support is unknown or possible, but this benchmark has no validated aligned serving profile
- `not_run`: framework is configured for the case, but this result artifact does not include a run
- `failed`: server or request failed; include the short root cause
- `invalid`: command used wrong shape, frames, model, dtype, cache, compile, or sampling params

If the failure is SGLang, treat it as a bug or bad profile and fix before finalizing the report. If the failure is another framework, verify the official docs/scripts before marking unsupported.

## Reports And Images

Formal reports use the fixed tracker issue, not ad hoc issues. For this repo, keep `mickqian/diffusion-bench-framework#1` as the canonical tracker and maintain one latest data-only formal report comment there.

Formal issue comments should be data-only: run metadata, framework versions, case tables, ratios, statuses, and reasons. Put debug analysis elsewhere.

Use the fixed artifact workflow for future reports:

- regenerate merged JSON, issue Markdown, dashboard Markdown, PNG, and SVG from `scripts/generate_h200_report_artifacts.sh`
- review `tmp/report/h200-framework-comparison-merged-local.issue.md` and `.png`
- delete stale formal tracker comments before appending the new generated issue Markdown
- keep the report shape stable so historical comments/images are comparable

For comparison images:

- group rows by case so SGLang and other frameworks are adjacent
- separate single-request and throughput sections
- show missing/failed cells explicitly
- show ratios relative to SGLang for the same case
- include source result JSON names and filter rules in the footer
- regenerate from a fixed script so future reports are reproducible
- include every framework in scope, even when the cell is `unsupported`, `no_profile`, `failed`, or `not_run`

Formal tracker-issue report (data-only comment on the fixed tracker `mickqian/diffusion-bench-framework#1`, one grouped block per case, stable layout across runs):

- run header: run timestamp, benchmark commit, sglang commit/version, run id, GPU count + model
- case metadata table: `| model | task | dims | steps | cfg |`
- framework comparison table: `| framework | profile | gpus | single_e2e_s | single/sglang | single_status | done/reqs | concurrency | p50_s | p50/sglang | p95_s | p99_s | qps | qps/sglang | throughput_status | reason |`
- delete stale formal comments before appending the new one; keep interpretation OUT of the comment (if a run looks unfair/regressed, write a separate investigation note)

## Publishing A Completed Run

After a full run, publish results so they are durable and comparable over time:

- commit the complete result JSONs (every case, every framework, single + throughput) into the repo
- update the GitHub Pages benchmark section: write `docs/data/latest-cross-framework.json` and append the run to `docs/data/historical-cross-framework.json`, then run `scripts/refresh_docs_data.py` to refresh the inline preview
- the published benchmark MUST state the **benchmark date** and the **exact version/commit of every framework** (sglang commit, vllm-omni commit, lightx2v commit, trtllm-visual version) — a latest-vs-latest comparison is uninterpretable later without the dated version set

## Output Discipline

When reporting back, include:

- which result JSONs were used
- which framework versions/commits were compared
- which cases were excluded and why
- whether data is benchmarked, failed, unsupported, or supported but not run
- exact paths to generated report/image artifacts

## Install & Environment Reference

Operational knobs relocated from the README. The runner installs conflicting frameworks into isolated virtualenvs; override the specs below only when intentionally changing a tracked ref.

**Pins are time-of-report snapshots, not permanent defaults.** The install specs below (e.g. `vllm-omni==0.18.0`, a fixed `LIGHTX2V_INSTALL_SPEC` commit, `tensorrt-llm==1.3.0rc18`) capture the competitor version tracked when that report ran, for reproducing *that* run only. **A fresh complete run defaults to latest-vs-latest** (see *Default Run Configuration* above): sglang `origin/main` against competitors' newest. Override every competitor spec to latest and record the resolved versions/commits in the result artifact: `VLLM_INSTALL_SPEC=vllm` (newest), `VLLM_OMNI_INSTALL_SPEC=git+https://github.com/vllm-project/vllm-omni.git` (main HEAD), `LIGHTX2V_INSTALL_SPEC=git+https://github.com/ModelTC/LightX2V.git` (main HEAD), newest `TRTLLM_INSTALL_SPEC`. Do not silently inherit an old default pin for a fresh run — that benchmarks stale competitors against a current sglang and is unfair.

**Compile policy (no-precision-loss best perf).** The default goal is each framework's best *lossless* performance, so compile is ON: run with `DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0`. In that mode the harness does not set `TORCH_COMPILE_DISABLE`, omits vLLM-Omni's `--enforce-eager --compilation-config '{"mode":0}'`, and defaults LightX2V `compile=true` (a profile's `lightx2v_config.compile=false` still opts a specific model out if its compile path breaks). sglang enables compile via profile serve_args (`--enable-torch-compile`, optionally `--performance-mode speed`) — every benched case's selected profile should carry it. Compile/warmup cost is paid during warmup and excluded from measured latency. Set `DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=1` only for a deliberately compile-free / eager comparison. SGLang is always launched with `--backend sglang` so it never silently benchmarks the vanilla-diffusers fallback (under `--backend auto` it falls back when `model_index.json` can't resolve — gated FLUX 403s, or `HF_HUB_OFFLINE`).

**trtllm-visual.** Installs `tensorrt-llm==1.3.0rc18` from the NVIDIA PyPI index; override `TRTLLM_INSTALL_SPEC` / `TRTLLM_PIP_EXTRA_INDEX_URL`. VisualGen (the `/v1/images/generations` path + `get_is_diffusion_model` auto-detect in `trtllm-serve`) exists **only in the 1.3.0 release candidates** — 1.2.x stable serves FLUX through the LLM engine and fails. Multi-GPU (CFG/Ulysses) is set via a `--extra_visual_gen_options` YAML, not `--tp_size`.

**LightX2V.** Pinned to the latest tracked upstream commit with LTX-2/2.3 runner support (`LIGHTX2V_INSTALL_SPEC`). Pins `transformers<5` (`LIGHTX2V_TRANSFORMERS_INSTALL_SPEC`) for the LTX-2 Gemma/SigLIP layout, then restores `safetensors>=0.8.0rc0`. H200 profiles use FA3/FlashInfer: the installer rebuilds flash-attn from source then uses a pinned torch 2.8/cu128 FA3 artifact (`LIGHTX2V_FA3_HF_REPO` / `_HF_REVISION` / `_HF_SUBDIR`; `LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC` to switch to a source build); also installs `sageattention` (for `sage_attn2` configs) and `hf-xet`. Single-file LTX checkpoints (LTX-2.3) get transformer metadata projected from the safetensors header into the generated config. Report LTX-2.3 LightX2V on the same 2-GPU budget as SGLang; H100 LTX profiles are hardware-specific.

**Reuse / cache.** `SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1` reuses an already-installed isolated venv. `DIFFUSION_BENCH_HF_CACHE_DIR` or `HF_HOME` redirects the HuggingFace cache when the default filesystem is small; fixed run scripts default to `/root/diffusion-bench-hf-cache/hub`.

**Run-script env vars (targeted reruns).** `DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS`; throughput: `THROUGHPUT_FRAMEWORKS`, `THROUGHPUT_CASES` / `THROUGHPUT_IMAGE_CASES` / `THROUGHPUT_VIDEO_CASES`, `IMAGE_NUM_REQUESTS` / `IMAGE_MAX_CONCURRENCY` / `VIDEO_NUM_REQUESTS` / `VIDEO_MAX_CONCURRENCY`; single-request: `SINGLE_E2E_FRAMEWORKS`, `SINGLE_E2E_CASES`, `SINGLE_E2E_SGLANG_PROFILE`; Wan video sweep: `SGLANG_VIDEO_REFRESH_PROFILE`, `RUN_SGLANG` / `RUN_VLLM_OMNI` / `RUN_LIGHTX2V`.

**Reproduction-script index.** `run_h200_single_e2e_*.sh` / `run_h200_throughput_*.sh` (+ `_common_` for the 3-framework suite), `run_h200_ltx_lightx2v_*` / `run_h100_ltx_lightx2v_*`, `run_h200_wan_vllm_omni_*`, `run_h200_video_parallelism_refresh_*`, `run_h200_cosmos3_*` (+ `probe_h200_cosmos3_*` profile probes), `run_trtllm_visual_h200.md` (runbook), `generate_h200_report_artifacts.sh`. Use `scripts/summarize_result_jsons.py runs/*.json` for a compact latency/failure table.
