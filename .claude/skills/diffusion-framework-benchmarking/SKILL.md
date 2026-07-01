---
name: diffusion-framework-benchmarking
description: Use when planning, running, extending, or interpreting diffusion framework benchmarks across SGLang-Diffusion, vLLM-Omni, LightX2V, diffusers, or similar systems, especially when comparing single-request latency, throughput, command profiles, framework versions, model support, fairness, failure reasons, or report/image generation in diffusion-bench-framework.
---

# Diffusion Framework Benchmarking

## Goal

Produce reproducible, fair, and actionable performance data for open-source diffusion serving frameworks. The benchmark is a guardrail for SGLang-Diffusion, so invalid comparisons are worse than missing data.

## Default Run Configuration (Complete Cross-Framework Run)

Unless the user asks otherwise, a complete cross-framework run defaults to:

- **Version policy: latest-vs-latest.** SGLang runs `origin/main` HEAD; every competitor runs its newest line — `vllm` newest, `vllm-omni` main HEAD, `LightX2V` main HEAD, newest `trtllm` release candidate. Override every competitor `*_INSTALL_SPEC` to latest and record each resolved version/commit in the result artifact. Pinned snapshots (the version sets in the Install reference and in `manifests/`) are **only** for reproducing a specific dated historical report — do not silently inherit an old pin for a fresh run.
- **Framework scope: all frameworks.** Include `sglang`, `vllm-omni`, `lightx2v`, and `trtllm-visual` (image cases). `diffusers` is an optional correctness/baseline reference, labeled as such. A complete run includes every framework in scope; cells a framework can't run are classified (`unsupported` / `no_profile` / `failed` / `not_run` / `invalid`), never silently dropped.
- **Execution environment: a freshly-acquired `rx devbox` H200.** Acquire a fresh H200 via the `rx` CLI (see the `rx-devbox` / `remote-development` skills) as the default machine; 2-GPU is the standard profile budget. Keep conflicting competitor frameworks in isolated virtualenvs. The local MacBook cannot run these benchmarks.

When the user explicitly narrows any of these (a single framework, a pinned ref to reproduce, an existing machine), follow that instead — these are defaults, not overrides of an explicit request.

## Non-Negotiables

- Do not use response cache, Cache-DiT, precomputed outputs, quantized checkpoints, reduced steps, or distilled weights unless the case explicitly compares those semantics.
- Disable torch compile for all frameworks unless the user explicitly asks for compile-inclusive numbers.
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
- attention backend choice is semantically consistent across the frameworks being compared, not just "whatever runs without crashing." Exact/full-precision attention (FlashAttention 2/3/4, FlashInfer, Torch SDPA) is one fair-swappable family — picking the fastest available member of that family for the hardware is fine. Quantized or approximate attention (SageAttention int8 QK, block-sparse/sparse attention, distilled attention) is a *different numeric class*. Do not silently substitute a quantized/approximate backend for a competitor when the reference (e.g. SGLang's backend for that case) runs full precision — that lets the competitor trade away accuracy the reference isn't trading away, which inflates its speed unfairly. If the only working backend on the target hardware is a different numeric class, either keep hunting for a same-class option, mark the cell `no_profile`/`failed` with the root cause, or run and report it as an explicitly labeled different-precision data point — never fold it into the main comparison unlabeled.
- latency metric is unified across frameworks on **client-side wall clock** — it is the only framework-agnostic, consistently-defined number. Server-side timers (e.g. SGLang's perf-dump `total_duration_ms`) are framework-specific and not emitted by every framework, so keep them only as per-framework diagnostic annotations, never as the cross-framework headline.
- single-request latency is a **steady-state median of several back-to-back requests after warmup**, not one isolated request. A single shot can absorb a cold/idle-path artifact — e.g. vLLM-Omni 0.24 stalls ~55s on spaced concurrency-1 requests while its engine runs in ~1s. Image cases repeat cheaply (≈5); video is expensive, so warm harder (≈3) and repeat fewer (≈2). If the repeats do not converge — or the single-request median exceeds the steady-state throughput p50 for the same case — the number is untrustworthy: mark the cell an anomaly and withhold the value (do not publish a stalled number as if it were latency). throughput records `num_requests`, concurrency, p50, p95, p99, and QPS; continuous load does not hit the isolated-request stall.
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
- diffusers is useful as a correctness and baseline reference, but it is not always a serving-optimized framework. Label it clearly and still use its fastest fair non-compile path.
- For Wan, LTX, Flux, Qwen-Image, and Z-Image, verify official model-family semantics first: scheduler, frame count, guidance fields, VAE dtype/offload, and whether the upstream implementation is one-stage or two-stage.

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

**Compile policy (cache-free fairness).** `diffusion-bench-compare` sets `TORCH_COMPILE_DISABLE=1` for all benchmarked subprocesses so cold compile time doesn't leak into runs. vLLM-Omni also gets `--enforce-eager --compilation-config '{"mode":0}'`; LightX2V configs force `compile=false` / `compile_shapes=[]`. SGLang is launched with `--backend sglang` so it never silently benchmarks the vanilla-diffusers fallback (under `--backend auto` it falls back when `model_index.json` can't resolve — gated FLUX 403s, or `HF_HUB_OFFLINE`).

**trtllm-visual.** Installs `tensorrt-llm==1.3.0rc18` from the NVIDIA PyPI index; override `TRTLLM_INSTALL_SPEC` / `TRTLLM_PIP_EXTRA_INDEX_URL`. VisualGen (the `/v1/images/generations` path + `get_is_diffusion_model` auto-detect in `trtllm-serve`) exists **only in the 1.3.0 release candidates** — 1.2.x stable serves FLUX through the LLM engine and fails. Multi-GPU (CFG/Ulysses) is set via a `--extra_visual_gen_options` YAML, not `--tp_size`.

**LightX2V.** Pinned to the latest tracked upstream commit with LTX-2/2.3 runner support (`LIGHTX2V_INSTALL_SPEC`). Pins `transformers<5` (`LIGHTX2V_TRANSFORMERS_INSTALL_SPEC`) for the LTX-2 Gemma/SigLIP layout, then restores `safetensors>=0.8.0rc0`. H200 profiles use FA3/FlashInfer: the installer rebuilds flash-attn from source then uses a pinned torch 2.8/cu128 FA3 artifact (`LIGHTX2V_FA3_HF_REPO` / `_HF_REVISION` / `_HF_SUBDIR`; `LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC` to switch to a source build); also installs `sageattention` (for `sage_attn2` configs) and `hf-xet`. Single-file LTX checkpoints (LTX-2.3) get transformer metadata projected from the safetensors header into the generated config. Report LTX-2.3 LightX2V on the same 2-GPU budget as SGLang; H100 LTX profiles are hardware-specific.

**Reuse / cache.** `SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1` reuses an already-installed isolated venv. `DIFFUSION_BENCH_HF_CACHE_DIR` or `HF_HOME` redirects the HuggingFace cache when the default filesystem is small; fixed run scripts default to `/root/diffusion-bench-hf-cache/hub`.

**Run-script env vars (targeted reruns).** `DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS`; throughput: `THROUGHPUT_FRAMEWORKS`, `THROUGHPUT_CASES` / `THROUGHPUT_IMAGE_CASES` / `THROUGHPUT_VIDEO_CASES`, `IMAGE_NUM_REQUESTS` / `IMAGE_MAX_CONCURRENCY` / `VIDEO_NUM_REQUESTS` / `VIDEO_MAX_CONCURRENCY`; single-request: `SINGLE_E2E_FRAMEWORKS`, `SINGLE_E2E_CASES`, `SINGLE_E2E_SGLANG_PROFILE`; Wan video sweep: `SGLANG_VIDEO_REFRESH_PROFILE`, `RUN_SGLANG` / `RUN_VLLM_OMNI` / `RUN_LIGHTX2V`.

**Reproduction-script index.** `run_h200_single_e2e_*.sh` / `run_h200_throughput_*.sh` (+ `_common_` for the 3-framework suite), `run_h200_ltx_lightx2v_*` / `run_h100_ltx_lightx2v_*`, `run_h200_wan_vllm_omni_*`, `run_h200_video_parallelism_refresh_*`, `run_h200_cosmos3_*` (+ `probe_h200_cosmos3_*` profile probes), `run_trtllm_visual_h200.md` (runbook), `generate_h200_report_artifacts.sh`. Use `scripts/summarize_result_jsons.py runs/*.json` for a compact latency/failure table.
