# Diffusion Bench Framework

Standalone harness for fair, reproducible comparison of diffusion serving frameworks on the same case:

- **SGLang-Diffusion**
- **vLLM-Omni**
- **LightX2V**
- **TensorRT-LLM VisualGen** (`trtllm-visual`, served via `trtllm-serve`)

It launches one serving stack per case, sends a single end-to-end request, optionally runs a fixed high-pressure throughput workload, writes a unified JSON result, and publishes a GitHub Pages one-pager.

It also includes a separate SGLang-Diffusion stability probe suite for periodic server/API robustness experiments. That suite is intentionally not part of the formal cross-framework latency comparison.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip && python3 -m pip install -e .

# SGLang-Diffusion must be in the same env to bench the `sglang` framework:
python3 -m pip install -e /path/to/sglang/python[diffusion]

# Dry-run (prints planned commands, runs nothing):
diffusion-bench-compare --dry-run --modes single_e2e throughput

# Run selected cases:
diffusion-bench-compare \
  --config configs/comparison_configs.json \
  --case-ids flux2_dev_t2i_1024 qwen_image_2512_t2i_1024 \
  --frameworks sglang trtllm-visual \
  --modes single_e2e throughput \
  --run-id "$(date -u +%Y%m%d-%H%M%S)" --output comparison-results.json
```

vLLM-Omni, LightX2V, and TensorRT-LLM VisualGen are installed into isolated virtualenvs by the runner (their deps conflict with SGLang). `trtllm-visual` is a generic, config-driven HTTP framework — its launch command and request shape come entirely from per-case `http_server` / `http_request` config blocks, so new OpenAI-compatible backends are added without editing `run_comparison.py`.

## How it works

```
diffusion-bench-compare → per case: launch stack → single e2e (+ optional throughput) → unified JSON → dashboard
```

- **Cases** — the bundled config is the formal tracking plan: one representative case per official model ID (FLUX.1/.2, Qwen-Image T2I+edit, Z-Image, Wan2.1/2.2, LTX-2/2.3, Cosmos3 Nano). Extra variants are for targeted investigations, not default coverage.
- **Command profiles** — each case tracks version- and hardware-specific best commands per framework (`--sglang-profile`, `DIFFUSION_BENCH_<FW>_PROFILE`; auto-selected by hardware else `default`). The runner records the resolved profile, framework ref, effective args, and actual server command in the result JSON.
- **Fairness** — cache-free, no Cache-DiT, same shape/seed/steps/guidance/dtype, same GPU count. torch.compile is ON for the competitors and OFF for SGLang by policy (its fused kernels match or beat compiler fusion, measured), so each framework runs its own fastest lossless path rather than a lowest-common-denominator one.
- **Not benchmarking the fallback** — `--backend sglang` does *not* guarantee this on its own. SGLang loads a component through a native/Diffusers fallback when it has no customized implementation for the requested configuration, and its loader only *refuses* to when `tp_size`/`sp_degree`/`ulysses_degree`/`ring_degree`/`kv_gather_degree` is above 1 or FSDP is on — under CFG parallelism alone, or on one GPU, it proceeds behind a single log line while the row still says "sglang". The runner therefore scans the server log for that path, records `native_fallback_components` on the result, and `scripts/merge_and_publish_run.sh` refuses to publish a run containing one.

## Where things live

- `src/diffusion_bench/` — `run_comparison.py` (server lifecycle + runner), `bench_serving.py` (async client), `generate_dashboard.py` / `build_report_artifacts.py` (data-only merged JSON + Markdown reports).
- `src/diffusion_bench/sglang_probe/` — SGLang-Diffusion stability probe matrix, runner, reporting, and reusable profiles.
- `configs/comparison_configs.json` — editable case/framework config.
- `scripts/` — reproducible per-run scripts (`run_h200_*.sh`, `run_b200_*.sh`), `install_comparison_frameworks.sh`, diagnostic probes (`probe_*.sh`), `merge_and_publish_run.sh`, `run_profile_vs_default.sh`, and `tests/run_all.sh`. One script per tracked run; see the script header for its knobs.
- `manifests/` — pinned run manifests (bench/SGLang/framework/hardware versions); the source of truth for reconstructing historical reports.
- `docs/sglang_probe/` — probe usage notes plus historical probe matrix/result summaries.
- `.claude/skills/` — operating discipline (see below).
- `docs/` — the GitHub Pages one-pager.

## GitHub Pages dashboard

The published one-pager (`docs/index.html`) reads two committed JSON files and renders **any** framework set they declare — no HTML edits to add a framework column:

- `docs/data/latest-cross-framework.json` — current spot-check. Declare `framework_order` and give each row a `cells` map keyed by framework.
- `docs/data/historical-cross-framework.json` — historical matrix; columns derive from the `cells` present per row.
- Register a new framework's bar color + label once in `FRAMEWORK_META` / `frameworkLabels` in `docs/index.html`; unregistered keys still render neutrally.
- After editing the JSON, run `python3 scripts/refresh_docs_data.py` to refresh the inline preview snapshots (the deployed page fetches the JSON directly).

## Skills

Operating discipline for long-term maintenance lives in `.claude/skills/` (auto-discovered by Claude Code when this repo is opened as the project):

| skill | use for |
|---|---|
| `diffusion-framework-benchmarking` | plan/run/interpret fair benchmarks; publish results + formal tracker-issue report (`mickqian/diffusion-bench-framework#1`); per-framework install pins + env-var reference |
| `diffusion-case-onboarding` | add model/framework cases + command profiles; maintain runner/config/result-schema without breaking comparability |
| `diffusion-regression-investigator` | investigate when SGLang looks slow or a comparison looks unfair |

Formal reports append one data-only comment to the tracker issue instead of opening new issues. Detailed operational knobs (LightX2V FA3 pins, `trtllm` install spec, throughput/single-e2e env vars, per-script overrides) live in the **benchmarking skill**, not here.
