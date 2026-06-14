# Diffusion Bench Framework

Standalone harness for fair, reproducible comparison of diffusion serving frameworks on the same case:

- **SGLang-Diffusion**
- **vLLM-Omni**
- **LightX2V**
- **TensorRT-LLM VisualGen** (`trtllm-visual`, served via `trtllm-serve`)

It launches one serving stack per case, sends a single end-to-end request, optionally runs a fixed high-pressure throughput workload, writes a unified JSON result, and publishes a GitHub Pages one-pager.

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
- **Fairness** — cache-free, no Cache-DiT, same shape/seed/steps/guidance/dtype, same GPU count; compile disabled unless explicitly comparing compile-inclusive numbers. SGLang runs with `--backend sglang` so it never silently benchmarks the diffusers fallback.

## Where things live

- `src/diffusion_bench/` — `run_comparison.py` (server lifecycle + runner), `bench_serving.py` (async client), `generate_dashboard.py` / `build_report_artifacts.py` / `generate_report_image.py`.
- `configs/comparison_configs.json` — editable case/framework config.
- `scripts/` — reproducible per-run scripts (`run_h200_*.sh`), `install_comparison_frameworks.sh`, `run_trtllm_visual_h200.md`, artifact generators. One script per tracked run; see the script header for its knobs.
- `manifests/` — pinned run manifests (bench/SGLang/framework/hardware versions); the source of truth for reconstructing historical reports.
- `skills/` — operating discipline (see below).
- `docs/` — the GitHub Pages one-pager.

## GitHub Pages dashboard

The published one-pager (`docs/index.html`) reads two committed JSON files and renders **any** framework set they declare — no HTML edits to add a framework column:

- `docs/data/latest-cross-framework.json` — current spot-check. Declare `framework_order` and give each row a `cells` map keyed by framework.
- `docs/data/historical-cross-framework.json` — historical matrix; columns derive from the `cells` present per row.
- Register a new framework's bar color + label once in `FRAMEWORK_META` / `frameworkLabels` in `docs/index.html`; unregistered keys still render neutrally.
- After editing the JSON, run `python3 scripts/refresh_docs_data.py` to refresh the inline preview snapshots (the deployed page fetches the JSON directly).

## Skills

Operating discipline for long-term maintenance lives in `skills/` (install/symlink into your Codex/Claude skills dir to auto-discover):

| skill | use for |
|---|---|
| `diffusion-framework-benchmarking` | plan/run/interpret fair benchmarks; **per-framework install pins + env-var reference** |
| `diffusion-case-onboarding` | add new model/framework cases and command profiles |
| `diffusion-bench-maintainer` | change runner/config/dashboard without breaking comparability |
| `diffusion-regression-investigator` | investigate when SGLang looks slow or a comparison looks unfair |
| `diffusion-performance-reporting` | append data-only comments to the fixed tracker issue (`mickqian/diffusion-bench-framework#1`) |

Formal reports append one data-only comment to the tracker issue instead of opening new issues. Detailed operational knobs (LightX2V FA3 pins, `trtllm` install spec, throughput/single-e2e env vars, per-script overrides) live in the **benchmarking skill**, not here.
