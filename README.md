# Diffusion Bench Framework

Standalone benchmark harness for comparing diffusion serving frameworks:

- SGLang-Diffusion
- vLLM-Omni
- LightX2V

The repo launches one serving stack per case, sends a single end-to-end request, optionally runs a fixed high-pressure throughput workload, and writes a unified JSON result file.

## Layout

- `src/diffusion_bench/run_comparison.py`: cross-framework server lifecycle and benchmark runner.
- `src/diffusion_bench/bench_serving.py`: fixed/VBench/random async serving benchmark client.
- `src/diffusion_bench/datasets.py`: benchmark dataset definitions.
- `src/diffusion_bench/generate_dashboard.py`: Markdown dashboard generator.
- `src/diffusion_bench/build_report_artifacts.py`: merges result JSONs and regenerates report/image artifacts.
- `src/diffusion_bench/generate_report_image.py`: PNG/SVG comparison image generator.
- `configs/comparison_configs.json`: editable copy of the default model, prompt, shape, seed, and framework settings.
- `scripts/install_comparison_frameworks.sh`: isolated venv installer for vLLM-Omni and LightX2V.
- `scripts/run_h200_single_e2e_20260510.sh`: reproducible H200 single-request run script for the 2026-05-10 report shape.
- `scripts/run_h200_throughput_20260511.sh`: reproducible H200 throughput run script for faster image cases.
- `scripts/run_h200_ltx_lightx2v_20260513.sh`: targeted H200 LightX2V LTX-2/LTX-2.3 rerun with the latest tracked upstream commit.
- `scripts/run_h100_ltx_lightx2v_20260513.sh`: targeted H100 LightX2V LTX-2/LTX-2.3 rerun when H200 is unavailable.
- `scripts/generate_h200_report_artifacts.sh`: fixed local entrypoint for merged JSON, issue Markdown, dashboard Markdown, and image output.
- `manifests/`: pinned run manifests tying report data to bench, SGLang, framework, and hardware versions.
- `skills/`: repo-local Codex skills for maintaining long-term performance tracking.
- `.github/ISSUE_TEMPLATE/performance-report.yml`: fallback template for bootstrapping a tracker issue in a new fork.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

SGLang-Diffusion must be installed in the main environment if you want to run the `sglang` framework. Install it with the diffusion extra, for example:

```bash
python3 -m pip install -e /path/to/sglang/python[diffusion]
```

vLLM-Omni and LightX2V are installed into isolated temporary virtualenvs by the runner because their dependencies conflict with SGLang.

`diffusion-bench-compare` forces `TORCH_COMPILE_DISABLE=1` for all benchmarked framework subprocesses so cold compile time does not leak into cross-framework runs. vLLM-Omni also gets `--enforce-eager --compilation-config '{"mode":0}'` because its diffusion runner uses `enforce_eager` to skip regional torch.compile. LightX2V config files force `compile=false` and `compile_shapes=[]`.

Set `SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1` only for reruns where the isolated framework venv has already been installed and should be reused.
The default installer pins LightX2V to the latest tracked upstream commit with LTX-2/LTX-2.3 runner support; override `LIGHTX2V_INSTALL_SPEC` only when intentionally changing that framework ref.
The LightX2V isolated venv also pins `transformers<5` by default because the tracked LTX-2 Gemma path expects the Transformers 4.x SigLIP module layout; override `LIGHTX2V_TRANSFORMERS_INSTALL_SPEC` only when validating a newer upstream stack.
Set `DIFFUSION_BENCH_HF_CACHE_DIR` or `HF_HOME` when the default HuggingFace cache filesystem is small or full. The fixed run scripts default to `/root/diffusion-bench-hf-cache/hub` so model downloads are kept separate from isolated framework venvs.
The H200 LightX2V profiles use FA3/FlashInfer paths. By default the installer rebuilds flash-attn from source for the isolated torch version, then uses a pinned torch 2.8/cu128 FA3 artifact via `LIGHTX2V_FA3_HF_REPO`, `LIGHTX2V_FA3_HF_REVISION`, and `LIGHTX2V_FA3_HF_SUBDIR`; set `LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC` only when intentionally switching back to a source build. It also installs `sageattention` for upstream LTX configs that select `sage_attn2`.
For single-file LTX checkpoints such as LTX-2.3, the harness projects transformer metadata from the safetensors header into the generated LightX2V config so server runs match upstream inference config semantics.
H100 LTX LightX2V profiles are hardware-specific: LTX-2 uses upstream block offload; LTX-2.3 keeps the best attempted full-offload profile, but currently still fails after warmup on 80GB GPUs and should be reported as no stable H100 server data.
`scripts/run_h200_throughput_20260511.sh` sets `DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS="--batching-max-size ${THROUGHPUT_MAX_CONCURRENCY} --batching-delay-ms 0"` by default so SGLang throughput runs use the same request concurrency as the benchmark client.
Set `THROUGHPUT_FRAMEWORKS="lightx2v"` and `THROUGHPUT_CASES="zimage_turbo_t2i_1024"` to reproduce a targeted throughput rerun without rerunning the whole matrix.
Set `SINGLE_E2E_FRAMEWORKS` and `SINGLE_E2E_CASES` the same way for targeted single-request reruns.

## Dry Run

```bash
diffusion-bench-compare --dry-run --modes single_e2e throughput
```

## Run Selected Cases

```bash
diffusion-bench-compare \
  --config configs/comparison_configs.json \
  --sglang-profile default \
  --case-ids qwen_image_2512_t2i_1024 qwen_image_edit_2511 \
  --frameworks sglang vllm-omni \
  --modes single_e2e throughput \
  --throughput-num-requests 32 \
  --throughput-max-concurrency 4 \
  --run-id "$(date -u +%Y%m%d-%H%M%S)" \
  --port 30200 \
  --output comparison-results.json
```

## Default Coverage

The bundled config is the formal tracking plan and keeps one representative case per official model ID. It currently covers FLUX.1, FLUX.2, Qwen-Image text-to-image and edit, Z-Image image generation, Wan2.1/Wan2.2 video generation, and LTX-2/LTX-2.3 video generation.

Extra variants such as true-CFG, alternate resolutions, non-representative LTX-2.3 pipeline/task variants, or single-GPU vs multi-GPU image runs should be added only for targeted investigations or explicit reruns, not as default tracker coverage. The representative LTX-2.3 default case is the two-stage pipeline.

Framework entries are included only when the local harness has a compatible serving path. Current coverage includes vLLM-Omni for supported image and LTX cases, and LightX2V for supported Wan, FLUX.2, and Z-Image cases.
If a framework needs a repackaged model ID for the same case, the formal report shows the actual per-framework model in the comparison table.

## Command Profiles

Each case can track version-specific best commands for every backend:

```json
"sglang": {
  "serve_args": "--model-type diffusion --warmup",
  "command_profiles": {
    "default": {
      "sglang_ref": "current-main",
      "serve_args": "--model-type diffusion --warmup",
      "notes": "Best known command for this SGLang line."
    },
    "v0.5.0-h100": {
      "sglang_ref": "v0.5.0",
      "hardware": ["h100"],
      "serve_args": "--warmup",
      "notes": "Example: older release where compile was not the best option."
    }
  }
}
```

Select a SGLang profile with `--sglang-profile <name>` or `SGLANG_BENCH_SGLANG_PROFILE=<name>`. Non-SGLang frameworks can use `DIFFUSION_BENCH_<FRAMEWORK>_PROFILE=<name>` or `DIFFUSION_BENCH_FRAMEWORK_PROFILE=<name>`. If no profile is explicit, the runner auto-selects the first profile whose `hardware` matches `--hardware-profile`, `SGLANG_BENCH_HARDWARE_PROFILE`, `GPU_CONFIG`, `RUNNER_LABELS`, or `nvidia-smi`, then falls back to `default`. Use separate profiles when H100/H200 need different GPU counts or launch args; do not reuse a profile that is known to OOM on one hardware class. The runner records the selected profile, profile source, hardware candidates, framework ref, effective `serve_args`, actual server command, and best-effort SGLang runtime metadata in `comparison-results.json`.

Framework entries may set `model_path` when the server needs a local mirror or snapshot path while the report should keep the official `model` ID in the case summary.

## Repo Skills

This repo includes Codex skills that encode the operating discipline for long-term benchmark maintenance:

- `skills/diffusion-bench-maintainer`: maintain runner/config/dashboard semantics without breaking comparability.
- `skills/diffusion-case-onboarding`: add new model/framework cases and command profiles.
- `skills/diffusion-framework-benchmarking`: plan, run, and interpret fair cross-framework diffusion benchmark data.
- `skills/diffusion-regression-investigator`: investigate when SGLang-Diffusion appears slower or a comparison looks unfair.
- `skills/diffusion-performance-reporting`: append data-only comments to the fixed performance tracker issue.

Install or symlink them into your Codex skills directory when you want them auto-discovered.

Formal benchmark reports should append one data-only comment to the fixed tracker issue instead of opening new issues. The canonical tracker for this repo is `mickqian/diffusion-bench-framework#1`.

Run manifests in `manifests/` are the source of truth for reconstructing historical reports. Each manifest records the benchmark commit, SGLang commit, framework install specs or observed commits, hardware profile, run IDs, and known framework failures.

## Generate Dashboard

```bash
diffusion-bench-dashboard \
  --results comparison-results.json \
  --output dashboard.md \
  --charts-dir comparison-charts \
  --report-repo mickqian/diffusion-bench-framework \
  --report-issue 1
```

The issue comment generated by `--report-issue` contains only structured performance data: run metadata, framework versions, GPU model, reproduce entrypoints, one grouped comparison block per case, selected profile, GPU count, latency, throughput p50/p95/p99, QPS, status, ratio-to-SGLang columns, and core sampling parameters. Keep debug analysis in separate investigation notes or dedicated follow-up issues, not in the tracker issue.

## Generate Report Artifacts

```bash
RESULTS="runs/single.json runs/throughput.json" \
OUTPUT_JSON=tmp/report/h200-framework-comparison-merged.json \
scripts/generate_h200_report_artifacts.sh
```

This writes a merged JSON, dashboard Markdown, data-only issue Markdown, and PNG/SVG comparison image from the same input JSONs. The default inputs are the local H200 single-request and throughput report artifacts under `tmp/report`.

## Notes

- No response cache or Cache-DiT is enabled by default.
- The CLI bundles a default config. Pass `--config configs/comparison_configs.json` when editing the repo copy.
- The default config uses one representative case per official model ID, model defaults for omitted sampling params, and overrides only resolution, seed, and framework-specific launch args.
- Command profiles are per case because the best serving command can change by model, framework version, and hardware.
- Hardware-specific profiles are first-class. Formal H100/H200 runs should either pass `--hardware-profile` or rely on detected GPU metadata, and the selected profile must be visible in the result JSON.
- Video models may require large GPUs and framework-specific runtime packages.
