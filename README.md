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
- `configs/comparison_configs.json`: editable copy of the default model, prompt, shape, seed, and framework settings.
- `scripts/install_comparison_frameworks.sh`: isolated venv installer for vLLM-Omni and LightX2V.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

SGLang-Diffusion must be installed in the environment if you want to run the `sglang` framework. vLLM-Omni and LightX2V are installed into isolated temporary virtualenvs by the runner because their dependencies conflict with SGLang.

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
  --port 30200 \
  --output comparison-results.json
```

## SGLang Command Profiles

Each case can track version-specific best commands for the `sglang` backend:

```json
"sglang": {
  "serve_args": "--enable-torch-compile --warmup",
  "command_profiles": {
    "default": {
      "sglang_ref": "current-main",
      "serve_args": "--enable-torch-compile --warmup",
      "notes": "Best known command for this SGLang line."
    },
    "v0.5.0-h100": {
      "sglang_ref": "v0.5.0",
      "serve_args": "--warmup",
      "notes": "Example: older release where compile was not the best option."
    }
  }
}
```

Select a profile with `--sglang-profile <name>` or `SGLANG_BENCH_SGLANG_PROFILE=<name>`. The runner records the selected profile, `sglang_ref`, effective `serve_args`, actual server command, and best-effort SGLang runtime metadata in `comparison-results.json`.

## Generate Dashboard

```bash
diffusion-bench-dashboard \
  --results comparison-results.json \
  --output dashboard.md \
  --charts-dir comparison-charts
```

## Notes

- No response cache or Cache-DiT is enabled by default.
- The CLI bundles a default config. Pass `--config configs/comparison_configs.json` when editing the repo copy.
- The default config uses model defaults for omitted sampling params and overrides only resolution, seed, and framework-specific launch args.
- SGLang command profiles are per case because the best serving command can change by model and SGLang version.
- Video models may require large GPUs and framework-specific runtime packages.
