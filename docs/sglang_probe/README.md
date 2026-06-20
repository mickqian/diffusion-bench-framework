# SGLang-Diffusion Stability Probe

This suite is for periodic robustness experiments, not the formal cross-framework latency comparison. It covers SGLang-Diffusion server mode, CLI generation, Python `DiffGenerator` APIs, LoRA flows, OpenAI-compatible HTTP APIs, invalid requests, and stress runs.

## Commands

Generate a reproducible matrix:

```bash
diffusion-bench-sglang-probe catalog \
  --repo /path/to/sglang \
  --profile release_full \
  --out-dir /tmp/sglang-probe-release
```

List planned cells without running them:

```bash
diffusion-bench-sglang-probe list \
  --repo /path/to/sglang \
  --profile daily_smoke \
  --out-dir /tmp/sglang-probe-daily
```

Run a bounded smoke profile:

```bash
diffusion-bench-sglang-probe run \
  --repo /path/to/sglang \
  --profile daily_smoke \
  --out-dir /scratch/sglang-probe-daily \
  --timeout-s 3600
```

Run one cell from an existing matrix:

```bash
diffusion-bench-sglang-probe run \
  --repo /path/to/sglang \
  --matrix /scratch/sglang-probe-release/matrix.json \
  --cell-id smoke_zimage_t2i_serve_raw_http_image_generation_baseline_baseline_smoke \
  --out-dir /scratch/sglang-probe-rerun \
  --timeout-s 3600
```

Summarize a run directory:

```bash
diffusion-bench-sglang-probe summarize --out-dir /scratch/sglang-probe-daily
```

## Profiles

- `daily_smoke`: cheap family/entrypoint smoke coverage.
- `weekly_pairwise`: smoke + bounded pairwise + targeted risky triples.
- `release_full`: full matrix used before release or large refactors.
- `stress`: long-running pressure cells only.

## Output Contract

Each run writes:

- `matrix.json`: generated parameter catalog and phase cells.
- `runs.jsonl`: append-only event stream.
- `runs/<cell_id>/cell.json`: selected cell config.
- `runs/<cell_id>/server.log` or `command.log`: launch/generation logs.
- `failures/*.json`: minimal failed request payloads.
- `summary.json`: aggregate failed cells, failed requests, and latency percentiles.

Secrets, cloud credentials, HF tokens, and machine allocation details are intentionally external. Pass them through the shell environment or your devbox bootstrap; do not commit them into profiles.
