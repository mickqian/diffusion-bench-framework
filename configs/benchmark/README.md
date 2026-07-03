# Benchmark configuration (source of truth)

This directory is the **explicit source of truth** for *what the cross-framework
benchmark runs*: which frameworks, which models/cases, which workloads, and each
framework's best-known command per case/hardware. The harness does **not** read
this directory directly — it reads the generated `configs/comparison_configs.json`.

Edit files here, then regenerate:

```bash
python3 scripts/build_benchmark_config.py
```

The build assembles everything into `configs/comparison_configs.json` **and** the
packaged copy `src/diffusion_bench/comparison_configs.json`, emits `MATRIX.md`,
and **fails if any case does not classify all four in-scope frameworks** — so a
framework (e.g. `trtllm-visual` on video) can never be silently dropped again.

## Layout

```
configs/benchmark/
├── README.md            # this file
├── MATRIX.md            # auto-generated coverage grid (case × framework)
├── frameworks.json      # the 4 in-scope frameworks + version policy + launch style
├── workloads.json       # single_e2e / throughput / warmup → benchmark_defaults
├── meta.json            # top-level fields (_comment, test_image_url)
└── cases/
    ├── _order.json      # order cases appear in the built config
    ├── image/<id>.json  # one file per image case
    └── video/<id>.json  # one file per video case
```

## Case file schema

Every case file lists **all four frameworks explicitly**. Each framework carries a
`status`; a `supported` framework also carries its best-known `command_profiles`.

```jsonc
{
  "id": "...", "model": "...", "task": "text-to-image",
  "width": 1024, "height": 1024, "seed": 42, "num_gpus": 2,
  "num_inference_steps": 50, "guidance_scale": 1.0,
  "frameworks": {
    "sglang":        { "status": "supported", "serve_args": "...", "command_profiles": { ... } },
    "vllm-omni":     { "status": "supported", "command_profiles": { ... } },
    "lightx2v":      { "status": "unsupported", "reason": "no serving path in tracked version" },
    "trtllm-visual": { "status": "not_run", "reason": "supported per coverage; profile+validation pending" }
  }
}
```

Statuses: `supported` (has a validated command), `unsupported` (no serving path),
`no_profile` (framework could run it but no command settled), `failed` (ran but
errored/OOM — keep the reason), `not_run` (in scope, not yet run), `invalid`
(comparison would not be apples-to-apples).

## Command-profile rule

A profile's `serve_args` must be a **specific command validated on the target
hardware** (ran, no OOM, fastest tried) — never hand-assembled flags. When the best
command is unknown, derive from `--performance-mode speed` (may OOM), validate on
the target GPU, then record. Cite the command's source + validated hardware in the
profile `notes`. (See the `diffusion-case-onboarding` and
`diffusion-framework-benchmarking` skills for the fairness/lossless methodology.)
