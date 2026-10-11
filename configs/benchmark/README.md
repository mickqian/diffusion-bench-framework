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
and **fails if any case does not classify every in-scope framework** — so a
framework (e.g. `trtllm-visual` on video) can never be silently dropped again.

## Layout

```
configs/benchmark/
├── README.md            # this file
├── MATRIX.md            # auto-generated coverage grid (case × framework)
├── frameworks.json      # the in-scope frameworks + version policy + launch style
├── workloads.json       # single_e2e / throughput / warmup → benchmark_defaults
├── meta.json            # top-level fields (_comment, test_image_url)
└── cases/
    ├── _order.json      # order cases appear in the built config
    ├── image/<id>.json  # one file per image case
    └── video/<id>.json  # one file per video case
```

## Case file schema

Every case file lists **every framework in frameworks.json explicitly**. Each framework carries a
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

## Per-hardware status

`status` is one global verdict, which conflates "this framework cannot run this
case" with "it could not run it on the box we tried". When only the second is
true, scope it with `status_by_hardware` — the entry for the first matching
hardware wins, and `status` applies everywhere else:

```jsonc
"lightx2v": {
  "status": "supported",
  "status_by_hardware": {
    "h100": { "status": "failed", "reason": "OOM on H100 80GB: ..." }
  },
  "command_profiles": { ... }
}
```

The command body is published as soon as *some* hardware is `supported`;
`config_guard.apply_hardware_statuses` then resolves it per box — the runner
never attempts a cell classified away on that hardware, and the report still
shows the classification with its reason rather than a bare `not_run`. Both
directions work: a `supported` override also promotes a cell whose baseline is
`unsupported`/`no_profile`.

Keys are the hardware tokens the RUNTIME derives from the GPU names
(`config_guard.HARDWARE_TOKENS`) — the same vocabulary `command_profiles[*].hardware`
matches against, so `h100`/`h200`/`b200`, never `hopper` or `80gb`. The build
rejects an unknown key, an invalid status, and a non-`supported` verdict with no
`reason`. Origin: LTX-2.3 two-stage marked lightx2v `failed` for an 80GB OOM, and
the global status then skipped the cell on a 179GB-per-GPU B200 too.

## Command-profile rule

A profile's `serve_args` must be a **specific command validated on the target
hardware** (ran, no OOM, fastest tried) — never hand-assembled flags. When the best
command is unknown, derive from `--performance-mode speed` (may OOM), validate on
the target GPU, then record. Cite the command's source + validated hardware in the
profile `notes`. (See the `diffusion-case-onboarding` and
`diffusion-framework-benchmarking` skills for the fairness/lossless methodology.)
