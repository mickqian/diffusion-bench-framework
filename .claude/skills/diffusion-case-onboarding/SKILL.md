---
name: diffusion-case-onboarding
description: Use when adding or updating benchmark cases, models, frameworks, shapes, sampling params, SGLang command profiles, or dependency installers in diffusion-bench-framework, or when maintaining the runner/config/result-schema/dashboard code without breaking comparability. The goal is to onboard comparable test cases and keep the harness reproducible for tracking SGLang-Diffusion against open-source diffusion serving frameworks.
---

# Diffusion Case Onboarding

## Goal

Add new benchmark coverage without corrupting historical comparability. A case should be explicit enough to rerun later and conservative enough not to encode accidental machine-specific behavior.

## Case Checklist

- Use the official model ID or a clearly named local path.
- The default tracker should include only one representative case per official model ID. Keep LTX-2.3 represented by a two-stage case; put true-CFG, alternate-shape, single-vs-multi-GPU, or non-representative pipeline variants behind explicit `--case-ids` reruns unless broader default coverage is requested.
- Specify task, prompt, resolution, frames, seed, and reference-image behavior.
- Keep sampling params omitted only when model defaults are intentionally being compared.
- If a README or upstream script uses explicit sampling params, encode them in the case.
- For video cases, always confirm `num_frames` and resolution match across frameworks.
- Do not enable caches unless the case is explicitly a cache benchmark.
- Use each framework's fastest **lossless** serving command; see the `diffusion-framework-benchmarking` SKILL for the lossless/lossy split. ON (lossless, use them): torch.compile, kernel fusion, each framework's fastest exact attention, TP/CFG/sequence parallelism, GPU-resident execution. OFF unless the case explicitly compares that semantic (lossy): response/Cache-DiT caches, quantized checkpoints, distilled/reduced-step, no-CFG variants, int8/approximate attention.

## Command Profiles

For every framework entry, maintain `command_profiles`:

- `sglang_ref` or `framework_ref`: tag, commit, package, config, or meaningful branch line.
- `serve_args`: the specific command that was **empirically validated on the target hardware** — it actually ran (no OOM) and was the fastest among the settings tried — then recorded here. Never hand-assemble flags you have not run. When the best command is unknown, derive a starting point from `--performance-mode speed` (+ `--enable-torch-compile` for the compiled DiT), but `speed` favors residency and **can OOM**, so validate it on the target GPU first; if it OOMs, fall back to a setting that fits and record that. Prefer a known-good concrete command (cookbook / `ci_perf` / model repro) over deriving from scratch.
- `hardware`: optional hardware selector such as `["h100"]` or `["h200"]` when a command is hardware-specific.
- `notes`: why this profile exists (especially when it differs from `default`), the command's source (cookbook / `ci_perf` / model repro), and the hardware it was validated on.
- Optional runtime overrides: `num_gpus`, `extra_env`, and `benchmark`.

Do not delete old profiles just because a newer command is better. Add a new profile when the best command changes by framework version, hardware class, or model implementation.
If a SGLang profile OOMs or fails on one hardware class, add a hardware-specific stable profile and rerun instead of letting the failed result stand as a valid comparison.

**Gotcha — profile selection picks the FIRST hardware match, not `default`.** `--hardware-profile <hw>` selects the first profile whose `hardware` list contains `<hw>` (falling back to `default` only if none match). So: (1) editing `default` is a no-op when a hardware-specific profile matches — patch the profile that is actually selected; (2) a stale or broken hardware profile silently *shadows* a newer good one for the same hardware (bit us: a known-OOM `h100-1gpu` shadowed a working `h100-2gpu-tp`; a `--extra_visual_gen_options` added only to `default` never ran because `h100-h200-2gpu-tp-speed` was selected). When adding/patching, dump the resolved profile for the target hardware and confirm it is the one you changed.

## Framework Checklist

- Add dependency installation only in `scripts/install_comparison_frameworks.sh` when the framework cannot share the main env.
- Keep health checks fail-fast when the server exits.
- Prefer official framework APIs over local patches.
- Store framework-specific settings under that framework entry; avoid global settings that only apply to one backend.

## Validation

At minimum, run a dry run and inspect generated commands. GPU benchmark validation should happen on a remote GPU machine, not on a local Mac.

## Repo Maintenance

When changing runner/config/result-schema (not just adding a case), comparability comes before convenience:

- inspect `configs/comparison_configs.json`, the packaged `src/diffusion_bench/comparison_configs.json`, runner code, README, and recent results before editing
- changing config semantics: update BOTH the editable and packaged config
- changing result JSON shape: update dashboard/reporting code and README examples in the same change
- backend command changes: add/update a per-case `command_profiles` entry, never overwrite historical intent
- runner changes: preserve fail-fast behavior; keep server logs and bench JSON paths discoverable
- after edits: inspect diffs and run only lightweight static checks unless a real benchmark is requested
