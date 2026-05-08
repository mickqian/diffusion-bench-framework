---
name: diffusion-case-onboarding
description: Use when adding or updating benchmark cases, models, frameworks, shapes, sampling params, SGLang command profiles, or dependency installers in diffusion-bench-framework. The goal is to onboard comparable test cases for tracking SGLang-Diffusion against open-source diffusion serving frameworks.
---

# Diffusion Case Onboarding

## Goal

Add new benchmark coverage without corrupting historical comparability. A case should be explicit enough to rerun later and conservative enough not to encode accidental machine-specific behavior.

## Case Checklist

- Use the official model ID or a clearly named local path.
- Specify task, prompt, resolution, frames, seed, and reference-image behavior.
- Keep sampling params omitted only when model defaults are intentionally being compared.
- If a README or upstream script uses explicit sampling params, encode them in the case.
- For video cases, always confirm `num_frames` and resolution match across frameworks.
- Do not enable caches unless the case is explicitly a cache benchmark.

## SGLang Profiles

For every `sglang` framework entry, maintain `command_profiles`:

- `sglang_ref`: tag, commit, or meaningful branch line.
- `serve_args`: full best-known SGLang serve args for that case/profile.
- `notes`: why this profile exists, especially when it differs from `default`.
- Optional runtime overrides: `num_gpus`, `extra_env`, and `benchmark`.

Do not delete old profiles just because a newer command is better. Add a new profile when the best command changes by SGLang version, hardware class, or model implementation.

## Framework Checklist

- Add dependency installation only in `scripts/install_comparison_frameworks.sh` when the framework cannot share the main env.
- Keep health checks fail-fast when the server exits.
- Prefer official framework APIs over local patches.
- Store framework-specific settings under that framework entry; avoid global settings that only apply to one backend.

## Validation

At minimum, run a dry run and inspect generated commands. GPU benchmark validation should happen on a remote GPU machine, not on a local Mac.

