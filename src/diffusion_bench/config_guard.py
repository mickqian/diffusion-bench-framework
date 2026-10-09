"""Guards that keep benchmark commands, configs, and published numbers aligned.

Born from a real incident set: a published run silently used commands that no
longer matched the declared policy/config, hardware-profile first-match was
patched on the wrong profile twice, and improvised serve_args produced numbers
the config never described. Every guard here turns one of those mistakes into
a machine check.

Selection semantics MIRROR ``run_comparison._select_command_profile`` (the
runtime authority): a profile matches a hardware candidate when the candidate
is one of the hardware tokens in the profile NAME or in any value of its
``hardware`` field; the first matching non-default profile in dict order wins,
else ``default``, else the first profile.
If you change the runtime selection, change this mirror in the same commit.
"""

from __future__ import annotations

import re

DEFAULT_PROFILE = "default"


# MIRRORS ``run_comparison._hardware_profile_candidates``. The runtime does not
# match a profile against the --hardware-profile string directly: it joins the
# override, the env var, gpu_config, runner_labels and the GPU names into one
# blob and looks for known hardware tokens in it. So `--hardware-profile
# blackwell` on a box of B200s yields the candidate ``b200`` -- "blackwell" is
# not a token and contributes nothing by itself.
#
# Keeping only the override string here (as this module used to) made the drift
# check compare against a different profile than the one that ran: a case whose
# Blackwell profile is named `b200-2gpu` matched at runtime and not in the
# mirror, so every correct row was reported as drift. If you change the runtime
# list, change this one in the same commit.
HARDWARE_TOKENS = (
    "gb300", "gb200", "b300", "b200", "h200", "h100", "a100",
    "l40", "l4", "rtx5090", "rtx4090", "rtx3090",
)
# One left-to-right scan in which a token claims its span; the tuple lists each
# token before any token it contains (`gb300` before `b300`, `l40` before `l4`).
# A substring test found both, so a GB300 box also derived `b300` and ran
# B300-only profiles, and a B300 box matched profiles named `gb300-*`. A profile
# now applies to a class only if it names that class.
_HARDWARE_TOKEN_RE = re.compile("|".join(t.replace("rtx", "rtx ?") for t in HARDWARE_TOKENS))


def hardware_tokens(text: str) -> set[str]:
    return {m.group(0).replace(" ", "") for m in _HARDWARE_TOKEN_RE.finditer(text.lower())}


def hardware_candidates(hardware_metadata: dict | None, override: str | None = None) -> list[str]:
    """Hardware tokens the runtime would derive, in the runtime's own order."""
    metadata = hardware_metadata or {}
    values = [
        override,
        metadata.get("hardware_profile_override"),
        metadata.get("gpu_config"),
        metadata.get("runner_labels"),
        *(metadata.get("gpus") or []),
    ]
    found = hardware_tokens(" ".join(str(value) for value in values if value))
    return [token for token in HARDWARE_TOKENS if token in found]


def profile_hardware_values(profile_cfg: dict) -> list[str]:
    hardware = (
        profile_cfg.get("hardware")
        or profile_cfg.get("hardware_profile")
        or profile_cfg.get("hardware_profiles")
    )
    if not hardware:
        return []
    if isinstance(hardware, str):
        return [hardware.lower()]
    return [str(v).lower() for v in hardware]


def profile_matches_hardware(name: str, profile_cfg: dict, candidate: str) -> bool:
    values = [name, *profile_hardware_values(profile_cfg)]
    return any(candidate in hardware_tokens(value) for value in values)


def select_profile(
    profiles: dict, hardware: str | list[str]
) -> tuple[str | None, dict | None, list[str]]:
    """Return (selected_name, selected_cfg, all_hardware_matches).

    ``all_hardware_matches`` lists every non-default profile that matched —
    more than one means first-match order is deciding, which is exactly the
    footgun where someone edits a matching-but-unselected profile.
    """
    candidates = [hardware] if isinstance(hardware, str) else list(hardware)
    candidates = [c.lower() for c in candidates if c]
    matches = [
        name
        for name, cfg in profiles.items()
        if name != DEFAULT_PROFILE
        and any(profile_matches_hardware(name, cfg, c) for c in candidates)
    ]
    if matches:
        return matches[0], profiles[matches[0]], matches
    if DEFAULT_PROFILE in profiles:
        return DEFAULT_PROFILE, profiles[DEFAULT_PROFILE], matches
    if profiles:
        first = next(iter(profiles))
        return first, profiles[first], matches
    return None, None, matches


# MIRRORS run_comparison.FRAMEWORK_PROFILE_RUNTIME_KEYS and _merge_nested: the
# keys a selected profile overrides in its framework entry, dicts merged deeply.
PROFILE_RUNTIME_KEYS = frozenset({
    "serve_args", "num_gpus", "extra_env", "benchmark", "model", "model_path",
    "lightx2v_config", "server_bin", "use_omni_arg", "required_help_args",
    "http_server", "http_request", "comfyui",
})


def _merge_nested(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolved_config(entry: dict, profile: dict | None) -> dict:
    """The framework config the harness runs: `entry` overlaid by the selected profile."""
    base = {key: value for key, value in entry.items() if key != "command_profiles"}
    runtime = {key: value for key, value in (profile or {}).items() if key in PROFILE_RUNTIME_KEYS}
    return _merge_nested(base, runtime)


UPSTREAM_RECIPE_KEYS = ("repo", "path", "commit", "commit_date", "section")


def upstream_recipe_problems(recipe) -> list[str]:
    """Why `recipe` cannot be checked for drift by scripts/check_competitor_recipes.py."""
    if not isinstance(recipe, dict):
        return ["upstream_recipe must be an object"]
    missing = [key for key in UPSTREAM_RECIPE_KEYS if not (isinstance(recipe.get(key), str) and recipe[key].strip())]
    problems = [f"upstream_recipe.{key} missing" for key in missing]
    if "commit" not in missing and not re.fullmatch(r"[0-9a-f]{7,40}", recipe["commit"]):
        problems.append("upstream_recipe.commit must be a 7-40 character hex sha")
    if "commit_date" not in missing and not re.fullmatch(r"20\d\d-\d\d-\d\d", recipe["commit_date"]):
        problems.append("upstream_recipe.commit_date must be YYYY-MM-DD")
    return problems


def serve_args_missing_tokens(config_serve_args: str, server_command: str) -> list[str]:
    """Tokens of the config's serve_args absent from the actually-run command."""
    have = server_command.split()
    return [tok for tok in config_serve_args.split() if tok not in have]


def verify_merged_commands(
    merged: dict,
    config: dict,
    hardware: str | list[str] | None = None,
    framework: str = "sglang",
) -> list[str]:
    """Cross-check every published row against the CURRENT config selection.

    Returns human-readable warnings for rows whose recorded profile is not the
    profile the config selects today, or whose recorded server command lacks
    tokens of the selected serve_args — i.e. the published number no longer
    describes what the config would run.
    """
    # Derive the same candidates the runtime did, from the run's own recorded
    # hardware, so the check compares against the profile that actually ran.
    if isinstance(hardware, str) or hardware is None:
        candidates = hardware_candidates(merged.get("hardware"), override=hardware)
        if not candidates:
            candidates = [hardware or "h100"]
    else:
        candidates = list(hardware)

    cases = {c.get("id"): c for c in config.get("cases", [])}
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for key in ("results", "throughput_results"):
        for row in merged.get(key, []) or []:
            if row.get("framework") != framework or row.get("error"):
                continue
            case_id = row.get("case_id")
            case = cases.get(case_id)
            if not case:
                continue
            fw_cfg = (case.get("frameworks") or {}).get(framework) or {}
            profiles = fw_cfg.get("command_profiles") or {}
            if not profiles:
                continue
            selected_name, selected_cfg, _ = select_profile(profiles, candidates)
            if selected_cfg is None:
                continue
            row_profile = (row.get("framework_metadata") or {}).get("profile")
            dedup = (case_id, str(row_profile))
            if dedup in seen:
                continue
            seen.add(dedup)
            if row_profile and row_profile != selected_name:
                warnings.append(
                    f"{case_id}/{framework}: published row ran profile "
                    f"{row_profile!r} but the config now selects "
                    f"{selected_name!r} on {'/'.join(candidates)} — re-run before publishing"
                )
                continue
            missing = serve_args_missing_tokens(
                selected_cfg.get("serve_args", ""), row.get("server_command") or ""
            )
            if missing:
                warnings.append(
                    f"{case_id}/{framework}: published row's command lacks "
                    f"config tokens {missing} (profile {selected_name!r}) — "
                    f"config changed after the run; re-run before publishing"
                )
    return warnings
