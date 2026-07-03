"""Guards that keep benchmark commands, configs, and published numbers aligned.

Born from a real incident set: a published run silently used commands that no
longer matched the declared policy/config, hardware-profile first-match was
patched on the wrong profile twice, and improvised serve_args produced numbers
the config never described. Every guard here turns one of those mistakes into
a machine check.

Selection semantics MIRROR ``run_comparison._select_command_profile`` (the
runtime authority): a profile matches a hardware candidate when the candidate
is a substring of the profile NAME or of any value in its ``hardware`` field;
the first matching non-default profile in dict order wins, else ``default``.
If you change the runtime selection, change this mirror in the same commit.
"""

from __future__ import annotations

DEFAULT_PROFILE = "default"


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
    values = [name.lower(), *profile_hardware_values(profile_cfg)]
    return any(candidate in value for value in values)


def select_profile(profiles: dict, hardware: str) -> tuple[str | None, dict | None, list[str]]:
    """Return (selected_name, selected_cfg, all_hardware_matches).

    ``all_hardware_matches`` lists every non-default profile that matched —
    more than one means first-match order is deciding, which is exactly the
    footgun where someone edits a matching-but-unselected profile.
    """
    candidate = hardware.lower()
    matches = [
        name
        for name, cfg in profiles.items()
        if name != DEFAULT_PROFILE and profile_matches_hardware(name, cfg, candidate)
    ]
    if matches:
        return matches[0], profiles[matches[0]], matches
    if DEFAULT_PROFILE in profiles:
        return DEFAULT_PROFILE, profiles[DEFAULT_PROFILE], matches
    return None, None, matches


def serve_args_missing_tokens(config_serve_args: str, server_command: str) -> list[str]:
    """Tokens of the config's serve_args absent from the actually-run command."""
    have = server_command.split()
    return [tok for tok in config_serve_args.split() if tok not in have]


def verify_merged_commands(
    merged: dict, config: dict, hardware: str = "h100", framework: str = "sglang"
) -> list[str]:
    """Cross-check every published row against the CURRENT config selection.

    Returns human-readable warnings for rows whose recorded profile is not the
    profile the config selects today, or whose recorded server command lacks
    tokens of the selected serve_args — i.e. the published number no longer
    describes what the config would run.
    """
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
            selected_name, selected_cfg, _ = select_profile(profiles, hardware)
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
                    f"{selected_name!r} on {hardware} — re-run before publishing"
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
