#!/usr/bin/env python3
"""Assemble the cookbook track (configs/benchmark/cookbook/cells/) into
configs/cookbook_configs.json and configs/benchmark/cookbook/COOKBOOK.md.

The headline matrix (configs/benchmark/cases/) runs OUR case shapes with each
framework's best command. That is the comparison we tune, and so the one where
our own setup hides our problems: LTX-2.5 two-stage cannot serve vLLM-Omni's
960x544 single-stage shape, and the MiniMax-H3 warmup cannot reach the 544 short
edge LightX2V and TRT-LLM ship as default -- neither showed in the headline
rows, because no headline case asks for those shapes (2xH200, 2026-10-30).

A cookbook cell takes the workload AND the command from one framework's own
published cookbook/recipe/config, verbatim, and runs every other framework
against that workload. Its rules, enforced here:

  * `cookbook` pins the source (repo, path, commit, commit_date, section) and
    copies the source's workload into `cookbook.workload`.
  * The built case must reproduce that workload; every field that differs (a
    4-GPU recipe on 2 GPUs, 80 frames -> 81, the source timing a pipeline cost
    where the harness times the client) needs a `deviations` entry whose
    `source`/`run` values match what the build sees. Stale entries fail too.
  * `forwards` gives each arm framework's DiT forward count: frameworks read
    "N steps" as N forwards or as N grid points (N-1 forwards).
  * Every arm carries a `tier`:
      verbatim       copied from a published cookbook/recipe/config; needs
                     `upstream_recipe`. The source framework's verbatim arm must
                     cite the cell's source file.
      matched-lossy  turns on the lossy features the source ships with
                     (SageAttention, AdaLN cache, FP8, Cache-DiT); lists them
                     in `lossy`.
      tuned          anything we chose ourselves; needs `notes`. Reported in
                     its own rows, never as the cell's SGLang number.
  * A profile's lightx2v_config / visual_gen_args / extra_env deep-merge over
    the base case's. The build resolves them into each arm and lists what was
    inherited; a verbatim arm fails unless each inherited key is named in its
    `allow_inherited` (harness plumbing such as the MP4 encode alignment).
  * Each cell needs an sglang arm; a cell with no verbatim sglang arm says why
    in `cookbook.sglang_verbatim_gap`.

Cells are built on a headline case (`base_case`) so request plumbing (prompt,
seed, encode alignment) is shared; `set` / `unset` take dotted paths into it.
Only the arm frameworks are kept, and they keep none of the base case's
profiles or serve_args, so a headline command cannot leak into a cookbook arm.

    python3 scripts/build_cookbook_config.py
"""
import copy
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from diffusion_bench.config_guard import (  # noqa: E402
    HARDWARE_TOKENS,
    _merge_nested,
    hardware_tokens,
    upstream_recipe_problems,
)

BENCH = os.path.join(REPO, "configs", "benchmark")
CASES = os.path.join(BENCH, "cases")
CELLS = os.path.join(BENCH, "cookbook", "cells")
OUT = os.path.join(REPO, "configs", "cookbook_configs.json")
OUT_MD = os.path.join(BENCH, "cookbook", "COOKBOOK.md")

TIERS = ("verbatim", "matched-lossy", "tuned")
# What every arm is timed by. A source that times something else (LightX2V's
# offline `RUN pipeline cost`, TRT-LLM's gen_latency) is a deviation.
RUN_METRIC = "client_e2e_median"
# How a cell is measured is part of its workload: a source that warms up with
# one full request and takes the median of three (vLLM-Omni's Qwen-Image-2.1
# recipe) is not the harness's few-step warmup and median of two, so each of
# these must be stated and either match or be a deviation.
REQUIRED_WORKLOAD = ("metric", "warmup", "repeats", "dit_forwards")
_BODY_DROP = {"status", "status_by_hardware", "reason", "command_profiles", "serve_args", "num_gpus"}
# Command configs resolved into each arm in full (see build_cell).
_RESOLVED_CONFIG_KEYS = ("lightx2v_config", "visual_gen_args", "extra_env")
_MISSING = object()


def load(path):
    with open(path) as f:
        return json.load(f)


def get_path(obj, dotted):
    cur = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def set_path(obj, dotted, value):
    keys = dotted.split(".")
    cur = obj
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = cur[key] = {}
        cur = nxt
    cur[keys[-1]] = copy.deepcopy(value)


def unset_path(obj, dotted):
    keys = dotted.split(".")
    parent = get_path(obj, ".".join(keys[:-1])) if len(keys) > 1 else obj
    if not isinstance(parent, dict) or keys[-1] not in parent:
        return False
    del parent[keys[-1]]
    return True


_ABS_PATH_RE = re.compile(r"(?<![\w:])/(?:scratch|data|personal|mnt|home|tmp|opt)\b[^\s\"',]*")


def _machine_paths(prof: dict) -> list[str]:
    """Absolute paths an arm's command reads: things that must exist on the box first."""
    text = json.dumps({k: prof.get(k) for k in ("serve_args", "lightx2v_config", "visual_gen_args", "extra_env")})
    return sorted(set(_ABS_PATH_RE.findall(text)))


def _inherited_leaves(base: dict, arm: dict, prefix: str) -> list[str]:
    """Dotted paths a deep merge takes from `base` because `arm` does not set them."""
    found = []
    for key, value in base.items():
        path = f"{prefix}.{key}"
        if key not in arm:
            found.append(path)
        elif isinstance(value, dict) and isinstance(arm[key], dict):
            found += _inherited_leaves(value, arm[key], path)
    return found


def base_cases():
    found = {}
    for path in glob.glob(os.path.join(CASES, "**", "*.json"), recursive=True):
        if not os.path.basename(path).startswith("_"):
            case = load(path)
            found[case["id"]] = case
    return found


def harness_timing(case: dict, defaults: dict) -> dict:
    """The warmup and repeat count the harness applies to `case`, as workload strings."""
    video = "video" in (case.get("task") or "")
    warm = {**defaults.get("warmup", {}), **((case.get("benchmark") or {}).get("warmup") or {})}
    requests = warm.get("video_num_requests" if video else "num_requests", warm.get("num_requests"))
    repeats = (defaults.get("single") or {}).get("video_repeats" if video else "image_repeats")
    return {
        "metric": RUN_METRIC,
        "warmup": f"{requests} request(s) at {warm.get('num_inference_steps')} steps",
        "repeats": f"median of {repeats}",
    }


def build_cell(cell: dict, bases: dict, in_scope: list[str],
               defaults: dict | None = None) -> tuple[dict | None, list[str]]:
    """(built case, errors) for one cookbook cell."""
    cid = cell.get("id") or "?"
    errors = []

    def err(msg):
        errors.append(f"{cid}: {msg}")

    if not cid.startswith("cb_"):
        err("id must start with `cb_` so a cookbook cell can never shadow a headline case")
    base = bases.get(cell.get("base_case"))
    if base is None:
        err(f"base_case {cell.get('base_case')!r} is not a headline case")
        return None, errors

    src = cell.get("cookbook") or {}
    for problem in upstream_recipe_problems(src):
        err(f"cookbook: {problem.replace('upstream_recipe', 'cookbook')}")
    src_fw = src.get("framework")
    if src_fw not in in_scope:
        err(f"cookbook.framework {src_fw!r} is not an in-scope framework")
    for key in ("source_hardware", "source_metric"):
        if not src.get(key):
            err(f"cookbook.{key} missing")
    # The GPU classes the source was written or measured for ([] = not stated).
    # The build cannot know the box a cell will run on; run_cookbook_cells.sh
    # says so when DBF_HARDWARE_PROFILE is not among these.
    source_gpu = src.get("source_gpu")
    if not isinstance(source_gpu, list) or not all(t in HARDWARE_TOKENS for t in source_gpu):
        err(f"cookbook.source_gpu must be a list of {HARDWARE_TOKENS} tokens ([] if the source names none)")
    workload = src.get("workload")
    if not isinstance(workload, dict) or not workload:
        err("cookbook.workload missing: copy the source's shape, frames, steps and GPU count")
        workload = {}
    for key in REQUIRED_WORKLOAD:
        if key not in workload:
            err(f"cookbook.workload.{key} missing: state the source's {key} (\"not stated\" if it gives none)")

    case = copy.deepcopy(base)
    for dotted, value in (cell.get("set") or {}).items():
        set_path(case, dotted, value)
    for dotted in cell.get("unset") or []:
        if not unset_path(case, dotted):
            err(f"unset {dotted!r}: not present in base case {base['id']}")
    case["id"] = cid
    case["label"] = cell.get("label") or cid
    case.pop("throughput", None)

    arms = cell.get("arms") or {}
    forwards = cell.get("forwards") or {}
    if "sglang" not in arms:
        err("no sglang arm")
    if src_fw and src_fw not in arms:
        err(f"no arm for the source framework {src_fw}")
    for fw in arms:
        if fw not in in_scope:
            err(f"arm framework {fw!r} is not in scope")
        # null = not verified for this framework; reported as such, never compared.
        if fw not in forwards or not (forwards[fw] is None or isinstance(forwards[fw], int)):
            err(f"forwards.{fw} missing: state the DiT forward count this arm runs (null if unverified)")

    base_fws = case.get("frameworks") or {}
    frameworks, statuses, reasons = {}, {}, {}
    arm_index = {}
    for fw, profiles in arms.items():
        if not profiles:
            err(f"{fw}: empty arm set")
            continue
        body = {k: v for k, v in (base_fws.get(fw) or {}).items() if k not in _BODY_DROP}
        body["serve_args"] = ""
        base_configs = {key: body.pop(key) for key in _RESOLVED_CONFIG_KEYS if isinstance(body.get(key), dict)}
        built_profiles = {}
        for name, prof in profiles.items():
            where = f"{fw}[{name}]"
            prof = copy.deepcopy(prof)
            # A profile's config deep-merges over the framework entry's, so a
            # headline key the source never set (Wan's `parallel`) would ride
            # along unseen. Resolve it here and name what was inherited.
            inherited = []
            for key, base_cfg in base_configs.items():
                arm_cfg = prof.get(key)
                if arm_cfg is False:
                    continue
                inherited += _inherited_leaves(base_cfg, arm_cfg or {}, key)
                prof[key] = _merge_nested(base_cfg, arm_cfg or {})
            if inherited:
                prof["inherited"] = inherited
            if fw == "lightx2v" and prof.get("tier") == "verbatim" and "use_compile" not in (
                profiles[name].get("lightx2v_config") or {}
            ):
                # Under DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0 (devbox_run_cases.sh)
                # the harness setdefaults use_compile=True; LightX2V's own default
                # is False (`config.get("use_compile", False)`, ce0f526ae62c).
                err(f"{where}: verbatim lightx2v arm must set lightx2v_config.use_compile (the source's "
                    "value, else LightX2V's default false) or the harness turns compile on")
            allowed = prof.get("allow_inherited") or []
            if prof.get("tier") == "verbatim":
                # A verbatim arm runs the source's config plus only the
                # plumbing it names; anything else from the headline case is a
                # leak (a 4-GPU case's parallel_vae_size reached a 1-GPU arm).
                leaked = [p for p in inherited if not any(p == a or p.startswith(a + ".") for a in allowed)]
                if leaked:
                    err(f"{where}: verbatim arm inherits {leaked} from base case {base['id']} -- `unset` them "
                        "in the cell, or list harness plumbing in allow_inherited")
            if not name.startswith("cb-"):
                err(f"{where}: arm names start with `cb-`")
            if hardware_tokens(name) or prof.get("hardware"):
                err(f"{where}: no hardware in an arm's name or `hardware` -- arms are only ever "
                    "selected explicitly, never by the box they land on")
            tier = prof.get("tier")
            if tier not in TIERS:
                err(f"{where}: tier {tier!r} not in {TIERS}")
            if "serve_args" not in prof:
                err(f"{where}: serve_args missing (write \"\" for none); nothing is inherited")
            if prof.get("num_gpus") != case.get("num_gpus"):
                err(f"{where}: num_gpus {prof.get('num_gpus')!r} != cell num_gpus {case.get('num_gpus')!r}")
            if tier == "verbatim":
                recipe = prof.get("upstream_recipe")
                for problem in upstream_recipe_problems(recipe):
                    err(f"{where}: verbatim arm: {problem}")
                if fw == src_fw and isinstance(recipe, dict) and (
                    recipe.get("repo"), recipe.get("path")) != (src.get("repo"), src.get("path")):
                    err(f"{where}: the source framework's verbatim arm must cite the cell's source "
                        f"{src.get('repo')}:{src.get('path')}")
            elif tier == "matched-lossy" and not prof.get("lossy"):
                err(f"{where}: matched-lossy arm must list its lossy features in `lossy`")
            # A cache sidecar or a patched kernel built on one box by hand is
            # gone with that box (2xH200, 2026-10-30): name how to rebuild it.
            paths = _machine_paths(prof)
            if (paths or tier == "matched-lossy") and not prof.get("requires"):
                err(f"{where}: reads {paths or 'lossy-feature artifacts'} -- list the steps that create them "
                    "in `requires` (e.g. `scripts/prepare_cookbook_h3.sh sgl-adaln`)")
            elif tier == "tuned" and not prof.get("notes"):
                err(f"{where}: tuned arm needs `notes` saying what was chosen and why")
            built_profiles[name] = prof
            arm_index.setdefault(fw, {})[name] = {
                k: prof[k] for k in ("tier", "lossy", "upstream_recipe") if k in prof
            }
        body["command_profiles"] = built_profiles
        frameworks[fw] = body
        if src_fw == fw and not any(p.get("tier") == "verbatim" for p in profiles.values()):
            err(f"{fw}: the source framework needs a verbatim arm")
    if "sglang" in arms and not any(p.get("tier") == "verbatim" for p in arms["sglang"].values()):
        if not src.get("sglang_verbatim_gap"):
            err("no verbatim sglang arm: add one from the sglang cookbook, or say why there is "
                "none in cookbook.sglang_verbatim_gap")
    for fw in in_scope:
        if fw not in frameworks:
            statuses[fw] = "not_run"
            reasons[fw] = "no arm in this cookbook cell"

    # Workload: what the source runs vs what this cell runs.
    actual = {}
    timing = harness_timing(case, defaults or {})
    for key in workload:
        if key == "dit_forwards":
            continue
        if key in timing:
            actual[key] = timing[key]
        else:
            actual[key] = get_path(case, key)
            if actual[key] is _MISSING:
                err(f"cookbook.workload.{key}: no such field in the built case")
    verified = [fw for fw in arms if forwards.get(fw) is not None]
    if "dit_forwards" in workload:
        for fw in verified:
            actual[f"dit_forwards.{fw}"] = forwards[fw]
    expected = {k: v for k, v in workload.items() if k != "dit_forwards"}
    if "dit_forwards" in workload:
        expected.update({f"dit_forwards.{fw}": workload["dit_forwards"] for fw in verified})

    deviations = {d.get("field"): d for d in cell.get("deviations") or []}
    for field, src_value in expected.items():
        run_value = actual.get(field, _MISSING)
        if run_value is _MISSING:
            continue
        dev = deviations.pop(field, None)
        if run_value == src_value:
            if dev is not None:
                err(f"deviation {field!r} is stale: the cell runs the source's value {src_value!r}")
            continue
        if dev is None:
            err(f"{field}: runs {run_value!r} where the source has {src_value!r} -- "
                "match the source or record a deviation")
            continue
        if dev.get("source") != src_value or dev.get("run") != run_value:
            err(f"deviation {field!r} says {dev.get('source')!r} -> {dev.get('run')!r}, "
                f"the build sees {src_value!r} -> {run_value!r}")
        if not dev.get("reason"):
            err(f"deviation {field!r} needs a reason")
    for field in deviations:
        err(f"deviation {field!r} names no field of cookbook.workload")

    case["frameworks"] = frameworks
    case["report_framework_statuses"] = statuses
    case["report_framework_reasons"] = reasons
    # Not `_`-prefixed: run_comparison copies these case fields into each
    # result's `case`, so a result says which cookbook, tier and deviations it
    # was measured under without this file at hand.
    case["track"] = "cookbook"
    case["cookbook"] = {**src, "forwards": forwards, "deviations": cell.get("deviations") or [],
                        "arms": arm_index}
    if cell.get("notes"):
        case["notes"] = cell["notes"]
    return case, errors


def build(cells_dir=CELLS, out=OUT, out_md=OUT_MD) -> int:
    in_scope = list(load(os.path.join(BENCH, "frameworks.json"))["frameworks"])
    meta = load(os.path.join(BENCH, "meta.json"))
    headline = load(os.path.join(REPO, "configs", "comparison_configs.json"))
    bases = base_cases()

    cases, errors = [], []
    for path in sorted(glob.glob(os.path.join(cells_dir, "*.json"))):
        built, cell_errors = build_cell(load(path), bases, in_scope, headline["benchmark_defaults"])
        errors += cell_errors
        if built is not None:
            cases.append(built)
    ids = [c["id"] for c in cases]
    for cid in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"{cid}: duplicate cell id")
    if errors:
        print("COOKBOOK BUILD FAILED:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    result = {**meta, "benchmark_defaults": headline["benchmark_defaults"], "cases": cases}
    with open(out, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        f.write("\n")
    print("wrote", os.path.relpath(out, REPO))
    _write_md(cases, out_md)
    print(f"\nOK: {len(cases)} cookbook cells.")
    return 0


def _cmd(prof: dict) -> str:
    parts = []
    if prof.get("serve_args"):
        parts.append(f"`{prof['serve_args']}`")
    for key in ("lightx2v_config", "visual_gen_args", "extra_env"):
        if prof.get(key):
            parts.append(f"{key} `{json.dumps(prof[key], separators=(',', ':'))}`")
    return "<br>".join(parts) or "(recipe default)"


def _write_md(cases, out_md):
    lines = [
        "# Cookbook cells (auto-generated — do not edit)",
        "",
        "Regenerated by scripts/build_cookbook_config.py from cookbook/cells/. Each cell runs one",
        "framework's published workload; compare a cell's verbatim arms first. `tuned` arms are",
        "ours and never stand in for a framework's cookbook number.",
        "",
    ]
    for c in cases:
        cb = c["cookbook"]
        lines += [
            f"## {c['id']}",
            "",
            f"{c['label']}  ",
            f"Source: **{cb['framework']}** `{cb['repo']}:{cb['path']}` @ `{cb['commit']}` "
            f"({cb['commit_date']}), section \"{cb['section']}\"; written for {cb['source_hardware']} "
            f"(GPU tokens: {', '.join(cb['source_gpu']) or 'none stated'}); "
            f"source metric: {cb['source_metric']}.",
        ]
        if cb.get("sglang_verbatim_gap"):
            lines.append(f"No verbatim sglang arm: {cb['sglang_verbatim_gap']}")
        lines.append("")
        if cb["deviations"]:
            lines += ["| deviation | source | run | reason |", "|---|---|---|---|"]
            for d in cb["deviations"]:
                lines.append(f"| {d['field']} | {d['source']} | {d['run']} | {d['reason']} |")
            lines.append("")
        lines += [
            "| framework | arm | tier | forwards | lossy | command | inherited from base case |",
            "|---|---|---|---|---|---|---|",
        ]
        for fw, body in c["frameworks"].items():
            for name, prof in body["command_profiles"].items():
                lines.append(
                    f"| {fw} | `{name}` | {prof['tier']} | {cb['forwards'].get(fw)} | "
                    f"{', '.join(prof.get('lossy') or [])} | {_cmd(prof)} | "
                    f"{', '.join(prof.get('inherited') or [])} |"
                )
        lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.relpath(out_md, REPO))


if __name__ == "__main__":
    sys.exit(build())
