#!/usr/bin/env python3
"""Assemble configs/benchmark/ (the explicit source of truth) into the
harness-consumed configs/comparison_configs.json and the packaged copy under
src/diffusion_bench/. Validates that EVERY case classifies ALL in-scope
frameworks, so a framework can never be silently dropped.

Source of truth (edit these):
  configs/benchmark/frameworks.json   frameworks in scope + version policy
  configs/benchmark/workloads.json    single_e2e / throughput / warmup -> benchmark_defaults
  configs/benchmark/meta.json         top-level (_comment, test_image_url)
  configs/benchmark/cases/<image|video>/<id>.json   one file per case, every framework explicit
  configs/benchmark/cases/_order.json order the cases appear in the built config

Regenerate after editing:  python3 scripts/build_benchmark_config.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from diffusion_bench import comfyui_client, fastvideo_client  # noqa: E402
from diffusion_bench.config_guard import (  # noqa: E402
    hardware_candidates,
    resolved_config,
    select_profile,
    upstream_recipe_problems,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "configs", "benchmark")
CASES = os.path.join(BENCH, "cases")
COMFYUI_WORKFLOWS = os.path.join(BENCH, "comfyui")
OUT_EDITABLE = os.path.join(REPO, "configs", "comparison_configs.json")
OUT_PACKAGED = os.path.join(REPO, "src", "diffusion_bench", "comparison_configs.json")

VALID_STATUS = {"supported", "unsupported", "no_profile", "failed", "not_run", "invalid"}


# The benchmark's published hardware. The lint checks the profile that the
# harness would actually SELECT there (first hardware match, else `default`)
# — patching an unselected profile is a recurring footgun. Every class we
# publish belongs here: the policy was linted on h100 only while a B200 run was
# being prepared, so a blackwell profile could have enabled compile unnoticed.
# Use the tokens the RUNTIME derives, not the string passed to
# --hardware-profile: it scans the GPU names for known tokens, so a B200 box
# selects by "b200". Linting "blackwell" only checked profiles named
# blackwell-*, and silently skipped cases whose Blackwell profile is named
# b200-* (minimax-h3), leaving them outside the compile policy. Each class is
# resolved through config_guard.hardware_candidates, as the runtime resolves a
# box's GPU names, so a GB300 box selects exactly what the lint checked.
DATACENTER_HARDWARE = ("h100", "h200", "b200", "b300", "gb200", "gb300")
POLICY_HARDWARE = DATACENTER_HARDWARE + ("rtx5090", "rtx4090")
# "Best lossless" policy: the selected sglang profile must run resident, and
# must NOT enable torch.compile — sglang's explicitly-fused kernels now match or
# beat compiler fusion on most diffusion models, so compile-on is the slower
# path as well as a long autotune before every measurement. A profile may opt
# out ONLY with a `policy_exception` string carrying measured evidence.
# The same switch has three spellings in this CLI and the guard only knew one
# of them. `--component-residency dit=layerwise-offload` and
# `--layerwise-offload-components all` are what the Qwen-Image-2.1 cookbook
# emits for consumer cards, and both slipped past a pattern that looked only for
# the dedicated `--*-offload` flags.
_OFFLOAD_ENABLE_RE = re.compile(
    r"--(?:text-encoder|image-encoder|vae|dit)-cpu-offload(?!\s+false)"
    r"|--dit-layerwise-offload(?!\s+false)"
    r"|--component-residency(?:\s+[\w.-]+=[\w-]+)*?\s+[\w.-]+=(?:layerwise-offload|cpu-offload)"
    r"|--layerwise-offload-components(?!\s+none)"
)

# The competitors get the same rule on datacenter cards (on the rtx classes
# offload is the capacity path): the SELECTED profile runs compiled and
# resident, or carries a dated `policy_exception`. MiniMax-H3's vLLM-Omni cell
# ran the recipe's 2x24/32 GB consumer command (TP2 + distributed layerwise
# offload + eager) on 183 GB B200s for seven weeks because nothing checked.
# `--dlo-*` only tunes distributed layerwise offload, so any of it means offload.
_VLLM_EAGER_RE = re.compile(
    r"--enforce-eager(?![\w-])"
    r"|--compilation-config[\s=]+\S*\"(?:mode|level)\"\s*:\s*0(?!\d)"
    r"|\"enforce_eager\"\s*:\s*true"
)
_VLLM_OFFLOAD_RE = re.compile(
    r"--enable-[\w-]*offload(?![\w-])|--diffusion-offload-config(?![\w-])|--dlo-[\w-]+"
)
# The harness strips none of these (compile-on only stops it ADDING
# --enforce-eager), and a stage/deploy YAML hides them from serve_args: the old
# cosmos3 stage config pinned enforce_eager and ran that cell compile-off.
_VLLM_CONFIG_FILE_RE = re.compile(r"--(?:stage-configs-path|deploy-config)[\s=]+(\S+)")
_VLLM_YAML_EAGER_RE = re.compile(r"^\s*enforce_eager\s*:\s*true\b", re.M | re.I)
_VLLM_YAML_OFFLOAD_RE = re.compile(
    r"^\s*enable_\w*offload\s*:\s*true\b|^([ \t]*)diffusion_offload_config\s*:\s*(?:\{|\n\1[ \t]+\w)",
    re.M | re.I,
)
_COMFYUI_OFFLOAD_RE = re.compile(r"--(?:lowvram|novram|cpu|cpu-vae|disable-smart-memory)(?![\w-])")
# LightX2V reads offload_granularity only when cpu_offload is on, and each
# `<component>_cpu_offload` defaults to cpu_offload, so the switches are those
# keys plus these two, which stream or release weights between requests.
_LIGHTX2V_OFFLOAD_KEYS = ("lazy_load", "unload_modules")

# Competitor cells that violated the policy when the lint landed (2026-09-30)
# with no dated evidence to cite. They print as errors; any other violation
# fails the build, and so do these under DIFFUSION_BENCH_STRICT_COMPETITOR_POLICY=1.
# Delete an entry when its cell is re-derived or justified: the list only shrinks.
KNOWN_COMPETITOR_VIOLATIONS = {
    ("cosmos3_nano_t2i_720p", "vllm-omni", "default"),
    ("cosmos3_nano_t2i_720p", "vllm-omni", "h200-2gpu-cfg"),
    ("ltx2_twostage_t2v", "lightx2v", "h100-1gpu"),
}
STRICT_COMPETITOR_POLICY_ENV = "DIFFUSION_BENCH_STRICT_COMPETITOR_POLICY"


SELECTED_ROWS = []  # (case, fw, hw, profile, args, exception, matches, recipe, config)
MISSING_RECIPES = {}  # (case, fw, profile) -> datacenter classes that select it


def _lint_serve_args_shape(cid: str, fw: str, name: str, args: str) -> list[str]:
    """Catch malformed serve_args before a run does.

    A bulk edit once produced `--enable-torch-compile false false`; the config
    built fine, the lint passed, and the breakage only surfaced hours later as
    "unrecognized arguments: false" on the flagship FLUX cases in a full matrix.
    """
    errs = []
    toks = args.split()
    for i in range(len(toks) - 2):
        if toks[i].startswith("--") and toks[i + 1] == toks[i + 2] and not toks[i + 1].startswith("--"):
            errs.append(
                f"{cid}/{fw}[{name}]: repeated value for {toks[i]} "
                f"({toks[i + 1]!r} twice) — malformed serve_args"
            )
    return errs


def _lint_model_override(cid: str, case_model, fw: str, body: dict) -> list[str]:
    """A framework running different weights needs a stated reason.

    ltx2 handed vLLM-Omni `rootonchair/LTX-2-19b-distilled` while sglang and
    LightX2V ran the full `Lightricks/LTX-2` -- a distilled checkpoint is
    explicitly lossy and would have made that cell unfairly fast. It came in
    with a bulk config migration and no rationale, and survived because nothing
    checked. Legitimate overrides exist (LightX2V reads Wan's original layout
    where the others read the Diffusers conversion), so the rule is a stated
    reason, not a ban.
    """
    errs = []
    seen = {body.get("model")} | {
        (prof or {}).get("model") for prof in (body.get("command_profiles") or {}).values()
    }
    for model in sorted(m for m in seen if m and m != case_model):
        if not str(body.get("model_override_reason") or "").strip():
            errs.append(
                f"{cid}/{fw}: runs {model!r} instead of the case's {case_model!r} "
                f"with no `model_override_reason` — state why the weights are "
                f"equivalent, or use the case's model"
            )
    return errs


def _recipe_label(source: dict) -> str:
    recipe = source.get("upstream_recipe")
    if recipe:
        return f"{recipe.get('repo')}@{recipe.get('commit')} ({recipe.get('commit_date')})"
    reason = str(source.get("upstream_recipe_reason") or "").strip()
    return f"none: {reason}" if reason else ""


def _config_label(fw: str, cfg: dict) -> str:
    """What decides a competitor's command besides serve_args."""
    parts = [f"{key}={value}" for key, value in sorted((cfg.get("extra_env") or {}).items())]
    if fw == "lightx2v":
        parts.append(json.dumps(cfg.get("lightx2v_config") or {}, separators=(",", ":"), ensure_ascii=False))
    if fw == "comfyui":
        spec = cfg.get("comfyui") or {}
        parts.append(f"workflow={spec.get('workflow')} compile={json.dumps(bool(spec.get('compile')))}")
    return " ".join(parts)


def _competitor_findings(fw: str, cfg: dict) -> list[str]:
    """Every switch in the resolved `cfg` that turns compile off or moves weights off the GPU."""
    args = cfg.get("serve_args") or ""
    found = []
    if fw == "vllm-omni":
        found += [m.group(0) for m in _VLLM_EAGER_RE.finditer(args)]
        found += [m.group(0) for m in _VLLM_OFFLOAD_RE.finditer(args)]
        for rel in _VLLM_CONFIG_FILE_RE.findall(args):
            path = os.path.join(REPO, rel)
            if not os.path.isfile(path):
                found.append(f"{rel} (unreadable, so what it enables is unknown)")
                continue
            text = open(path).read()
            for pattern in (_VLLM_YAML_EAGER_RE, _VLLM_YAML_OFFLOAD_RE):
                m = pattern.search(text)
                if m:
                    found.append(f"{m.group(0).strip().splitlines()[0]} in {rel}")
    elif fw == "lightx2v":
        lcfg = cfg.get("lightx2v_config") or {}
        found += [
            f"lightx2v_config.{key}={json.dumps(value)}"
            for key, value in lcfg.items()
            if value and (key == "cpu_offload" or key.endswith("_cpu_offload") or key in _LIGHTX2V_OFFLOAD_KEYS)
        ]
        # The key LightX2V reads; upstream's MiniMax-H3 presets ship "use_compile": false.
        if "use_compile" in lcfg and not lcfg["use_compile"]:
            found.append("lightx2v_config.use_compile=false")
    elif fw == "comfyui":
        found += [m.group(0) for m in _COMFYUI_OFFLOAD_RE.finditer(args)]
        if not (cfg.get("comfyui") or {}).get("compile"):
            found.append("comfyui.compile=false")
    elif fw == "fastvideo":
        try:
            overrides = {k: v.lower() for k, v in fastvideo_client.serve_overrides(args.split()).items()}
        except ValueError as exc:
            return [f"serve_args are not dotted overrides ({exc})"]
        # Every offload switch defaults to true, so only an explicit false is resident.
        for part in fastvideo_client.OFFLOAD_PARTS:
            key = f"generator.engine.offload.{part}"
            if overrides.get(key) != "false":
                found.append(f"--{key} {overrides.get(key, '(unset: true)')}")
        found += [f"--{key} true" for key in fastvideo_client.DEFERRED_LOAD_KEYS if overrides.get(key) == "true"]
        if not any(overrides.get(key) == "true" for key in fastvideo_client.COMPILE_KEYS):
            found.append("DiT compile off (" + " / ".join(fastvideo_client.COMPILE_KEYS) + ")")
    return found


def _lint_competitor_policy(cid: str, fw: str, body: dict) -> list[tuple[tuple | None, str]]:
    """(cell, error) per selected profile that runs eager or offloaded on
    datacenter hardware without dated evidence. `cell` is (case, fw, profile),
    or None for an error KNOWN_COMPETITOR_VIOLATIONS may not excuse."""
    profiles = body.get("command_profiles") or {}
    groups = {}  # profile -> (source, findings, classes, datacenter classes)
    for hw in POLICY_HARDWARE:
        name, prof, matches = select_profile(profiles, hardware_candidates(None, override=hw))
        name, source = (name, prof) if prof is not None else ("inline", body)
        cfg = resolved_config(body, prof)
        if name not in groups:
            groups[name] = (source, _competitor_findings(fw, cfg), [], [])
        _, findings, classes, datacenter = groups[name]
        classes.append(hw)
        if hw in DATACENTER_HARDWARE:
            datacenter.append(hw)
        status = "yes" if source.get("policy_exception") else (
            "VIOLATION" if hw in DATACENTER_HARDWARE and findings else ""
        )
        SELECTED_ROWS.append(
            (cid, fw, hw, name, cfg.get("serve_args") or "", status, matches,
             _recipe_label(source), _config_label(fw, cfg))
        )
    errors = []
    for name, (source, findings, classes, datacenter) in groups.items():
        exception = source.get("policy_exception")
        if exception and not re.search(r"20\d\d-\d\d", str(exception)):
            errors.append((None, f"{cid}/{fw}[{name}]: policy_exception must cite dated evidence (no 20YY-MM found)"))
        if not datacenter:
            continue
        if not (source.get("upstream_recipe") or str(source.get("upstream_recipe_reason") or "").strip()):
            MISSING_RECIPES[(cid, fw, name)] = datacenter
        if findings and not exception:
            errors.append((
                (cid, fw, name),
                f"{cid}/{fw}[{name}] (selected on {', '.join(datacenter)}): runs eager or "
                f"offloaded ({', '.join(findings)}) and no policy_exception",
            ))
    return errors


def _lint_upstream_recipes(cid: str, fw: str, body: dict) -> list[str]:
    """A recorded recipe must be complete, or the drift check cannot ask about it."""
    errs = []
    for name, source in [("inline", body), *(body.get("command_profiles") or {}).items()]:
        if "upstream_recipe" in source:
            errs += [f"{cid}/{fw}[{name}]: {p}" for p in upstream_recipe_problems(source["upstream_recipe"])]
        if "upstream_recipe_reason" in source and not str(source["upstream_recipe_reason"]).strip():
            errs.append(f"{cid}/{fw}[{name}]: upstream_recipe_reason is empty")
    return errs


def _lint_sglang_policy(cid: str, body: dict) -> list[str]:
    errs = []
    profiles = body.get("command_profiles") or {}
    if not profiles:
        return errs
    for hw in POLICY_HARDWARE:
        name, prof, matches = select_profile(profiles, hardware_candidates(None, override=hw))
        SELECTED_ROWS.append(
            (cid, "sglang", hw, name, prof.get("serve_args", ""),
             "yes" if prof.get("policy_exception") else "", matches, _recipe_label(prof), "")
        )
        exception = prof.get("policy_exception")
        if exception:
            if not re.search(r"20\d\d-\d\d", str(exception)):
                errs.append(
                    f"{cid}/sglang[{name}]: policy_exception must cite dated "
                    f"evidence (no 20YY-MM found)"
                )
            continue
        args = prof.get("serve_args", "")
        if re.search(r"--enable-torch-compile(?!\s+false)", args):
            errs.append(
                f"{cid}/sglang[{name}] (selected on {hw}): enables "
                f"torch.compile and no policy_exception"
            )
        m = _OFFLOAD_ENABLE_RE.search(args)
        if m:
            errs.append(
                f"{cid}/sglang[{name}] (selected on {hw}): offload enabled "
                f"({m.group(0)!r}) and no policy_exception"
            )
    return errs


def _inline_comfyui(cid: str, case: dict, body: dict) -> list[str]:
    """Inline every ComfyUI workflow template and dry-render it for this case.

    The template is looked up by name at build time so the built config is
    self-contained, and it is rendered with the case's own parameters so a
    placeholder with no value, a graph with no nonce (every request after the
    first would be a cache hit) or a checkpoint the spec does not map fails
    here rather than forty minutes into a GPU run.
    """
    errs = []
    base = body.get("comfyui") or {}
    blocks = [("inline", base)] + [
        (name, (prof or {}).get("comfyui"))
        for name, prof in (body.get("command_profiles") or {}).items()
    ]
    for name, block in blocks:
        if not block:
            continue
        where = f"{cid}/comfyui[{name}]"
        effective = {**base, **block, "params": {**(base.get("params") or {}), **(block.get("params") or {})}}
        workflow = block.get("workflow")
        if not workflow:
            if name == "inline":
                errs.append(f"{where}: no `workflow` template named")
            continue
        path = os.path.join(COMFYUI_WORKFLOWS, workflow)
        if not os.path.isfile(path):
            errs.append(f"{where}: workflow template {workflow!r} not found under configs/benchmark/comfyui/")
            continue
        graph = load(path)
        block["workflow_graph"] = graph
        try:
            rendered = comfyui_client.render_workflow(graph, comfyui_client.workflow_params(case, effective))
        except KeyError as exc:
            errs.append(f"{where}: {exc.args[0]}")
            continue
        nodes = rendered.values()
        if not any(comfyui_client.NONCE_INPUT in (n.get("inputs") or {}) for n in nodes):
            errs.append(f"{where}: no node carries {comfyui_client.NONCE_INPUT}; every repeat would be a cache hit")
        if not any(n.get("class_type") in comfyui_client.SAMPLER_CLASSES for n in nodes):
            errs.append(f"{where}: no sampler node")
        mapped = {os.path.basename(rel) for rel in (effective.get("models") or {})}
        wanted = {
            v for n in nodes for v in (n.get("inputs") or {}).values()
            if isinstance(v, str) and v.endswith(".safetensors")
        }
        for missing in sorted(wanted - mapped):
            errs.append(f"{where}: workflow loads {missing!r} but `models` does not map it")
    return errs


def load(path):
    return json.load(open(path))


def build():
    frameworks_meta = load(os.path.join(BENCH, "frameworks.json"))["frameworks"]
    in_scope = list(frameworks_meta.keys())
    workloads = load(os.path.join(BENCH, "workloads.json"))["workloads"]
    meta = load(os.path.join(BENCH, "meta.json"))
    order = load(os.path.join(CASES, "_order.json")).get("order", [])

    # workloads -> benchmark_defaults (the shape the harness reads)
    benchmark_defaults = {
        "throughput": {k: v for k, v in workloads["throughput"].items() if k != "description"},
        "warmup": {k: v for k, v in workloads["warmup"].items() if k != "description"},
        "single": {
            "image_repeats": workloads["single_e2e"]["image_repeats"],
            "video_repeats": workloads["single_e2e"]["video_repeats"],
        },
    }

    case_files = glob.glob(os.path.join(CASES, "**", "*.json"), recursive=True)
    cases_by_id = {}
    errors = []
    policy_errors = []
    competitor_errors = []
    for path in sorted(case_files):
        if os.path.basename(path).startswith("_"):
            continue
        c = load(path)
        cid = c.get("id")
        rel = os.path.relpath(path, REPO)
        if not cid:
            errors.append(f"{rel}: missing 'id'")
            continue

        fw_src = c.get("frameworks", {})
        # validate: every in-scope framework present + valid status
        missing = [fw for fw in in_scope if fw not in fw_src]
        if missing:
            errors.append(f"{cid}: frameworks not classified (silently dropped): {missing}")
        frameworks, statuses, reasons = {}, {}, {}
        for fw, entry in fw_src.items():
            status = entry.get("status")
            if status not in VALID_STATUS:
                errors.append(f"{cid}/{fw}: bad status {status!r} (allowed: {sorted(VALID_STATUS)})")
                continue
            if status == "supported":
                body = {k: v for k, v in entry.items() if k != "status"}
                if not (body.get("command_profiles") or body.get("serve_args") is not None):
                    errors.append(f"{cid}/{fw}: status=supported but no command_profiles/serve_args")
                for _pn, _pf in (body.get("command_profiles") or {"": body}).items():
                    errors.extend(
                        _lint_serve_args_shape(cid, fw, _pn or "inline", _pf.get("serve_args") or "")
                    )
                errors.extend(_lint_model_override(cid, c.get("model"), fw, body))
                errors.extend(_lint_upstream_recipes(cid, fw, body))
                if fw == "sglang":
                    policy_errors.extend(_lint_sglang_policy(cid, body))
                else:
                    competitor_errors.extend(_lint_competitor_policy(cid, fw, body))
                if fw == "comfyui":
                    errors.extend(_inline_comfyui(cid, c, body))
                frameworks[fw] = body
            else:
                statuses[fw] = status
                if entry.get("reason"):
                    reasons[fw] = entry["reason"]

        out = {k: v for k, v in c.items() if k != "frameworks"}
        out["frameworks"] = frameworks
        if statuses:
            out["report_framework_statuses"] = statuses
        if reasons:
            out["report_framework_reasons"] = reasons
        cases_by_id[cid] = out

    unknown_order = [cid for cid in order if cid not in cases_by_id]
    if unknown_order:
        errors.append(f"_order.json lists unknown case ids: {unknown_order}")
    ordered = [cases_by_id[cid] for cid in order if cid in cases_by_id]
    for cid in sorted(cases_by_id):  # any case file not in _order.json -> append (and warn)
        if cid not in order:
            errors.append(f"{cid}: not in cases/_order.json (append it)")
            ordered.append(cases_by_id[cid])

    if errors:
        print("BUILD FAILED — matrix/validation errors:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        sys.exit(1)

    known = [msg for cell, msg in competitor_errors if cell in KNOWN_COMPETITOR_VIOLATIONS]
    policy_errors += [msg for cell, msg in competitor_errors if cell not in KNOWN_COMPETITOR_VIOLATIONS]
    policy_errors += [
        f"{'/'.join(cell)}: listed in KNOWN_COMPETITOR_VIOLATIONS but no longer "
        f"violates -- delete the entry"
        for cell in sorted(KNOWN_COMPETITOR_VIOLATIONS - {cell for cell, _ in competitor_errors})
    ]
    if os.environ.get(STRICT_COMPETITOR_POLICY_ENV) == "1":
        policy_errors += known
        known = []

    if policy_errors:
        print(
            "BUILD FAILED — best-lossless policy violations (add the measured "
            "fastest command, or a `policy_exception` with evidence):",
            file=sys.stderr,
        )
        for e in policy_errors:
            print("  -", e, file=sys.stderr)
        sys.exit(1)

    result = {**meta, "benchmark_defaults": benchmark_defaults, "cases": ordered}
    for out_path in (OUT_EDITABLE, OUT_PACKAGED):
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print("wrote", os.path.relpath(out_path, REPO))

    _write_matrix(ordered, in_scope)
    print(f"\nOK: {len(ordered)} cases x {len(in_scope)} frameworks fully classified.")
    # Both copies are rewritten above, so the working tree is always consistent
    # and `git add configs/` looks complete while silently leaving the packaged
    # copy -- what an install of the package reads -- behind. That happened ten
    # times before it was noticed, so say the command here, at the moment of the
    # mistake. scripts/tests/test_config_copies_in_sync.py is the backstop.
    print(
        "\nstage BOTH copies:\n"
        f"  git add {os.path.relpath(OUT_EDITABLE, REPO)} "
        f"{os.path.relpath(OUT_PACKAGED, REPO)}"
    )

    _write_selected(in_scope)
    if MISSING_RECIPES:
        print(
            f"WARNING: {len(MISSING_RECIPES)} competitor profile(s) run on datacenter "
            "hardware with no `upstream_recipe` (or `upstream_recipe_reason`), so "
            "scripts/check_competitor_recipes.py cannot tell when their command goes stale:",
            file=sys.stderr,
        )
        for (cid, fw, name), classes in sorted(MISSING_RECIPES.items()):
            print(f"  - {cid}/{fw}[{name}] on {', '.join(classes)}", file=sys.stderr)
    if known:
        print(
            f"ERROR: {len(known)} pre-existing competitor policy violation(s), not "
            f"fatal while listed in KNOWN_COMPETITOR_VIOLATIONS "
            f"({STRICT_COMPETITOR_POLICY_ENV}=1 makes them fatal):",
            file=sys.stderr,
        )
        for e in known:
            print("  -", e, file=sys.stderr)


def _write_matrix(ordered, in_scope):
    """Emit an always-current coverage matrix so the grid is never stale."""
    sym = {"supported": "profile", "unsupported": "n/a", "no_profile": "no-cmd",
           "failed": "FAIL", "not_run": "TODO", "invalid": "BAD"}
    lines = ["# Benchmark coverage matrix", "",
             "Auto-generated by `scripts/build_benchmark_config.py` — do not edit by hand.",
             "Scope matrix (has a command profile), NOT run-results. `profile`=configured to run · `TODO`=not_run · `no-cmd`=no_profile · `n/a`=unsupported · `FAIL`/`BAD`=failed/invalid.",
             "", "| case | task | " + " | ".join(in_scope) + " |",
             "|---|---|" + "|".join(["---"] * len(in_scope)) + "|"]
    for c in ordered:
        statuses = c.get("report_framework_statuses", {})
        row = [c["id"], c.get("task", "")]
        for fw in in_scope:
            st = "supported" if fw in c.get("frameworks", {}) else statuses.get(fw, "not_run")
            row.append(sym.get(st, st))
        lines.append("| " + " | ".join(row) + " |")
    path = os.path.join(BENCH, "MATRIX.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, REPO))



def _write_selected(in_scope):
    """SELECTED.md: per case and framework, the profile the harness will
    ACTUALLY run on the policy hardware — review this, not the raw case files.
    `AMBIGUOUS` marks cases where >1 profile matches and dict order decides
    (editing a matching-but-unselected profile is the classic footgun)."""
    rank = {fw: i for i, fw in enumerate(in_scope)}
    rows = sorted(SELECTED_ROWS, key=lambda r: (r[0], rank.get(r[1], len(rank)), r[1], r[2]))
    lines = [
        "# Selected commands (auto-generated — do not edit)",
        "",
        "Regenerated by scripts/build_benchmark_config.py. The `selected` profile is",
        "what `--hardware-profile <hw>` will actually run; edit THAT profile.",
        "`exception`: `yes` carries a dated `policy_exception`; `VIOLATION` runs eager or",
        "offloaded on datacenter hardware without one (see KNOWN_COMPETITOR_VIOLATIONS).",
        "`config` is what decides a competitor's command besides serve_args.",
        "",
        "| case | framework | hw | selected profile | exception | ambiguous matches | upstream recipe | serve_args | config |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, fw, hw, name, args, exc, matches, recipe, config in rows:
        amb = ", ".join(m for m in matches[1:]) if len(matches) > 1 else ""
        lines.append(
            f"| {cid} | {fw} | {hw} | `{name}` | {exc} | "
            f"{('AMBIGUOUS: ' + amb) if amb else ''} | {recipe} | "
            f"{f'`{args}`' if args else ''} | {f'`{config}`' if config else ''} |"
        )
    out = os.path.join(BENCH, "SELECTED.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(out, REPO))
    ambiguous = {}
    for cid, fw, hw, name, _, _, matches, _, _ in rows:
        if len(matches) > 1:
            ambiguous.setdefault((cid, fw, name, tuple(matches)), []).append(hw)
    for (cid, fw, name, matches), classes in ambiguous.items():
        print(
            f"WARNING: {cid}/{fw} on {', '.join(classes)}: {len(matches)} profiles "
            f"match ({', '.join(matches)}); dict order selected {name!r}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    build()
