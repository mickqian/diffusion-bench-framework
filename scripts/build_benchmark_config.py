#!/usr/bin/env python3
"""Assemble configs/benchmark/ (the explicit source of truth) into the
harness-consumed configs/comparison_configs.json and the packaged copy under
src/diffusion_bench/. Validates that EVERY case classifies ALL in-scope
frameworks, so a framework can never be silently dropped.

Source of truth (edit these):
  configs/benchmark/frameworks.json   frameworks in scope + version policy
  configs/benchmark/workloads.json    single_e2e / throughput / warmup -> benchmark_defaults
  configs/benchmark/meta.json         top-level (_comment, test_image_url)
  configs/benchmark/cases/<image|video>/<id>.json   one file per case, all 4 frameworks explicit
  configs/benchmark/cases/_order.json order the cases appear in the built config

Regenerate after editing:  python3 scripts/build_benchmark_config.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from diffusion_bench import comfyui_client  # noqa: E402
from diffusion_bench.config_guard import select_profile  # noqa: E402

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
# b200-* (minimax-h3), leaving them outside the compile policy.
POLICY_HARDWARE = ("h100", "h200", "b200", "b300", "rtx5090", "rtx4090")
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


SELECTED_ROWS = []  # (case, hw, profile, args, exception?, ambiguous-matches)


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


def _lint_sglang_policy(cid: str, body: dict) -> list[str]:
    errs = []
    profiles = body.get("command_profiles") or {}
    if not profiles:
        return errs
    for hw in POLICY_HARDWARE:
        name, prof, matches = select_profile(profiles, hw)
        if prof is None:
            continue
        SELECTED_ROWS.append(
            (cid, hw, name, prof.get("serve_args", ""),
             bool(prof.get("policy_exception")), matches)
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
                if fw == "sglang":
                    policy_errors.extend(_lint_sglang_policy(cid, body))
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

    _write_selected()


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



def _write_selected():
    """SELECTED.md: per case, the sglang profile the harness will ACTUALLY run
    on the policy hardware — review this, not the raw case files. `AMBIGUOUS`
    marks cases where >1 profile matches and dict order decides (editing a
    matching-but-unselected profile is the classic footgun)."""
    lines = [
        "# Selected sglang commands (auto-generated — do not edit)",
        "",
        "Regenerated by scripts/build_benchmark_config.py. The `selected` profile is",
        "what `--hardware-profile <hw>` will actually run; edit THAT profile.",
        "",
        "| case | hw | selected profile | exception | ambiguous matches | serve_args |",
        "|---|---|---|---|---|---|",
    ]
    for cid, hw, name, args, has_exc, matches in sorted(SELECTED_ROWS):
        amb = ", ".join(m for m in matches[1:]) if len(matches) > 1 else ""
        lines.append(
            f"| {cid} | {hw} | `{name}` | {'yes' if has_exc else ''} | "
            f"{('AMBIGUOUS: ' + amb) if amb else ''} | `{args}` |"
        )
    out = os.path.join(BENCH, "SELECTED.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(out, REPO))
    ambiguous = [r for r in SELECTED_ROWS if len(r[5]) > 1]
    for cid, hw, name, _, _, matches in ambiguous:
        print(
            f"WARNING: {cid} on {hw}: {len(matches)} profiles match "
            f"({', '.join(matches)}); dict order selected {name!r}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    build()
