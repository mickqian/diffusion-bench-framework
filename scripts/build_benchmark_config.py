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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "configs", "benchmark")
CASES = os.path.join(BENCH, "cases")
OUT_EDITABLE = os.path.join(REPO, "configs", "comparison_configs.json")
OUT_PACKAGED = os.path.join(REPO, "src", "diffusion_bench", "comparison_configs.json")

VALID_STATUS = {"supported", "unsupported", "no_profile", "failed", "not_run", "invalid"}


# The benchmark's published hardware. The lint checks the profile that the
# harness would actually SELECT there (first hardware match, else `default`)
# — patching an unselected profile is a recurring footgun.
POLICY_HARDWARE = ("h100",)
# "Best lossless" policy: the selected sglang profile must run compiled and
# resident. A profile may opt out ONLY with a `policy_exception` string
# carrying measured evidence (e.g. zimage: compile measured slower).
_OFFLOAD_ENABLE_RE = re.compile(
    r"--(?:text-encoder|image-encoder|vae|dit)-cpu-offload(?!\s+false)"
    r"|--dit-layerwise-offload(?!\s+false)"
)


def _selected_profile(profiles: dict, hardware: str):
    for name, p in profiles.items():
        hw = p.get("hardware") or []
        if hardware in hw:
            return name, p
    if "default" in profiles:
        return "default", profiles["default"]
    return None, None


def _lint_sglang_policy(cid: str, body: dict) -> list[str]:
    errs = []
    profiles = body.get("command_profiles") or {}
    if not profiles:
        return errs
    for hw in POLICY_HARDWARE:
        name, prof = _selected_profile(profiles, hw)
        if prof is None:
            continue
        if prof.get("policy_exception"):
            continue
        args = prof.get("serve_args", "")
        if not re.search(r"--enable-torch-compile(?!\s+false)", args):
            errs.append(
                f"{cid}/sglang[{name}] (selected on {hw}): missing "
                f"--enable-torch-compile and no policy_exception"
            )
        m = _OFFLOAD_ENABLE_RE.search(args)
        if m:
            errs.append(
                f"{cid}/sglang[{name}] (selected on {hw}): offload enabled "
                f"({m.group(0)!r}) and no policy_exception"
            )
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
                if fw == "sglang":
                    policy_errors.extend(_lint_sglang_policy(cid, body))
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


if __name__ == "__main__":
    build()
