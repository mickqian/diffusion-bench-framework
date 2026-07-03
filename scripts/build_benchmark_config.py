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
from diffusion_bench.config_guard import select_profile  # noqa: E402

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


SELECTED_ROWS = []  # (case, hw, profile, args, exception?, ambiguous-matches)


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
