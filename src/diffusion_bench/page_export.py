"""Turn a merged benchmark result into the sections the Pages site reads.

The dated `scripts/export_pages_data_*.py` exporters each hardcode one run's
paths, labels, and hardware; they stay as the record of how a historical run was
published. This module is the parameterized version the recurring job uses: it
takes a merged artifact plus run metadata and produces the same section shape,
reading case labels from the config instead of a per-script dict.
"""

from __future__ import annotations

from typing import Any

FRAMEWORK_ORDER = ("sglang", "vllm-omni", "lightx2v", "trtllm-visual")
FRAMEWORK_LABELS = {
    "sglang": "SGLang-Diffusion",
    "vllm-omni": "vLLM-Omni",
    "lightx2v": "LightX2V",
    "trtllm-visual": "TensorRT-LLM VisualGen",
}


def case_details(case: dict) -> str:
    shape = f"{case['width']}x{case['height']}"
    if case.get("num_frames"):
        shape += f"x{case['num_frames']}f"
    parts = [shape, f"{case['num_inference_steps']} steps"]
    if case.get("true_cfg_scale"):
        parts.append(f"true-cfg {case['true_cfg_scale']}")
    else:
        gs = case.get("guidance_scale")
        parts.append(f"cfg {gs:g}" if isinstance(gs, (int, float)) else "cfg ?")
    return " · ".join(parts)


def case_label(case: dict) -> str:
    """Display name. Cases carry `label`; fall back to the id so a newly added
    case still publishes instead of raising in an unattended run."""
    return case.get("label") or case["id"]


def _rows_by_mode(merged: dict, mode: str) -> tuple[dict, dict]:
    key_field = "results" if mode == "single_e2e" else "throughput_results"
    ok_rows: dict[tuple[str, str], dict] = {}
    failed_rows: dict[tuple[str, str], str] = {}
    for row in merged.get(key_field, []):
        key = (row["case_id"], row["framework"])
        has_latency = isinstance(row.get("latency_s"), (int, float))
        has_qps = isinstance((row.get("metrics") or {}).get("throughput_rps"), (int, float))
        if row.get("error") or not (has_latency or has_qps):
            failed_rows[key] = row.get("error") or "no data recorded"
        else:
            ok_rows[key] = row
    return ok_rows, failed_rows


def _classify_cell(case_id: str, fw: str, ok_rows, failed_rows, config_cases) -> dict | None:
    """Non-OK cell dict, or None when the cell has a real measurement."""
    key = (case_id, fw)
    if key in failed_rows:
        return {"status": "failed", "reason": failed_rows[key]}
    if key in ok_rows:
        return None
    case_cfg = config_cases[case_id]
    statuses = case_cfg.get("report_framework_statuses") or {}
    reasons = case_cfg.get("report_framework_reasons") or {}
    cell = {"status": statuses.get(fw, "not_run")}
    if reasons.get(fw):
        cell["reason"] = reasons[fw]
    return cell


def _profile_of(row: dict) -> str:
    return (row.get("framework_metadata") or {}).get("profile") or "default"


def build_single_e2e_rows(merged: dict, config_cases: dict, case_order) -> tuple[list, int, int]:
    ok_rows, failed_rows = _rows_by_mode(merged, "single_e2e")
    rows, wins, comparable = [], 0, 0
    for case_id in case_order:
        cfg = config_cases[case_id]
        cells: dict[str, Any] = {}
        competitor_lat: dict[str, float] = {}
        for fw in FRAMEWORK_ORDER:
            override = _classify_cell(case_id, fw, ok_rows, failed_rows, config_cases)
            if override is not None:
                cells[fw] = override
                continue
            row = ok_rows[(case_id, fw)]
            cells[fw] = {
                "client_latency_s": round(row["latency_s"], 3),
                "profile": _profile_of(row),
                "gpus": f"{row.get('num_gpus', cfg['num_gpus'])} GPU",
            }
            if fw != "sglang":
                competitor_lat[fw] = row["latency_s"]
        sgl = ok_rows.get((case_id, "sglang"))
        entry = {
            "case_id": case_id,
            "case": case_label(cfg),
            "details": case_details(cfg),
            "gpus": f"{cfg['num_gpus']} GPU",
            "mode": "single_e2e",
            "cells": cells,
        }
        if sgl is None:
            entry["winner"] = (
                min(competitor_lat, key=competitor_lat.get) if competitor_lat else "n/a"
            )
            if competitor_lat:
                comparable += 1
        else:
            entry["winner"] = "sglang"
            if competitor_lat:
                comparable += 1
                best = min(competitor_lat.values())
                if sgl["latency_s"] <= best:
                    wins += 1
                    entry["winner_speedup"] = f"{best / sgl['latency_s']:.2f}x"
                else:
                    # honesty guard: never report a loss as a win
                    entry["winner"] = min(competitor_lat, key=competitor_lat.get)
                    entry["winner_speedup"] = f"{sgl['latency_s'] / best:.2f}x"
        rows.append(entry)
    return rows, wins, comparable


def build_throughput_rows(merged: dict, config_cases: dict, case_order) -> tuple[list, int, int]:
    """Throughput is opt-in per case, so only the cases that actually produced
    throughput rows appear — an absent case is out of scope, not a failure."""
    ok_rows, failed_rows = _rows_by_mode(merged, "throughput")
    measured = {cid for (cid, _fw) in list(ok_rows) + list(failed_rows)}
    rows, wins, comparable = [], 0, 0
    for case_id in case_order:
        if case_id not in measured:
            continue
        cfg = config_cases[case_id]
        cells: dict[str, Any] = {}
        competitor_qps: dict[str, float] = {}
        for fw in FRAMEWORK_ORDER:
            override = _classify_cell(case_id, fw, ok_rows, failed_rows, config_cases)
            if override is not None:
                cells[fw] = override
                continue
            row = ok_rows[(case_id, fw)]
            metrics = row.get("metrics") or {}
            cell = {"profile": _profile_of(row), "gpus": f"{row.get('num_gpus', cfg['num_gpus'])} GPU"}
            for src, dst in (
                ("throughput_rps", "qps"),
                ("p50_s", "p50_s"),
                ("p95_s", "p95_s"),
                ("p99_s", "p99_s"),
                ("num_requests", "num_requests"),
                ("max_concurrency", "concurrency"),
            ):
                if isinstance(metrics.get(src), (int, float)):
                    cell[dst] = round(metrics[src], 4)
            cells[fw] = cell
            if fw != "sglang" and isinstance(metrics.get("throughput_rps"), (int, float)):
                competitor_qps[fw] = metrics["throughput_rps"]
        sgl = ok_rows.get((case_id, "sglang"))
        sgl_qps = ((sgl or {}).get("metrics") or {}).get("throughput_rps")
        entry = {
            "case_id": case_id,
            "case": case_label(cfg),
            "details": case_details(cfg),
            "gpus": f"{cfg['num_gpus']} GPU",
            "mode": "throughput",
            "cells": cells,
        }
        if not isinstance(sgl_qps, (int, float)):
            entry["winner"] = (
                max(competitor_qps, key=competitor_qps.get) if competitor_qps else "n/a"
            )
            if competitor_qps:
                comparable += 1
        else:
            entry["winner"] = "sglang"
            if competitor_qps:
                comparable += 1
                best = max(competitor_qps.values())
                if sgl_qps >= best:
                    wins += 1
                    entry["winner_speedup"] = f"{sgl_qps / best:.2f}x"
                else:
                    entry["winner"] = max(competitor_qps, key=competitor_qps.get)
                    entry["winner_speedup"] = f"{best / sgl_qps:.2f}x"
        rows.append(entry)
    return rows, wins, comparable


def framework_versions(merged: dict) -> dict:
    """Resolved versions per framework, as recorded by the runner."""
    out: dict[str, str] = {}
    runtime = merged.get("framework_runtime") or {}
    for fw, info in runtime.items():
        if isinstance(info, dict):
            ver = info.get("version") or info.get("resolved") or info.get("commit")
            if ver:
                out[fw] = str(ver)
        elif info:
            out[fw] = str(info)
    sgl = merged.get("sglang_runtime") or {}
    if sgl.get("version") or merged.get("commit_sha"):
        out.setdefault(
            "sglang",
            str(sgl.get("version") or "")
            + (f" @ {merged['commit_sha'][:9]}" if merged.get("commit_sha") else ""),
        )
    return {k: v for k, v in out.items() if v}


def build_sections(
    merged: dict,
    config: dict,
    *,
    run_id: str,
    run_label: str,
    gpu: str,
    date: str,
    reproduce: str | None = None,
) -> list[dict]:
    config_cases = {c["id"]: c for c in config["cases"]}
    case_order = [c["id"] for c in config["cases"]]
    versions = framework_versions(merged)
    sections = []

    single_rows, s_wins, s_cmp = build_single_e2e_rows(merged, config_cases, case_order)
    if single_rows:
        sections.append(
            {
                "id": f"{run_id}_single_e2e",
                "run": run_id,
                "run_label": run_label,
                "date": date,
                "gpu": gpu,
                "title": run_label,
                "subtitle": (
                    f"Steady-state single-request latency, best lossless config per framework "
                    f"(compile on, no caches, no quantization). SGLang-Diffusion fastest in "
                    f"{s_wins}/{s_cmp} comparable cases."
                ),
                "mode": "single_e2e",
                "framework_versions": versions,
                "summary": {"sglang_wins": s_wins, "comparable": s_cmp},
                "rows": single_rows,
            }
        )
        if reproduce:
            sections[-1]["reproduce"] = reproduce

    tput_rows, t_wins, t_cmp = build_throughput_rows(merged, config_cases, case_order)
    if tput_rows:
        sections.append(
            {
                "id": f"{run_id}_throughput",
                "run": run_id,
                "run_label": run_label,
                "date": date,
                "gpu": gpu,
                "title": f"{run_label} — throughput",
                "subtitle": (
                    "High-pressure throughput on a representative pair (one popular image model, "
                    "one video+audio case) — the matrix is deliberately not swept, see "
                    "configs/benchmark/workloads.json. "
                    f"SGLang-Diffusion highest QPS in {t_wins}/{t_cmp} comparable cases."
                ),
                "mode": "throughput",
                "framework_versions": versions,
                "summary": {"sglang_wins": t_wins, "comparable": t_cmp},
                "rows": tput_rows,
            }
        )
        if reproduce:
            sections[-1]["reproduce"] = reproduce
    return sections
