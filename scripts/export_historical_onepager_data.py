#!/usr/bin/env python3
"""Export public-safe historical benchmark data for the one-pager."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "comparison_configs.json"
OUTPUT_PATH = ROOT / "docs" / "data" / "historical-cross-framework.json"
FRAMEWORKS = ("sglang", "vllm-omni", "lightx2v")
FRAMEWORK_LABELS = {
    "sglang": "SGLang-Diffusion",
    "vllm-omni": "vLLM-Omni",
    "lightx2v": "LightX2V",
}

FULL_MATRIX_PATH = ROOT / "tmp" / "report" / "h200-framework-comparison-merged-local.json"
COSMOS_HISTORY_PATH = (
    ROOT / "tmp" / "report" / "cosmos3-origin-main-64a1dec-vs-vllm-2gpu-video-20260602.json"
)
FLUX_FASTEST_DIR = ROOT / "reports" / "flux-fastest-20260610"

CASE_ORDER = (
    "flux1_dev_t2i_1024",
    "flux2_dev_t2i_1024",
    "qwen_image_2512_t2i_1024",
    "qwen_image_edit_2511",
    "zimage_turbo_t2i_1024",
    "wan21_t2v_1_3b_480p",
    "wan21_i2v_14b_480p",
    "wan22_t2v_a14b_720p",
    "wan22_ti2v_5b_704p",
    "ltx2_twostage_t2v",
    "ltx2.3_twostage_t2v_2gpus",
    "wan21_i2v_14b_720p",
    "wan22_i2v_a14b_720p",
    "cosmos3_nano_t2i_720p",
    "cosmos3_nano_t2v_720p_189f",
    "cosmos3_nano_i2v_720p_189f",
)

CASE_LABELS = {
    "flux1_dev_t2i_1024": "FLUX.1-dev T2I",
    "flux2_dev_t2i_1024": "FLUX.2-dev T2I",
    "qwen_image_2512_t2i_1024": "Qwen-Image-2512 T2I",
    "qwen_image_edit_2511": "Qwen-Image-Edit-2511",
    "zimage_turbo_t2i_1024": "Z-Image-Turbo T2I",
    "wan21_t2v_1_3b_480p": "Wan2.1 T2V 1.3B 480p",
    "wan21_i2v_14b_480p": "Wan2.1 I2V 14B 480p",
    "wan22_t2v_a14b_720p": "Wan2.2 T2V A14B 720p",
    "wan22_ti2v_5b_704p": "Wan2.2 TI2V 5B 704p",
    "wan21_i2v_14b_720p": "Wan2.1 I2V 14B 720p",
    "wan22_i2v_a14b_720p": "Wan2.2 I2V A14B 720p",
    "ltx2_twostage_t2v": "LTX-2 two-stage T2V",
    "ltx2.3_twostage_t2v_2gpus": "LTX-2.3 two-stage T2V",
    "cosmos3_nano_t2i_720p": "Cosmos3 Nano T2I",
    "cosmos3_nano_t2v_720p_189f": "Cosmos3 Nano T2V",
    "cosmos3_nano_i2v_720p_189f": "Cosmos3 Nano I2V",
}

NO_PROFILE_REASONS = {
    ("ltx2.3_twostage_t2v_2gpus", "vllm-omni"): "No validated vLLM-Omni LTX-2.3 two-stage profile is tracked.",
    ("flux1_dev_t2i_1024", "lightx2v"): "Tracked LightX2V version has no FLUX.1 serving path.",
    ("qwen_image_2512_t2i_1024", "lightx2v"): "Tracked LightX2V version has no Qwen-Image T2I serving path.",
    ("qwen_image_edit_2511", "lightx2v"): "Tracked LightX2V version has no Qwen-Image-Edit serving path.",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def round_float(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def case_configs() -> dict[str, dict[str, Any]]:
    data = load_json(CONFIG_PATH)
    return {case["id"]: case for case in data.get("cases", [])}


def case_details(case_id: str, config: dict[str, Any]) -> dict[str, str]:
    width = config.get("width")
    height = config.get("height")
    frames = config.get("num_frames")
    dims = f"{width}x{height}x{frames}f" if frames else f"{width}x{height}"
    steps = config.get("num_inference_steps")
    cfg = config.get("guidance_scale")
    cfg2 = config.get("guidance_scale_2")
    parts = [dims]
    if steps is not None:
        parts.append(f"{steps} steps")
    if cfg is not None:
        cfg_label = f"cfg {cfg:g}"
        if cfg2 is not None:
            cfg_label += f"/{cfg2:g}"
        parts.append(cfg_label)
    return {
        "case_id": case_id,
        "case": CASE_LABELS.get(case_id, case_id.replace("_", " ")),
        "details": " · ".join(parts),
    }


def profile(entry: dict[str, Any]) -> str | None:
    metadata = entry.get("framework_metadata") or {}
    return metadata.get("profile") or metadata.get("description")


def source_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def cell_from_entry(entry: dict[str, Any] | None, case_id: str, framework: str) -> dict[str, Any]:
    if entry is None:
        reason = NO_PROFILE_REASONS.get((case_id, framework), "No validated aligned serving profile is tracked.")
        return {"status": "no_profile", "reason": reason}

    error = entry.get("error")
    cell: dict[str, Any] = {
        "gpus": entry.get("num_gpus"),
        "profile": profile(entry) or "default",
    }
    if error:
        cell.update({"status": "failed", "reason": str(error).splitlines()[0][:220]})
        return cell

    cell.update(
        {
            "status": "ok",
            "latency_s": round_float(entry.get("latency_s")),
        }
    )
    metrics = entry.get("metrics") or {}
    if metrics:
        cell["server_latency_s"] = round_float(metrics.get("server_latency_s"))
        cell["qps"] = round_float(metrics.get("throughput_rps") or metrics.get("throughput_qps"), 4)
        cell["p50_s"] = round_float(metrics.get("latency_p50_s") or metrics.get("latency_p50"))
        cell["p95_s"] = round_float(metrics.get("latency_p95_s") or metrics.get("latency_p95"))
        cell["p99_s"] = round_float(metrics.get("latency_p99_s") or metrics.get("latency_p99"))
        cell["requests"] = metrics.get("num_success") or metrics.get("num_requests")
        cell["concurrency"] = metrics.get("max_concurrency")
    return {k: v for k, v in cell.items() if v is not None}


def add_ratios(cells: dict[str, dict[str, Any]], mode: str) -> tuple[str | None, int]:
    sg = cells.get("sglang") or {}
    sg_value = sg.get("qps") if mode == "throughput" else sg.get("latency_s")
    comparable = 0
    winner: str | None = None

    if sg_value:
        best_value = sg_value
        winner = "sglang"
        for framework, cell in cells.items():
            value = cell.get("qps") if mode == "throughput" else cell.get("latency_s")
            if not value:
                continue
            comparable += 1
            if mode == "throughput":
                cell["ratio_to_sglang"] = round(value / sg_value, 3)
                if value > best_value:
                    best_value = value
                    winner = framework
            else:
                cell["ratio_to_sglang"] = round(value / sg_value, 3)
                if value < best_value:
                    best_value = value
                    winner = framework
    return winner, comparable


def group_entries(entries: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in entries:
        framework = entry.get("framework")
        case_id = entry.get("case_id")
        if framework in FRAMEWORKS and case_id:
            grouped[case_id][framework] = entry
    return grouped


def section_from_merged(
    data: dict[str, Any],
    key: str,
    mode: str,
    title: str,
    subtitle: str,
    source_manifest: str,
    reproduce: str,
    configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped = group_entries(data.get(key) or [])
    rows: list[dict[str, Any]] = []
    sg_wins = 0
    other_wins = 0
    comparable_rows = 0

    for case_id in CASE_ORDER:
        if case_id not in configs or case_id not in grouped:
            continue
        cells = {
            framework: cell_from_entry(grouped[case_id].get(framework), case_id, framework)
            for framework in FRAMEWORKS
        }
        winner, comparable = add_ratios(cells, mode)
        if comparable >= 2:
            comparable_rows += 1
            if winner == "sglang":
                sg_wins += 1
            elif winner:
                other_wins += 1
        rows.append(
            {
                **case_details(case_id, configs[case_id]),
                "mode": mode,
                "winner": winner,
                "cells": cells,
            }
        )

    return {
        "id": source_manifest.removesuffix(".json").replace("/", "_") + "_" + mode,
        "title": title,
        "subtitle": subtitle,
        "mode": mode,
        "source_manifest": source_manifest,
        "source_run_id": data.get("run_id"),
        "reproduce": reproduce,
        "summary": {
            "rows": len(rows),
            "comparable_rows": comparable_rows,
            "sglang_diffusion_wins": sg_wins,
            "other_wins": other_wins,
        },
        "rows": rows,
    }


def flux_series(run_id: str, framework: str) -> str | None:
    if framework == "sglang":
        if "sgld-before" in run_id:
            return "sglang_before"
        if "sgld-now" in run_id:
            return "sglang_now"
    if framework == "vllm-omni":
        return "vllm_omni_now"
    if framework == "lightx2v":
        return "lightx2v_now"
    return None


def flux_hardware(run_id: str) -> str:
    if run_id.startswith("b200-"):
        return "B200"
    if run_id.startswith("h200-"):
        return "H200"
    return "unknown"


def flux_gpu_label(entry: dict[str, Any]) -> str:
    gpus = entry.get("num_gpus")
    prof = profile(entry) or ""
    if gpus == 2 and "tp" in prof:
        return "2 GPU TP"
    if gpus:
        return f"{gpus} GPU"
    return "-"


def build_flux_section(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_paths = sorted((FLUX_FASTEST_DIR / "raw").glob("*.json"))
    failure_paths = sorted((FLUX_FASTEST_DIR / "failures").glob("*.json"))
    for path in raw_paths + failure_paths:
        data = load_json(path)
        run_id = data.get("run_id") or path.stem
        hardware = flux_hardware(run_id)
        for entry in data.get("results") or []:
            series = flux_series(run_id, entry.get("framework"))
            case_id = entry.get("case_id")
            if not series or case_id not in ("flux1_dev_t2i_1024", "flux2_dev_t2i_1024"):
                continue
            key = (hardware, case_id, series)
            cloned = dict(entry)
            cloned["source_result"] = source_label(path)
            cloned["run_id"] = run_id
            cloned["series"] = series
            cloned["hardware"] = hardware
            prev = candidates.get(key)
            lat = cloned.get("latency_s")
            prev_lat = prev.get("latency_s") if prev else None
            if prev is None or (lat is not None and (prev_lat is None or lat < prev_lat)):
                candidates[key] = cloned

    rows = []
    for hardware in ("H200", "B200"):
        for case_id in ("flux1_dev_t2i_1024", "flux2_dev_t2i_1024"):
            cells: dict[str, dict[str, Any]] = {}
            for series in ("sglang_before", "sglang_now", "vllm_omni_now", "lightx2v_now"):
                entry = candidates.get((hardware, case_id, series))
                if entry:
                    cell = cell_from_entry(entry, case_id, entry.get("framework", ""))
                    cell["gpus"] = flux_gpu_label(entry)
                    cell["source_result"] = entry.get("source_result")
                    cells[series] = cell
                else:
                    cells[series] = {"status": "not_run", "reason": "No successful recorded profile in this report."}

            winner = None
            best = None
            for series, cell in cells.items():
                value = cell.get("latency_s")
                if value is not None and (best is None or value < best):
                    best = value
                    winner = series
            if cells.get("sglang_now", {}).get("latency_s"):
                sg = cells["sglang_now"]["latency_s"]
                for cell in cells.values():
                    if cell.get("latency_s"):
                        cell["ratio_to_sglang_now"] = round(cell["latency_s"] / sg, 3)
            rows.append(
                {
                    **case_details(case_id, configs[case_id]),
                    "hardware": hardware,
                    "mode": "single_e2e",
                    "torch_compile": "on",
                    "winner": winner,
                    "cells": cells,
                }
            )

    return {
        "id": "flux_fastest_20260610",
        "title": "FLUX fastest history",
        "subtitle": "Best recorded 1GPU/2GPU TP profile per framework series; torch compile allowed; no cache.",
        "mode": "single_e2e",
        "source_report": "reports/flux-fastest-20260610",
        "reproduce": "scripts/run_flux_fastest_20260610.sh",
        "summary": {
            "rows": len(rows),
            "sglang_now_wins": sum(1 for row in rows if row["winner"] == "sglang_now"),
            "other_wins": sum(1 for row in rows if row["winner"] not in (None, "sglang_now")),
        },
        "rows": rows,
    }


def build_data() -> dict[str, Any]:
    configs = case_configs()
    sections = []

    if FULL_MATRIX_PATH.exists():
        full = load_json(FULL_MATRIX_PATH)
        sections.append(
            section_from_merged(
                full,
                "results",
                "single_e2e",
                "H200 full matrix",
                "Single-request latency across image and video cases from the May formal matrix.",
                "manifests/20260510-h200-single-e2e.json + manifests/20260514-h200-wan-vllm-omni.json",
                "scripts/generate_h200_report_artifacts.sh",
                configs,
            )
        )
        sections.append(
            section_from_merged(
                full,
                "throughput_results",
                "throughput",
                "H200 high-pressure throughput",
                "32 image requests at concurrency 4 and 8 video requests at concurrency 2 where tracked.",
                "manifests/20260511-h200-throughput-fast.json + manifests/20260514-h200-wan-vllm-omni.json",
                "scripts/generate_h200_report_artifacts.sh",
                configs,
            )
        )

    if COSMOS_HISTORY_PATH.exists():
        cosmos = load_json(COSMOS_HISTORY_PATH)
        sections.append(
            section_from_merged(
                cosmos,
                "results",
                "single_e2e",
                "Cosmos3 H200 history",
                "Earlier Cosmos3 Nano SGLang-Diffusion main vs vLLM-Omni PR #3454 comparison.",
                "manifests/20260601-h200-cosmos3.json",
                "scripts/run_h200_cosmos3_20260601.sh",
                configs,
            )
        )

    sections.append(build_flux_section(configs))

    full_single = next((s for s in sections if s["id"].endswith("_single_e2e")), None)
    full_throughput = next((s for s in sections if s["id"].endswith("_throughput")), None)

    return {
        "id": "historical-cross-framework-20260610",
        "updated_at": date.today().isoformat(),
        "frameworks": FRAMEWORK_LABELS,
        "policy": {
            "public_source": "Formal manifests plus committed reports/raw; tmp source JSON is used only when referenced by a manifest.",
            "latency_source": "client-side latency for headline comparisons; server-side latency is diagnostic only.",
            "cache": "No response cache, no Cache-DiT, no temporal cache in these rows.",
        },
        "summary": {
            "sections": len(sections),
            "h200_full_matrix_single_rows": full_single["summary"]["rows"] if full_single else 0,
            "h200_full_matrix_sglang_wins": full_single["summary"]["sglang_diffusion_wins"] if full_single else 0,
            "h200_full_matrix_other_wins": full_single["summary"]["other_wins"] if full_single else 0,
            "h200_throughput_sglang_wins": full_throughput["summary"]["sglang_diffusion_wins"] if full_throughput else 0,
            "h200_throughput_other_wins": full_throughput["summary"]["other_wins"] if full_throughput else 0,
        },
        "investigation": {
            "headline": "The latest local evidence does not show SGLang-Diffusion generally behind; the regression is concentrated in H200 FLUX.1 2GPU TP.",
            "status": "needs stage breakdown / trace to close",
            "evidence": [
                "Latest H200 aligned spot-check: SGLang-Diffusion wins 3 of 4 same-GPU cases; vLLM-Omni wins only FLUX.1 2GPU TP by 1.08x.",
                "H200 FLUX.1 single-GPU favors SGLang-Diffusion: 6.566s vs vLLM-Omni 7.288s, so the base single-GPU path is not slower.",
                "H200 FLUX.1 TP scaling is weak for SGLang-Diffusion: 6.566s to 6.400s, about 2.5%; vLLM-Omni scales 7.288s to 5.912s, about 18.9%.",
                "H200 FLUX.2 2GPU TP favors SGLang-Diffusion: 12.507s vs vLLM-Omni 16.654s, so the issue is not a blanket TP failure.",
                "B200 FLUX rows favor SGLang-Diffusion now after performance-mode speed: FLUX.1 3.608s vs vLLM-Omni 4.224s; FLUX.2 6.256s vs 12.183s.",
                "Most May H200 full-matrix image/video rows favor SGLang-Diffusion over vLLM-Omni and LightX2V; this makes the H200 FLUX.1 TP miss a narrow optimization target.",
            ],
            "likely_causes": [
                "H200 FLUX.1 TP communication or partition overhead cancels the denoiser compute savings at this shape.",
                "The current best profile may need to prefer 1GPU for FLUX.1/H200 unless throughput pressure or memory constraints require TP.",
                "vLLM-Omni may be parallelizing the FLUX.1 compiled path with lower per-step overhead on H200.",
            ],
            "next_checks": [
                "Collect stage breakdown for text encoder, denoise, VAE decode, and response encoding on 1GPU vs 2GPU.",
                "Trace TP collectives and denoiser kernels to separate compute speedup from communication overhead.",
                "Compare denoise-only latency and CUDA graph/compile coverage for SGLang-Diffusion vs vLLM-Omni.",
                "Keep client-side latency as the public comparison metric and use server-side time only as breakdown.",
            ],
        },
        "sections": sections,
    }


def main() -> None:
    data = build_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
