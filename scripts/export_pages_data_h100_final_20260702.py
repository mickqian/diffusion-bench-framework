#!/usr/bin/env python3
"""Publish the H100x4 final (compile-on) run to the Pages data files.

Does two things, idempotently:

1. Rebuilds ``docs/data/latest-cross-framework.json`` from
   ``reports/h100x4-final-20260702/merged_final.json`` (single-request rows,
   statuses from the case config, winner/speedup computed here).
2. Migrates the previous "latest" run (B200 2026-07-01) into
   ``docs/data/historical-cross-framework.json`` as its own run tab, so it
   stays on the site.

Run ``python3 scripts/refresh_docs_data.py`` afterwards to refresh the inline
file:// preview snapshots embedded in docs/index.html.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "h100x4-final-20260702"
CONFIG = ROOT / "configs" / "comparison_configs.json"
LATEST = ROOT / "docs" / "data" / "latest-cross-framework.json"
HISTORICAL = ROOT / "docs" / "data" / "historical-cross-framework.json"

FRAMEWORK_ORDER = ["sglang", "vllm-omni", "lightx2v", "trtllm-visual"]

CASE_ORDER = [
    "flux1_dev_t2i_1024",
    "flux2_dev_t2i_1024",
    "qwen_image_2512_t2i_1024",
    "qwen_image_2512_t2i_1024_truecfg",
    "zimage_turbo_t2i_1024",
    "wan21_t2v_1_3b_480p",
    "wan22_ti2v_5b_704p",
    "ltx2.3_twostage_t2v_2gpus",
    "cosmos3_nano_t2v_720p_189f",
]

CASE_LABELS = {
    "flux1_dev_t2i_1024": "FLUX.1-dev T2I",
    "flux2_dev_t2i_1024": "FLUX.2-dev T2I",
    "qwen_image_2512_t2i_1024": "Qwen-Image-2512 T2I (no-CFG)",
    "qwen_image_2512_t2i_1024_truecfg": "Qwen-Image-2512 T2I (true-CFG)",
    "zimage_turbo_t2i_1024": "Z-Image-Turbo T2I",
    "wan21_t2v_1_3b_480p": "Wan2.1 T2V 1.3B 480p",
    "wan22_ti2v_5b_704p": "Wan2.2 TI2V 5B 704p",
    "ltx2.3_twostage_t2v_2gpus": "LTX-2.3 two-stage T2V",
    "cosmos3_nano_t2v_720p_189f": "Cosmos3 Nano T2V 720p",
}

# What actually ran (devbox venvs + session records); merged_final's
# sglang_runtime.git_commit is the container-image commit, not the runtime.
FRAMEWORKS = {
    "sglang": {
        "label": "SGLang-Diffusion",
        "commit": "c05c48b",
        "note": (
            "origin/main + PR #29774 TP shard + txt_mlp replicated "
            "(auto plan in #30004); torch.compile + resident DiT; "
            "CFG-parallel for true-CFG"
        ),
    },
    "vllm-omni": {
        "label": "vLLM-Omni",
        "vllm_omni": "0.24.0rc2.dev8+g4d2ee151",
        "vllm": "0.24.0",
        "note": "FlashInfer/FLASH_ATTN exact; compile on (no --enforce-eager)",
    },
    "lightx2v": {
        "label": "LightX2V",
        "commit": "7efd05f",
        "note": "main; FA2/FA3 exact; compile on",
    },
    "trtllm-visual": {
        "label": "TensorRT-LLM VisualGen",
        "tensorrt_llm": "1.3.0rc18",
        "note": "_torch backend; VANILLA(SDPA) exact attention (fastest on Hopper); cache-free",
    },
}

POLICY = {
    "latency_source": (
        "client-side wall clock, steady-state median of back-to-back "
        "measured requests after warmup; server-side timers kept only as "
        "sglang diagnostics"
    ),
    "selection": "best lossless profile per case (h100-* speed/compile where tuned, else default)",
    "cache": "no response cache, no Cache-DiT",
    "torch_compile": "ENABLED for every framework ('no precision loss, best perf' policy)",
    "attention": (
        "each framework runs its fastest exact-precision attention "
        "(FA/FlashInfer/SDPA); no quantized/approximate substitution"
    ),
    "qwen_note": (
        "Qwen-Image is reported under both semantics: no-CFG (guidance 1.0, "
        "one forward/step) and true-CFG (true_cfg_scale 4 + negative prompt, "
        "two forwards/step, served with CFG-parallel). The two rows are not "
        "comparable to each other."
    ),
}


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def build_latest() -> dict:
    merged = load(REPORT / "merged_final.json")
    config_cases = {
        c["id"]: c
        for c in load(CONFIG)["cases"]
    }
    ok_rows: dict[tuple[str, str], dict] = {}
    failed_rows: dict[tuple[str, str], str] = {}
    for row in merged["results"]:
        if row.get("mode") != "single_e2e":
            continue
        key = (row["case_id"], row["framework"])
        if row.get("error") or not isinstance(row.get("latency_s"), (int, float)):
            failed_rows[key] = row.get("error") or "no latency recorded"
        else:
            ok_rows[key] = row

    rows = []
    wins = 0
    comparable = 0
    for case_id in CASE_ORDER:
        case_cfg = config_cases[case_id]
        sgl = ok_rows[(case_id, "sglang")]
        statuses = case_cfg.get("report_framework_statuses") or {}
        reasons = case_cfg.get("report_framework_reasons") or {}
        cells = {}
        competitor_lat = {}
        for fw in FRAMEWORK_ORDER:
            row = ok_rows.get((case_id, fw))
            if row:
                cells[fw] = {
                    "client_latency_s": round(row["latency_s"], 3),
                    "profile": (row.get("framework_metadata") or {}).get("profile")
                    or "default",
                }
                if fw != "sglang":
                    competitor_lat[fw] = row["latency_s"]
            elif (case_id, fw) in failed_rows:
                cells[fw] = {"status": "failed", "reason": failed_rows[(case_id, fw)]}
            else:
                status = statuses.get(fw, "not_run")
                cell = {"status": status}
                if reasons.get(fw):
                    cell["reason"] = reasons[fw]
                cells[fw] = cell
        entry = {
            "case_id": case_id,
            "case": CASE_LABELS[case_id],
            "details": case_details(case_cfg),
            "gpus": f"{sgl['num_gpus']} GPU",
            "winner": "sglang",
            "cells": cells,
        }
        if competitor_lat:
            comparable += 1
            best = min(competitor_lat.values())
            if sgl["latency_s"] <= best:
                wins += 1
                entry["winner_speedup"] = f"{best / sgl['latency_s']:.2f}x"
            else:  # honesty guard: never report a loss as a win
                entry["winner"] = min(competitor_lat, key=competitor_lat.get)
                entry["winner_speedup"] = f"{sgl['latency_s'] / best:.2f}x"
        rows.append(entry)

    return {
        "id": "h100x4-final-20260702",
        "title": (
            "H100 SGLang-Diffusion vs vLLM-Omni vs LightX2V vs TensorRT-LLM "
            "VisualGen (best lossless, compile on)"
        ),
        "updated_at": "2026-07-03",
        "source_report": "reports/h100x4-final-20260702/",
        "source_script": "configs/benchmark/ (explicit matrix) + scripts/build_benchmark_config.py",
        "hardware": {"label": "4x NVIDIA H100 80GB", "gpu_name": "NVIDIA H100 80GB HBM3", "gpus": 4},
        "policy": POLICY,
        "framework_order": FRAMEWORK_ORDER,
        "frameworks": FRAMEWORKS,
        "summary": {
            "cases": len(rows),
            "comparable_rows": comparable,
            "sglang_diffusion_wins": wins,
            "other_wins": comparable - wins,
            "note": (
                "Steady-state single-request latency on H100, best lossless "
                "config per framework (torch.compile on everywhere). SGLang "
                "fastest on every comparable case, including both Qwen-Image "
                "semantics after the TP selective-sharding work (sglang "
                "PRs #29774/#30004). Throughput tables live in the report."
            ),
        },
        "rows": rows,
    }


def _h100_framework_versions() -> dict:
    return {
        "sglang": "c05c48b+#29774",
        "vllm-omni": "0.24.0rc2",
        "lightx2v": "7efd05f",
        "trtllm-visual": "1.3.0rc18",
    }


def h100_single_e2e_section(latest: dict) -> dict:
    """Convert the freshly built latest rows into a history section."""
    rows = []
    for row in latest["rows"]:
        gpus = int(str(row["gpus"]).split()[0])
        sgl_lat = row["cells"]["sglang"]["client_latency_s"]
        cells = {}
        for fw, cell in row["cells"].items():
            if "client_latency_s" in cell:
                lat = cell["client_latency_s"]
                cells[fw] = {
                    "gpus": gpus,
                    "profile": cell.get("profile", "default"),
                    "status": "ok",
                    "latency_s": lat,
                    "ratio_to_sglang": round(lat / sgl_lat, 3),
                }
            else:
                cells[fw] = dict(cell)
        rows.append(
            {
                "case_id": row["case_id"],
                "case": row["case"],
                "details": row["details"],
                "mode": "single_e2e",
                "winner": row["winner"],
                "cells": cells,
            }
        )
    return {
        "id": "h100x4-final-20260702_single_e2e",
        "run": "h100x4-final-20260702",
        "run_label": "H100 best-lossless (compile on)",
        "date": "2026-07-03",
        "gpu": "4x NVIDIA H100 80GB",
        "title": "H100 best-lossless (compile on)",
        "subtitle": latest["summary"]["note"],
        "mode": "single_e2e",
        "source_manifest": latest["source_report"],
        "reproduce": latest["source_script"],
        "framework_versions": _h100_framework_versions(),
        "summary": {
            "rows": len(rows),
            "comparable_rows": latest["summary"]["comparable_rows"],
            "sglang_diffusion_wins": latest["summary"]["sglang_diffusion_wins"],
            "other_wins": latest["summary"]["other_wins"],
        },
        "rows": rows,
    }


def h100_throughput_section(latest: dict) -> dict:
    """Parse the committed dashboard's High-Pressure Throughput table."""
    text = (REPORT / "dashboard_final.md").read_text(encoding="utf-8")
    lines = text.split("## High-Pressure Throughput", 1)[1].splitlines()
    table = [
        line for line in lines if line.startswith("|") and "---" not in line
    ]
    data_rows = table[1:]  # drop header
    assert len(data_rows) == len(CASE_ORDER), (len(data_rows), len(CASE_ORDER))
    latest_rows = {r["case_id"]: r for r in latest["rows"]}
    rows = []
    wins = comparable = 0
    for case_id, line in zip(CASE_ORDER, data_rows):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        # Model | sglang qps | sglang p95 | vllm qps | vllm p95 |
        # trtllm qps | trtllm p95 | lightx2v qps | lightx2v p95
        pairs = {
            "sglang": (cols[1], cols[2]),
            "vllm-omni": (cols[3], cols[4]),
            "trtllm-visual": (cols[5], cols[6]),
            "lightx2v": (cols[7], cols[8]),
        }
        base_cells = latest_rows[case_id]["cells"]
        sgl_qps = float(pairs["sglang"][0])
        cells = {}
        qps_by_fw = {}
        for fw in FRAMEWORK_ORDER:
            qps_str, p95_str = pairs[fw]
            if qps_str != "N/A":
                qps = float(qps_str)
                qps_by_fw[fw] = qps
                cells[fw] = {
                    "status": "ok",
                    "qps": qps,
                    "p95_s": float(p95_str),
                    "ratio_to_sglang": round(qps / sgl_qps, 3),
                }
            else:
                # carry the single_e2e classification for the missing cell
                prev = base_cells.get(fw, {})
                status = prev.get("status", "not_run")
                cells[fw] = {"status": status if "client_latency_s" not in prev else "not_run"}
                if prev.get("reason") and "client_latency_s" not in prev:
                    cells[fw]["reason"] = prev["reason"]
        winner = max(qps_by_fw, key=qps_by_fw.get)
        if len(qps_by_fw) > 1:
            comparable += 1
            if winner == "sglang" or qps_by_fw["sglang"] >= max(
                v for k, v in qps_by_fw.items() if k != "sglang"
            ):
                wins += 1
                winner = "sglang"
        rows.append(
            {
                "case_id": case_id,
                "case": latest_rows[case_id]["case"],
                "details": latest_rows[case_id]["details"],
                "mode": "throughput",
                "winner": winner,
                "cells": cells,
            }
        )
    return {
        "id": "h100x4-final-20260702_throughput",
        "run": "h100x4-final-20260702",
        "run_label": "H100 best-lossless (compile on)",
        "date": "2026-07-03",
        "gpu": "4x NVIDIA H100 80GB",
        "title": "H100 high-pressure throughput",
        "subtitle": "Image requests at concurrency 2; qps and p95 from the committed report dashboard.",
        "mode": "throughput",
        "source_manifest": latest["source_report"],
        "reproduce": latest["source_script"],
        "framework_versions": _h100_framework_versions(),
        "summary": {
            "rows": len(rows),
            "comparable_rows": comparable,
            "sglang_diffusion_wins": wins,
            "other_wins": comparable - wins,
        },
        "rows": rows,
    }


def b200_to_history_section(old_latest: dict) -> dict:
    fw_versions = {}
    for fw, meta in (old_latest.get("frameworks") or {}).items():
        fw_versions[fw] = (
            meta.get("commit")
            or meta.get("vllm_omni")
            or meta.get("tensorrt_llm")
            or ""
        )
    rows = []
    for row in old_latest.get("rows", []):
        gpus = int(str(row.get("gpus", "0")).split()[0] or 0)
        sgl_lat = (row["cells"].get("sglang") or {}).get("client_latency_s")
        cells = {}
        for fw, cell in row["cells"].items():
            if "client_latency_s" in cell:
                lat = cell["client_latency_s"]
                new = {
                    "gpus": gpus,
                    "profile": cell.get("profile", "default"),
                    "status": "ok",
                    "latency_s": lat,
                }
                if sgl_lat:
                    new["ratio_to_sglang"] = round(lat / sgl_lat, 3)
                cells[fw] = new
            else:
                cells[fw] = dict(cell)
        rows.append(
            {
                "case_id": row["case_id"],
                "case": row["case"],
                "details": row["details"],
                "mode": "single_e2e",
                "winner": row.get("winner", "sglang"),
                "cells": cells,
            }
        )
    summary = old_latest.get("summary", {})
    return {
        "id": f"{old_latest['id']}_single_e2e",
        "run": old_latest["id"],
        "run_label": "B200 cross-framework",
        "date": old_latest.get("updated_at", ""),
        "gpu": old_latest.get("hardware", {}).get("label", ""),
        "title": "B200 cross-framework",
        "subtitle": summary.get("note", ""),
        "mode": "single_e2e",
        "source_manifest": old_latest.get("source_report", ""),
        "reproduce": old_latest.get("source_script", ""),
        "framework_versions": fw_versions,
        "summary": {
            "rows": len(rows),
            "comparable_rows": summary.get("comparable_rows"),
            "sglang_diffusion_wins": summary.get("sglang_diffusion_wins"),
            "other_wins": summary.get("other_wins"),
        },
        "rows": rows,
    }


def main() -> int:
    old_latest = load(LATEST)
    historical = load(HISTORICAL)

    if old_latest.get("id") == "b200-cross-framework-20260701":
        runs = {s.get("run") or s.get("id") for s in historical.get("sections", [])}
        if old_latest["id"] not in runs:
            historical["sections"].insert(0, b200_to_history_section(old_latest))
            historical["updated_at"] = "2026-07-03"
            summary = historical.setdefault("summary", {})
            summary["sections"] = len(historical["sections"])
            dump(HISTORICAL, historical)
            print(f"migrated {old_latest['id']} into {HISTORICAL.name}")
        else:
            print(f"{old_latest['id']} already in {HISTORICAL.name}, skipping")

    latest = build_latest()
    dump(LATEST, latest)
    print(
        f"wrote {LATEST.name}: {latest['summary']['cases']} cases, "
        f"{latest['summary']['sglang_diffusion_wins']}/"
        f"{latest['summary']['comparable_rows']} comparable wins"
    )

    # The site's visible per-run view is the history tabs (the focused
    # "latest" hero was removed), so the H100 run must be a section too.
    historical = load(HISTORICAL)
    runs = {s.get("run") or s.get("id") for s in historical.get("sections", [])}
    if "h100x4-final-20260702" not in runs:
        historical["sections"][0:0] = [
            h100_single_e2e_section(latest),
            h100_throughput_section(latest),
        ]
        historical["updated_at"] = "2026-07-03"
        historical.setdefault("summary", {})["sections"] = len(historical["sections"])
        dump(HISTORICAL, historical)
        print(f"inserted h100x4-final-20260702 sections into {HISTORICAL.name}")
    else:
        print(f"h100x4-final-20260702 already in {HISTORICAL.name}, skipping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
