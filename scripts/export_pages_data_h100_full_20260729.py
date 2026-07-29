#!/usr/bin/env python3
"""Publish the H100x4 full 18-case cross-framework run to the Pages data files.

This supersedes the 2026-07-03 H100 run (9 image-only cases): same policy
(best lossless, compile on everywhere, latest-vs-latest), but the full
18-case matrix (8 image + 10 video) across sglang / vllm-omni / lightx2v /
trtllm-visual, both single_e2e and throughput modes.

Does two things, idempotently:

1. Rebuilds ``docs/data/latest-cross-framework.json`` from
   ``reports/h100x4-full-20260728/merged.json``.
2. Migrates the previous "latest" H100 run (2026-07-03) into
   ``docs/data/historical-cross-framework.json`` as its own section, and
   replaces the H100 section(s) there with this run.

Run ``python3 scripts/refresh_docs_data.py`` afterwards to refresh the inline
file:// preview snapshots embedded in docs/index.html.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "h100x4-full-20260728"
CONFIG = ROOT / "configs" / "comparison_configs.json"
LATEST = ROOT / "docs" / "data" / "latest-cross-framework.json"
HISTORICAL = ROOT / "docs" / "data" / "historical-cross-framework.json"

FRAMEWORK_ORDER = ["sglang", "vllm-omni", "lightx2v", "trtllm-visual"]

CASE_ORDER = [
    "flux1_dev_t2i_1024",
    "flux2_dev_t2i_1024",
    "qwen_image_2512_t2i_1024",
    "qwen_image_2512_t2i_1024_truecfg",
    "qwen_image_edit_2511",
    "zimage_turbo_t2i_1024",
    "ideogram4_t2i_1024_2gpu_tp",
    "cosmos3_nano_t2i_720p",
    "wan21_t2v_1_3b_480p",
    "wan21_i2v_14b_480p",
    "wan21_i2v_14b_720p",
    "wan22_t2v_a14b_720p",
    "wan22_i2v_a14b_720p",
    "wan22_ti2v_5b_704p",
    "ltx2_twostage_t2v",
    "ltx2.3_twostage_t2v_2gpus",
    "cosmos3_nano_t2v_720p_189f",
    "cosmos3_nano_i2v_720p_189f",
]

CASE_LABELS = {
    "flux1_dev_t2i_1024": "FLUX.1-dev T2I",
    "flux2_dev_t2i_1024": "FLUX.2-dev T2I",
    "qwen_image_2512_t2i_1024": "Qwen-Image-2512 T2I (no-CFG)",
    "qwen_image_2512_t2i_1024_truecfg": "Qwen-Image-2512 T2I (true-CFG)",
    "qwen_image_edit_2511": "Qwen-Image-Edit-2511",
    "zimage_turbo_t2i_1024": "Z-Image-Turbo T2I",
    "ideogram4_t2i_1024_2gpu_tp": "Ideogram-4 (FP8) T2I",
    "cosmos3_nano_t2i_720p": "Cosmos3-Nano T2I 720p",
    "wan21_t2v_1_3b_480p": "Wan2.1 T2V 1.3B 480p",
    "wan21_i2v_14b_480p": "Wan2.1 I2V 14B 480p",
    "wan21_i2v_14b_720p": "Wan2.1 I2V 14B 720p",
    "wan22_t2v_a14b_720p": "Wan2.2 T2V A14B 720p",
    "wan22_i2v_a14b_720p": "Wan2.2 I2V A14B 720p",
    "wan22_ti2v_5b_704p": "Wan2.2 TI2V 5B 704p",
    "ltx2_twostage_t2v": "LTX-2 two-stage T2V",
    "ltx2.3_twostage_t2v_2gpus": "LTX-2.3 two-stage T2V",
    "cosmos3_nano_t2v_720p_189f": "Cosmos3-Nano T2V 720p",
    "cosmos3_nano_i2v_720p_189f": "Cosmos3-Nano I2V 720p",
}

# Cells with a known, non-generic root cause: overrides the generic
# ok_rows/failed_rows classification below. "broken" = confirmed bug in the
# framework's own code (tracked separately, not a benchmark/config issue).
# "profile_issue" = our command profile needs work (framework itself likely
# supports the case); not the framework's fault, but not a clean data point.
KNOWN_ISSUES = {
    # NOTE 2026-07-29: the original (1,)-placeholder crashes on ideogram4 and
    # both Wan2.2-A14B variants were root-caused (offload_during_compile's
    # residency strategy kept releasing restored weights on dual-DiT use-site
    # switches) and fixed in sgl-project/sglang#32743; ideogram4 and
    # wan22_t2v single_e2e now carry measured numbers. What remains below is
    # what the fix exposed underneath: pure 80GB capacity limits.
    ("wan22_i2v_a14b_720p", "sglang"): (
        "failed",
        "Capacity: after the #32743 fix, I2V A14B still cannot load on "
        "4x80GB -- sibling ranks place 20-35GB of loader staging on each "
        "other's GPUs, and with both 14B experts plus the CLIP image "
        "encoder the load-time peak exceeds 80GB (a layerwise-offload "
        "profile fails the same way, so the limit is in the loading path, "
        "not steady-state residency). T2V (one fewer resident encoder, "
        "no image-cond channels) fits and wins at 209.9s. Tracked "
        "separately as an sglang loader fix.",
    ),
    ("wan22_t2v_a14b_720p", "sglang", "throughput"): (
        "failed",
        "Capacity at concurrency 2: with both 14B experts resident (the "
        "profile that makes single-request 209.9s the fastest of all "
        "frameworks), one request's 720p x 81f VAE decode overlapping the "
        "other's denoise deterministically exceeds 80GB (~73GB used, "
        "346MiB short). Not flaky -- a retry hits the same wall. Needs an "
        "upstream capacity lever (decode chunking under pressure or "
        "partial residency) to lift.",
    ),
    ("cosmos3_nano_t2i_720p", "vllm-omni"): (
        "profile_issue",
        "vllm-omni stage config is missing a now-required "
        "'engine_input_source' field (orchestrator init fails before any "
        "request runs); the correct value for vllm-omni's current main "
        "branch was not determined in this run -- needs a profile update, "
        "not a framework limitation.",
    ),
    ("ltx2_twostage_t2v", "vllm-omni"): (
        "profile_issue",
        "vllm-omni rejects the current serve_args combination: "
        "'LTX CFG parallelism only supports CFG-only guidance without "
        "rescale.' The profile needs to drop or reconfigure the guidance "
        "rescale path for LTX-2 CFG-parallel; not attempted in this run.",
    ),
    ("cosmos3_nano_t2i_720p", "trtllm-visual"): (
        "profile_issue",
        "trtllm-visual server starts and passes health checks but returns "
        "400 Bad Request on every generation request; likely a request "
        "parameter mismatch for this case, not investigated further.",
    ),
    ("flux2_dev_t2i_1024", "trtllm-visual"): (
        "profile_issue",
        "trtllm-visual OOMs at startup (CUDA out of memory, ~79GB/79GB "
        "used on a single H100) loading FLUX.2; likely needs an explicit "
        "CPU-offload or multi-GPU profile for this model, not attempted "
        "in this run.",
    ),
}

FRAMEWORKS = {
    "sglang": {
        "label": "SGLang-Diffusion",
        "commit": "161fffe (+#32743 fix for re-measured cells)",
        "note": (
            "origin/main HEAD; compile or measured-eager per case per the "
            "tuned command_profiles; Ulysses/SP or CFG-parallel per case. "
            "Cells re-measured on 2026-07-29 additionally carry the "
            "sgl-project/sglang#32743 dual-DiT residency fix."
        ),
    },
    "vllm-omni": {
        "label": "vLLM-Omni",
        "note": "main HEAD (git+https://github.com/vllm-project/vllm-omni.git); compile on",
    },
    "lightx2v": {
        "label": "LightX2V",
        "note": "main HEAD (git+https://github.com/ModelTC/LightX2V.git); FA2/FA3 exact; compile on",
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
    "version_policy": (
        "latest-vs-latest: sglang runs origin/main HEAD, every competitor "
        "runs its newest main/release line (no pinned snapshots)"
    ),
    "qwen_note": (
        "Qwen-Image is reported under both semantics: no-CFG (guidance 1.0, "
        "one forward/step) and true-CFG (true_cfg_scale 4 + negative prompt, "
        "two forwards/step, served with CFG-parallel). The two rows are not "
        "comparable to each other."
    ),
    "bimodal_latency_note": (
        "During this run, two compile-mode sglang cells (flux1, qwen "
        "no-CFG) showed a bimodal latency distribution across independent "
        "server restarts (each state internally consistent over its own 5 "
        "repeats -- not per-request noise); raw evidence in "
        "reports/h100x4-full-20260728/bimodal-evidence-*.json. The "
        "published flux1 number is the reproduced fast-mode measurement. "
        "The qwen no-CFG cell was subsequently re-measured under its final "
        "eager profile (see the profile's policy_exception), where repeats "
        "are tight (4.55-4.57s) and the bimodal observation no longer "
        "applies. Other sglang image cells were independently re-verified "
        "stable (<2% delta). Root cause of the compile-mode bimodality is "
        "tracked separately."
    ),
    "post_run_fixes_note": (
        "2026-07-29 follow-up: the initial pass surfaced two sglang bugs "
        "and one measured mis-configuration; all three were fixed and the "
        "affected cells re-measured on the same machine and sglang base "
        "(161fffe) plus the fix commit. (1) ideogram4 and Wan2.2-A14B "
        "crashed reading (1,)-placeholder weights -- root-caused to "
        "offload_during_compile's residency interaction and fixed in "
        "sgl-project/sglang#32743; ideogram4 now measures 4.00s and "
        "wan22 T2V 209.9s (fastest of all frameworks). (2) qwen no-CFG "
        "was published at 5.74s under torch.compile, which is a measured "
        "net loss for this model (inductor GEMMs lose to cublas nvjet, "
        "and the #31849 dynamo workaround additionally drops the fused "
        "qk-norm-rope kernel when compiling); the profile now runs true "
        "eager at 4.56s, near-parity with vLLM's 4.48s. The remaining "
        "Wan2.2-A14B gaps are pure 80GB capacity limits, documented "
        "per-cell."
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


def _rows_by_mode(merged: dict, mode: str) -> dict:
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


def _classify_cell(case_id, fw, ok_rows, failed_rows, config_cases, mode) -> dict | None:
    """Return a non-OK cell dict, or None if this is a normal OK/measured cell."""
    key = (case_id, fw)
    mode_key = (case_id, fw, mode)
    if mode_key in KNOWN_ISSUES:
        status, reason = KNOWN_ISSUES[mode_key]
        return {"status": status, "reason": reason}
    if key in KNOWN_ISSUES:
        status, reason = KNOWN_ISSUES[key]
        return {"status": status, "reason": reason}
    if key in failed_rows:
        return {"status": "failed", "reason": failed_rows[key]}
    if key in ok_rows:
        return None
    case_cfg = config_cases[case_id]
    statuses = case_cfg.get("report_framework_statuses") or {}
    reasons = case_cfg.get("report_framework_reasons") or {}
    status = statuses.get(fw, "not_run")
    cell = {"status": status}
    if reasons.get(fw):
        cell["reason"] = reasons[fw]
    return cell


def build_single_e2e_rows(merged: dict, config_cases: dict) -> tuple[list[dict], int, int]:
    ok_rows, failed_rows = _rows_by_mode(merged, "single_e2e")
    rows = []
    wins = comparable = 0
    for case_id in CASE_ORDER:
        sgl = ok_rows.get((case_id, "sglang"))
        cells = {}
        competitor_lat = {}
        for fw in FRAMEWORK_ORDER:
            override = _classify_cell(case_id, fw, ok_rows, failed_rows, config_cases, "single_e2e")
            if override is not None:
                cells[fw] = override
                continue
            row = ok_rows[(case_id, fw)]
            cells[fw] = {
                "client_latency_s": round(row["latency_s"], 3),
                "profile": (row.get("framework_metadata") or {}).get("profile") or "default",
            }
            if fw != "sglang":
                competitor_lat[fw] = row["latency_s"]
        if sgl is None:
            entry = {
                "case_id": case_id,
                "case": CASE_LABELS[case_id],
                "details": case_details(config_cases[case_id]),
                "gpus": f"{config_cases[case_id]['num_gpus']} GPU",
                "winner": min(competitor_lat, key=competitor_lat.get) if competitor_lat else "n/a",
                "cells": cells,
            }
            if competitor_lat:
                comparable += 1
            rows.append(entry)
            continue
        entry = {
            "case_id": case_id,
            "case": CASE_LABELS[case_id],
            "details": case_details(config_cases[case_id]),
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
    return rows, wins, comparable


def build_throughput_rows(merged: dict, config_cases: dict, latest_rows: dict) -> tuple[list[dict], int, int]:
    ok_rows, failed_rows = _rows_by_mode(merged, "throughput")
    rows = []
    wins = comparable = 0
    for case_id in CASE_ORDER:
        cells = {}
        qps_by_fw = {}
        base_cells = latest_rows[case_id]["cells"]
        for fw in FRAMEWORK_ORDER:
            override = _classify_cell(case_id, fw, ok_rows, failed_rows, config_cases, "throughput")
            if override is not None:
                # carry a friendlier status through if the single_e2e cell
                # already explains this fw/case combo and throughput has no
                # extra info to add (avoids duplicating long reason strings).
                prior = base_cells.get(fw, {})
                if override.get("status") == "failed" and prior.get("status") in (
                    "broken",
                    "profile_issue",
                    "not_run",
                ):
                    cells[fw] = {"status": prior["status"]}
                    if prior.get("reason"):
                        cells[fw]["reason"] = prior["reason"]
                else:
                    cells[fw] = override
                continue
            row = ok_rows[(case_id, fw)]
            metrics = row.get("metrics") or {}
            qps = metrics.get("throughput_rps")
            p95 = metrics.get("latency_p95_s")
            qps_by_fw[fw] = qps
            cells[fw] = {"status": "ok", "qps": round(qps, 3) if qps is not None else None}
            if p95 is not None:
                cells[fw]["p95_s"] = round(p95, 3)
        if not qps_by_fw:
            rows.append(
                {
                    "case_id": case_id,
                    "case": CASE_LABELS[case_id],
                    "details": case_details(config_cases[case_id]),
                    "mode": "throughput",
                    "winner": "n/a",
                    "cells": cells,
                }
            )
            continue
        for fw, qps in qps_by_fw.items():
            if fw != "sglang" and qps:
                cells[fw]["ratio_to_sglang"] = (
                    round(qps / qps_by_fw["sglang"], 3) if qps_by_fw.get("sglang") else None
                )
        winner = max(qps_by_fw, key=lambda k: qps_by_fw[k] or 0)
        if len(qps_by_fw) > 1:
            comparable += 1
            if winner == "sglang":
                wins += 1
        rows.append(
            {
                "case_id": case_id,
                "case": CASE_LABELS[case_id],
                "details": case_details(config_cases[case_id]),
                "mode": "throughput",
                "winner": winner,
                "cells": cells,
            }
        )
    return rows, wins, comparable


def build_latest() -> dict:
    merged = load(REPORT / "merged.json")
    config_cases = {c["id"]: c for c in load(CONFIG)["cases"]}
    single_rows, wins, comparable = build_single_e2e_rows(merged, config_cases)
    return {
        "id": "h100x4-full-20260728",
        "title": (
            "H100 SGLang-Diffusion vs vLLM-Omni vs LightX2V vs TensorRT-LLM "
            "VisualGen (full 18-case matrix, best lossless, compile on)"
        ),
        "updated_at": "2026-07-29",
        "source_report": "reports/h100x4-full-20260728/",
        "source_script": "configs/benchmark/ (explicit matrix) + scripts/build_benchmark_config.py",
        "hardware": {"label": "4x NVIDIA H100 80GB", "gpu_name": "NVIDIA H100 80GB HBM3", "gpus": 4},
        "policy": POLICY,
        "framework_order": FRAMEWORK_ORDER,
        "frameworks": FRAMEWORKS,
        "summary": {
            "cases": len(single_rows),
            "comparable_rows": comparable,
            "sglang_diffusion_wins": wins,
            "other_wins": comparable - wins,
            "note": (
                "Steady-state single-request latency on H100, best lossless "
                "config per framework (compile by default, measured-eager "
                "where compile is a proven net loss), latest main/HEAD for "
                "every framework. Full 18-case image+video matrix. The two "
                "sglang bugs the initial pass surfaced were fixed "
                "(sgl-project/sglang#32743) and the affected cells "
                "re-measured -- see policy.post_run_fixes_note; remaining "
                "gaps are documented per-cell (mostly 80GB capacity limits "
                "and competitor profile issues). Throughput tables live in "
                "the report."
            ),
        },
        "rows": single_rows,
        "_throughput_rows_and_stats": build_throughput_rows(merged, config_cases, {r["case_id"]: r for r in single_rows}),
    }


def _h100_framework_versions() -> dict:
    return {
        "sglang": "161fffe (origin/main)",
        "vllm-omni": "main HEAD",
        "lightx2v": "main HEAD",
        "trtllm-visual": "1.3.0rc18",
    }


def h100_single_e2e_section(latest: dict) -> dict:
    rows = []
    for row in latest["rows"]:
        gpus_str = str(row.get("gpus", "0"))
        gpus = int(gpus_str.split()[0]) if gpus_str.split()[0].isdigit() else 0
        sgl_cell = row["cells"].get("sglang") or {}
        sgl_lat = sgl_cell.get("client_latency_s")
        cells = {}
        for fw, cell in row["cells"].items():
            if "client_latency_s" in cell:
                lat = cell["client_latency_s"]
                new_cell = {
                    "gpus": gpus,
                    "profile": cell.get("profile", "default"),
                    "status": "ok",
                    "latency_s": lat,
                }
                if sgl_lat:
                    new_cell["ratio_to_sglang"] = round(lat / sgl_lat, 3)
                cells[fw] = new_cell
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
        "id": "h100x4-full-20260728_single_e2e",
        "run": "h100x4-full-20260728",
        "run_label": "H100 full 18-case matrix (best-lossless)",
        "date": "2026-07-29",
        "gpu": "4x NVIDIA H100 80GB",
        "title": "H100 full 18-case matrix (best-lossless)",
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


def h100_throughput_section(latest: dict, throughput_stats: tuple) -> dict:
    rows, wins, comparable = throughput_stats
    return {
        "id": "h100x4-full-20260728_throughput",
        "run": "h100x4-full-20260728",
        "run_label": "H100 full 18-case matrix (best-lossless)",
        "date": "2026-07-29",
        "gpu": "4x NVIDIA H100 80GB",
        "title": "H100 high-pressure throughput",
        "subtitle": "4 requests at concurrency 2 (image and video); qps and p95 from client-observed completions.",
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


def previous_h100_to_history_section(old_latest: dict) -> dict:
    """Archive the 2026-07-03 H100 (9-case) run as its own history section."""
    fw_versions = {}
    for fw, meta in (old_latest.get("frameworks") or {}).items():
        fw_versions[fw] = meta.get("commit") or meta.get("vllm_omni") or meta.get("tensorrt_llm") or ""
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
        "run_label": "H100 best-lossless (measured-fastest, 9-case)",
        "date": old_latest.get("updated_at", ""),
        "gpu": old_latest.get("hardware", {}).get("label", ""),
        "title": "H100 best-lossless (measured-fastest, 9-case)",
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

    if old_latest.get("id") and old_latest["id"] != "h100x4-full-20260728":
        runs = {s.get("run") or s.get("id") for s in historical.get("sections", [])}
        if old_latest["id"] not in runs:
            historical["sections"].insert(0, previous_h100_to_history_section(old_latest))
            summary = historical.setdefault("summary", {})
            summary["sections"] = len(historical["sections"])
            print(f"archived {old_latest['id']} into {HISTORICAL.name}")
        else:
            print(f"{old_latest['id']} already in {HISTORICAL.name}, skipping archive")

    latest = build_latest()
    throughput_stats = latest.pop("_throughput_rows_and_stats")
    dump(LATEST, latest)
    print(
        f"wrote {LATEST.name}: {latest['summary']['cases']} cases, "
        f"{latest['summary']['sglang_diffusion_wins']}/"
        f"{latest['summary']['comparable_rows']} comparable wins"
    )

    # Replace semantics: this run supersedes any prior h100x4-* section(s)
    # (previous H100 run was already archived above under its own id).
    historical["sections"] = [
        s for s in historical.get("sections", [])
        if not str(s.get("run") or s.get("id", "")).startswith("h100x4-full-")
    ]
    historical["sections"][0:0] = [
        h100_single_e2e_section(latest),
        h100_throughput_section(latest, throughput_stats),
    ]
    historical["updated_at"] = "2026-07-29"
    historical.setdefault("summary", {})["sections"] = len(historical["sections"])
    dump(HISTORICAL, historical)
    print(f"replaced H100 sections in {HISTORICAL.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
