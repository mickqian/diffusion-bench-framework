"""Build merged JSON and Markdown reports for benchmark runs."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from diffusion_bench.generate_dashboard import (
    build_issue_report_comment,
    generate_dashboard,
)


FRAMEWORK_ORDER = {"sglang": 0, "vllm-omni": 1, "lightx2v": 2, "trtllm-visual": 3, "diffusers": 4}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _git_head() -> str | None:
    try:
        ret = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    return ret.stdout.strip() or None


def _case_order(config_path: Path) -> dict[str, int]:
    config = _load(config_path)
    return {case["id"]: idx for idx, case in enumerate(config.get("cases", []))}


def _entry_sort_key(case_rank: dict[str, int], entry: dict) -> tuple[int, int, str, str]:
    case_id = str(entry.get("case_id") or "")
    framework = str(entry.get("framework") or "")
    return (
        case_rank.get(case_id, 10_000),
        FRAMEWORK_ORDER.get(framework, 100),
        case_id,
        framework,
    )


def _merge_dict(dst: dict, src: dict | None) -> dict:
    if not isinstance(src, dict):
        return dst
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value)
        elif value is not None:
            dst[key] = copy.deepcopy(value)
    return dst


def _merge_entries(runs: list[tuple[Path, dict]], key: str, case_rank: dict[str, int]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for path, data in runs:
        for entry in data.get(key, []) or []:
            entry_key = (str(entry.get("case_id") or ""), str(entry.get("framework") or ""))
            cloned = copy.deepcopy(entry)
            cloned.setdefault("source_result", path.as_posix())
            by_key[entry_key] = cloned
    return sorted(by_key.values(), key=lambda entry: _entry_sort_key(case_rank, entry))


def merge_results(paths: list[Path], output_json: Path, config_path: Path, run_id: str | None = None) -> dict:
    runs = [(path, _load(path)) for path in paths]
    if not runs:
        raise ValueError("at least one result JSON is required")

    case_rank = _case_order(config_path)
    merged: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_head() or runs[-1][1].get("commit_sha") or "unknown",
        "run_id": run_id or output_json.stem,
        "hardware": {},
        "sglang_runtime": {},
        "framework_runtime": {},
        "benchmark_env": {},
        "benchmark_framework_args": {},
        "benchmark_modes": [],
        "torch_compile_disabled": None,
        "reproduce_script": "scripts/generate_h200_report_artifacts.sh",
        "source_results": [],
        "results": _merge_entries(runs, "results", case_rank),
        "throughput_results": _merge_entries(runs, "throughput_results", case_rank),
    }

    modes = set()
    for path, data in runs:
        merged["source_results"].append(
            {
                "path": path.as_posix(),
                "run_id": data.get("run_id"),
                "commit_sha": data.get("commit_sha"),
                "timestamp": data.get("timestamp"),
            }
        )
        if data.get("hardware"):
            merged["hardware"] = copy.deepcopy(data["hardware"])
        if data.get("sglang_runtime"):
            merged["sglang_runtime"] = copy.deepcopy(data["sglang_runtime"])
        _merge_dict(merged["framework_runtime"], data.get("framework_runtime"))
        _merge_dict(merged["benchmark_env"], data.get("benchmark_env"))
        _merge_dict(merged["benchmark_framework_args"], data.get("benchmark_framework_args"))
        if data.get("torch_compile_disabled") is not None:
            merged["torch_compile_disabled"] = bool(data["torch_compile_disabled"])
        modes.update(data.get("benchmark_modes") or [])
        if data.get("results"):
            modes.add("single_e2e")
        if data.get("throughput_results"):
            modes.add("throughput")

    merged["benchmark_modes"] = sorted(modes)
    _flag_isolated_request_anomalies(merged)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def _flag_isolated_request_anomalies(merged: dict, threshold: float = 1.8) -> None:
    # A single isolated request that is much slower than the steady-state
    # throughput p50 for the same case/framework is an isolated-request artifact
    # (idle/scheduler stall), not compute -- flag it so the report renders it as
    # an anomaly instead of a trustworthy latency. This catches the case the
    # harness's own repeat-spread check misses: every measured request stalled
    # (internally consistent but uniformly inflated), so there is no fast sample.
    p50_by_key: dict[tuple, float] = {}
    for entry in merged.get("throughput_results") or []:
        if entry.get("error"):
            continue
        metrics = entry.get("metrics") or {}
        p50 = metrics.get("latency_p50_s") or metrics.get("latency_p50")
        if p50:
            p50_by_key[(entry.get("case_id"), entry.get("framework"))] = float(p50)
    for entry in merged.get("results") or []:
        if entry.get("error"):
            continue
        metrics = entry.setdefault("metrics", {})
        if metrics.get("latency_unstable"):
            continue
        latency = entry.get("latency_s")
        p50 = p50_by_key.get((entry.get("case_id"), entry.get("framework")))
        if latency and p50 and float(latency) >= threshold * p50:
            metrics["latency_unstable"] = True
            metrics["latency_anomaly_reason"] = (
                f"single-request latency {float(latency):.1f}s is "
                f"{float(latency) / p50:.1f}x the steady-state throughput p50 "
                f"({p50:.1f}s) -- isolated-request stall, not compute"
            )
    return


def write_artifacts(
    merged: dict,
    dashboard_md: Path,
    issue_md: Path,
) -> None:
    dashboard_md.parent.mkdir(parents=True, exist_ok=True)
    dashboard, _ = generate_dashboard(merged, history=[], charts_dir=None)
    dashboard_md.write_text(dashboard, encoding="utf-8")

    issue_md.parent.mkdir(parents=True, exist_ok=True)
    issue_md.write_text(build_issue_report_comment(merged), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate merged diffusion benchmark report artifacts")
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/comparison_configs.json"), type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dashboard-md", required=True, type=Path)
    parser.add_argument("--issue-md", required=True, type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    merged = merge_results(args.results, args.output_json, args.config, args.run_id)
    write_artifacts(merged, args.dashboard_md, args.issue_md)
    print(args.output_json)
    print(args.dashboard_md)
    print(args.issue_md)


if __name__ == "__main__":
    main()
