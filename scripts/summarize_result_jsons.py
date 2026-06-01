#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _first_error(result):
    error = result.get("error")
    if not error:
        return "-"
    return " ".join(str(error).split())


def _result_profile(result):
    metadata = result.get("framework_metadata") or {}
    return metadata.get("profile") or metadata.get("sglang_profile") or "-"


def _throughput_metric(result, key):
    return _fmt((result.get("metrics") or {}).get(key))


def main():
    parser = argparse.ArgumentParser(description="Summarize diffusion benchmark JSON result rows.")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.results:
        data = json.loads(path.read_text())
        for result in data.get("results", []):
            rows.append(
                [
                    result.get("case_id"),
                    result.get("framework"),
                    result.get("mode", "single_e2e"),
                    _result_profile(result),
                    _fmt(result.get("num_gpus")),
                    _fmt(result.get("latency_s")),
                    "-",
                    "-",
                    "-",
                    "-",
                    _first_error(result),
                    path.name,
                ]
            )
        for result in data.get("throughput_results", []):
            rows.append(
                [
                    result.get("case_id"),
                    result.get("framework"),
                    result.get("mode", "throughput"),
                    _result_profile(result),
                    _fmt(result.get("num_gpus")),
                    _fmt(result.get("latency_s")),
                    _throughput_metric(result, "throughput_rps"),
                    _throughput_metric(result, "latency_p50_s"),
                    _throughput_metric(result, "latency_p95_s"),
                    _throughput_metric(result, "latency_p99_s"),
                    _first_error(result),
                    path.name,
                ]
            )

    headers = [
        "case",
        "framework",
        "mode",
        "profile",
        "gpus",
        "latency_s",
        "qps",
        "p50_s",
        "p95_s",
        "p99_s",
        "error",
        "file",
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


if __name__ == "__main__":
    main()
