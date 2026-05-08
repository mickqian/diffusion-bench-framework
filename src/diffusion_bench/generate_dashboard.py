"""Generate a Markdown dashboard for diffusion cross-framework comparisons.

Reads current comparison results + historical data from sgl-project/ci-data repo
and produces a Markdown report with tables and trend charts saved as PNG files.

Usage:
    diffusion-bench-dashboard \
        --results comparison-results.json \
        --output dashboard.md \
        --charts-dir comparison-charts/ \
        --history-dir history/           # optional, local history JSONs
        --fetch-history                  # fetch from GitHub API instead
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# History fetching (from sgl-project/ci-data repo via GitHub API)
# ---------------------------------------------------------------------------

CI_DATA_REPO_OWNER = "sgl-project"
CI_DATA_REPO_NAME = "ci-data"
CI_DATA_BRANCH = "main"
HISTORY_PREFIX = "diffusion-comparisons"
MAX_HISTORY_RUNS = 14

# Base URL for chart images pushed to sgl-project/ci-data
CHARTS_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{CI_DATA_REPO_OWNER}/{CI_DATA_REPO_NAME}"
    f"/{CI_DATA_BRANCH}/{HISTORY_PREFIX}/charts"
)


def _github_get(url: str, token: str) -> dict | list | None:
    """Simple GET to GitHub API."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"  Warning: GitHub API request failed ({e.code}): {url}")
        return None
    except Exception as e:
        print(f"  Warning: GitHub API request error: {e}")
        return None


def fetch_history_from_github(token: str) -> list[dict]:
    """Fetch recent comparison result JSONs from sgl-project/ci-data repo."""
    print("Fetching historical comparison data from GitHub...")
    url = (
        f"https://api.github.com/repos/{CI_DATA_REPO_OWNER}/{CI_DATA_REPO_NAME}"
        f"/contents/{HISTORY_PREFIX}?ref={CI_DATA_BRANCH}"
    )
    listing = _github_get(url, token)
    if not listing or not isinstance(listing, list):
        print("  No historical data found.")
        return []

    # Filter JSON files and sort by name (date prefix) descending
    json_files = sorted(
        [f for f in listing if f["name"].endswith(".json")],
        key=lambda f: f["name"],
        reverse=True,
    )[:MAX_HISTORY_RUNS]

    history = []
    for entry in json_files:
        raw_url = entry.get("download_url")
        if not raw_url:
            continue
        data = _github_get(raw_url, token)
        if data and isinstance(data, dict):
            history.append(data)
    print(f"  Loaded {len(history)} historical run(s).")
    return history


def load_history_from_dir(history_dir: str) -> list[dict]:
    """Load historical JSONs from a local directory."""
    if not os.path.isdir(history_dir):
        return []
    files = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )[:MAX_HISTORY_RUNS]
    history = []
    for fname in files:
        try:
            with open(os.path.join(history_dir, fname)) as f:
                history.append(json.load(f))
        except Exception:
            pass
    return history


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------


def _fmt_latency(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _fmt_speedup(sglang_lat: float | None, other_lat: float | None) -> str:
    if sglang_lat is None or other_lat is None or sglang_lat <= 0:
        return "N/A"
    ratio = other_lat / sglang_lat
    return f"{ratio:.2f}x"


def _short_date(ts: str) -> str:
    """Extract short date from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d")
    except Exception:
        return ts[:10]


def _short_sha(sha: str) -> str:
    return sha[:7] if sha and sha != "unknown" else "?"


def _assess_risk(
    cid: str,
    current_cases: dict[str, dict[str, float | None]],
    history: list[dict],
    other_frameworks: list[str],
) -> tuple[str, str]:
    """Assess risk for a given case, returning (emoji, reason).

    Rules (checked in order):
    - N/A latency → ❌ broken
    - History exists: SGLang latency >5% vs avg of last 3 runs → ⚠️ regression
    - Competitor exists & SGLang slower → 🔴 competitive risk
    - SGLang faster than all competitors by >20% → 🟢 strong advantage
    - SGLang faster than all competitors by ≤20% → 🟡 moderate advantage
    - Default → ✅ stable
    """
    sg_lat = current_cases.get(cid, {}).get("sglang")

    # Broken: sglang latency is N/A
    if sg_lat is None:
        return "❌", f"{cid}: SGLang latency is N/A (broken)"

    # Check regression against 3-run historical average
    if history:
        hist_lats: list[float] = []
        for run in history[:3]:
            run_cases = _extract_case_results(run)
            h_lat = run_cases.get(cid, {}).get("sglang")
            if h_lat is not None:
                hist_lats.append(h_lat)
        if hist_lats:
            avg_3 = sum(hist_lats) / len(hist_lats)
            if avg_3 > 0 and (sg_lat - avg_3) / avg_3 > 0.05:
                pct = (sg_lat - avg_3) / avg_3 * 100
                return (
                    "⚠️",
                    f"{cid}: SGLang regression +{pct:.1f}% vs 3-run avg "
                    f"({sg_lat:.2f}s vs {avg_3:.2f}s)",
                )

    # Check competitive risk
    if other_frameworks:
        competitor_lats: dict[str, float] = {}
        for ofw in other_frameworks:
            olat = current_cases.get(cid, {}).get(ofw)
            if olat is not None:
                competitor_lats[ofw] = olat

        if competitor_lats:
            # SGLang slower than any competitor?
            for ofw, olat in competitor_lats.items():
                if sg_lat > olat:
                    return (
                        "🔴",
                        f"{cid}: SGLang slower than {ofw} "
                        f"({sg_lat:.2f}s vs {olat:.2f}s)",
                    )

            # SGLang faster — check margin
            min_competitor = min(competitor_lats.values())
            advantage = (min_competitor - sg_lat) / min_competitor
            if advantage > 0.20:
                return "🟢", ""
            else:
                return "🟡", ""

    # Default: stable
    return "✅", ""


def _trend_emoji(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return ""
    diff_pct = (current - previous) / previous * 100
    if diff_pct < -2:
        return " :arrow_down:"  # faster (good)
    elif diff_pct > 2:
        return " :arrow_up:"  # slower (bad)
    return " :left_right_arrow:"


def _extract_case_results(run_data: dict) -> dict[str, dict[str, float | None]]:
    """Extract {case_id: {framework: latency}} from a run."""
    mapping: dict[str, dict[str, float | None]] = {}
    for r in run_data.get("results", []):
        cid = r["case_id"]
        fw = r["framework"]
        if cid not in mapping:
            mapping[cid] = {}
        mapping[cid][fw] = r.get("latency_s")
    return mapping


def _extract_throughput_results(run_data: dict) -> dict[str, dict[str, dict]]:
    mapping: dict[str, dict[str, dict]] = {}
    for r in run_data.get("throughput_results", []):
        cid = r["case_id"]
        fw = r["framework"]
        if cid not in mapping:
            mapping[cid] = {}
        mapping[cid][fw] = r.get("metrics", {})
    return mapping


def _sanitize_filename(name: str) -> str:
    """Sanitize a case ID to be a safe filename."""
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def generate_dashboard(
    current: dict,
    history: list[dict],
    charts_dir: str | None = None,
) -> tuple[str, list[str]]:
    """Generate full markdown dashboard.

    Returns (markdown_string, alert_reasons) where alert_reasons is a list of
    human-readable strings for cases that need attention (empty if all is well).

    If charts_dir is provided, saves chart PNGs as files to that directory
    and references them via raw.githubusercontent URLs. Otherwise, charts
    are omitted.

    Returns the markdown string.
    """
    lines: list[str] = []
    lines.append("# Diffusion Cross-Framework Performance Dashboard\n")
    ts = current.get("timestamp", datetime.now(timezone.utc).isoformat())
    sha = current.get("commit_sha", "unknown")
    lines.append(f"*Generated: {_short_date(ts)} | Commit: `{_short_sha(sha)}`*\n")

    current_cases = _extract_case_results(current)
    case_ids = list(current_cases.keys())

    # ---- Regression detection ----
    REGRESSION_THRESHOLD = 0.05  # 5%
    regressions: list[str] = []
    if history:
        prev_cases = _extract_case_results(history[0])
        for cid in case_ids:
            for fw in ("sglang", "vllm-omni"):
                cur = current_cases.get(cid, {}).get(fw)
                prev = prev_cases.get(cid, {}).get(fw)
                if cur and prev and prev > 0:
                    pct = (cur - prev) / prev
                    if pct > REGRESSION_THRESHOLD:
                        regressions.append(
                            f"**{cid}** ({fw}): {prev:.2f}s -> {cur:.2f}s "
                            f"(+{pct*100:.1f}%)"
                        )

    if regressions:
        lines.append("> [!WARNING]\n> **Performance Regression Detected**\n>")
        for reg in regressions:
            lines.append(f"> - {reg}")
        lines.append("\n")

    # Discover all frameworks present in results
    all_frameworks = []
    seen_fw = set()
    for r in current.get("results", []) + current.get("throughput_results", []):
        fw = r["framework"]
        if fw not in seen_fw:
            all_frameworks.append(fw)
            seen_fw.add(fw)
    # Ensure sglang is first
    if "sglang" in all_frameworks:
        all_frameworks.remove("sglang")
        all_frameworks.insert(0, "sglang")
    other_frameworks = [fw for fw in all_frameworks if fw != "sglang"]

    # ---- Section 1: Cross-Framework Comparison (current run) ----
    lines.append("## Cross-Framework Performance Comparison\n")

    # Compute risk assessments for all cases
    risk_map: dict[str, tuple[str, str]] = {}
    for cid in case_ids:
        risk_map[cid] = _assess_risk(cid, current_cases, history, other_frameworks)

    # Dynamic header
    header = "| Model | Risk |"
    sep = "|-------|------|"
    for fw in all_frameworks:
        header += f" {fw} (s) |"
        sep += "---------|"
    for ofw in other_frameworks:
        header += f" vs {ofw} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    # One row per case (deduplicated by case_id)
    seen_cases = set()
    for r in current.get("results", []):
        cid = r["case_id"]
        if cid in seen_cases:
            continue
        seen_cases.add(cid)

        case_fws = current_cases.get(cid, {})
        sg_lat = case_fws.get("sglang")

        risk_emoji, _ = risk_map.get(cid, ("✅", ""))
        row = f"| {r['model'].split('/')[-1]} | {risk_emoji} |"
        # Latency columns -- bold the fastest
        lats = {fw: case_fws.get(fw) for fw in all_frameworks}
        valid_lats = [v for v in lats.values() if v is not None]
        min_lat = min(valid_lats) if valid_lats else None
        for fw in all_frameworks:
            lat = lats[fw]
            if lat is not None and min_lat is not None and lat == min_lat:
                row += f" **{_fmt_latency(lat)}** |"
            else:
                row += f" {_fmt_latency(lat)} |"
        # Speedup columns
        for ofw in other_frameworks:
            row += f" {_fmt_speedup(sg_lat, case_fws.get(ofw))} |"
        lines.append(row)

    throughput_cases = _extract_throughput_results(current)
    if throughput_cases:
        lines.append("\n## High-Pressure Throughput\n")
        header = "| Model |"
        sep = "|-------|"
        for fw in all_frameworks:
            header += f" {fw} (req/s) | {fw} p95 (s) |"
            sep += "---------|---------|"
        lines.append(header)
        lines.append(sep)

        seen_cases = set()
        for r in current.get("throughput_results", []):
            cid = r["case_id"]
            if cid in seen_cases:
                continue
            seen_cases.add(cid)
            row = f"| {r['model'].split('/')[-1]} |"
            case_fws = throughput_cases.get(cid, {})
            for fw in all_frameworks:
                metrics = case_fws.get(fw, {})
                throughput = metrics.get("throughput_rps")
                p95 = metrics.get("latency_p95_s")
                row += f" {_fmt_latency(throughput)} | {_fmt_latency(p95)} |"
            lines.append(row)

    # ---- Section 2: Cross-Framework Speedup Trend (only if multiple frameworks) ----
    if history and other_frameworks:
        lines.append("\n## SGLang vs vLLM-Omni Speedup Over Time\n")

        header = "| Date |"
        sep = "|------|"
        for cid in case_ids:
            header += f" {cid} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        all_runs = [current] + history
        for run in all_runs:
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            row = f"| {date} |"
            for cid in case_ids:
                sg = run_cases.get(cid, {}).get("sglang")
                vl = run_cases.get(cid, {}).get("vllm-omni")
                row += f" {_fmt_speedup(sg, vl)} |"
            lines.append(row)

    # ---- Section 4: Matplotlib Trend Charts (saved as PNG files) ----
    if history and charts_dir:
        all_runs = list(reversed([current] + history))  # chronological order

        def _chart_label(run: dict) -> str:
            d = _short_date(run.get("timestamp", ""))
            s = _short_sha(run.get("commit_sha", ""))
            return f"{d}\n({s})"

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(charts_dir, exist_ok=True)

            # Per-case latency trend charts
            for cid in case_ids:
                labels = []
                sg_vals = []
                vl_vals = []
                for run in all_runs:
                    run_cases = _extract_case_results(run)
                    sg = run_cases.get(cid, {}).get("sglang")
                    vl = run_cases.get(cid, {}).get("vllm-omni")
                    if sg is None:
                        continue
                    labels.append(_chart_label(run))
                    sg_vals.append(sg)
                    vl_vals.append(vl)

                if not sg_vals:
                    continue

                has_vl = any(v is not None for v in vl_vals)
                fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

                # SGLang line
                ax.plot(
                    range(len(sg_vals)),
                    sg_vals,
                    "o-",
                    color="#2563eb",
                    linewidth=2,
                    markersize=6,
                    label="SGLang",
                )
                for i, v in enumerate(sg_vals):
                    ax.annotate(
                        f"{v:.2f}s",
                        (i, v),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#2563eb",
                    )

                # vLLM-Omni line (if data exists)
                if has_vl:
                    vl_clean = [v if v is not None else float("nan") for v in vl_vals]
                    ax.plot(
                        range(len(vl_clean)),
                        vl_clean,
                        "s--",
                        color="#dc2626",
                        linewidth=2,
                        markersize=5,
                        label="vLLM-Omni",
                    )
                    for i, v in enumerate(vl_vals):
                        if v is not None:
                            ax.annotate(
                                f"{v:.2f}s",
                                (i, v),
                                textcoords="offset points",
                                xytext=(0, -14),
                                ha="center",
                                fontsize=8,
                                color="#dc2626",
                            )

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel("Latency (s)")
                ax.set_title(f"Latency Trend -- {cid}", fontsize=11, fontweight="bold")
                ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
                ax.grid(True, alpha=0.3)
                all_vals = sg_vals + [v for v in vl_vals if v is not None]
                y_min = min(all_vals)
                y_max = max(all_vals)
                y_range = y_max - y_min if y_max > y_min else max(y_max * 0.1, 0.1)
                ax.set_ylim(
                    bottom=max(0, y_min - y_range * 0.3),
                    top=y_max + y_range * 0.3,
                )

                filename = f"latency_{_sanitize_filename(cid)}.png"
                chart_path = os.path.join(charts_dir, filename)
                fig.savefig(chart_path, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved chart: {chart_path}")

                chart_url = f"{CHARTS_RAW_BASE_URL}/{filename}"
                lines.append(f"\n### Latency Trend: {cid}\n")
                lines.append(f"![Latency Trend {cid}]({chart_url})\n")

            # Speedup trend chart (only if multiple frameworks)
            if other_frameworks:
                fig, ax = plt.subplots(figsize=(max(6, len(all_runs) * 1.2), 4))
                colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c"]
                for ci_idx, cid in enumerate(case_ids):
                    speedups = []
                    run_labels = []
                    for run in all_runs:
                        run_cases = _extract_case_results(run)
                        sg = run_cases.get(cid, {}).get("sglang")
                        vl = run_cases.get(cid, {}).get("vllm-omni")
                        if sg and vl and sg > 0:
                            speedups.append(vl / sg)
                        else:
                            speedups.append(None)
                        run_labels.append(_chart_label(run))
                    clean = [v if v is not None else float("nan") for v in speedups]
                    ax.plot(
                        range(len(clean)),
                        clean,
                        "o-",
                        color=colors[ci_idx % len(colors)],
                        linewidth=2,
                        markersize=5,
                        label=cid,
                    )

                ax.set_xticks(range(len(run_labels)))
                ax.set_xticklabels(run_labels, fontsize=7)
                ax.set_ylabel("Speedup (x)")
                ax.set_title(
                    "SGLang Speedup Over vLLM-Omni", fontsize=11, fontweight="bold"
                )
                ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
                ax.legend(loc="upper left", fontsize=7)
                ax.grid(True, alpha=0.3)

                filename = "speedup_trend.png"
                chart_path = os.path.join(charts_dir, filename)
                fig.savefig(chart_path, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved chart: {chart_path}")

                chart_url = f"{CHARTS_RAW_BASE_URL}/{filename}"
                lines.append("\n### Speedup Trend (SGLang vs vLLM-Omni)\n")
                lines.append(f"![Speedup Trend]({chart_url})\n")

        except ImportError:
            lines.append("\n*Charts unavailable (matplotlib not installed)*\n")

    # ---- SGLang Performance Trend (raw data table, at the end) ----
    if history:
        lines.append(f"\n## SGLang Performance Trend (Last {len(history) + 1} Runs)\n")

        header = "| Date | Commit |"
        sep = "|------|--------|"
        for cid in case_ids:
            header += f" {cid} (s) |"
            sep += "---------|"
        header += " Trend |"
        sep += "-------|"
        lines.append(header)
        lines.append(sep)

        all_runs = [current] + history
        for i, run in enumerate(all_runs):
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            sha_s = _short_sha(run.get("commit_sha", ""))
            row = f"| {date} | `{sha_s}` |"
            for cid in case_ids:
                lat = run_cases.get(cid, {}).get("sglang")
                row += f" {_fmt_latency(lat)} |"
            if i + 1 < len(all_runs):
                prev_cases = _extract_case_results(all_runs[i + 1])
                emojis = []
                for cid in case_ids:
                    cur = run_cases.get(cid, {}).get("sglang")
                    prev = prev_cases.get(cid, {}).get("sglang")
                    emojis.append(_trend_emoji(cur, prev))
                row += " ".join(emojis) + " |"
            else:
                row += " -- |"
            lines.append(row)

    # ---- Risk Notification ----
    alert_cases = [
        (cid, emoji, reason)
        for cid, (emoji, reason) in risk_map.items()
        if emoji in ("⚠️", "🔴", "❌")
    ]
    if alert_cases:
        lines.append("\n> [!CAUTION]")
        lines.append("> **Action Required — Performance Alert**")
        lines.append(">")
        lines.append("> The following cases need attention:")
        for _cid, _emoji, reason in alert_cases:
            lines.append(f"> - {reason}")
        lines.append("")

    # Footer
    lines.append("\n---")
    lines.append(
        "*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*"
    )

    alert_reasons = [reason for _, _, reason in alert_cases]
    return "\n".join(lines) + "\n", alert_reasons


def _md_cell(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if text else "-"


def _fmt_report_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return _md_cell(value)


def _fmt_report_rps(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return _md_cell(value)


def _fmt_report_ratio(value: object, baseline: object) -> str:
    try:
        numerator = float(value)
        denominator = float(baseline)
    except (TypeError, ValueError):
        return "-"
    if denominator <= 0:
        return "-"
    return f"{numerator / denominator:.3f}x"


def _case_id(entry: dict) -> str:
    return str(entry.get("case_id") or "")


def _framework_name(entry: dict) -> str:
    return str(entry.get("framework") or "")


def _framework_sort_key(framework: str) -> tuple[int, str]:
    preferred = {
        "sglang": 0,
        "vllm-omni": 1,
        "lightx2v": 2,
        "diffusers": 3,
    }
    return preferred.get(framework, 100), framework


def _group_results_by_case(entries: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for entry in entries:
        grouped.setdefault(_case_id(entry), {})[_framework_name(entry)] = entry
    return grouped


def _ordered_case_ids(results: dict) -> list[str]:
    ordered = []
    seen = set()
    for entry in results.get("results", []) + results.get("throughput_results", []):
        case_id = _case_id(entry)
        if case_id and case_id not in seen:
            ordered.append(case_id)
            seen.add(case_id)
    return ordered


def _format_dims(entry: dict) -> str:
    width = entry.get("width")
    height = entry.get("height")
    frames = entry.get("num_frames")
    if width and height and frames:
        return f"{width}x{height}x{frames}"
    if width and height:
        return f"{width}x{height}"
    return "-"


def _format_cfg(entry: dict) -> str:
    parts = []
    if entry.get("guidance_scale") is not None:
        parts.append(f"gs={entry['guidance_scale']}")
    if entry.get("guidance_scale_2") is not None:
        parts.append(f"gs2={entry['guidance_scale_2']}")
    if entry.get("true_cfg_scale") is not None:
        parts.append(f"true={entry['true_cfg_scale']}")
    if entry.get("negative_prompt_set"):
        parts.append("neg=1")
    return ",".join(parts) if parts else "-"


def _result_status(entry: dict | None) -> str:
    if not entry:
        return "-"
    metrics = entry.get("metrics") or {}
    failed_requests = int(metrics.get("failed_requests", 0) or 0)
    completed_requests = int(metrics.get("completed_requests", 0) or 0)
    num_requests = int(metrics.get("num_requests", 0) or 0)
    if failed_requests or (num_requests and completed_requests != num_requests):
        return "partial"
    error = str(entry.get("error") or "")
    if not error:
        return "ok"
    if "partial failure" in error:
        return "partial"
    return "failed"


def _throughput_p50(entry: dict | None) -> object:
    metrics = (entry or {}).get("metrics") or {}
    return (
        metrics.get("latency_p50_s")
        or metrics.get("latency_p50")
        or (entry or {}).get("latency_s")
    )


def _throughput_p95(entry: dict | None) -> object:
    metrics = (entry or {}).get("metrics") or {}
    return metrics.get("latency_p95_s") or metrics.get("latency_p95")


def _throughput_rps(entry: dict | None) -> object:
    return ((entry or {}).get("metrics") or {}).get("throughput_rps")


def _successful_metric(entry: dict | None, key: str) -> object:
    if _result_status(entry) != "ok":
        return None
    return (entry or {}).get(key)


def _successful_throughput_metric(entry: dict | None, getter) -> object:
    if _result_status(entry) != "ok":
        return None
    return getter(entry)


def build_issue_report_comment(results: dict) -> str:
    single_by_case = _group_results_by_case(results.get("results", []))
    throughput_by_case = _group_results_by_case(results.get("throughput_results", []))
    gpus = (results.get("hardware") or {}).get("gpus") or []
    gpu_model = "; ".join(sorted(set(gpus)))
    sglang_runtime = results.get("sglang_runtime") or {}

    lines = [
        f"## Diffusion Benchmark Data - {_md_cell(results.get('timestamp'))}",
        "",
        "| bench_commit | sglang_commit | sglang_version | run_id | gpu_count | gpu_model |",
        "| --- | --- | --- | --- | ---: | --- |",
        "| "
        + " | ".join(
            [
                _md_cell(results.get("commit_sha")),
                _md_cell(sglang_runtime.get("git_commit")),
                _md_cell(sglang_runtime.get("package_version")),
                _md_cell(results.get("run_id")),
                str(len(gpus)),
                _md_cell(gpu_model),
            ]
        )
        + " |",
        "",
        "Ratio columns are framework value divided by SGLang value for the same case.",
    ]

    for case_id in _ordered_case_ids(results):
        single_fws = single_by_case.get(case_id, {})
        throughput_fws = throughput_by_case.get(case_id, {})
        frameworks = sorted(
            set(single_fws) | set(throughput_fws), key=_framework_sort_key
        )
        case_entry = (
            single_fws.get("sglang")
            or throughput_fws.get("sglang")
            or single_fws.get(frameworks[0])
            or throughput_fws.get(frameworks[0])
        )
        sglang_single = single_fws.get("sglang")
        sglang_throughput = throughput_fws.get("sglang")
        sglang_single_latency = _successful_metric(sglang_single, "latency_s")
        sglang_p50 = _successful_throughput_metric(sglang_throughput, _throughput_p50)
        sglang_rps = _successful_throughput_metric(sglang_throughput, _throughput_rps)

        lines.extend(
            [
                "",
                f"### {_md_cell(case_id)}",
                "",
                "| model | task | dims | steps | cfg |",
                "| --- | --- | --- | ---: | --- |",
                "| "
                + " | ".join(
                    [
                        _md_cell(case_entry.get("model")),
                        _md_cell(case_entry.get("task")),
                        _md_cell(_format_dims(case_entry)),
                        _md_cell(case_entry.get("num_inference_steps")),
                        _md_cell(_format_cfg(case_entry)),
                    ]
                )
                + " |",
                "",
                "| framework | gpus | single_e2e_s | single/sglang | single_status | throughput_p50_s | p50/sglang | throughput_p95_s | throughput_rps | rps/sglang | throughput_status |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for framework in frameworks:
            single_entry = single_fws.get(framework)
            throughput_entry = throughput_fws.get(framework)
            entry = single_entry or throughput_entry or {}
            single_latency = (single_entry or {}).get("latency_s")
            throughput_p50 = _throughput_p50(throughput_entry)
            throughput_rps = _throughput_rps(throughput_entry)
            row = [
                _md_cell(framework),
                _md_cell(entry.get("num_gpus")),
                _fmt_report_float(single_latency),
                _fmt_report_ratio(single_latency, sglang_single_latency),
                _result_status(single_entry),
                _fmt_report_float(throughput_p50),
                _fmt_report_ratio(throughput_p50, sglang_p50),
                _fmt_report_float(_throughput_p95(throughput_entry)),
                _fmt_report_rps(throughput_rps),
                _fmt_report_ratio(throughput_rps, sglang_rps),
                _result_status(throughput_entry),
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _append_issue_report_comment(results: dict, repo: str, issue: str) -> None:
    import subprocess

    result = subprocess.run(
        ["gh", "issue", "comment", issue, "--repo", repo, "--body-file", "-"],
        input=build_issue_report_comment(results),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0:
        print(f"Appended formal benchmark report to issue #{issue}")
    else:
        print(
            f"Warning: failed to append report to issue #{issue} "
            f"(rc={result.returncode}): {result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion cross-framework comparison dashboard"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to comparison-results.json from current run",
    )
    parser.add_argument(
        "--output",
        default="dashboard.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--charts-dir",
        default="comparison-charts",
        help="Directory to save chart PNG files (default: comparison-charts/)",
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Local directory containing historical comparison JSONs",
    )
    parser.add_argument(
        "--fetch-history",
        action="store_true",
        help="Fetch history from ci-data GitHub repo",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Also write to $GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--report-issue",
        default=os.environ.get("DIFFUSION_BENCH_REPORT_ISSUE"),
        help="Append a data-only benchmark report comment to this existing issue number",
    )
    parser.add_argument(
        "--report-repo",
        default=os.environ.get("DIFFUSION_BENCH_REPORT_REPO")
        or os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repo for --report-issue, e.g. owner/name",
    )

    args = parser.parse_args()

    # Load current results
    with open(args.results) as f:
        current = json.load(f)
    print(f"Loaded current results: {len(current.get('results', []))} entries")

    # Load history
    history: list[dict] = []
    if args.fetch_history:
        token = os.environ.get("GH_PAT_FOR_NIGHTLY_CI_DATA") or os.environ.get(
            "GITHUB_TOKEN"
        )
        if token:
            history = fetch_history_from_github(token)
        else:
            print("Warning: No GitHub token available, skipping history fetch")
    elif args.history_dir:
        history = load_history_from_dir(args.history_dir)
        print(f"Loaded {len(history)} historical run(s) from {args.history_dir}")

    # Generate dashboard
    markdown, _alert_reasons = generate_dashboard(
        current, history, charts_dir=args.charts_dir
    )

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(markdown)
    print(f"Dashboard written to {args.output}")

    # Write to GitHub Step Summary
    if args.step_summary:
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(markdown)
            print("Dashboard appended to $GITHUB_STEP_SUMMARY")
        else:
            print("Warning: $GITHUB_STEP_SUMMARY not set, skipping")

    if args.report_issue:
        if not args.report_repo:
            raise SystemExit("--report-repo is required when --report-issue is set")
        _append_issue_report_comment(current, args.report_repo, args.report_issue)
    else:
        print("No report issue configured — skipping issue comment.")


if __name__ == "__main__":
    main()
