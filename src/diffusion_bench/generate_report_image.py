"""Render a compact PNG/SVG comparison poster from merged benchmark results."""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by CLI environment
    raise SystemExit("Pillow is required to generate report images") from exc


FRAMEWORK_ORDER = ("sglang", "vllm-omni", "lightx2v")
FRAMEWORK_LABELS = {
    "sglang": "SGLang-Diffusion",
    "vllm-omni": "vLLM-Omni",
    "lightx2v": "LightX2V",
}
FRAMEWORK_COLORS = {
    "sglang": "#e8524d",
    "vllm-omni": "#4f7fad",
    "lightx2v": "#59a14f",
}
FRAMEWORK_TINTS = {
    "sglang": "#fde5e3",
    "vllm-omni": "#e6eef7",
    "lightx2v": "#eaf7ed",
}

PREFERRED_CASE_ORDER = (
    "flux1_dev_t2i_1024",
    "flux2_dev_t2i_1024",
    "qwen_image_2512_t2i_1024",
    "qwen_image_edit_2511",
    "zimage_turbo_t2i_1024",
    "ltx2_twostage_t2v",
    "ltx2.3_twostage_t2v_2gpus",
    "wan21_t2v_1_3b_480p",
    "wan21_i2v_14b_480p",
    "wan22_ti2v_5b_704p",
    "wan21_i2v_14b_720p",
    "wan22_t2v_a14b_720p",
    "wan22_i2v_a14b_720p",
)
CASE_LABELS = {
    "flux1_dev_t2i_1024": "FLUX.1-dev T2I",
    "flux2_dev_t2i_1024": "FLUX.2-dev T2I",
    "qwen_image_2512_t2i_1024": "Qwen-Image-2512 T2I",
    "qwen_image_edit_2511": "Qwen-Image-Edit-2511",
    "zimage_turbo_t2i_1024": "Z-Image-Turbo T2I",
    "ltx2_twostage_t2v": "LTX-2 two-stage T2V",
    "ltx2.3_twostage_t2v_2gpus": "LTX-2.3 two-stage T2V",
    "wan21_t2v_1_3b_480p": "Wan2.1 T2V 1.3B 480p",
    "wan21_i2v_14b_480p": "Wan2.1 I2V 14B 480p",
    "wan22_ti2v_5b_704p": "Wan2.2 TI2V 5B 704p",
    "wan21_i2v_14b_720p": "Wan2.1 I2V 14B 720p",
    "wan22_t2v_a14b_720p": "Wan2.2 T2V A14B 720p",
    "wan22_i2v_a14b_720p": "Wan2.2 I2V A14B 720p",
}

BG = "#f6f8fb"
PANEL = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#dfe4ea"
GRID = "#eef1f5"
DANGER = "#b91c1c"
GOOD = "#166534"
PENDING = "#f3f4f6"
FAIL_BG = "#fff1f2"
SUPPORTED_BG = "#f0fdf4"
NO_PROFILE_BG = "#eef2ff"
CARD_DARK = "#111827"

WIDTH = 2000
MARGIN = 72
PANEL_X = 72
PANEL_W = 1856
PAD = 32
CASE_X = PANEL_X + PAD
METRIC_XS = {
    "sglang": 780,
    "vllm-omni": 1078,
    "lightx2v": 1376,
}
METRIC_W = 258
WINNER_X = 1670
WINNER_W = 208
ROW_SINGLE_H = 90
ROW_THROUGHPUT_H = 118
PANEL_HEADER_H = 102
HEADER_H = 190


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else None,
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else None,
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_TITLE = _font(48, bold=True)
FONT_SUBTITLE = _font(27)
FONT_LEGEND = _font(23, bold=True)
FONT_STAT_NUM = _font(34, bold=True)
FONT_STAT_LABEL = _font(22, bold=True)
FONT_STAT_META = _font(17)
FONT_SECTION = _font(31, bold=True)
FONT_HEADER = _font(18, bold=True)
FONT_CASE = _font(22, bold=True)
FONT_META = _font(16, bold=True)
FONT_VALUE = _font(22, bold=True)
FONT_RATIO = _font(17, bold=True)
FONT_STATUS = _font(18, bold=True)
FONT_STATUS_SMALL = _font(15, bold=True)
FONT_FOOTER = _font(18, bold=True)


def _case_configs(config: dict) -> dict[str, dict]:
    return {case["id"]: case for case in config.get("cases", [])}


def _ordered_cases(config: dict, results: dict) -> list[str]:
    available = set()
    for entry in results.get("results", []) + results.get("throughput_results", []):
        case_id = entry.get("case_id")
        if case_id:
            available.add(case_id)
    for case in config.get("cases", []):
        case_id = case.get("id")
        frameworks = case.get("frameworks") or {}
        if case_id and "sglang" in frameworks and len(frameworks) >= 2:
            available.add(case_id)

    ordered = [case_id for case_id in PREFERRED_CASE_ORDER if case_id in available]
    seen = set(ordered)
    for case in config.get("cases", []):
        case_id = case.get("id")
        if case_id in available and case_id not in seen:
            ordered.append(case_id)
            seen.add(case_id)
    return ordered


def _by_case(entries: Iterable[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for entry in entries:
        framework = entry.get("framework")
        if framework in FRAMEWORK_ORDER:
            grouped.setdefault(entry.get("case_id", ""), {})[framework] = entry
    return grouped


def _ok(entry: dict | None) -> bool:
    return bool(entry) and not entry.get("error")


def _successful_latency(entry: dict | None) -> float | None:
    if not _ok(entry):
        return None
    value = entry.get("latency_s")
    return float(value) if value is not None else None


def _throughput_metrics(
    entry: dict | None,
) -> tuple[float | None, float | None, float | None]:
    if not _ok(entry):
        return None, None, None
    metrics = entry.get("metrics") or {}
    p50 = metrics.get("latency_p50_s") or metrics.get("latency_p50")
    p99 = metrics.get("latency_p99_s") or metrics.get("latency_p99")
    qps = metrics.get("throughput_rps") or metrics.get("throughput_qps")
    return (
        float(p50) if p50 is not None else None,
        float(p99) if p99 is not None else None,
        float(qps) if qps is not None else None,
    )


def _case_entry(case_id: str, entries: dict[str, dict], case_cfg: dict | None) -> dict:
    if entries.get("sglang"):
        return entries["sglang"]
    if entries:
        return next(iter(entries.values()))
    return case_cfg or {}


def _clean_number(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _case_title(case_id: str) -> str:
    if case_id in CASE_LABELS:
        return CASE_LABELS[case_id]
    return case_id.replace("_", " ")


def _case_meta(entry: dict, case_cfg: dict | None) -> str:
    parts = []
    width = entry.get("width") or (case_cfg or {}).get("width")
    height = entry.get("height") or (case_cfg or {}).get("height")
    frames = entry.get("num_frames") or (case_cfg or {}).get("num_frames")
    if width and height and frames:
        parts.append(f"{width}x{height}x{frames}f")
    elif width and height:
        parts.append(f"{width}x{height}")

    steps = entry.get("num_inference_steps") or (case_cfg or {}).get("num_inference_steps")
    if steps is not None:
        parts.append(f"{_clean_number(steps)} steps")

    guidance = entry.get("guidance_scale")
    if guidance is None:
        guidance = (case_cfg or {}).get("guidance_scale")
    guidance2 = entry.get("guidance_scale_2")
    if guidance2 is None:
        guidance2 = (case_cfg or {}).get("guidance_scale_2")
    true_cfg = entry.get("true_cfg_scale")
    if true_cfg is None:
        true_cfg = (case_cfg or {}).get("true_cfg_scale")
    if guidance is not None and guidance2 is not None:
        parts.append(f"cfg {_clean_number(guidance)}/cfg2 {_clean_number(guidance2)}")
    elif guidance is not None:
        parts.append(f"cfg {_clean_number(guidance)}")
    elif true_cfg is not None:
        parts.append(f"true cfg {_clean_number(true_cfg)}")

    num_gpus = entry.get("num_gpus") or (case_cfg or {}).get("num_gpus")
    if num_gpus:
        parts.append(f"{num_gpus} GPU")
    return " - ".join(parts) if parts else "-"


def _hardware_label(results: dict) -> str:
    override = (results.get("hardware") or {}).get("hardware_profile_override")
    if override:
        return str(override).upper()
    gpus = (results.get("hardware") or {}).get("gpus") or []
    if gpus and any(name in str(gpus[0]) for name in ("GB300", "GB200", "B300", "B200")):
        return "Blackwell"
    if gpus and "H200" in str(gpus[0]):
        return "H200"
    if gpus and "H100" in str(gpus[0]):
        return "H100"
    return "GPU"


def _single_winner(case_entries: dict[str, dict]) -> str | None:
    values = {
        framework: _successful_latency(case_entries.get(framework))
        for framework in FRAMEWORK_ORDER
    }
    values = {key: value for key, value in values.items() if value is not None}
    if len(values) < 2:
        return None
    return min(values, key=lambda key: values[key])


def _throughput_winner(case_entries: dict[str, dict]) -> str | None:
    values = {
        framework: _throughput_metrics(case_entries.get(framework))[2]
        for framework in FRAMEWORK_ORDER
    }
    values = {key: value for key, value in values.items() if value is not None}
    if len(values) < 2:
        return None
    return max(values, key=lambda key: values[key])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _rounded(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_header(
    draw: ImageDraw.ImageDraw,
    results: dict,
    single_summary: tuple[int, int, int],
    throughput_summary: tuple[int, int],
    throughput_meta: str,
) -> None:
    _rounded(draw, (0, 0, WIDTH, HEADER_H), 0, PANEL)
    draw.line((0, HEADER_H, WIDTH, HEADER_H), fill="#e5e7eb", width=1)

    _draw_text(draw, (MARGIN, 32), "SGLang-Diffusion vs Other Frameworks", FONT_TITLE)
    compile_label = (
        "torch compile disabled"
        if results.get("torch_compile_disabled", True)
        else "torch compile allowed"
    )
    subtitle = (
        f"{_hardware_label(results)} - single request latency + high-pressure "
        f"throughput - no cache - {compile_label}"
    )
    _draw_text(draw, (MARGIN, 95), subtitle, FONT_SUBTITLE, "#4b5563")

    for framework, x in zip(FRAMEWORK_ORDER, (MARGIN, 360, 590), strict=True):
        _rounded(draw, (x, 154, x + 24, 178), 6, FRAMEWORK_COLORS[framework])
        _draw_text(draw, (x + 36, 151), FRAMEWORK_LABELS[framework], FONT_LEGEND, "#374151")

    wins, comparable, pending = single_summary
    qps_wins, qps_comparable = throughput_summary
    _draw_stat_card(
        draw,
        1308,
        f"{wins}/{comparable}" if comparable else "0/0",
        "single req wins",
        f"{comparable + pending} rows - {pending} pending",
    )
    _draw_stat_card(
        draw,
        1626,
        f"{qps_wins}/{qps_comparable}" if qps_comparable else "0/0",
        "QPS wins",
        throughput_meta,
    )


def _draw_stat_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    value: str,
    label: str,
    meta: str,
) -> None:
    _rounded(draw, (x, 46, x + 288, 150), 18, CARD_DARK)
    _draw_text(draw, (x + 30, 58), value, FONT_STAT_NUM, "#ffffff")
    _draw_text(draw, (x + 30, 96), label, FONT_STAT_LABEL, "#ffffff")
    _draw_text(draw, (x + 30, 125), meta, FONT_STAT_META, "#d1d5db")


def _draw_panel_header(
    draw: ImageDraw.ImageDraw,
    y: int,
    title: str,
    hint: str,
    winner_header: str,
) -> None:
    _draw_text(draw, (PANEL_X + PAD, y + 26), title, FONT_SECTION)
    _draw_text(draw, (PANEL_X + PANEL_W - PAD, y + 33), hint, FONT_HEADER, MUTED, anchor="ra")
    header_y = y + 74
    _draw_text(draw, (CASE_X, header_y), "Case", FONT_HEADER, MUTED)
    for framework, x in METRIC_XS.items():
        _draw_text(draw, (x + 12, header_y), FRAMEWORK_LABELS[framework], FONT_HEADER, MUTED)
    _draw_text(draw, (WINNER_X + 8, header_y), winner_header, FONT_HEADER, MUTED)
    draw.line((CASE_X, y + PANEL_HEADER_H, PANEL_X + PANEL_W - PAD, y + PANEL_HEADER_H), fill="#e5e7eb", width=1)


def _draw_case_cell(
    draw: ImageDraw.ImageDraw,
    case_id: str,
    entry: dict,
    case_cfg: dict | None,
    y: int,
) -> None:
    _draw_text(draw, (CASE_X, y + 18), _case_title(case_id), FONT_CASE)
    _draw_text(draw, (CASE_X, y + 49), _case_meta(entry, case_cfg), FONT_META, MUTED)


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s" if value < 10 else f"{value:.2f}s"


def _draw_missing(draw: ImageDraw.ImageDraw, framework: str, y: int, row_h: int) -> None:
    x = METRIC_XS[framework]
    _draw_text(draw, (x + METRIC_W / 2, y + row_h / 2 - 2), "-", FONT_VALUE, "#9ca3af", anchor="mm")


def _missing_status(case_cfg: dict | None, framework: str) -> str:
    statuses = (case_cfg or {}).get("report_framework_statuses") or {}
    if framework in statuses:
        return str(statuses[framework])
    if case_cfg and framework in (case_cfg.get("frameworks") or {}):
        return "not_run"
    return "no_profile"


def _short_error_label(entry: dict) -> str:
    error = str(entry.get("error") or "")
    if "server exited" in error or "health check" in error:
        return "server failed"
    if "task FAILED" in error or "task failed" in error:
        return "task failed"
    return "failed"


def _draw_status_cell(
    draw: ImageDraw.ImageDraw,
    framework: str,
    y: int,
    row_h: int,
    title: str,
    detail: str | None,
    kind: str,
) -> None:
    x = METRIC_XS[framework]
    if kind == "fail":
        fill, color = FAIL_BG, DANGER
    elif kind == "unsupported":
        fill, color = PENDING, MUTED
    elif kind == "no_profile":
        fill, color = NO_PROFILE_BG, "#3730a3"
    else:
        fill, color = SUPPORTED_BG, GOOD
    box_h = 48
    _rounded(draw, (x, y + (row_h - box_h) / 2, x + METRIC_W, y + (row_h + box_h) / 2), 10, fill)
    if detail:
        _draw_text(draw, (x + 12, y + row_h / 2 - 2), title, FONT_STATUS_SMALL, color, anchor="lm")
        _draw_text(draw, (x + 12, y + row_h / 2 + 20), detail, FONT_STATUS_SMALL, "#4b5563", anchor="lm")
    else:
        _draw_text(draw, (x + 12, y + row_h / 2 + 1), title, FONT_STATUS, color, anchor="lm")


def _draw_missing_status_cell(
    draw: ImageDraw.ImageDraw,
    framework: str,
    case_cfg: dict | None,
    y: int,
    row_h: int,
) -> None:
    status = _missing_status(case_cfg, framework)
    if status == "unsupported":
        _draw_status_cell(draw, framework, y, row_h, "unsupported", None, "unsupported")
    elif status == "no_profile":
        _draw_status_cell(draw, framework, y, row_h, "no profile", "not run", "no_profile")
    else:
        _draw_status_cell(draw, framework, y, row_h, "not run", "configured", "supported")


def _draw_single_metric(
    draw: ImageDraw.ImageDraw,
    framework: str,
    entry: dict | None,
    case_cfg: dict | None,
    y: int,
    sglang_latency: float | None,
    winner: str | None,
) -> None:
    x = METRIC_XS[framework]
    value = _successful_latency(entry)
    if value is None:
        if entry and entry.get("error"):
            _draw_status_cell(draw, framework, y, ROW_SINGLE_H, _short_error_label(entry), None, "fail")
        else:
            _draw_missing_status_cell(draw, framework, case_cfg, y, ROW_SINGLE_H)
        return

    if framework == "sglang" or framework == winner:
        _rounded(draw, (x, y + 5, x + METRIC_W, y + 63), 10, FRAMEWORK_TINTS[framework])
    ratio = value / sglang_latency if sglang_latency else None
    ratio_text = f"{ratio:.3f}x" if ratio is not None else "-"
    ratio_color = TEXT if framework == "sglang" else DANGER if ratio and ratio >= 1 else GOOD
    _draw_text(draw, (x + 12, y + 18), _format_seconds(value), FONT_VALUE)
    _draw_text(draw, (x + 12, y + 45), ratio_text, FONT_RATIO, ratio_color)


def _draw_throughput_metric(
    draw: ImageDraw.ImageDraw,
    framework: str,
    entry: dict | None,
    case_cfg: dict | None,
    y: int,
    sglang_qps: float | None,
    winner: str | None,
) -> None:
    x = METRIC_XS[framework]
    p50, p99, qps = _throughput_metrics(entry)
    if qps is None:
        if entry and entry.get("error"):
            _draw_status_cell(draw, framework, y, ROW_THROUGHPUT_H, _short_error_label(entry), None, "fail")
        else:
            _draw_missing_status_cell(draw, framework, case_cfg, y, ROW_THROUGHPUT_H)
        return

    if framework == "sglang" or framework == winner:
        _rounded(draw, (x, y + 11, x + METRIC_W, y + 89), 10, FRAMEWORK_TINTS[framework])
    ratio = qps / sglang_qps if sglang_qps else None
    ratio_text = f"{ratio:.3f}x qps" if ratio is not None else "-"
    ratio_color = TEXT if framework == "sglang" else GOOD if ratio and ratio >= 1 else DANGER
    _draw_text(draw, (x + 12, y + 27), f"{qps:.4f} qps", FONT_VALUE)
    _draw_text(draw, (x + 12, y + 54), ratio_text, FONT_RATIO, ratio_color)
    if p50 is not None and p99 is not None:
        _draw_text(draw, (x + 12, y + 80), f"p50/p99 {p50:.2f}s/{p99:.2f}s", FONT_META, MUTED)


def _draw_winner_pill(
    draw: ImageDraw.ImageDraw,
    winner: str | None,
    y: int,
    row_h: int,
) -> None:
    pill_y = y + 18 if row_h == ROW_SINGLE_H else y + 28
    if not winner:
        _rounded(draw, (WINNER_X, pill_y, WINNER_X + WINNER_W, pill_y + 42), 21, PENDING)
        _draw_text(draw, (WINNER_X + WINNER_W / 2, pill_y + 27), "pending", FONT_STATUS_SMALL, MUTED, anchor="mm")
        return
    _rounded(draw, (WINNER_X, pill_y, WINNER_X + WINNER_W, pill_y + 42), 21, FRAMEWORK_TINTS[winner])
    _rounded(draw, (WINNER_X + 16, pill_y + 14, WINNER_X + 30, pill_y + 28), 4, FRAMEWORK_COLORS[winner])
    _draw_text(draw, (WINNER_X + 46, pill_y + 27), FRAMEWORK_LABELS[winner], FONT_STATUS, TEXT, anchor="lm")


def _single_summary(cases: list[str], single: dict[str, dict[str, dict]]) -> tuple[int, int, int]:
    comparable = 0
    wins = 0
    for case_id in cases:
        winner = _single_winner(single.get(case_id, {}))
        if not winner:
            continue
        comparable += 1
        if winner == "sglang":
            wins += 1
    return wins, comparable, len(cases) - comparable


def _throughput_summary(
    throughput_cases: list[str], throughput: dict[str, dict[str, dict]]
) -> tuple[int, int]:
    comparable = 0
    wins = 0
    for case_id in throughput_cases:
        winner = _throughput_winner(throughput.get(case_id, {}))
        if not winner:
            continue
        comparable += 1
        if winner == "sglang":
            wins += 1
    return wins, comparable


def _throughput_run_meta(
    throughput_cases: list[str], throughput: dict[str, dict[str, dict]]
) -> str:
    pairs = set()
    for case_id in throughput_cases:
        for entry in throughput.get(case_id, {}).values():
            metrics = entry.get("metrics") or {}
            requests = metrics.get("num_requests")
            concurrency = metrics.get("max_concurrency")
            if requests is not None and concurrency is not None:
                pairs.add((int(requests), int(concurrency)))
    if not pairs:
        return "-"
    if len(pairs) == 1:
        requests, concurrency = next(iter(pairs))
        return f"{requests} requests - max concurrency {concurrency}"
    return "; ".join(
        f"{requests} req @ c{concurrency}"
        for requests, concurrency in sorted(pairs, reverse=True)
    )


def _source_footer(results: dict) -> list[str]:
    sources = results.get("source_results") or []
    names = [Path(str(item.get("path") or "")).name for item in sources if item.get("path")]
    if not names:
        return [f"Data: {results.get('run_id', '-')}."]
    single = names[0] if names else "-"
    throughput = next((name for name in names[1:] if "throughput" in name), "-")
    primary = {single}
    if throughput != "-":
        primary.add(throughput)
    extra_names = [name for name in names if name not in primary]
    extra = ", ".join(extra_names) if extra_names else "-"
    return [
        f"Data: single-e2e {single}; throughput {throughput}.",
        f"Additional inputs: {extra}.",
        "Comparable win ratios exclude failed, unsupported, no-profile, and not-run cells.",
    ]


def render_report_image(
    results_path: Path,
    config_path: Path,
    output_png: Path,
    output_svg: Path | None = None,
) -> None:
    results = _load_json(results_path)
    config = _load_json(config_path)
    case_cfgs = _case_configs(config)
    cases = _ordered_cases(config, results)
    single = _by_case(results.get("results", []))
    throughput = _by_case(results.get("throughput_results", []))
    throughput_cases = [case_id for case_id in cases if case_id in throughput]

    single_panel_h = PANEL_HEADER_H + len(cases) * ROW_SINGLE_H + 14
    throughput_panel_h = PANEL_HEADER_H + max(1, len(throughput_cases)) * ROW_THROUGHPUT_H + 14
    footer_h = 112
    single_y = 250
    throughput_y = single_y + single_panel_h + 54
    footer_y = throughput_y + throughput_panel_h + 40
    height = footer_y + footer_h + 44

    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)

    _draw_header(
        draw,
        results,
        _single_summary(cases, single),
        _throughput_summary(throughput_cases, throughput),
        _throughput_run_meta(throughput_cases, throughput),
    )

    _rounded(draw, (PANEL_X, single_y, PANEL_X + PANEL_W, single_y + single_panel_h), 18, PANEL, BORDER)
    _draw_panel_header(
        draw,
        single_y,
        "Single Request Latency",
        "Lower is better - latency and latency/SGLang-Diffusion",
        "Fastest",
    )
    rows_y = single_y + PANEL_HEADER_H
    for idx, case_id in enumerate(cases):
        row_y = rows_y + idx * ROW_SINGLE_H
        if idx:
            draw.line((CASE_X, row_y, PANEL_X + PANEL_W - PAD, row_y), fill=GRID, width=1)
        case_cfg = case_cfgs.get(case_id)
        case_entries = single.get(case_id, {})
        entry = _case_entry(case_id, case_entries, case_cfg)
        winner = _single_winner(case_entries)
        _draw_case_cell(draw, case_id, entry, case_cfg, row_y)
        sglang_latency = _successful_latency(case_entries.get("sglang"))
        for framework in FRAMEWORK_ORDER:
            _draw_single_metric(
                draw,
                framework,
                case_entries.get(framework),
                case_cfg,
                row_y,
                sglang_latency,
                winner,
            )
        _draw_winner_pill(draw, winner, row_y, ROW_SINGLE_H)

    _rounded(
        draw,
        (PANEL_X, throughput_y, PANEL_X + PANEL_W, throughput_y + throughput_panel_h),
        18,
        PANEL,
        BORDER,
    )
    _draw_panel_header(
        draw,
        throughput_y,
        "High-Pressure Throughput",
        "Higher QPS is better - qps ratio plus p50/p99 latency",
        "Highest QPS",
    )
    rows_y = throughput_y + PANEL_HEADER_H
    if throughput_cases:
        for idx, case_id in enumerate(throughput_cases):
            row_y = rows_y + idx * ROW_THROUGHPUT_H
            if idx:
                draw.line((CASE_X, row_y, PANEL_X + PANEL_W - PAD, row_y), fill=GRID, width=1)
            case_cfg = case_cfgs.get(case_id)
            case_entries = throughput.get(case_id, {})
            entry = _case_entry(case_id, case_entries, case_cfg)
            winner = _throughput_winner(case_entries)
            _draw_case_cell(draw, case_id, entry, case_cfg, row_y + 8)
            sglang_qps = _throughput_metrics(case_entries.get("sglang"))[2]
            for framework in FRAMEWORK_ORDER:
                _draw_throughput_metric(
                    draw,
                    framework,
                    case_entries.get(framework),
                    case_cfg,
                    row_y,
                    sglang_qps,
                    winner,
                )
            _draw_winner_pill(draw, winner, row_y, ROW_THROUGHPUT_H)
    else:
        _draw_text(draw, (CASE_X, rows_y + 48), "No throughput results in this artifact.", FONT_VALUE, MUTED)

    _rounded(draw, (PANEL_X, footer_y, PANEL_X + PANEL_W, footer_y + footer_h), 14, "#eef3f8", BORDER)
    for idx, line in enumerate(_source_footer(results)):
        _draw_text(draw, (PANEL_X + 28, footer_y + 22 + idx * 30), line, FONT_FOOTER, "#4b5563")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)

    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        embedded_png = base64.b64encode(output_png.read_bytes()).decode("ascii")
        title = html.escape(f"Diffusion benchmark report: {results.get('run_id', '-')}")
        output_svg.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}">',
                    f"<title>{title}</title>",
                    f'<image href="data:image/png;base64,{embedded_png}" width="{WIDTH}" height="{height}"/>',
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diffusion benchmark comparison image")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/comparison_configs.json"), type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-svg", type=Path)
    args = parser.parse_args()

    render_report_image(args.results, args.config, args.output_png, args.output_svg)
    print(args.output_png)
    if args.output_svg:
        print(args.output_svg)


if __name__ == "__main__":
    main()
