"""Render a FLUX regression/competition comparison image."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required to generate report images") from exc


CASES = ("flux1_dev_t2i_1024", "flux2_dev_t2i_1024")
CASE_LABELS = {
    "flux1_dev_t2i_1024": "FLUX.1-dev T2I",
    "flux2_dev_t2i_1024": "FLUX.2-dev T2I",
}
SERIES_ORDER = ("sglang-before", "sglang-now", "vllm-omni", "lightx2v")
SERIES_LABELS = {
    "sglang-before": "SGLang-Diffusion Apr 10",
    "sglang-now": "SGLang-Diffusion now",
    "vllm-omni": "vLLM-Omni now",
    "lightx2v": "LightX2V now",
}
SERIES_COLORS = {
    "sglang-before": "#f59e0b",
    "sglang-now": "#e8524d",
    "vllm-omni": "#4f7fad",
    "lightx2v": "#59a14f",
}
SERIES_TINTS = {
    "sglang-before": "#fff7ed",
    "sglang-now": "#fde5e3",
    "vllm-omni": "#e6eef7",
    "lightx2v": "#eaf7ed",
}

BG = "#f6f8fb"
PANEL = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#dfe4ea"
GRID = "#eef1f5"
DANGER = "#b91c1c"
GOOD = "#166534"
FAIL_BG = "#fff1f2"
MISSING_BG = "#f3f4f6"
CARD_DARK = "#111827"

WIDTH = 2200
MARGIN = 72
PANEL_X = 72
PANEL_W = WIDTH - PANEL_X * 2
PAD = 34
HEADER_H = 206
PANEL_HEADER_H = 104
ROW_H = 132
CASE_X = PANEL_X + PAD
COL_X = {
    "sglang-before": 520,
    "sglang-now": 880,
    "vllm-omni": 1240,
    "lightx2v": 1600,
}
COL_W = 310
FASTEST_X = 1960
FASTEST_W = 150


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


FONT_TITLE = _font(48, True)
FONT_SUBTITLE = _font(26)
FONT_LEGEND = _font(22, True)
FONT_SECTION = _font(31, True)
FONT_HEADER = _font(18, True)
FONT_CASE = _font(23, True)
FONT_META = _font(16, True)
FONT_VALUE = _font(24, True)
FONT_RATIO = _font(17, True)
FONT_SMALL = _font(15, True)
FONT_STATUS = _font(18, True)
FONT_STAT_NUM = _font(34, True)
FONT_STAT_LABEL = _font(22, True)
FONT_STAT_META = _font(17)
FONT_FOOTER = _font(18, True)


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _rounded(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _short_commit(value: str | None) -> str:
    return value[:7] if value else "unknown"


def _hardware_label(data: dict) -> str:
    override = str((data.get("hardware") or {}).get("hardware_profile_override") or "").lower()
    gpus = " ".join(str(gpu) for gpu in (data.get("hardware") or {}).get("gpus") or [])
    text = f"{override} {gpus}".lower()
    if any(token in text for token in ("gb300", "gb200", "b300", "b200", "blackwell")):
        return "Blackwell"
    if "h200" in text:
        return "Hopper H200"
    if "h100" in text:
        return "Hopper H100"
    return override.upper() if override else "GPU"


def _series_key(data: dict, entry: dict, before_commit: str) -> str | None:
    framework = entry.get("framework")
    if framework == "sglang":
        commit = (data.get("sglang_runtime") or {}).get("git_commit") or ""
        run_id = str(data.get("run_id") or "").lower()
        if before_commit and commit.startswith(before_commit):
            return "sglang-before"
        if any(token in run_id for token in ("before", "old", "apr10", "apr-10")):
            return "sglang-before"
        return "sglang-now"
    if framework in ("vllm-omni", "lightx2v"):
        return framework
    return None


def _entry_profile(entry: dict) -> str:
    metadata = entry.get("framework_metadata") or {}
    return str(metadata.get("profile") or metadata.get("sglang_profile") or "-")


def _entry_meta(entry: dict) -> str:
    profile = _entry_profile(entry)
    num_gpus = entry.get("num_gpus")
    parts = [profile]
    if num_gpus:
        parts.append(f"{num_gpus} GPU")
    return " - ".join(parts)


def _collect_best(paths: list[Path], before_commit: str) -> tuple[dict, dict]:
    best: dict[tuple[str, str, str], dict] = {}
    sources = {
        "sglang-before": set(),
        "sglang-now": set(),
        "vllm-omni": set(),
        "lightx2v": set(),
    }
    for path in paths:
        data = _load(path)
        hardware = _hardware_label(data)
        for entry in data.get("results") or []:
            if entry.get("mode") != "single_e2e" or entry.get("case_id") not in CASES:
                continue
            series = _series_key(data, entry, before_commit)
            if series is None:
                continue
            cloned = dict(entry)
            cloned["_source"] = path.as_posix()
            cloned["_hardware"] = hardware
            cloned["_run_id"] = data.get("run_id")
            if series.startswith("sglang"):
                cloned["_runtime_commit"] = (data.get("sglang_runtime") or {}).get("git_commit")
            key = (hardware, entry["case_id"], series)
            prev = best.get(key)
            latency = cloned.get("latency_s")
            prev_latency = prev.get("latency_s") if prev else None
            if latency is not None and (prev_latency is None or latency < prev_latency):
                best[key] = cloned
            elif prev is None:
                best[key] = cloned
            sources[series].add(path.as_posix())
    return best, sources


def _case_meta(entry: dict | None) -> str:
    if not entry:
        return "-"
    parts = []
    width = entry.get("width")
    height = entry.get("height")
    if width and height:
        parts.append(f"{width}x{height}")
    steps = entry.get("num_inference_steps")
    if steps:
        parts.append(f"{steps} steps")
    guidance = entry.get("guidance_scale")
    if guidance is not None:
        parts.append(f"cfg {guidance:g}" if isinstance(guidance, (int, float)) else f"cfg {guidance}")
    return " - ".join(parts) if parts else "-"


def _successful_latency(entry: dict | None) -> float | None:
    if not entry or entry.get("error"):
        return None
    value = entry.get("latency_s")
    return float(value) if isinstance(value, (int, float)) else None


def _winner(entries: dict[str, dict]) -> str | None:
    values = {
        series: _successful_latency(entry)
        for series, entry in entries.items()
    }
    values = {series: value for series, value in values.items() if value is not None}
    if len(values) < 2:
        return None
    return min(values, key=lambda series: values[series])


def _short_error(entry: dict | None) -> str:
    if not entry:
        return "-"
    error = str(entry.get("error") or "")
    lowered = error.lower()
    if "timeout" in lowered:
        return "timeout"
    if "out of memory" in lowered or "oom" in lowered:
        return "oom"
    if "unknown command" in lowered or "unrecognized" in lowered:
        return "bad args"
    if error:
        return "failed"
    return "-"


def _draw_header(draw: ImageDraw.ImageDraw, best: dict, sources: dict) -> None:
    _rounded(draw, (0, 0, WIDTH, HEADER_H), 0, PANEL)
    draw.line((0, HEADER_H, WIDTH, HEADER_H), fill="#e5e7eb", width=1)
    _draw_text(draw, (MARGIN, 32), "FLUX Fastest Performance Comparison", FONT_TITLE)
    _draw_text(
        draw,
        (MARGIN, 96),
        "single request latency - best of recorded 1GPU/2GPU TP profiles - no cache - torch compile allowed",
        FONT_SUBTITLE,
        "#4b5563",
    )
    x = MARGIN
    for series in SERIES_ORDER:
        _rounded(draw, (x, 156, x + 24, 180), 6, SERIES_COLORS[series])
        _draw_text(draw, (x + 36, 153), SERIES_LABELS[series], FONT_LEGEND, "#374151")
        x += 420 if series == "sglang-before" else 310

    comparable = 0
    now_wins = 0
    for hardware in sorted({key[0] for key in best}):
        for case_id in CASES:
            entries = {
                series: best.get((hardware, case_id, series))
                for series in SERIES_ORDER
            }
            if sum(1 for entry in entries.values() if _successful_latency(entry) is not None) >= 2:
                comparable += 1
                if _winner(entries) == "sglang-now":
                    now_wins += 1
    _draw_stat_card(
        draw,
        1578,
        f"{now_wins}/{comparable}" if comparable else "0/0",
        "current wins",
        f"{sum(len(v) for v in sources.values())} source files",
    )
    _draw_stat_card(
        draw,
        1860,
        f"{len({key[0] for key in best})}",
        "hardware",
        "Hopper + Blackwell",
    )


def _draw_stat_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    value: str,
    label: str,
    meta: str,
) -> None:
    _rounded(draw, (x, 42, x + 240, 148), 16, CARD_DARK)
    _draw_text(draw, (x + 30, 56), value, FONT_STAT_NUM, "#ffffff")
    _draw_text(draw, (x + 30, 98), label, FONT_STAT_LABEL, "#ffffff")
    _draw_text(draw, (x + 30, 124), meta, FONT_STAT_META, "#cbd5e1")


def _draw_panel_header(draw: ImageDraw.ImageDraw, y: int, hardware: str) -> None:
    _draw_text(draw, (CASE_X, y + 28), hardware, FONT_SECTION)
    _draw_text(draw, (CASE_X, y + 68), "Case", FONT_HEADER, MUTED)
    for series, x in COL_X.items():
        _draw_text(draw, (x + 12, y + 68), SERIES_LABELS[series], FONT_HEADER, MUTED)
    _draw_text(draw, (FASTEST_X + 10, y + 68), "Fastest", FONT_HEADER, MUTED)
    draw.line((CASE_X, y + PANEL_HEADER_H - 4, PANEL_X + PANEL_W - PAD, y + PANEL_HEADER_H - 4), fill=GRID, width=1)


def _draw_case_cell(draw: ImageDraw.ImageDraw, y: int, case_id: str, entries: dict[str, dict]) -> None:
    first_entry = next((entry for entry in entries.values() if entry), None)
    _draw_text(draw, (CASE_X, y + 26), CASE_LABELS[case_id], FONT_CASE)
    _draw_text(draw, (CASE_X, y + 59), _case_meta(first_entry), FONT_META, MUTED)


def _draw_result_cell(
    draw: ImageDraw.ImageDraw,
    series: str,
    entry: dict | None,
    current_latency: float | None,
    winner: str | None,
    y: int,
) -> None:
    x = COL_X[series]
    latency = _successful_latency(entry)
    if latency is None:
        fill = FAIL_BG if entry and entry.get("error") else MISSING_BG
        _rounded(draw, (x, y + 12, x + COL_W, y + 96), 10, fill)
        _draw_text(draw, (x + 18, y + 44), _short_error(entry), FONT_STATUS, DANGER if entry and entry.get("error") else MUTED)
        if entry:
            _draw_text(draw, (x + 18, y + 70), _entry_profile(entry), FONT_SMALL, MUTED)
        return

    if series == winner:
        _rounded(draw, (x, y + 10, x + COL_W, y + 100), 10, SERIES_TINTS[series])
    elif series == "sglang-now":
        _rounded(draw, (x, y + 10, x + COL_W, y + 100), 10, "#fff7f7")

    _draw_text(draw, (x + 18, y + 36), f"{latency:.3f}s", FONT_VALUE)
    if current_latency:
        ratio = latency / current_latency
        ratio_color = TEXT if series == "sglang-now" else GOOD if ratio < 1 else DANGER
        _draw_text(draw, (x + 18, y + 62), f"{ratio:.3f}x vs now", FONT_RATIO, ratio_color)
    _draw_text(draw, (x + 18, y + 88), _entry_meta(entry), FONT_SMALL, MUTED)


def _draw_winner(draw: ImageDraw.ImageDraw, y: int, winner: str | None) -> None:
    if not winner:
        _rounded(draw, (FASTEST_X, y + 32, FASTEST_X + FASTEST_W, y + 76), 22, MISSING_BG)
        _draw_text(draw, (FASTEST_X + FASTEST_W / 2, y + 54), "pending", FONT_STATUS, MUTED, anchor="mm")
        return
    _rounded(draw, (FASTEST_X, y + 32, FASTEST_X + FASTEST_W, y + 76), 22, SERIES_TINTS[winner])
    _rounded(draw, (FASTEST_X + 18, y + 47, FASTEST_X + 32, y + 61), 4, SERIES_COLORS[winner])
    label = "current" if winner == "sglang-now" else SERIES_LABELS[winner].split()[0]
    _draw_text(draw, (FASTEST_X + 44, y + 54), label, FONT_STATUS, TEXT, anchor="lm")


def _draw_footer(draw: ImageDraw.ImageDraw, y: int, paths: list[Path]) -> None:
    _rounded(draw, (PANEL_X, y, PANEL_X + PANEL_W, y + 92), 12, "#eef6fb", "#dbe7f0")
    text = (
        "Data: generated from raw diffusion-bench result JSONs. "
        "Cells choose the fastest successful profile for each hardware/case/series; failures remain visible when no successful run exists."
    )
    _draw_text(draw, (PANEL_X + 28, y + 28), text, FONT_FOOTER, "#475569")
    suffix = ", ".join(path.name for path in paths[:4])
    if len(paths) > 4:
        suffix += f", +{len(paths) - 4} more"
    _draw_text(draw, (PANEL_X + 28, y + 58), suffix, FONT_FOOTER, "#475569")


def render_flux_regression_image(
    result_paths: list[Path],
    output_png: Path,
    output_svg: Path | None = None,
    before_commit: str = "8227187",
) -> None:
    best, sources = _collect_best(result_paths, before_commit)
    hardware_order = sorted(
        {key[0] for key in best},
        key=lambda label: (0 if "Hopper" in label else 1 if "Blackwell" in label else 2, label),
    )
    panel_h = PANEL_HEADER_H + ROW_H * len(CASES) + 34
    height = HEADER_H + 58 + panel_h * max(1, len(hardware_order)) + 124
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    _draw_header(draw, best, sources)

    y = HEADER_H + 58
    for hardware in hardware_order:
        _rounded(draw, (PANEL_X, y, PANEL_X + PANEL_W, y + panel_h), 16, PANEL, BORDER)
        _draw_panel_header(draw, y + 20, hardware)
        row_y = y + 20 + PANEL_HEADER_H
        for case_id in CASES:
            entries = {series: best.get((hardware, case_id, series)) for series in SERIES_ORDER}
            current_latency = _successful_latency(entries.get("sglang-now"))
            winner = _winner(entries)
            _draw_case_cell(draw, row_y, case_id, entries)
            for series in SERIES_ORDER:
                _draw_result_cell(draw, series, entries.get(series), current_latency, winner, row_y)
            _draw_winner(draw, row_y, winner)
            draw.line((CASE_X, row_y + ROW_H - 1, PANEL_X + PANEL_W - PAD, row_y + ROW_H - 1), fill=GRID, width=1)
            row_y += ROW_H
        y += panel_h + 42

    _draw_footer(draw, height - 108, result_paths)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)

    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        encoded = base64.b64encode(output_png.read_bytes()).decode("ascii")
        output_svg.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}">',
                    f'<image href="data:image/png;base64,{encoded}" width="{WIDTH}" height="{height}"/>',
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a FLUX fastest comparison image")
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument("--sglang-before-commit", default="8227187")
    args = parser.parse_args()

    render_flux_regression_image(
        args.results,
        args.output_png,
        args.output_svg,
        before_commit=args.sglang_before_commit,
    )
    print(args.output_png)
    if args.output_svg:
        print(args.output_svg)


if __name__ == "__main__":
    main()
