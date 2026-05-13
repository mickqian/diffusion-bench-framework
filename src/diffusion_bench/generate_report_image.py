"""Render a compact PNG/SVG comparison image from merged benchmark results."""

from __future__ import annotations

import argparse
import base64
import html
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by CLI environment
    raise SystemExit("Pillow is required to generate report images") from exc


FRAMEWORK_ORDER = ("sglang", "vllm-omni", "lightx2v", "diffusers")
FRAMEWORK_LABELS = {
    "sglang": "SGLang",
    "vllm-omni": "vLLM-Omni",
    "lightx2v": "LightX2V",
    "diffusers": "diffusers",
}

BG = "#f7f8fb"
TEXT = "#172033"
MUTED = "#667085"
BORDER = "#d8dee9"
HEADER = "#e7edf7"
ROW_ALT = "#ffffff"
ROW = "#fdfefe"
OK = "#d9f2e6"
WARN = "#fff4ce"
BAD = "#ffe1df"
BEST = "#bce8d1"


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


FONT_TITLE = _font(34, bold=True)
FONT_SUBTITLE = _font(18)
FONT_SECTION = _font(23, bold=True)
FONT_HEADER = _font(18, bold=True)
FONT_CELL = _font(17)
FONT_CELL_BOLD = _font(17, bold=True)
FONT_SMALL = _font(14)
FONT_SMALL_BOLD = _font(14, bold=True)


def _framework_sort(framework: str) -> tuple[int, str]:
    try:
        return FRAMEWORK_ORDER.index(framework), framework
    except ValueError:
        return len(FRAMEWORK_ORDER), framework


def _case_configs(config: dict) -> dict[str, dict]:
    return {case["id"]: case for case in config.get("cases", [])}


def _ordered_cases(config: dict, results: dict) -> list[str]:
    configured = []
    for case in config.get("cases", []):
        frameworks = case.get("frameworks") or {}
        if "sglang" in frameworks and len(frameworks) >= 2:
            configured.append(case["id"])

    seen = set(configured)
    for entry in results.get("results", []) + results.get("throughput_results", []):
        case_id = entry.get("case_id")
        if case_id and case_id not in seen:
            configured.append(case_id)
            seen.add(case_id)
    return configured


def _frameworks_for_report(config: dict, results: dict) -> list[str]:
    frameworks = set(FRAMEWORK_ORDER[:3])
    for case in config.get("cases", []):
        frameworks.update((case.get("frameworks") or {}).keys())
    for entry in results.get("results", []) + results.get("throughput_results", []):
        if entry.get("framework"):
            frameworks.add(entry["framework"])
    return sorted(frameworks, key=_framework_sort)


def _by_case(entries: Iterable[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for entry in entries:
        grouped.setdefault(entry.get("case_id", ""), {})[entry.get("framework", "")] = entry
    return grouped


def _ok(entry: dict | None) -> bool:
    return bool(entry) and not entry.get("error")


def _single_latency(entry: dict | None) -> float | None:
    if not _ok(entry):
        return None
    value = entry.get("latency_s")
    return float(value) if value is not None else None


def _throughput_metrics(entry: dict | None) -> tuple[float | None, float | None, float | None]:
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


def _gpu_label(entry: dict | None) -> str:
    if not entry:
        return ""
    num_gpus = entry.get("num_gpus")
    return f" | {num_gpus}GPU" if num_gpus else ""


def _status_text(entry: dict | None, case_cfg: dict | None, framework: str) -> str:
    if entry and entry.get("error"):
        return "failed"
    if entry:
        return "ok"
    if case_cfg and framework in (case_cfg.get("frameworks") or {}):
        return "not run"
    return "not configured"


def _cell_fill(status: str, is_best: bool = False) -> str:
    if is_best:
        return BEST
    if status == "ok":
        return OK
    if status == "not run":
        return WARN
    if status == "failed":
        return BAD
    return ROW_ALT


def _draw_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str = BORDER) -> None:
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=1)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
    fill: str = TEXT,
    max_lines: int = 2,
    line_gap: int = 4,
) -> int:
    if not text:
        return y
    avg_chars = max(8, int(width / max(8, font.size * 0.55)))
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        wrapped.extend(textwrap.wrap(raw_line, width=avg_chars) or [""])
    if len(wrapped) > max_lines:
        wrapped = wrapped[: max_lines - 1] + [wrapped[max_lines - 1].rstrip(".") + "..."]
    line_height = int(font.size * 1.25)
    for line in wrapped:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: str = TEXT,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=font, fill=fill)


def _single_cell(entry: dict | None, sg_latency: float | None) -> tuple[str, float | None]:
    latency = _single_latency(entry)
    if latency is None:
        return "", None
    ratio = latency / sg_latency if sg_latency and sg_latency > 0 else None
    ratio_text = f"\n{ratio:.2f}x vs SG" if ratio is not None and entry.get("framework") != "sglang" else "\n1.00x baseline"
    return f"{latency:.3f}s{_gpu_label(entry)}{ratio_text}", latency


def _throughput_cell(entry: dict | None, sg_qps: float | None) -> tuple[str, float | None]:
    p50, p99, qps = _throughput_metrics(entry)
    if p50 is None and qps is None:
        return "", None
    ratio = qps / sg_qps if qps is not None and sg_qps and sg_qps > 0 else None
    ratio_text = f" ({ratio:.2f}x)" if ratio is not None and entry and entry.get("framework") != "sglang" else ""
    parts = []
    if p50 is not None:
        parts.append(f"p50 {p50:.3f}s{_gpu_label(entry)}")
    if p99 is not None:
        parts.append(f"p99 {p99:.3f}s")
    if qps is not None:
        parts.append(f"{qps:.4f} qps{ratio_text}")
    return "\n".join(parts), qps


def _source_footer(results: dict, results_path: Path) -> str:
    sources = results.get("source_results") or []
    if sources:
        names = [Path(str(item.get("path") or "")).name for item in sources if item.get("path")]
    else:
        names = [results_path.name]
    return "Sources: " + ", ".join(names[:4]) + (" ..." if len(names) > 4 else "")


def render_report_image(results_path: Path, config_path: Path, output_png: Path, output_svg: Path | None = None) -> None:
    results = _load_json(results_path)
    config = _load_json(config_path)
    case_cfgs = _case_configs(config)
    cases = _ordered_cases(config, results)
    frameworks = _frameworks_for_report(config, results)
    single = _by_case(results.get("results", []))
    throughput = _by_case(results.get("throughput_results", []))
    throughput_cases = [case_id for case_id in cases if case_id in throughput]

    width = 1720
    margin = 42
    case_w = 360
    gap = 12
    fw_w = int((width - margin * 2 - case_w - gap * len(frameworks)) / len(frameworks))
    title_h = 126
    section_h = 46
    header_h = 42
    single_row_h = 72
    throughput_row_h = 92
    footer_h = 56
    height = (
        margin
        + title_h
        + section_h
        + header_h
        + len(cases) * single_row_h
        + 28
        + section_h
        + header_h
        + max(1, len(throughput_cases)) * throughput_row_h
        + footer_h
    )

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    y = margin
    draw.text((margin, y), "Diffusion Framework Benchmark", font=FONT_TITLE, fill=TEXT)
    y += 44
    subtitle = "H200 comparison: single-request latency and throughput. Lower latency is better; higher QPS is better."
    draw.text((margin, y), subtitle, font=FONT_SUBTITLE, fill=MUTED)
    y += 28
    meta = f"run_id: {results.get('run_id', '-')}   generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    draw.text((margin, y), meta, font=FONT_SMALL, fill=MUTED)
    y += 54

    def draw_table_header(y0: int, section: str, row_h: int) -> int:
        draw.text((margin, y0 + 8), section, font=FONT_SECTION, fill=TEXT)
        y0 += section_h
        x = margin
        _draw_rect(draw, (x, y0, x + case_w, y0 + header_h), HEADER)
        _draw_centered(draw, "case", (x, y0, x + case_w, y0 + header_h), FONT_HEADER)
        x += case_w + gap
        for framework in frameworks:
            _draw_rect(draw, (x, y0, x + fw_w, y0 + header_h), HEADER)
            _draw_centered(
                draw,
                FRAMEWORK_LABELS.get(framework, framework),
                (x, y0, x + fw_w, y0 + header_h),
                FONT_HEADER,
            )
            x += fw_w + gap
        return y0 + header_h

    y = draw_table_header(y, "Single Request Latency", single_row_h)
    for idx, case_id in enumerate(cases):
        row_y = y + idx * single_row_h
        case_cfg = case_cfgs.get(case_id)
        row_fill = ROW_ALT if idx % 2 else ROW
        x = margin
        _draw_rect(draw, (x, row_y, x + case_w, row_y + single_row_h - 8), row_fill)
        _draw_wrapped(draw, case_id, x + 14, row_y + 13, case_w - 28, FONT_CELL_BOLD, max_lines=2)

        sg_latency = _single_latency(single.get(case_id, {}).get("sglang"))
        values = {
            fw: _single_latency(single.get(case_id, {}).get(fw))
            for fw in frameworks
        }
        valid = [value for value in values.values() if value is not None]
        best = min(valid) if valid else None

        x += case_w + gap
        for fw in frameworks:
            entry = single.get(case_id, {}).get(fw)
            status = _status_text(entry, case_cfg, fw)
            is_best = values.get(fw) is not None and best is not None and values[fw] == best
            _draw_rect(draw, (x, row_y, x + fw_w, row_y + single_row_h - 8), _cell_fill(status, is_best))
            text, _ = _single_cell(entry, sg_latency)
            if text:
                _draw_wrapped(draw, text, x + 12, row_y + 12, fw_w - 24, FONT_CELL_BOLD if is_best else FONT_CELL, max_lines=2)
            else:
                _draw_centered(draw, status, (x, row_y, x + fw_w, row_y + single_row_h - 8), FONT_SMALL, MUTED)
            x += fw_w + gap

    y += len(cases) * single_row_h + 28
    y = draw_table_header(y, "Throughput: p50 / p99 / QPS", throughput_row_h)
    if throughput_cases:
        for idx, case_id in enumerate(throughput_cases):
            row_y = y + idx * throughput_row_h
            case_cfg = case_cfgs.get(case_id)
            row_fill = ROW_ALT if idx % 2 else ROW
            x = margin
            _draw_rect(draw, (x, row_y, x + case_w, row_y + throughput_row_h - 8), row_fill)
            _draw_wrapped(draw, case_id, x + 14, row_y + 16, case_w - 28, FONT_CELL_BOLD, max_lines=2)

            sg_qps = _throughput_metrics(throughput.get(case_id, {}).get("sglang"))[2]
            qps_values = {
                fw: _throughput_metrics(throughput.get(case_id, {}).get(fw))[2]
                for fw in frameworks
            }
            valid_qps = [value for value in qps_values.values() if value is not None]
            best_qps = max(valid_qps) if valid_qps else None

            x += case_w + gap
            for fw in frameworks:
                entry = throughput.get(case_id, {}).get(fw)
                status = _status_text(entry, case_cfg, fw)
                is_best = qps_values.get(fw) is not None and best_qps is not None and qps_values[fw] == best_qps
                _draw_rect(draw, (x, row_y, x + fw_w, row_y + throughput_row_h - 8), _cell_fill(status, is_best))
                text, _ = _throughput_cell(entry, sg_qps)
                if text:
                    _draw_wrapped(draw, text, x + 12, row_y + 10, fw_w - 24, FONT_SMALL_BOLD if is_best else FONT_SMALL, max_lines=3)
                else:
                    _draw_centered(draw, status, (x, row_y, x + fw_w, row_y + throughput_row_h - 8), FONT_SMALL, MUTED)
                x += fw_w + gap
    else:
        _draw_wrapped(draw, "No throughput results in this artifact.", margin + 14, y + 14, width - margin * 2 - 28, FONT_CELL, MUTED, max_lines=1)

    footer_y = height - footer_h + 12
    footer = _source_footer(results, results_path)
    draw.text((margin, footer_y), footer, font=FONT_SMALL, fill=MUTED)
    draw.text((margin, footer_y + 20), "Generated by diffusion-bench-report-image.", font=FONT_SMALL, fill=MUTED)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)

    if output_svg:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        escaped_footer = html.escape(footer)
        embedded_png = base64.b64encode(output_png.read_bytes()).decode("ascii")
        output_svg.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                    f'<title>{escaped_footer}</title>',
                    f'<image href="data:image/png;base64,{embedded_png}" width="{width}" height="{height}"/>',
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
