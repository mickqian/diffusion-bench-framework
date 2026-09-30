"""Summarise the reproduction's cells from the harness JSON, its runlog and the
nvidia-smi samples.

Per-step time is taken the same way for both frameworks: the difference between
the measured 40-step requests and the steady 3-step warmups, divided by the 37
extra steps. Fixed costs (encoders, VAE, save, polling) cancel, and it needs no
framework-specific timer -- sglang's "average time per step" line is taken before
a device synchronise and reads low, and LightX2V only prints stage timers when
its profiler is on, which itself adds synchronisations.

    python3 summarize_cells.py <cells_dir> <tag> [<tag> ...]
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

WARMUP_STEPS = 3
WARMUP_LATENCIES = re.compile(r"Warmup latencies: (.+)$")
SGLANG_STAGE = re.compile(r"\[(\w+Stage|denoising_stage|decoding_stage)\] finished in ([0-9.]+) seconds")
SGLANG_STEPS = re.compile(r"^\s*\[server\]\s+infer_steps: (\d+)")
LX2V_PROFILE = re.compile(r"\[Profile\].* - (?:Level\d_Log )?(Run DiT|Run VAE Decoder|Run Encoders|Run Text Encoder|Run Image Encoder|Run VAE Encoder|RUN pipeline) cost ([0-9.]+) seconds")
LX2V_STEP = re.compile(r"step_index: (\d+) / (\d+)")


def peak_mib(smi_csv: Path) -> dict[int, int]:
    peaks: dict[int, int] = {}
    if not smi_csv.exists():
        return peaks
    with smi_csv.open() as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                idx, used = int(row[1]), int(row[2])
            except ValueError:
                continue
            peaks[idx] = max(peaks.get(idx, 0), used)
    return peaks


def parse_runlog(runlog: Path) -> dict:
    out: dict = {"warmups_s": [], "sglang_denoise_s": [], "sglang_steps": [], "lx2v": {}, "lx2v_last_step": None}
    if not runlog.exists():
        return out
    for line in runlog.read_text(errors="replace").splitlines():
        m = WARMUP_LATENCIES.search(line)
        if m:
            out["warmups_s"] = [float(x.strip().rstrip("s")) for x in m.group(1).split(",")]
        m = SGLANG_STEPS.search(line)
        if m:
            out["sglang_steps"].append(int(m.group(1)))
        m = SGLANG_STAGE.search(line)
        if m and m.group(1) == "DenoisingStage":
            out["sglang_denoise_s"].append(float(m.group(2)))
        m = LX2V_PROFILE.search(line)
        if m:
            out["lx2v"].setdefault(m.group(1), []).append(float(m.group(2)))
        m = LX2V_STEP.search(line)
        if m:
            out["lx2v_last_step"] = (int(m.group(1)), int(m.group(2)))
    return out


def summarise(cells_dir: Path, tag: str) -> dict:
    result_json = cells_dir / f"{tag}.json"
    data = json.loads(result_json.read_text()) if result_json.exists() else {}
    results = data.get("results") or [{}]
    r = results[0]
    metrics = r.get("metrics") or {}
    samples = metrics.get("latency_samples_s") or ([r["latency_s"]] if r.get("latency_s") else [])
    log = parse_runlog(cells_dir / f"{tag}.runlog")
    steady_warmups = log["warmups_s"][1:] or log["warmups_s"]
    e2e = statistics.median(samples) if samples else None
    warm = statistics.median(steady_warmups) if steady_warmups else None
    steps = r.get("num_inference_steps") or 40
    per_step = (e2e - warm) / (steps - WARMUP_STEPS) if e2e is not None and warm is not None else None
    peaks = peak_mib(cells_dir / f"{tag}.smi.csv")
    return {
        "tag": tag,
        "framework": r.get("framework"),
        "case": r.get("case_id"),
        "gpus": r.get("num_gpus"),
        "error": (r.get("error") or "")[:160] or None,
        "e2e_40_s": round(e2e, 2) if e2e is not None else None,
        "e2e_40_samples_s": samples,
        "e2e_3_steady_s": round(warm, 2) if warm is not None else None,
        "warmups_s": log["warmups_s"],
        "per_step_s": round(per_step, 3) if per_step is not None else None,
        "sglang_denoise_stage_s": log["sglang_denoise_s"],
        "sglang_infer_steps_seen": sorted(set(log["sglang_steps"])),
        "lightx2v_profile_s": log["lx2v"],
        "lightx2v_last_step": log["lx2v_last_step"],
        "peak_mib_per_gpu": peaks,
        "peak_mib_max": max(peaks.values()) if peaks else None,
        "server_command": r.get("server_command"),
        "server_startup_s": metrics.get("server_startup_s"),
    }


def main() -> None:
    cells_dir = Path(sys.argv[1])
    rows = [summarise(cells_dir, tag) for tag in sys.argv[2:]]
    print(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
