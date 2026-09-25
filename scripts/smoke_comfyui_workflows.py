#!/usr/bin/env python3
"""Run every ComfyUI workflow once at a couple of steps, through the harness's own path.

A template mistake -- a renamed input, a checkpoint a loader rejects, a node this
ComfyUI does not have -- otherwise surfaces as a failed cell after a full warmup
inside a multi-hour round. This resolves each case's ComfyUI profile, lays out its
workspace, launches the server and sends one request exactly as run_comparison
does, only with the step count cut down, then stops the server.

  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python3 scripts/smoke_comfyui_workflows.py \
      [--hardware-profile b200] [case_id ...]

The first request loads the weights, so the time printed is a load, not a latency.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_bench import run_comparison as rc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case ids (default: every case with a ComfyUI cell)")
    ap.add_argument("--config", default="configs/comparison_configs.json")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--port", type=int, default=18300)
    ap.add_argument("--hardware-profile", default=None)
    args = ap.parse_args()

    config = json.load(open(args.config))
    hardware = rc._collect_hardware_metadata()
    if args.hardware_profile:
        hardware["hardware_profile_override"] = args.hardware_profile
    log_dir = Path("comparison-logs")
    log_dir.mkdir(exist_ok=True)

    statuses = []
    port = args.port
    for case in config["cases"]:
        raw = (case.get("frameworks") or {}).get("comfyui")
        if not raw or (args.cases and case["id"] not in args.cases):
            continue
        fw_cfg = rc._resolve_framework_config("comfyui", raw, None, hardware)
        smoke_case = rc._case_for_framework(case, fw_cfg)
        smoke_case["num_inference_steps"] = args.steps
        base_url = f"http://{rc.DEFAULT_HOST}:{port}"
        started, proc, status, detail = time.time(), None, "ok", ""
        try:
            env = rc._framework_env("comfyui", rc._apply_benchmark_env(os.environ.copy()))
            env.update(fw_cfg.get("extra_env", {}))
            env = rc._visible_gpus_env(env, int(smoke_case.get("num_gpus") or 1))
            rc._prepare_comfyui_workspace(smoke_case, fw_cfg, config)
            cmd = rc.build_server_cmd("comfyui", smoke_case, fw_cfg, port)
            with open(log_dir / f"smoke_{case['id']}_comfyui.log", "w") as log:
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid
                )
                rc.wait_for_health(base_url, "comfyui", timeout=600, proc=proc)
                latency = rc.send_request_comfyui(base_url, smoke_case, config)
            detail = f"{latency:.1f}s incl. load, {args.steps} steps, {smoke_case.get('num_gpus')} GPU"
        except Exception as exc:  # noqa: BLE001 - every case reports, then the run fails
            status, detail = "FAIL", f"{type(exc).__name__}: {str(exc)[:600]}"
        finally:
            if proc:
                rc.kill_server(proc)
        print(f"SMOKE {status:4s} {case['id']:34s} {time.time() - started:6.0f}s  {detail}", flush=True)
        statuses.append(status)
        port += 1
    return 0 if statuses and all(s == "ok" for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
