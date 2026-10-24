#!/usr/bin/env python3
"""A case retired after a run measured it still publishes that run's rows.

The page builds rows only for cases in the config, so retiring LTX-2 and
LTX-2.3 for LTX-2.5 (2026-10-24) would have dropped the 2026-10-22 round's
measured rows from its own publish, against the user's decision to keep them.
configs/benchmark/retired_cases.json freezes their definitions; the page adds
one only for a case the run measured and the config no longer has.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench.page_export import build_sections  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:300]) if detail and not cond else ''}")
    if not cond:
        fail = 1


config = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
retired = json.loads((ROOT / "configs" / "benchmark" / "retired_cases.json").read_text())["cases"]
current = {c["id"] for c in config["cases"]}
check("the retired file names LTX-2 and LTX-2.3",
      {c["id"] for c in retired} == {"ltx2_twostage_t2v", "ltx2.3_twostage_t2v_2gpus"}, [c["id"] for c in retired])
check("no retired case is back in the matrix", not current & {c["id"] for c in retired})
for c in retired:
    check(f"{c['id']}: carries what a row shows",
          all(k in c for k in ("label", "num_gpus", "width", "height", "num_inference_steps", "guidance_scale",
                               "report_framework_statuses")), sorted(c))


def row(case_id, fw, latency):
    return {"case_id": case_id, "framework": fw, "latency_s": latency, "num_gpus": 2,
            "framework_metadata": {"profile": "b300"}}


merged = {
    "hardware": {"hardware_profile_override": "b300"},
    "results": [row("ltx2_twostage_t2v", "sglang", 2.87), row("ltx2_twostage_t2v", "trtllm-visual", 3.55),
                row("ltx2.5_twostage_t2v_2gpus", "sglang", 5.24), row("ltx2.5_twostage_t2v_2gpus", "vllm-omni", 6.0)],
    "throughput_results": [],
}
args = dict(run_id="t", run_label="t", gpu="4x B300", date="2026-10-24")
rows = {r["case_id"]: r for r in build_sections(merged, config, **args, retired_cases=retired)[0]["rows"]}
check("the retired case's measured row publishes", "ltx2_twostage_t2v" in rows, sorted(rows))
check("...after the live cases", list(rows)[-1] == "ltx2_twostage_t2v", list(rows))
check("a retired case the run did not measure stays out", "ltx2.3_twostage_t2v_2gpus" not in rows)
ltx2 = rows.get("ltx2_twostage_t2v", {"cells": {}})
check("its label and shape come from the frozen definition",
      ltx2.get("case") == "LTX-2 two-stage T2V · 2 GPUs" and ltx2.get("details", "").startswith("768x512x121f"), ltx2)
check("its unmeasured frameworks keep their classification",
      ltx2["cells"].get("lightx2v", {}).get("status") == "unsupported", ltx2["cells"].get("lightx2v"))
check("its ratio is against its own sglang row", ltx2["cells"].get("trtllm-visual", {}).get("ratio_to_sglang") == 1.237,
      ltx2["cells"].get("trtllm-visual"))
without = {r["case_id"] for r in build_sections(merged, config, **args)[0]["rows"]}
check("without the frozen definitions the row is dropped (why the file exists)", "ltx2_twostage_t2v" not in without)

sys.exit(fail)
