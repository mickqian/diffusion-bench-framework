#!/usr/bin/env python3
"""One-time migration: split the monolithic comparison_configs.json into the
explicit configs/benchmark/ tree (one file per case, every case carrying an
explicit status for all four in-scope frameworks). Run once; thereafter edit
the per-case files and regenerate with build_benchmark_config.py.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "configs", "comparison_configs.json")
OUT = os.path.join(REPO, "configs", "benchmark")
CASES_DIR = os.path.join(OUT, "cases")

FRAMEWORKS = ["sglang", "vllm-omni", "lightx2v", "trtllm-visual"]
IMAGE_TASKS = {"text-to-image", "image-edit", "image-to-image"}


def domain_of(task: str) -> str:
    return "image" if task in IMAGE_TASKS else "video"


def main():
    cfg = json.load(open(SRC))
    for d in ("image", "video"):
        os.makedirs(os.path.join(CASES_DIR, d), exist_ok=True)

    for case in cfg.get("cases", []):
        cid = case["id"]
        task = case.get("task", "")
        statuses = case.get("report_framework_statuses", {})
        reasons = case.get("report_framework_reasons", {})
        fw_in = case.get("frameworks", {})

        fw_out = {}
        for fw in FRAMEWORKS:
            if fw in statuses:
                # an explicit report status (failed/oom/unsupported/...) wins over a
                # bare frameworks entry; keep the config as provenance if present
                fw_out[fw] = {"status": statuses[fw], **(fw_in.get(fw, {}))}
                if fw in reasons:
                    fw_out[fw]["reason"] = reasons[fw]
            elif fw in fw_in:
                fw_out[fw] = {"status": "supported", **fw_in[fw]}
            else:
                # was silently absent -> make it explicit and auditable
                fw_out[fw] = {
                    "status": "not_run",
                    "reason": "Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported.",
                }

        out = {k: v for k, v in case.items()
               if k not in ("frameworks", "report_framework_statuses", "report_framework_reasons")}
        out["frameworks"] = fw_out

        path = os.path.join(CASES_DIR, domain_of(task), f"{cid}.json")
        json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
        print("wrote", os.path.relpath(path, REPO))

    print(f"\n{len(cfg.get('cases', []))} cases decomposed into {os.path.relpath(CASES_DIR, REPO)}/")


if __name__ == "__main__":
    main()
