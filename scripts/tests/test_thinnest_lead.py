#!/usr/bin/env python3
"""The report must show where SGLang-Diffusion's lead is thinnest, not only a win count.

For weeks the round summary said "SGLang-Diffusion fastest in 14/15 cases" while
an external team measured SGLang slower on MiniMax-H3 in the few-step, four-GPU
regime. A count cannot show a margin: 14/15 reads the same whether the wins are
2.5x or 1.03x, and the one loss is not named. So each round publishes, per mode,
the three comparable cases with the smallest margin over the fastest competitor
-- losses first -- with sglang's stage sum and unattributed tail beside them.

"Comparable" is the headline's own rule (two or more frameworks measured), so the
list and the count cannot disagree about which rows exist. A row sglang did not
measure is still comparable there, and here it has no lead at all.
"""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.generate_dashboard import build_issue_report_comment  # noqa: E402
from diffusion_bench.page_export import build_sections, thinnest_lead  # noqa: E402

spec = importlib.util.spec_from_file_location("publish_bench_run", ROOT / "scripts" / "publish_bench_run.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def case_cfg(cid, gpus=2):
    return {"id": cid, "num_gpus": gpus, "width": 1024, "height": 1024,
            "num_inference_steps": 8, "guidance_scale": 1.0}


def single(cid, fw, lat, metrics=None, gpus=2):
    if lat is None:
        return {"case_id": cid, "framework": fw, "latency_s": None, "error": "server exited"}
    return {"case_id": cid, "framework": fw, "latency_s": lat, "num_gpus": gpus,
            "metrics": dict(metrics or {})}


def tput(cid, fw, qps):
    return {"case_id": cid, "framework": fw, "num_gpus": 2,
            "metrics": {"throughput_rps": qps, "num_requests": 4, "max_concurrency": 2}}


def sections_for(cases, results=(), throughput=()):
    merged = {"results": list(results), "throughput_results": list(throughput)}
    config = {"cases": [case_cfg(c) for c in cases]}
    secs = build_sections(merged, config, run_id="t", run_label="t", gpu="2x Test GPU", date="2026-09-30")
    return {s["mode"]: s for s in secs}, merged


def ids(entries):
    return [e["case_id"] for e in entries]


TAIL = {"diagnostic_client_latency_s": 2.95, "diagnostic_stage_sum_s": 2.41,
        "unattributed_tail_s": 0.54, "unattributed_tail_frac": 0.183}

print("single_e2e ordering")
# tie_b precedes tie_a, so case order and name order disagree
CASES = ["win_big", "loss", "solo", "tie_b", "tie_a", "even", "rival_tie"]
secs, merged = sections_for(CASES, [
    single("win_big", "sglang", 1.0), single("win_big", "vllm-omni", 3.0),
    single("loss", "sglang", 2.0, TAIL), single("loss", "vllm-omni", 1.9),
    # sglang alone: no competitor ran, so there is no margin to rank
    single("solo", "sglang", 0.5), single("solo", "vllm-omni", None),
    single("tie_b", "sglang", 2.0), single("tie_b", "vllm-omni", 2.1),
    single("tie_a", "sglang", 4.0), single("tie_a", "lightx2v", 4.2),
    single("even", "sglang", 1.0), single("even", "vllm-omni", 1.0),
    single("rival_tie", "sglang", 1.0), single("rival_tie", "vllm-omni", 1.2),
    single("rival_tie", "lightx2v", 1.2),
])
sec = secs["single_e2e"]
top = thinnest_lead(sec, merged, "2x Test GPU")
check("three cases, the loss first", ids(top) == ["loss", "even", "tie_b"], str(ids(top)))
full = thinnest_lead(sec, merged, "2x Test GPU", n=10)
check(
    "equal margins keep case order",
    ids(full) == ["loss", "even", "tie_b", "tie_a", "rival_tie", "win_big"],
    str(ids(full)),
)
check("a single-framework row is excluded", "solo" not in ids(full))
check(
    "same rows as the headline counts",
    len(full) == pub._headline_summary(sec, types.SimpleNamespace(gpu="x"))["comparable_rows"],
)
loss = top[0]
check("the loss carries both sides", loss["sglang"]["latency_s"] == 2.0
      and loss["fastest_competitor"] == {"framework": "vllm-omni", "gpus": 2, "latency_s": 1.9},
      json.dumps(loss))
check("ratio is competitor/sglang, as on the page", loss["ratio_to_sglang"] == 0.95)
check("hardware recorded", loss["hardware"] == "2x Test GPU")
check("tail fields carried from sglang's metrics",
      all(loss["sglang"].get(k) == v for k, v in TAIL.items()), json.dumps(loss["sglang"]))
check("absent tail fields stay absent, not zero", "unattributed_tail_s" not in top[1]["sglang"])
rival = next(e for e in full if e["case_id"] == "rival_tie")
check("tied competitors resolve in framework order",
      rival["fastest_competitor"]["framework"] == "vllm-omni")

print("rows sglang did not measure")
secs, merged = sections_for(["ok", "no_sgl_two", "no_sgl_one"], [
    single("ok", "sglang", 1.0), single("ok", "vllm-omni", 1.2),
    single("no_sgl_two", "sglang", None), single("no_sgl_two", "vllm-omni", 5.0),
    single("no_sgl_two", "lightx2v", 6.0),
    single("no_sgl_one", "vllm-omni", 5.0),
])
got = thinnest_lead(secs["single_e2e"], merged, "2x Test GPU")
check("comparable without sglang ranks first", ids(got) == ["no_sgl_two", "ok"], str(ids(got)))
check("...with sglang's status and no ratio",
      got[0]["sglang"] == {"status": "failed"} and "ratio_to_sglang" not in got[0], json.dumps(got[0]))

print("throughput ranks the other way")
secs, merged = sections_for(["t_win", "t_loss", "t_close"], throughput=[
    tput("t_win", "sglang", 0.5), tput("t_win", "vllm-omni", 0.3),
    tput("t_loss", "sglang", 0.2), tput("t_loss", "vllm-omni", 0.25),
    tput("t_close", "sglang", 0.1), tput("t_close", "vllm-omni", 0.09),
], results=[single("t_loss", "sglang", 9.0, TAIL), single("t_loss", "vllm-omni", 9.5)])
got = thinnest_lead(secs["throughput"], merged, "2x Test GPU")
check("higher qps is the better side", ids(got) == ["t_loss", "t_close", "t_win"], str(ids(got)))
check("qps compared, not latency", got[0]["metric"] == "qps" and got[0]["sglang"]["qps"] == 0.2)
check("single_e2e tail never lands on a throughput row", "unattributed_tail_s" not in got[0]["sglang"])

print("tracker issue table")
REAL = ["qwen_image_2512_t2i_1024", "zimage_turbo_t2i_1024", "flux1_dev_t2i_1024",
        "ltx2.3_twostage_t2v_2gpus"]
run = {
    "timestamp": "2026-09-30T00:00:00+00:00",
    "commit_sha": "0" * 40,
    "hardware": {"gpus": ["Test GPU"] * 4},
    "results": [
        single(REAL[0], "sglang", 2.776, TAIL), single(REAL[0], "vllm-omni", 2.638),
        single(REAL[1], "sglang", 0.469), single(REAL[1], "vllm-omni", 0.486),
        single(REAL[2], "sglang", 2.392), single(REAL[2], "vllm-omni", 3.352),
        single(REAL[3], "sglang", 7.753),
    ],
}
md = build_issue_report_comment(run)
lines = md.splitlines()
head = next((i for i, ln in enumerate(lines) if ln == "### Thinnest lead - single_e2e"), None)
check("the section exists", head is not None)
if head is not None:
    body = []
    for ln in lines[head + 1:]:
        if ln.startswith("#"):
            break
        body.append(ln)
    table = [ln for ln in body if ln.strip()]
    check("data only: nothing but the table", all(ln.startswith("|") for ln in table), str(table[:1]))
    check("before the per-case tables", head < lines.index(f"### {REAL[0]}"))
    rows = [ln.split(" | ")[0].lstrip("| ") for ln in table[2:]]
    check("rows ranked, sglang-only case excluded", rows == REAL[:3], str(rows))
    first = table[2]
    check("the loss shows its ratio", "| 0.950x |" in first, first)
    check("stage sum and tail shown when present", "| 2.950 | 2.410 | 0.540 |" in first, first)
    check("and '-' when absent", table[3].endswith("| - | - | - |"), table[3])
check("no throughput table without throughput data", "### Thinnest lead - throughput" not in md)

print("latest-cross-framework.json")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    merged = {"results": [single("loss", "sglang", 2.0, TAIL), single("loss", "vllm-omni", 1.9),
                          single("even", "sglang", 1.0), single("even", "vllm-omni", 1.0)],
              "throughput_results": []}
    (d / "merged.json").write_text(json.dumps(merged))
    (d / "config.json").write_text(json.dumps({"cases": [case_cfg(c) for c in CASES]}))
    (d / "hist.json").write_text(json.dumps({"sections": []}))
    (d / "latest.json").write_text(json.dumps({"sections": []}))
    (d / "run.sh").write_text("")
    # the real data files are never touched
    pub.ROOT, pub.HISTORICAL, pub.LATEST = d, d / "hist.json", d / "latest.json"
    pub.subprocess = types.SimpleNamespace(run=lambda *a, **k: None)  # no inline refresh
    sys.argv = ["publish_bench_run.py", "--merged", str(d / "merged.json"),
                "--config", str(d / "config.json"), "--run-id", "t", "--label", "t",
                "--gpu", "2x Test GPU", "--reproduce", "run.sh"]
    rc = pub.main()
    latest = json.loads((d / "latest.json").read_text())
check("publish succeeds", rc == 0)
check("thinnest_lead is a list in the latest file",
      ids(latest.get("thinnest_lead") or []) == ["loss", "even"], str(latest.get("thinnest_lead")))
check("the headline is still the count",
      latest.get("summary", {}).get("sglang_diffusion_wins") == 1
      and latest["summary"].get("comparable_rows") == 2, str(latest.get("summary")))

sys.exit(fail)
