#!/usr/bin/env python3
"""ComfyUI requests must do the same work every time, and only that work.

ComfyUI caches node outputs keyed on their inputs, and the harness repeats an
identical request (same prompt, same seed) to take a median. Without a per-
request change the second request is answered from cache and times nothing,
so every template carries a nonce. The nonce must also be the ONLY thing that
changes between two requests -- anything else (a seed, a size) would make the
repeats measure different work.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diffusion_bench import comfyui_client as cc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


# typed substitution: a whole-string placeholder keeps its type
out = cc.render_workflow({"a": "{{n}}", "b": "x-{{n}}", "c": ["1", 0]}, {"n": 7})
check("whole-string placeholder keeps an int an int", out["a"] == 7)
check("embedded placeholder interpolates", out["b"] == "x-7")
check("node links are left alone", out["c"] == ["1", 0])
try:
    cc.render_workflow({"a": "{{missing}}"}, {})
    check("an unknown placeholder is an error", False)
except KeyError:
    check("an unknown placeholder is an error", True)

cfg = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
seen = 0
for case in cfg["cases"]:
    fw = (case.get("frameworks") or {}).get("comfyui")
    if not fw:
        continue
    blocks = {"inline": fw["comfyui"]}
    for name, prof in (fw.get("command_profiles") or {}).items():
        if prof.get("comfyui"):
            blocks[name] = {**fw["comfyui"], **prof["comfyui"]}
    for name, spec in blocks.items():
        seen += 1
        where = f"{case['id']}[{name}]"
        check(f"{where}: workflow inlined by the build", isinstance(spec.get("workflow_graph"), dict))
        first = cc.render_workflow(spec["workflow_graph"], cc.workflow_params(case, spec))
        second = cc.render_workflow(spec["workflow_graph"], cc.workflow_params(case, spec))

        def strip(graph):
            return {
                nid: {**node, "inputs": {k: v for k, v in node["inputs"].items() if k != cc.NONCE_INPUT}}
                for nid, node in graph.items()
            }

        nonces = [n["inputs"][cc.NONCE_INPUT] for n in first.values() if cc.NONCE_INPUT in n["inputs"]]
        check(f"{where}: carries a nonce", bool(nonces))
        check(f"{where}: two requests differ only in the nonce", strip(first) == strip(second))
        check(
            f"{where}: the nonce changes per request",
            {n["inputs"].get(cc.NONCE_INPUT) for n in first.values()}
            != {n["inputs"].get(cc.NONCE_INPUT) for n in second.values()},
        )
        steps = [n["inputs"].get("steps") for n in first.values() if n["class_type"] in cc.SAMPLER_CLASSES | {"BasicScheduler", "Flux2Scheduler", "LTXVScheduler", "Ideogram4Scheduler"}]
        check(f"{where}: the case's steps reach the sampling schedule", case["num_inference_steps"] in steps, f"{steps}")

check("the built config has ComfyUI cells", seen >= 10, f"{seen}")
sys.exit(fail)
