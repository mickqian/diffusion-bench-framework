#!/usr/bin/env python3
"""The cookbook track: its lint, its generated files, and its runner's guards.

A cookbook cell claims to run one framework's published workload and command
verbatim. Each check below is a way that claim went wrong, or nearly did, in
the 2xH200 cookbook round (2026-10-30): a headline 4-GPU TRT-LLM config leaked
`parallel_vae_size: 4` and `torch_compile_config` into "verbatim" 1-GPU arms,
the harness turned LightX2V compile on where the source leaves it off, and
the image's sglang tree was the pin on one box and another commit on the next.
"""
import copy
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bcc", ROOT / "scripts" / "build_cookbook_config.py")
bcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcc)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        fail = 1


IN_SCOPE = list(json.load(open(ROOT / "configs/benchmark/frameworks.json"))["frameworks"])
BASES = bcc.base_cases()
RECIPE = {"repo": "ModelTC/LightX2V", "path": "configs/z_image/z_image_turbo_t2i.json",
          "commit": "4fe984c0", "commit_date": "2026-10-10", "section": "whole file"}
SGL = {"repo": "sgl-project/sglang", "path": "docs/src/snippets/configs/TongyiMAI/z-image-turbo.jsx",
       "commit": "2b8e4cde9564", "commit_date": "2026-10-23", "section": "builder"}
GOOD = {
    "id": "cb_test",
    "base_case": "zimage_turbo_t2i_1024",
    "set": {"num_gpus": 1, "width": 1664, "height": 928},
    "unset": ["frameworks.vllm-omni.extra_env.DIFFUSION_ATTENTION_BACKEND"],
    "cookbook": {"framework": "lightx2v", **RECIPE, "source_hardware": "1 GPU", "source_metric": "pipeline cost",
                 "source_gpu": [],
                 "workload": {"width": 1664, "height": 928, "num_gpus": 1, "dit_forwards": 9, "metric": "pipeline cost",
                              "warmup": "2 request(s) at 3 steps", "repeats": "one run"}},
    "forwards": {"lightx2v": 9, "sglang": 9},
    "deviations": [{"field": "metric", "source": "pipeline cost", "run": "client_e2e_median", "reason": "one client"},
                   {"field": "repeats", "source": "one run", "run": "median of 5", "reason": "same median"}],
    "arms": {
        "lightx2v": {"cb-lx-verbatim": {"tier": "verbatim", "num_gpus": 1, "upstream_recipe": RECIPE,
                                        "serve_args": "", "lightx2v_config": {
                                            "attn_type": "flash_attn3", "enable_cfg": False,
                                            "sample_guide_scale": 0.0, "use_compile": False}}},
        "sglang": {"cb-sgl-cookbook": {"tier": "verbatim", "num_gpus": 1, "upstream_recipe": SGL,
                                       "serve_args": "--ulysses-degree=1 --ring-degree=1"}},
    },
}


DEFAULTS = json.load(open(ROOT / "configs/comparison_configs.json"))["benchmark_defaults"]


def build(mutate=None):
    cell = copy.deepcopy(GOOD)
    if mutate:
        mutate(cell)
    return bcc.build_cell(cell, BASES, IN_SCOPE, DEFAULTS)


def errors_of(mutate):
    return build(mutate)[1]


def expect_error(name, mutate, needle):
    errs = errors_of(mutate)
    check(name, any(needle in e for e in errs), errs or "no error")


print("lint")
case, errs = build()
check("a well-formed cell builds clean", not errs, errs)
check("only arm frameworks are kept", set(case["frameworks"]) == {"lightx2v", "sglang"}, list(case["frameworks"]))
check("every other framework is classified not_run",
      set(case["report_framework_statuses"]) == set(IN_SCOPE) - {"lightx2v", "sglang"})
check("result-visible track and tiers", case["track"] == "cookbook"
      and case["cookbook"]["arms"]["sglang"]["cb-sgl-cookbook"]["tier"] == "verbatim")
check("no headline profile survives", all(
    set(b["command_profiles"]) <= {"cb-lx-verbatim", "cb-sgl-cookbook"} for b in case["frameworks"].values()))
check("headline serve_args are not inherited", all(b["serve_args"] == "" for b in case["frameworks"].values()))

expect_error("a field that differs from the source needs a deviation",
             lambda c: c["set"].update(width=1024), "width: runs 1024 where the source has 1664")
expect_error("a deviation for a field that matches is stale",
             lambda c: c["deviations"].append({"field": "width", "source": 1664, "run": 1664, "reason": "x"}),
             "is stale")
expect_error("a deviation must quote what the build sees",
             lambda c: c["deviations"][0].update(source="something else"), "the build sees")
expect_error("a deviation naming no workload field fails",
             lambda c: c["deviations"].append({"field": "fps", "source": 1, "run": 2, "reason": "x"}),
             "names no field")
expect_error("forwards differing from the source need a deviation",
             lambda c: c["forwards"].update(sglang=8), "dit_forwards.sglang")
check("forwards null (unverified) is not compared",
      not errors_of(lambda c: c["forwards"].update(sglang=None)))
expect_error("forwards must be stated for every arm framework",
             lambda c: c["forwards"].pop("sglang"), "forwards.sglang missing")
expect_error("a verbatim arm needs upstream_recipe",
             lambda c: c["arms"]["sglang"]["cb-sgl-cookbook"].pop("upstream_recipe"), "verbatim arm: upstream_recipe")
expect_error("the source framework's verbatim arm must cite the cell's source",
             lambda c: c["arms"]["lightx2v"]["cb-lx-verbatim"].update(upstream_recipe={**RECIPE, "path": "other.json"}),
             "must cite the cell's source")
expect_error("arm names carry no hardware token",
             lambda c: c["arms"]["sglang"].update({"cb-sgl-h200": c["arms"]["sglang"]["cb-sgl-cookbook"]}),
             "no hardware")
expect_error("an arm needs serve_args, nothing is inherited",
             lambda c: c["arms"]["sglang"]["cb-sgl-cookbook"].pop("serve_args"), "serve_args missing")
expect_error("arm num_gpus must equal the cell's",
             lambda c: c["arms"]["sglang"]["cb-sgl-cookbook"].update(num_gpus=2), "num_gpus 2 != cell num_gpus 1")


def no_sglang_verbatim(c):
    c["arms"]["sglang"] = {"cb-sgl-tuned": {"tier": "tuned", "num_gpus": 1, "serve_args": "", "notes": "ours"}}


expect_error("no verbatim sglang arm needs a stated gap", no_sglang_verbatim, "sglang_verbatim_gap")
check("a stated gap makes it legal", not errors_of(
    lambda c: (no_sglang_verbatim(c), c["cookbook"].update(sglang_verbatim_gap="no 2-GPU command"))))
expect_error("a tuned arm needs notes", lambda c: (no_sglang_verbatim(c), c["arms"]["sglang"]["cb-sgl-tuned"].pop("notes"),
                                                   c["cookbook"].update(sglang_verbatim_gap="x")), "needs `notes`")
expect_error("a matched-lossy arm lists its lossy features",
             lambda c: c["arms"]["sglang"].update({"cb-sgl-sage": {"tier": "matched-lossy", "num_gpus": 1,
                                                                    "serve_args": "", "requires": ["x"]}}),
             "lossy features")
expect_error("unknown tier", lambda c: c["arms"]["sglang"]["cb-sgl-cookbook"].update(tier="best"), "tier 'best'")
expect_error("ids start with cb_", lambda c: c.update(id="zimage_x"), "must start with `cb_`")
expect_error("unset of a missing path fails", lambda c: c["unset"].append("frameworks.nope.x"), "unset")

expect_error("timing is part of the workload: warmup must be stated",
             lambda c: c["cookbook"]["workload"].pop("warmup"), "workload.warmup missing")
expect_error("a source warmup the harness does not run is a deviation",
             lambda c: c["cookbook"]["workload"].update(warmup="1 request"), "warmup: runs '2 request(s) at 3 steps'")
expect_error("source_gpu must be hardware tokens",
             lambda c: c["cookbook"].update(source_gpu=["hopper"]), "source_gpu")
expect_error("an arm reading a machine path names how to create it",
             lambda c: c["arms"]["sglang"]["cb-sgl-cookbook"].update(
                 serve_args="--minimax-h3-adaln-cache-path /scratch/bench/x.safetensors"), "`requires`")
expect_error("a matched-lossy arm names its prerequisites",
             lambda c: c["arms"]["sglang"].update({"cb-sgl-sage": {"tier": "matched-lossy", "num_gpus": 1,
                                                                    "serve_args": "", "lossy": ["sage_attn"]}}),
             "`requires`")

# The headline Z-Image vLLM entry sets DIFFUSION_ATTENTION_BACKEND; a verbatim
# vLLM arm whose recipe does not must not pick it up silently.
VO = {"repo": "vllm-project/vllm-omni", "path": "recipes/Tongyi-MAI/Z-Image-Turbo.md", "commit": "c586c28e",
      "commit_date": "2026-10-25", "section": "x"}


def vllm_arm(c, unset=True):
    c["forwards"]["vllm-omni"] = 9
    c["arms"]["vllm-omni"] = {"cb-vllm-verbatim": {"tier": "verbatim", "num_gpus": 1, "upstream_recipe": VO,
                                                   "serve_args": ""}}
    if not unset:
        c["unset"] = []


expect_error("a verbatim arm may not inherit a headline setting", lambda c: vllm_arm(c, unset=False),
             "inherits ['extra_env.DIFFUSION_ATTENTION_BACKEND']")
check("unset in the cell removes the leak", not errors_of(vllm_arm))
check("allow_inherited names harness plumbing", not errors_of(lambda c: (
    vllm_arm(c, unset=False),
    c["arms"]["vllm-omni"]["cb-vllm-verbatim"].update(allow_inherited=["extra_env.DIFFUSION_ATTENTION_BACKEND"]))))
case, _ = build(lambda c: vllm_arm(c, unset=False))
check("tuned/other arms still report what they inherit",
      case["frameworks"]["vllm-omni"]["command_profiles"]["cb-vllm-verbatim"]["inherited"]
      == ["extra_env.DIFFUSION_ATTENTION_BACKEND"])
expect_error("a verbatim lightx2v arm pins use_compile",
             lambda c: c["arms"]["lightx2v"]["cb-lx-verbatim"]["lightx2v_config"].pop("use_compile"),
             "use_compile")
case, _ = build()
case, _ = build(lambda c: (c["arms"]["lightx2v"]["cb-lx-verbatim"]["lightx2v_config"].pop("enable_cfg"),
                           c["arms"]["lightx2v"]["cb-lx-verbatim"].update(allow_inherited=["lightx2v_config.enable_cfg"])))
check("lightx2v config is resolved in full into the arm",
      "lightx2v_config" not in case["frameworks"]["lightx2v"]
      and case["frameworks"]["lightx2v"]["command_profiles"]["cb-lx-verbatim"]["lightx2v_config"]["enable_cfg"] is False)

print("generated files")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "build_cookbook_config.py", tmp / "scripts")
    shutil.copytree(ROOT / "configs", tmp / "configs")
    shutil.copytree(ROOT / "src", tmp / "src")
    proc = subprocess.run([sys.executable, str(tmp / "scripts" / "build_cookbook_config.py")],
                          capture_output=True, text=True)
    check("the committed cells build", proc.returncode == 0, proc.stderr[-2000:])
    for rel in ("configs/cookbook_configs.json", "configs/benchmark/cookbook/COOKBOOK.md"):
        check(f"{rel} matches its sources (run scripts/build_cookbook_config.py)",
              (tmp / rel).read_text() == (ROOT / rel).read_text())
    headline = {c["id"] for c in json.load(open(ROOT / "configs/comparison_configs.json"))["cases"]}
    check("no cookbook cell is in the headline config", not any(i.startswith("cb_") for i in headline))

print("runner guards")
with tempfile.TemporaryDirectory() as td:
    tree = pathlib.Path(td) / "sglang"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    env = {**os.environ, "DBF_REPO_DIR": str(ROOT), "DBF_LOG_DIR": td}
    env.pop("CB_SGLANG_COMMIT", None)
    run = lambda *args, **extra: subprocess.run(
        ["bash", str(ROOT / "scripts/run_cookbook_cells.sh"), "t", "0", "39000", *args],
        env={**env, **extra}, capture_output=True, text=True)
    cell = "cb_qwen21_t2i_1024_50s_vllm"
    r = run(cell, "sglang:cb-sgl-cookbook")
    check("an sglang arm needs CB_SGLANG_COMMIT", r.returncode != 0 and "CB_SGLANG_COMMIT" in r.stderr, r.stderr)
    r = run(cell, "sglang:cb-sgl-cookbook", CB_SGLANG_COMMIT="3a01d4cc", DBF_SGLANG_PYTHONPATH=str(tree))
    check("an sglang tree off the pin is refused", r.returncode == 2 and "not CB_SGLANG_COMMIT" in r.stderr, r.stderr)
    r = run(cell, "sglang:cb-nope")
    check("an unknown arm is refused before anything runs", r.returncode == 2 and "has no arm" in r.stderr, r.stderr)
    r = run("cb_nope")
    check("an unknown cell is refused", r.returncode == 2 and "no cookbook cell" in r.stderr, r.stderr)
    r = run("cb_ltx25_t2v_544_1gpu_vllm", "sglang:cb-nope", DBF_HARDWARE_PROFILE="h200")
    check("a box outside the source's GPU classes is called out",
          "HARDWARE DEVIATION cb_ltx25_t2v_544_1gpu_vllm" in r.stderr, r.stderr)
    r = run(cell, "sglang:cb-nope", DBF_HARDWARE_PROFILE="h200")
    check("a box the source was written for is not", "HARDWARE DEVIATION" not in r.stderr, r.stderr)

sys.exit(fail)
