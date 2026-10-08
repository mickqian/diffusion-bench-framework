#!/usr/bin/env python3
"""The competitors' SELECTED command must be compiled and resident on datacenter cards.

The build enforced "best lossless" for sglang only. MiniMax-H3's vLLM-Omni
command was the recipe's 2x24/32 GB consumer path (TP2 + distributed layerwise
offload + --enforce-eager) and ran on 183 GB B200s in every round from
2026-08-07 to 2026-09-25, and SELECTED.md -- the table a round is reviewed
from -- listed sglang rows only, so nobody saw it.

Checked here: each spelling of eager/offload per framework fails on datacenter
hardware, a dated `policy_exception` passes and an undated one does not, the rtx
classes are not linted, the pre-existing violations are held in a list that only
shrinks, and SELECTED.md carries every framework's rows.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_benchmark_config as bbc  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:300]) if detail and not cond else ''}")
    if not cond:
        fail = 1


def lint(fw, body):
    bbc.SELECTED_ROWS.clear()
    bbc.MISSING_RECIPES.clear()
    return bbc._lint_competitor_policy("case", fw, body)


def cells(errors):
    return [cell for cell, _ in errors]


def profile(**fields):
    return {"command_profiles": {"default": fields}}


# FastVideo's offload switches all default to true, so resident means each one false.
FV_RESIDENT = " ".join(
    f"--generator.engine.offload.{part} false" for part in ("dit", "dit_layerwise", "text_encoder", "image_encoder", "vae")
)
FV_COMPILED = "--generator.pipeline.experimental.inference_torch_compile true"


with tempfile.TemporaryDirectory() as td:
    eager_yaml = os.path.join(td, "stage.yaml")
    pathlib.Path(eager_yaml).write_text("stage_args:\n  - engine_args:\n      enforce_eager: true\n")
    offload_yaml = os.path.join(td, "deploy.yaml")
    pathlib.Path(offload_yaml).write_text("stages:\n  diffusion_offload_config:\n    mode: layer\n")
    resident_yaml = os.path.join(td, "resident.yaml")
    pathlib.Path(resident_yaml).write_text("stage_args:\n  - engine_args:\n      enforce_eager: false\n")

    violating = [
        ("--enforce-eager", "vllm-omni", profile(serve_args="--num-gpus 2 --enforce-eager")),
        ("--compilation-config mode 0", "vllm-omni", profile(serve_args='--compilation-config {"mode":0}')),
        ("--enable-cpu-offload", "vllm-omni", profile(serve_args="--enable-cpu-offload")),
        ("--enable-layerwise-offload", "vllm-omni", profile(serve_args="--enable-layerwise-offload")),
        ("--enable-distributed-layerwise-offload", "vllm-omni",
         profile(serve_args="--tensor-parallel-size 2 --enable-distributed-layerwise-offload")),
        ("--diffusion-offload-config", "vllm-omni",
         profile(serve_args='--diffusion-offload-config {"mode":"layer","components":["dit"]}')),
        ("--dlo-*", "vllm-omni", profile(serve_args="--dlo-resident-layers 20")),
        ("enforce_eager in a stage YAML", "vllm-omni", profile(serve_args=f"--stage-configs-path {eager_yaml}")),
        ("offload config in a deploy YAML", "vllm-omni", profile(serve_args=f"--deploy-config {offload_yaml}")),
        ("a deploy YAML the lint cannot read", "vllm-omni", profile(serve_args="--deploy-config does/not/exist.yaml")),
        ("cpu_offload", "lightx2v", profile(lightx2v_config={"cpu_offload": True, "offload_granularity": "block"})),
        ("t5_cpu_offload", "lightx2v", profile(lightx2v_config={"t5_cpu_offload": True})),
        # The key LightX2V reads, and what upstream's MiniMax-H3 presets ship.
        ("use_compile false", "lightx2v", profile(lightx2v_config={"use_compile": False})),
        # The profile inherits the entry's config, as the runtime merges it.
        ("cpu_offload inherited from the entry", "lightx2v",
         {"lightx2v_config": {"cpu_offload": True}, "command_profiles": {"default": {}}}),
        ("--lowvram", "comfyui", {"comfyui": {"compile": True}, **profile(serve_args="--lowvram")}),
        ("--novram", "comfyui", {"comfyui": {"compile": True}, **profile(serve_args="--novram")}),
        ("--cpu", "comfyui", {"comfyui": {"compile": True}, **profile(serve_args="--cpu")}),
        ("compile false", "comfyui", {"comfyui": {"compile": False}, **profile(serve_args="--gpu-only")}),
        ("offload left at its default", "fastvideo", profile(serve_args=FV_COMPILED)),
        ("text encoder offloaded", "fastvideo",
         profile(serve_args=f"{FV_RESIDENT} {FV_COMPILED} --generator.engine.offload.text_encoder true")),
        ("lazy module load", "fastvideo",
         profile(serve_args=f"{FV_RESIDENT} {FV_COMPILED} --generator.engine.offload.lazy_module_load true")),
        ("H3 sequential load", "fastvideo",
         profile(serve_args=f"{FV_RESIDENT} {FV_COMPILED} --generator.pipeline.experimental.h3_sequential_load true")),
        ("eager DiT", "fastvideo", profile(serve_args=FV_RESIDENT)),
        ("serve_args that are not dotted overrides", "fastvideo", profile(serve_args="--num-gpus 4 true")),
    ]
    for label, fw, body in violating:
        errors = lint(fw, body)
        check(f"{fw} {label}: fails on every datacenter class",
              cells(errors) == [("case", fw, "default")]
              and all(f"selected on {', '.join(bbc.DATACENTER_HARDWARE)}" in msg for _, msg in errors),
              errors)
        dated = json.loads(json.dumps(body))
        dated["command_profiles"]["default"]["policy_exception"] = "Capacity, measured 2026-09-30 on H100."
        check(f"{fw} {label}: passes with a dated policy_exception", lint(fw, dated) == [], lint(fw, dated))

    resident = [
        ("resident usp + tiled VAE", "vllm-omni", profile(serve_args="--num-gpus 2 --usp 2 --vae-use-tiling")),
        ("--compilation-config mode 3", "vllm-omni", profile(serve_args='--compilation-config {"mode":3}')),
        ("--no-enforce-eager", "vllm-omni", profile(serve_args="--no-enforce-eager")),
        ("a stage YAML with enforce_eager false", "vllm-omni",
         profile(serve_args=f"--stage-configs-path {resident_yaml}")),
        # LightX2V reads offload_granularity only when cpu_offload is on.
        ("offload_granularity with cpu_offload false", "lightx2v",
         profile(lightx2v_config={"cpu_offload": False, "t5_cpu_offload": False, "offload_granularity": "model"})),
        ("use_compile true", "lightx2v", profile(lightx2v_config={"use_compile": True, "attn_type": "flash_attn3"})),
        # LightX2V reads only use_compile, so a `compile` key changes nothing.
        ("a dead `compile` key", "lightx2v", profile(lightx2v_config={"compile": False})),
        ("--gpu-only with compile", "comfyui", {"comfyui": {"compile": True}, **profile(serve_args="--gpu-only")}),
        ("--backend pytorch", "trtllm-visual", profile(serve_args="--backend pytorch")),
        ("resident with regional compile", "fastvideo", profile(serve_args=f"{FV_RESIDENT} {FV_COMPILED}")),
        # `fastvideo serve` reads --key=value, turns `-` into `_` and casts True/False case-insensitively.
        ("resident with whole-DiT compile, spelled differently", "fastvideo",
         profile(serve_args=FV_RESIDENT.replace(" false", "=False").replace("dit_layerwise", "dit-layerwise")
                 + " --generator.engine.compile.enabled=True")),
    ]
    for label, fw, body in resident:
        check(f"{fw} {label}: passes", lint(fw, body) == [], lint(fw, body))

undated = profile(serve_args="--enforce-eager", policy_exception="it OOMs")
errors = lint("vllm-omni", undated)
check("an undated policy_exception is an error no allowlist can excuse",
      cells(errors) == [None] and "dated evidence" in errors[0][1], errors)

consumer = {
    "command_profiles": {
        "default": {"serve_args": "--num-gpus 2 --usp 2"},
        "rtx5090-2gpu": {"hardware": ["rtx5090"], "serve_args": "--enforce-eager --enable-distributed-layerwise-offload"},
        "consumer-dlo-eager-20260807": {"serve_args": "--enforce-eager --dlo-resident-layers 20"},
    }
}
check("an rtx-only offload profile is not linted", lint("vllm-omni", consumer) == [], lint("vllm-omni", consumer))
selected = {(hw, name) for _, _, hw, name, *_ in bbc.SELECTED_ROWS}
check("the rtx5090 row is still listed in SELECTED", ("rtx5090", "rtx5090-2gpu") in selected, sorted(selected))
check("a provenance-only profile without `hardware` is never selected",
      not any(name == "consumer-dlo-eager-20260807" for _, name in selected), sorted(selected))
check("a profile with no upstream_recipe is reported",
      ("case", "vllm-omni", "default") in bbc.MISSING_RECIPES, bbc.MISSING_RECIPES)
lint("vllm-omni", profile(serve_args="", upstream_recipe_reason="no upstream recipe for this model"))
check("an upstream_recipe_reason silences that warning", not bbc.MISSING_RECIPES, bbc.MISSING_RECIPES)

recipe = {"repo": "o/r", "path": "a.md", "commit": "7266fc613", "commit_date": "2026-09-28", "section": "s"}
check("a complete upstream_recipe validates",
      bbc._lint_upstream_recipes("case", "vllm-omni", profile(upstream_recipe=recipe)) == [])
bad = bbc._lint_upstream_recipes("case", "vllm-omni", profile(upstream_recipe={**recipe, "commit_date": "Sep 28"}))
check("a malformed upstream_recipe is a build error", bad and "YYYY-MM-DD" in bad[0], bad)


def build(tmp, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != bbc.STRICT_COMPETITOR_POLICY_ENV}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(tmp / "scripts" / "build_benchmark_config.py")],
                          capture_output=True, text=True, env=env)


def fresh_tree(td):
    tmp = pathlib.Path(td)
    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "build_benchmark_config.py", tmp / "scripts")
    shutil.copytree(ROOT / "configs", tmp / "configs")
    shutil.copytree(ROOT / "src", tmp / "src")
    return tmp


def edit_case(tmp, rel, fw, name, change):
    path = tmp / "configs" / "benchmark" / "cases" / rel
    data = json.loads(path.read_text())
    change(data["frameworks"][fw]["command_profiles"][name])
    path.write_text(json.dumps(data, indent=2) + "\n")


with tempfile.TemporaryDirectory() as td:
    tmp = fresh_tree(td)
    proc = build(tmp)
    check("the repo builds with the known violations reported, not fatal", proc.returncode == 0,
          proc.stdout + proc.stderr)
    for cid, fw, name in sorted(bbc.KNOWN_COMPETITOR_VIOLATIONS):
        check(f"known violation printed: {cid}/{fw}[{name}]", f"{cid}/{fw}[{name}]" in proc.stderr)
    selected = (tmp / "configs" / "benchmark" / "SELECTED.md").read_text()
    check("the committed SELECTED.md is what the sources produce",
          selected == (ROOT / "configs" / "benchmark" / "SELECTED.md").read_text(),
          "run scripts/build_benchmark_config.py")
    check("SELECTED.md has a framework column", "| case | framework | hw |" in selected)
    for fw in ("sglang", "vllm-omni", "lightx2v", "trtllm-visual", "comfyui"):
        check(f"SELECTED.md lists {fw} rows", f"| {fw} | h100 |" in selected and f"| {fw} | gb300 |" in selected)
    check("the H3 vLLM-Omni cell on B200 is the measured resident Blackwell command",
          "| minimax_h3_t2va_5s | vllm-omni | b200 | `blackwell-2gpu` |  |  | vllm-project/vllm-omni@" in selected)
    check("its H100 cell is the offload profile with its exception",
          "| minimax_h3_t2va_5s | vllm-omni | h100 | `h100-2gpu-dlo` | yes |" in selected)
    check("a known violation is marked in SELECTED.md",
          "| ltx2_twostage_t2v | lightx2v | h100 | `h100-1gpu` | VIOLATION |" in selected)

    strict = build(tmp, {bbc.STRICT_COMPETITOR_POLICY_ENV: "1"})
    check("strict mode fails on the known violations",
          strict.returncode == 1 and all(f"{c}/{f}[{n}]" in strict.stderr for c, f, n in bbc.KNOWN_COMPETITOR_VIOLATIONS),
          strict.stderr)

with tempfile.TemporaryDirectory() as td:
    tmp = fresh_tree(td)
    edit_case(tmp, "video/minimax_h3_t2va_5s.json", "vllm-omni", "default",
              lambda p: p.update(serve_args=p["serve_args"] + " --enforce-eager"))
    proc = build(tmp)
    check("a new violation fails the build",
          proc.returncode == 1 and "minimax_h3_t2va_5s/vllm-omni[default]" in proc.stderr, proc.stderr)

with tempfile.TemporaryDirectory() as td:
    tmp = fresh_tree(td)
    edit_case(tmp, "video/ltx2_twostage_t2v.json", "lightx2v", "h100-1gpu",
              lambda p: p.update(policy_exception="Capacity: OOM measured 2026-09-30 on H100."))
    proc = build(tmp)
    check("a known violation that is fixed must leave the list",
          proc.returncode == 1 and "no longer violates" in proc.stderr, proc.stderr)

sys.exit(fail)
