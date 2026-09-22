#!/usr/bin/env python3
"""The best-lossless guard must see every profile the repo can actually select.

Two holes, both found while adding the Qwen-Image-2.1 hardware sweep:

* `POLICY_HARDWARE` listed only h100 and b200, so the case's `rtx5090-*` and
  `rtx4090-*` profiles -- which enable offload, and must therefore justify
  themselves -- were never linted at all. A profile the guard cannot select is
  a profile the guard cannot check.
* `_OFFLOAD_ENABLE_RE` knew one spelling of "turn offload on". The Qwen-Image-2.1
  cookbook emits two others, `--component-residency dit=layerwise-offload` and
  `--layerwise-offload-components all`, and both walked straight past it.

The first check is the self-maintaining one: declare a profile for a new
hardware class and it fails until that class is linted too.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_benchmark_config as bbc  # noqa: E402
from diffusion_bench.config_guard import profile_hardware_values  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          f"{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        fail = 1


declared = set()
for path in (ROOT / "configs" / "benchmark" / "cases").rglob("*.json"):
    if path.name == "_order.json":
        continue
    case = json.loads(path.read_text())
    for fw in (case.get("frameworks") or {}).values():
        for cfg in (fw.get("command_profiles") or {}).values():
            declared.update(profile_hardware_values(cfg))

unlinted = sorted(
    hw for hw in declared
    if not any(hw in linted or linted in hw for linted in bbc.POLICY_HARDWARE)
)
check(
    "every hardware class with a declared profile is linted",
    not unlinted,
    f"{unlinted} appear in `hardware` lists but no POLICY_HARDWARE entry selects them",
)

# The three spellings that all mean "offload is on". A guard that knows one of
# them lets the other two publish an offloaded cell as if it were resident.
on = [
    "--dit-layerwise-offload true",
    "--text-encoder-cpu-offload true",
    "--component-residency dit=resident text_encoder=layerwise-offload vae=resident",
    "--component-residency dit=cpu-offload",
    "--layerwise-offload-components all",
]
off = [
    "--dit-layerwise-offload false",
    "--tp-size 2 --performance-mode speed",
    "--component-residency dit=resident vae=resident",
]
for args in on:
    check(f"caught: {args[:58]}", bool(bbc._OFFLOAD_ENABLE_RE.search(args)))
for args in off:
    check(f"not flagged: {args[:58]}", not bbc._OFFLOAD_ENABLE_RE.search(args))

# An offloading profile must carry dated evidence, and the new consumer ones do.
case = json.loads(
    (ROOT / "configs" / "benchmark" / "cases" / "image"
     / "qwen_image_21_t2i_1024.json").read_text()
)
for name, cfg in case["frameworks"]["sglang"]["command_profiles"].items():
    if bbc._OFFLOAD_ENABLE_RE.search(cfg.get("serve_args", "")):
        exc = str(cfg.get("policy_exception", ""))
        check(f"{name} justifies its offload", bool(exc), "no policy_exception")
        check(f"{name} cites a date", bool(exc) and "2026-" in exc, exc[:70])

sys.exit(fail)
