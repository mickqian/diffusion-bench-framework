#!/usr/bin/env python3
"""Build a copy of the benchmark config with sglang's TUNING flags stripped.

Every one of the 15 b200 cases pins at least one flag that describes HOW to run
rather than WHAT to run -- parallelism on 8, component residency on 8, attention
backend on 3. Each pin is a claim that we know better than the runtime, and each
one has to be re-validated on every new hardware generation or it silently
becomes the wrong answer.

It has already been the wrong answer once. Cosmos3 T2I pinned `--tp-size 2`,
which makes `_has_explicit_parallel_policy()` true and so SUPPRESSES
`_enable_cfg_parallel_if_supported()` -- the runtime would have picked CFG
parallelism for that model on 2 GPUs, which measured 25% faster. The pin was
costing us the thing it was supposed to secure.

So: strip the tuning flags, keep the ones that select the model and the harness
contract, and measure both. Three outcomes per case, all of them useful --

  default == profile   the pin earns nothing; drop it and let the runtime adapt
  default  > profile   sglang's auto-selection has a real gap worth fixing there
  default  < profile   the pin is actively harming, as Cosmos3 T2I was

Usage:
    python3 scripts/compare_profile_vs_default.py \
        --config configs/comparison_configs.json \
        --out /tmp/config_bare.json [--hardware b200]

Then run the harness twice per case, once with each config, interleaved.
`scripts/run_profile_vs_default.sh` does that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# WHAT to run: selects the model, the pipeline, or the harness contract.
POLICY_FLAGS = {
    "--model-type",
    "--warmup-mode",
    "--pipeline-class-name",
    "--model-variant",
    "--model-path",
}
# HOW to run it: the runtime has an auto path for each of these.
TUNING_FLAGS = {
    "--tp-size",
    "--cfg-parallel-size",
    "--enable-cfg-parallel",
    "--ulysses-degree",
    "--ring-degree",
    "--sp-degree",
    "--kv-gather-degree",
    "--attention-backend",
    "--dit-layerwise-offload",
    "--performance-mode",
    "--enable-torch-compile",
    "--ltx2-two-stage-device-mode",
    "--batching-max-size",
    "--enable-batching-metrics",
    # Component residency. The first pass missed these and the "unclassified
    # flag" note caught them -- they are the same category as
    # --dit-layerwise-offload and are exactly what the runtime's
    # keep_resident_* deployment config decides on its own.
    "--dit-cpu-offload",
    "--text-encoder-cpu-offload",
    "--pin-cpu-memory",
    "--layerwise-offload-components",
    "--dit-offload-prefetch-size",
    "--dit-layerwise-resident-layers",
}


def strip_tuning(serve_args: str) -> tuple[str, list[str]]:
    """Return (policy-only args, the tuning flags that were removed)."""
    tokens = serve_args.split()
    kept: list[str] = []
    removed: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            # A bare value whose flag we already handled; carry it with the flag.
            kept.append(tok)
            i += 1
            continue
        has_value = i + 1 < len(tokens) and not tokens[i + 1].startswith("--")
        value = tokens[i + 1] if has_value else ""
        if tok in TUNING_FLAGS:
            removed.append(f"{tok} {value}".strip())
        else:
            kept.append(tok)
            if has_value:
                kept.append(value)
            if tok not in POLICY_FLAGS:
                print(f"    note: keeping unclassified flag {tok}")
        i += 2 if has_value else 1
    return " ".join(kept), removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/comparison_configs.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hardware", default="b200")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    total = 0
    for case in cfg.get("cases", []):
        fw = (case.get("frameworks") or {}).get("sglang")
        if not isinstance(fw, dict):
            continue
        cid = case.get("id", "?")
        touched: list[str] = []
        if isinstance(fw.get("serve_args"), str):
            fw["serve_args"], removed = strip_tuning(fw["serve_args"])
            touched += removed
        for prof in (fw.get("command_profiles") or {}).values():
            if isinstance(prof, dict) and isinstance(prof.get("serve_args"), str):
                prof["serve_args"], removed = strip_tuning(prof["serve_args"])
                touched += removed
        uniq = sorted({t.split()[0] for t in touched})
        if uniq:
            total += 1
            print(f"  {cid:34s} dropped: {' '.join(uniq)}")
        else:
            print(f"  {cid:34s} (nothing pinned)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n{total} case(s) had sglang tuning flags; bare config written to {args.out}")


if __name__ == "__main__":
    main()
