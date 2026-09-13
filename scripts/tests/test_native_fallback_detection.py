#!/usr/bin/env python3
"""A cell that fell back to a native implementation must say so.

sglang loads a component through a native/Diffusers fallback when it has no
customized implementation for it. The loader REFUSES to do that only when
tp_size, sp_degree, ulysses_degree, ring_degree or kv_gather_degree is > 1, or
FSDP is on (transformer_loader.validate_native_fallback). With cfg-parallel
alone -- or on a single GPU -- none of those is set and the fallback proceeds
with nothing louder than a log line.

That is a benchmark-invalidating swap: the row still says "sglang" while the
number describes Diffusers. Cosmos3 on `--attention-backend fa2` is a live
example; only the `--tp-size 2` in that profile turned it into a startup error
instead of a quiet measurement. The cosmos3 T2I profile is moving to
`--tp-size 1 --cfg-parallel-size 2`, which removes exactly that protection, so
the harness has to notice for itself.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import NATIVE_FALLBACK_MARKERS  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def matches(line):
    return any(p in line for p in NATIVE_FALLBACK_MARKERS)


# the three lines sglang actually emits on that path (component_loader.py)
REAL = [
    "Component: transformer doesn't have a customized version yet, using native version",
    "Error while loading customized transformer, falling back to native version",
    "RuntimeError: Native Diffusers fallback for transformer component 'transformer' "
    "cannot honor requested distributed execution: tp_size=2.",
]
for line in REAL:
    check(f"detects: {line[:52]}...", matches(line))

# lines that must NOT trip it, or every run is flagged and the flag means nothing
DECOYS = [
    "INFO: server started, all components loaded natively by sglang",
    "using native cuda graphs for the DiT",
    "native_fallback_components: []",
    "capturing native-resolution latents",
]
for line in DECOYS:
    check(f"ignores: {line[:52]}", not matches(line))

# the detection has to run where server output is read, and the finding has to
# reach the result -- a warning printed into a log nobody greps is what this
# replaces.
src = (ROOT / "src" / "diffusion_bench" / "run_comparison.py").read_text()
check(
    "scanned on the server log stream",
    "NATIVE_FALLBACK_MARKERS" in src.split("def _log_pipe")[1][:600],
)
check("recorded on the result", 'metrics["native_fallback_components"]' in src)
check("and stated out loud during the run", "through a native fallback" in src)

sys.exit(fail)
