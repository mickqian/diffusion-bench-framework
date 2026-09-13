#!/usr/bin/env python3
"""Every framework must be sent the same request inside the measured window.

The cross-framework metric is client wall clock, so anything the harness asks
one framework to do inside that window is charged to that framework alone.

It was charging sglang. `run_single_request` passed `perf_dump_path` on every
measured request when (and only when) the framework was sglang, which asks the
server for a per-stage performance dump. No competitor's send path even accepts
the argument. That asymmetry is what this test pins, and it is reason enough to
move the dump regardless of what it costs.

What it costs is a separate question, and not one five samples can answer:

    zimage sglang,    measured, dump on  : 0.695, 0.478, 0.474, 0.472, 0.473
    zimage sglang,    warmup,   dump off : 0.523, 0.471, 0.472
    zimage vllm-omni, warmup,   dump off : 0.485, 0.690, 0.489, 0.484

The first line looks like the dump's one-time setup landing on request 1 -- but
the third is the same outlier with no dump, in a different framework, on the
same case minutes later. The dump is still collected (it is how the client-side
read stall was caught), in its own request outside the window.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "diffusion_bench" / "run_comparison.py"
src = SOURCE.read_text()
tree = ast.parse(src)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "run_single_request"
)

# The measured loop is the `for` whose body sends the repeats.
loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
check("run_single_request has a measured loop", len(loops) == 1)
loop = loops[0]

sends = [
    n for n in ast.walk(loop)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "send_request"
]
check("the loop sends exactly one request", len(sends) == 1)
for call in sends:
    kwargs = {k.arg for k in call.keywords}
    check(
        "no per-framework argument inside the measured window",
        "perf_dump_path" not in kwargs,
        f"kwargs={sorted(kwargs) or 'none'}",
    )

# The diagnostic must still be collected, just not in there.
outside = [
    n for n in ast.walk(fn)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == "send_request"
    and any(k.arg == "perf_dump_path" for k in n.keywords)
]
check("the server-side dump is still collected", len(outside) == 1)
check(
    "collected outside the measured loop",
    all(n.lineno > loop.end_lineno for n in outside),
    f"loop ends at line {loop.end_lineno}",
)

# And it must not be able to cost a whole video request.
consts = {
    n.targets[0].id: n.value.value
    for n in tree.body
    if isinstance(n, ast.Assign)
    and isinstance(n.targets[0], ast.Name)
    and isinstance(n.value, ast.Constant)
}
budget = consts.get("PERF_DUMP_MAX_S")
check(
    f"an expensive case skips the annotation (PERF_DUMP_MAX_S={budget})",
    budget is not None and 10.0 <= budget <= 600.0,
)
body = src[src.index("def run_single_request") :]
check("the skip is recorded, not silent", "server_latency_note" in body)

# Guard the general rule: the only framework branch inside the measured loop
# should be none at all.
branches = [
    n for n in ast.walk(loop)
    if isinstance(n, ast.Compare)
    and any(
        isinstance(c, ast.Constant) and c.value in {"sglang", "vllm-omni", "lightx2v"}
        for c in n.comparators
    )
]
check("no framework-specific branch inside the loop", not branches)

sys.exit(fail)
