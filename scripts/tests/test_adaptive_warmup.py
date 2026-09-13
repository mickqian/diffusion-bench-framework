#!/usr/bin/env python3
"""Warm until consecutive requests agree -- at the shape that will be MEASURED.

A fixed warmup count is not a guarantee of steady state. On 4xB200 cosmos3's
five MEASURED requests still decayed 1.09 -> 0.72s after the configured
warmups, so its median scored the warm-up rather than the model.

The first version of this check was vacuous. The fixed warmups deliberately run
a reduced-step request (3 steps against the case's 50) because they exist to
load weights and trigger compile cheaply -- and the convergence test compared
those cheap requests with each other. Two of them agree almost immediately, so
the loop never fired, and when it did it settled a shape nobody measures. The
measured request still opened cold: cosmos3 1.054s, then 0.827-0.846s.

The tempting alternative was to hand sglang `--warmup-resolutions` so it settles
at the measured shape. That would have been an advantage vLLM-Omni has no
equivalent for: neither framework warms at the request's shape by default --
vLLM-Omni's startup dummy run is a fixed 512x512, 2-step request (hardcoded
since #6094, 2026-08-18), and sglang's is the model's default resolution. So the
rule is applied in the harness, to everyone, and a framework that settles
immediately pays nothing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import (  # noqa: E402
    WARMUP_CONVERGE_PCT,
    WARMUP_EXTRA_BUDGET_S,
    WARMUP_EXTRA_MAX,
)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def converged(xs):
    if len(xs) < 2 or min(xs[-2:]) <= 0:
        return False
    return (max(xs[-2:]) - min(xs[-2:])) / min(xs[-2:]) * 100 <= WARMUP_CONVERGE_PCT


def simulate(full_shape_seq, est, budget=WARMUP_EXTRA_BUDGET_S):
    """Replay the harness's adaptive loop over the measured-shape latencies."""
    got, extra, spent = [], 0, 0.0
    while extra < WARMUP_EXTRA_MAX and spent + est <= budget and not converged(got):
        if extra >= len(full_shape_seq):
            break
        extra += 1
        elapsed = full_shape_seq[extra - 1]
        got.append(elapsed)
        spent += elapsed
        est = elapsed
    return got, extra, spent


# --- the bug this rewrite exists for -------------------------------------
# cosmos3: two 3-step warmups at 0.33s agree within 1%, so the old loop stopped
# and the 50-step request opened at 1.054s. The new loop converges on the
# 50-step shape instead, so the measured window opens at ~0.84s.
cheap = [0.332, 0.330]
check(
    "cheap warmups converge trivially",
    converged(cheap),
    f"{cheap} differ {(max(cheap) - min(cheap)) / min(cheap) * 100:.1f}% -- says nothing about 50 steps",
)
got, extra, spent = simulate([1.054, 0.844, 0.827], est=0.33 * 50 / 3)
check(
    "measured shape is warmed until it settles",
    extra == 3 and converged(got),
    f"{got} in {spent:.1f}s",
)

# --- a framework already at steady state pays the minimum -----------------
got, extra, _ = simulate([0.478, 0.477], est=0.5)
check("flat (vllm-omni-like) settles in two", extra == 2 and converged(got), str(got))

# --- never spin forever ---------------------------------------------------
got, extra, _ = simulate([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], est=1.0)
check("never settles -> bounded", extra == WARMUP_EXTRA_MAX and not converged(got), str(got))

# --- the budget must be checked BEFORE the first full-shape request -------
# Without `spent + est <= budget`, `spent` starts at zero and an expensive video
# case is always billed one full request -- 690s added to every wan22 cell.
got, extra, spent = simulate([690.0, 690.0], est=45.0 * 40 / 3)
check(
    "a 690s video is not billed a warmup it cannot afford",
    extra == 0 and spent == 0.0,
    f"estimate {45.0 * 40 / 3:.0f}s > budget {WARMUP_EXTRA_BUDGET_S:.0f}s",
)
# but a mid-cost video still gets warmed
got, extra, spent = simulate([34.8, 34.6], est=4.0 * 50 / 3)
check(
    "a 35s video still gets a measured-shape warmup",
    extra == 2 and converged(got),
    f"{got} in {spent:.1f}s",
)

# --- thresholds have to be sane ------------------------------------------
check(
    f"thresholds converge={WARMUP_CONVERGE_PCT}% max={WARMUP_EXTRA_MAX} budget={WARMUP_EXTRA_BUDGET_S}s",
    2.0 <= WARMUP_CONVERGE_PCT <= 10.0
    and 2 <= WARMUP_EXTRA_MAX <= 8
    and 30.0 <= WARMUP_EXTRA_BUDGET_S <= 600.0,
)

# --- the harness must actually send the case, not the reduced-step copy ---
src = (ROOT / "src" / "diffusion_bench" / "run_comparison.py").read_text()
body = src[src.index("def _run_warmups") : src.index("def run_single_request")]
check(
    "the adaptive phase targets the measured case",
    "full_case, full_lats = case, steady" in body,
    "not the reduced-step warmup_case",
)
check(
    "both warmup curves are returned",
    body.rstrip().endswith("return lats, steady"),
)

sys.exit(fail)
