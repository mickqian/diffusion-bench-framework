#!/usr/bin/env python3
"""Warm until consecutive requests agree — same rule for every framework.

A fixed warmup count is not a guarantee of steady state. On 4xB200 cosmos3's
five MEASURED requests still decayed 1.09 -> 0.72s after the configured
warmups, so its median scored the warm-up rather than the model.

The tempting fix was to hand sglang `--warmup-resolutions` so it settles at the
measured shape. That would have been an advantage vLLM-Omni has no equivalent
for: neither framework warms at the request's shape by default -- vLLM-Omni's
startup dummy run is a fixed 512x512, 2-step request (hardcoded since #6094,
2026-08-18), and sglang's is the model's default resolution. So the rule is
applied in the harness, to everyone, and a framework that settles immediately
pays nothing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import (  # noqa: E402
    WARMUP_CONVERGE_PCT,
    WARMUP_EXTRA_MAX,
)

fail = 0


def simulate(sequence, fixed=2):
    """Replay the harness's loop over a framework's warmup latencies."""
    lats = list(sequence[:fixed])
    extra = 0
    i = fixed
    while (
        len(lats) >= 2
        and extra < WARMUP_EXTRA_MAX
        and min(lats[-2:]) > 0
        and (max(lats[-2:]) - min(lats[-2:])) / min(lats[-2:]) * 100 > WARMUP_CONVERGE_PCT
    ):
        if i >= len(sequence):
            break
        lats.append(sequence[i])
        i += 1
        extra += 1
    return lats, extra


def check(name, sequence, want_extra, want_converged):
    global fail
    lats, extra = simulate(sequence)
    spread = (max(lats[-2:]) - min(lats[-2:])) / min(lats[-2:]) * 100
    converged = spread <= WARMUP_CONVERGE_PCT
    ok = extra == want_extra and converged == want_converged
    if not ok:
        fail = 1
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name:32s} extra={extra} "
        f"final pair differ {spread:5.1f}%  -> {lats}"
    )


# cosmos3's real decay, used as the warmup curve it would have produced
check("decaying (cosmos3-like)", [1.33, 1.11, 0.84, 0.81, 0.96, 0.82, 0.81], 2, True)
# a framework already at steady state pays nothing
check("flat (vllm-omni-like)", [0.478, 0.477, 0.480], 0, True)
check("flat video (minimax-like)", [113.09, 113.13, 113.17], 0, True)
# never spin forever on a series that will not settle
check("never settles", [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], WARMUP_EXTRA_MAX, False)
# and stop cleanly when the sequence runs out
check("sequence exhausted", [1.0, 2.0], 0, False)

# the thresholds themselves have to be sane: too loose and it never fires,
# too tight and every framework pays for noise
ok = 2.0 <= WARMUP_CONVERGE_PCT <= 10.0 and 1 <= WARMUP_EXTRA_MAX <= 8
print(f"  {'ok  ' if ok else 'FAIL'} thresholds converge={WARMUP_CONVERGE_PCT}% max_extra={WARMUP_EXTRA_MAX}")
if not ok:
    fail = 1

sys.exit(fail)
