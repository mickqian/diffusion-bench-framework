#!/usr/bin/env python3
"""Dispersed repeats must be visible, and only real stalls withheld.

Reading medians alone on the B200 image cases says "sglang is 3-7% slower than
vLLM-Omni". The samples say something different: sglang's repeats spread 45-63%
(zimage 0.47-0.68s, qwen 2.66-4.27s) while every competitor sits at 0.3-2%, and
sglang's FAST mode beats the competitor's median. A median summarises a
distribution it does not describe, so the samples and the spread are always
recorded.

Dispersed is not the same as untrustworthy: a bimodal framework really does
produce both modes, so the value is still published and only the reader is
pointed at the samples. Withholding stays for repeats that do not converge at
all -- the vLLM-Omni idle stall, where one sample was 100x another.
"""
import sys

sys.path.insert(0, "src")

DISPERSED_PCT = 25.0
UNSTABLE_RATIO = 3.0


def classify(samples):
    lo, hi = min(samples), max(samples)
    spread = (hi - lo) / lo * 100
    return spread, spread >= DISPERSED_PCT, (hi / lo) >= UNSTABLE_RATIO


fail = 0


def check(name, samples, want_dispersed, want_withheld):
    global fail
    spread, dispersed, withheld = classify(samples)
    ok = dispersed == want_dispersed and withheld == want_withheld
    if not ok:
        fail = 1
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name:34s} spread={spread:8.1f}% "
        f"dispersed={dispersed!s:5s} withheld={withheld}"
    )


# measured on 4xB200, 2026-09-12 -- sglang's short image cases
check("sglang zimage", [0.51, 0.68, 0.68, 0.47, 0.47], True, False)
check("sglang qwen t2i", [2.69, 2.79, 2.66, 4.27], True, False)
check("sglang cosmos3 t2i", [1.33, 1.11, 0.84, 0.81, 0.96], True, False)

# well-behaved: must not be flagged, or the flag means nothing
check("sglang qwen edit", [4.224, 4.23, 4.467], False, False)
check("sglang minimax (video)", [113.086, 113.13, 113.173], False, False)
check("vllm-omni zimage", [0.476, 0.478, 0.486], False, False)
check("vllm-omni qwen t2i", [2.703, 2.705, 2.71], False, False)
check("trtllm qwen t2i", [3.445, 3.464, 3.494], False, False)

# the case withholding exists for
check("a genuine idle stall", [55.4, 55.5, 0.48], True, True)

# the boundary is a threshold, not a coincidence
check("just under the line", [1.0, 1.24], False, False)
check("just over the line", [1.0, 1.26], True, False)

sys.exit(fail)
