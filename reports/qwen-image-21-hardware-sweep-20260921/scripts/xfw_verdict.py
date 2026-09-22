#!/usr/bin/env python3
"""Turn a cross-framework ABAB log into a paired verdict.

Under 5% this repo requires paired rounds that agree in sign, so report the
per-round diff and the sign agreement, not just two medians. Also report every
individual sample, because LightX2V's client latency is bimodal and a median
alone hides which mode a round landed in.
"""
import re, statistics, sys, pathlib

path = pathlib.Path(sys.argv[1])
hw = sys.argv[2] if len(sys.argv) > 2 else "?"
rounds = {}
for m in re.finditer(r"round=(\d+) arm=(\w+) ::\s+RESULT \S+(?: \S+)? p50=([0-9.]+)s \[([^\]]*)\]", path.read_text()):
    r, arm, p50, raw = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    rounds.setdefault(r, {})[arm] = (p50, [float(x) for x in raw.split(",")])

full = sorted(r for r, v in rounds.items() if {"sglang", "lightx2v"} <= set(v))
if not full:
    print("no complete rounds"); raise SystemExit(1)

print(f"  {hw}: {len(full)} paired round(s)\n")
print(f"  {'round':>5s} {'sglang':>9s} {'lightx2v':>9s} {'diff':>8s}  samples")
diffs = []
for r in full:
    s, l = rounds[r]["sglang"][0], rounds[r]["lightx2v"][0]
    d = (l - s) / l * 100
    diffs.append(d)
    print(f"  {r:5d} {s:9.3f} {l:9.3f} {d:+7.1f}%  lx2v={rounds[r]['lightx2v'][1]}")
sg = [rounds[r]["sglang"][0] for r in full]
lx = [rounds[r]["lightx2v"][0] for r in full]
agree = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
print(f"\n  medians: sglang {statistics.median(sg):.3f}s, lightx2v {statistics.median(lx):.3f}s")
print(f"  paired diffs: {', '.join(f'{d:+.1f}%' for d in diffs)}  ->  "
      f"{'SAME SIGN' if agree else 'MIXED SIGN (no resolvable difference)'}")

# every lightx2v sample, and how close each lands to a 0.5 s boundary
allsamp = [x for r in full for x in rounds[r]["lightx2v"][1]]
off = [abs(x - round(x / 0.5) * 0.5) for x in allsamp]
print(f"\n  lightx2v samples ({len(allsamp)}): {sorted(allsamp)}")
print(f"  worst distance from a 0.5s multiple: {max(off):.3f}s  (tick hypothesis)")
sgsamp = [x for r in full for x in rounds[r]["sglang"][1]]
print(f"  sglang samples  ({len(sgsamp)}): {sorted(sgsamp)}")
# sglang is NOT the control for the tick: 4.51-4.53 happens to sit near 4.5
# too, so that comparison proves nothing. The discriminating fact is that
# LightX2V's SERVER-side times (4.443-4.477, 14.10-14.20) are nowhere near a
# 0.5s multiple while its CLIENT times always are -- the quantisation is added
# after the pipeline finishes, and no per-framework control is needed to see it.
print("  (the tick is evidenced against LightX2V's own server-side times,")
print("   4.443-4.477s and 14.10-14.20s, which are not near 0.5s multiples.)")
