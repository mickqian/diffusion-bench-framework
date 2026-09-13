#!/usr/bin/env python3
"""Every async-generation poll loop must use the shared interval.

The poll interval is a quantum on the measured latency -- the number recorded
is the first tick AFTER the work finished. At 1s that was visible in the
nightly data: every sglang video result sat at N x (1s + ~4ms of loop
overhead), and ltx2.3 (13-17s) bounced between apparent regressions and
recoveries that were mostly the rounding boundary moving.

A stray `time.sleep(1)` in one framework's loop silently reintroduces it for
that framework only, which is worse than having it everywhere: the frameworks
stop being measured on the same ruler.
"""
import ast
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "diffusion_bench" / "run_comparison.py"
tree = ast.parse(SOURCE.read_text())
lines = SOURCE.read_text().splitlines()

fail = 0
polls = []
for node in ast.walk(tree):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        continue
    if node.func.id != "_poll_get":
        continue
    # the sleep immediately preceding the poll in the same loop body
    window = lines[max(0, node.lineno - 4) : node.lineno - 1]
    sleep = next((ln.strip() for ln in reversed(window) if "time.sleep(" in ln), None)
    polls.append((node.lineno, sleep))

print(f"found {len(polls)} poll site(s)")
if len(polls) < 4:
    print(f"  FAIL expected at least 4 (sglang video, vLLM-Omni x2, LightX2V, generic)")
    fail = 1

ALLOWED = {"time.sleep(POLL_INTERVAL_S)", "time.sleep(LIGHTX2V_POLL_INTERVAL_S)"}
for lineno, sleep in polls:
    ok = sleep in ALLOWED
    if not ok:
        fail = 1
    print(f"  {'ok  ' if ok else 'FAIL'} line {lineno}: {sleep}")

# the constant itself must stay small enough to keep the shortest video case
# (ltx2.3, ~13s) under a couple of percent
ns = {n.targets[0].id: n.value.value for n in tree.body
      if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
      and isinstance(n.value, ast.Constant) and n.targets[0].id == "POLL_INTERVAL_S"}
interval = ns.get("POLL_INTERVAL_S")
err = (interval / 13.0) * 100 if interval else None
ok = interval is not None and err < 2.0
if not ok:
    fail = 1
print(f"  {'ok  ' if ok else 'FAIL'} POLL_INTERVAL_S={interval} -> <={err:.1f}% on a 13s case"
      if interval else "  FAIL POLL_INTERVAL_S not found")

sys.exit(fail)
