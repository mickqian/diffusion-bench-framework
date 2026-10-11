#!/usr/bin/env python3
"""A server that moves off the requested HTTP port must fail the start at once.

sglang settles a taken port by moving to another one ("Port 60009 was
unavailable, using port 5446 instead") unless --strict-ports is passed. The
harness kept polling 60009 and a FastH3 arm on 4xH200 sat out the whole
2400 s health timeout with the server ready on 5446 (2026-10-11).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import _moved_http_port, wait_for_health  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


line = "[10-11 09:24:14] Port 60009 was unavailable, using port 5446 instead"
msg = _moved_http_port(line, 60009)
check("detects the HTTP port moving", msg is not None and "5446" in msg, str(msg))
check("ignores an internal port moving", _moved_http_port(line, 60011) is None)
check("ignores unrelated lines", _moved_http_port("Uvicorn running on http://127.0.0.1:60009", 60009) is None)

t0 = time.time()
try:
    wait_for_health("http://127.0.0.1:9", "sglang", timeout=30, startup_errors=[msg])
    check("wait_for_health raises on a startup error", False)
except RuntimeError as e:
    check("wait_for_health raises on a startup error", "5446" in str(e), str(e))
check("without waiting out the timeout", time.time() - t0 < 5, f"{time.time() - t0:.1f}s")

sys.exit(fail)
