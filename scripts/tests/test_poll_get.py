#!/usr/bin/env python3
"""`_poll_get` must survive a blip and still give up on a dead server.

A transient status-check failure is not a result: a wan22 run that had already
produced a 690s video was thrown away because one poll was late while the
server was busy generating. But tolerance must not become "wait forever" --
a server that is really gone has to surface before the caller's deadline.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from diffusion_bench import run_comparison as rc  # noqa: E402

fail = 0


def check(name, cond):
    global fail
    print(("ok    " if cond else "FAIL  ") + name)
    if not cond:
        fail = 1


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


# 1. immediate success
calls = {"n": 0}


def ok_get(url, timeout=None):
    calls["n"] += 1
    return FakeResp({"status": "completed"})


real_get = requests.get
requests.get = ok_get
r = rc._poll_get("http://x/status", time.time() + 5)
check("returns on first success", r.json()["status"] == "completed" and calls["n"] == 1)

# 2. two transient failures, then success
calls["n"] = 0


def flaky_get(url, timeout=None):
    calls["n"] += 1
    if calls["n"] <= 2:
        raise requests.Timeout("read timed out")
    return FakeResp({"status": "completed"})


requests.get = flaky_get
t0 = time.time()
r = rc._poll_get("http://x/status", time.time() + 30)
check("survives a blip and returns", r.json()["status"] == "completed" and calls["n"] == 3)
check("backs off between retries", time.time() - t0 >= 2)

# 3. never recovers -> raises before running forever
calls["n"] = 0


def dead_get(url, timeout=None):
    calls["n"] += 1
    raise requests.ConnectionError("connection refused")


requests.get = dead_get
t0 = time.time()
raised = None
try:
    rc._poll_get("http://x/status", time.time() + 5)
except TimeoutError as exc:
    raised = exc
elapsed = time.time() - t0
check("gives up on a dead server", raised is not None)
check("names the underlying error", raised is not None and "ConnectionError" in str(raised))
check("respects the deadline (<10s for a 5s budget)", elapsed < 10)

# 4. a real HTTP error is NOT swallowed as transient
class BadResp(FakeResp):
    def raise_for_status(self):
        raise requests.HTTPError("500 Server Error")


def http_error_get(url, timeout=None):
    return BadResp({})


requests.get = http_error_get
raised = None
try:
    rc._poll_get("http://x/status", time.time() + 5)
except requests.HTTPError as exc:
    raised = exc
check("surfaces an HTTP error immediately", raised is not None)

requests.get = real_get
sys.exit(fail)
