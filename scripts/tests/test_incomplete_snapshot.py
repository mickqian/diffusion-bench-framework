#!/usr/bin/env python3
"""A half-built model cache must be named as such, not blamed on the framework.

`/cluster-storage/models` is shared and is not stable under a run: a
Z-Image-Turbo snapshot was re-materialised at 09:12 while a server started at
09:11. What came out named the wrong component at every step --

    FileNotFoundError on a transformer shard
    -> sglang's native loader fails
    -> "Native Diffusers fallback for transformer component 'transformer'
        cannot honor requested distributed execution: tp_size=2"
    -> "sglang server exited before health check passed (exit 1)"

which reads as "sglang has no native implementation for this model". It is not
even a framework problem, and it cost twenty minutes to walk back -- including a
wrong inference that the NEXT cell had silently run Diffusers, which it had not.

Two defences, both tested here: refuse to launch against a snapshot that is
mid-download, and when a server dies anyway, put the read failure in the error
rather than leaving the framework's downstream complaint to speak for it.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import _incomplete_snapshot  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def make_repo(root: Path, model: str, *, broken=False, incomplete=False):
    repo = root / ("models--" + model.replace("/", "--"))
    blobs, snap = repo / "blobs", repo / "snapshots" / "abc123"
    (snap / "transformer").mkdir(parents=True)
    blobs.mkdir(parents=True)
    good = blobs / "deadbeef"
    good.write_text("weights")
    os.symlink(good, snap / "transformer" / "shard1.safetensors")
    if broken:
        os.symlink(blobs / "missing-blob", snap / "transformer" / "shard2.safetensors")
    if incomplete:
        (blobs / "cafe1234.incomplete").write_text("partial")
    return repo


MODEL = "Tongyi-MAI/Z-Image-Turbo"

with tempfile.TemporaryDirectory() as d:
    cache = Path(d)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)

    make_repo(cache, MODEL)
    check("a complete snapshot passes", _incomplete_snapshot(MODEL) is None)

    # not cached at all is not a problem -- the framework fetches it
    check("an uncached model passes", _incomplete_snapshot("org/never-seen") is None)
    # a local path is the caller's business
    check("a local path passes", _incomplete_snapshot("/models/local") is None)
    check("an empty model passes", _incomplete_snapshot("") is None)

with tempfile.TemporaryDirectory() as d:
    cache = Path(d)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    make_repo(cache, MODEL, broken=True)
    why = _incomplete_snapshot(MODEL)
    check("a dangling symlink is caught", why is not None)
    if why:
        check("it names the model", MODEL in why)
        check("it names the file", "shard2.safetensors" in why, why[:90])
        check("it says this is not the framework's fault",
              "not a framework failure" in why)
        check("it says what to do", "re-run" in why)

with tempfile.TemporaryDirectory() as d:
    cache = Path(d)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    make_repo(cache, MODEL, incomplete=True)
    why = _incomplete_snapshot(MODEL)
    check("a partial blob is caught", why is not None and "partial blob" in why)

# HF_HOME is the other spelling the harness sets
with tempfile.TemporaryDirectory() as d:
    cache = Path(d)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    os.environ["HF_HOME"] = str(cache)
    make_repo(cache, MODEL, broken=True)
    check("HF_HOME is honoured too", _incomplete_snapshot(MODEL) is not None)
    os.environ.pop("HF_HOME", None)
    check("no cache configured is not an error", _incomplete_snapshot(MODEL) is None)

# the second defence: the read failure has to reach the reported error
src = (ROOT / "src" / "diffusion_bench" / "run_comparison.py").read_text()
check("the launch is gated on the check", "cache_problem = _incomplete_snapshot(" in src)
check("read failures are collected from the server log",
      'checkpoint_errors.append' in src)
check("and reach the recorded error",
      "the real cause" in src and 'single_result["error"] = message' in src)

sys.exit(fail)
