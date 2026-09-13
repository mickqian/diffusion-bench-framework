#!/usr/bin/env python3
"""A finished measurement must survive whatever happens to the ones after it.

The run artifact used to be written only after every framework had finished. So
when the per-case `timeout` fired on wan22 -- 90 minutes in, with sglang and
vLLM-Omni both already measured -- the process was killed before writing and
all of it was lost. The artifact is now checkpointed after each framework.

Two properties matter beyond "it writes a file": the checkpoint must be atomic
(a timeout landing mid-write must not leave a truncated JSON where a good one
was), and it must be labelled, so a reader can tell an interrupted run from a
complete one instead of concluding the missing frameworks were skipped.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.run_comparison import _write_run_artifact  # noqa: E402

fail = 0


def check(name, cond):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fail = 1


META = {"run_id": "t", "hardware": {"gpus": []}, "benchmark_modes": ["single_e2e"]}
R1 = [{"case_id": "c", "framework": "sglang", "latency_s": 1.0, "model": "/local/path"}]
R2 = R1 + [{"case_id": "c", "framework": "vllm-omni", "latency_s": 2.0, "model": "/local/path"}]

with tempfile.TemporaryDirectory() as d:
    out = str(Path(d) / "nested" / "run.json")

    # checkpoint after the first framework
    _write_run_artifact(out, R1, [], partial=True, **META)
    got = json.loads(Path(out).read_text())
    check("checkpoint writes (creating dirs)", Path(out).exists())
    check("checkpoint is labelled partial", got.get("partial") is True)
    check("checkpoint holds the finished result", len(got["results"]) == 1)
    check("checkpoint carries run metadata", got.get("run_id") == "t")
    check("no hub lookup on a checkpoint", "model_revisions" not in got)

    # second framework overwrites in place
    _write_run_artifact(out, R2, [], partial=True, **META)
    got = json.loads(Path(out).read_text())
    check("checkpoint accumulates", len(got["results"]) == 2)

    # final write
    final = _write_run_artifact(out, R2, [], partial=False, **META)
    got = json.loads(Path(out).read_text())
    check("final write clears the partial flag", got.get("partial") is False)
    check("final write resolves model revisions", "model_revisions" in got)
    check("local paths resolve without the hub",
          got["model_revisions"].get("/local/path") == "local-path")
    check("the writer returns what it wrote", final["results"] == got["results"])

    # atomicity: no stray temp file left behind
    leftovers = [p.name for p in Path(out).parent.iterdir() if p.name.endswith(".tmp")]
    check("no .tmp left behind", not leftovers)

    # a valid JSON is readable at every point -- never a truncated file
    check("artifact always parses", isinstance(json.loads(Path(out).read_text()), dict))

sys.exit(fail)
