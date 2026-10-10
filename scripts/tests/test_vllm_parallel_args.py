"""The harness adds --tensor-parallel-size only when a vLLM-Omni command names no parallelism.

`--usp 2` was not recognised, so the harness appended `--tensor-parallel-size 2`
and Qwen-Image-2.1's "2-GPU" vLLM-Omni cell ran USP2 x TP2 on four GPUs
(GB300, 2026-10-10). Needs the runtime importable (i.e. `requests` installed).
"""
import sys

from diffusion_bench.run_comparison import _build_vllm_cmd

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))
    fail |= not cond


CASE = {"id": "c", "model": "Qwen/Qwen-Image-2.1", "num_gpus": 2}


def cmd(serve_args):
    return _build_vllm_cmd(CASE, {"serve_args": serve_args}, 40001)


for args in ("--usp 2", "--usp=2", "--ring 2", "--num-gpus 2 --usp 2", "--cfg-parallel-size 2"):
    c = cmd(args)
    check(f"{args!r} keeps its own parallelism", "--tensor-parallel-size" not in c, " ".join(c))
c = cmd("")
check("no parallel flag on 2 GPUs gets TP2", c[c.index("--tensor-parallel-size") + 1] == "2" if "--tensor-parallel-size" in c else False, " ".join(c))
sys.exit(fail)
