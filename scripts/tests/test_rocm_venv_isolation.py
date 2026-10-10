"""A ROCm-built competitor venv must not run on the host image's ROCm SDK.

sglang's ROCm images export ROCM_HOME/ROCM_PATH/LD_LIBRARY_PATH at their own SDK.
On the 2026-10-10 MI355X box that made vLLM-Omni's venv (vllm 0.31.0+rocm723)
load the image's libamd_smi.so, fail `import torch` on an undefined symbol, and
fail its platform detection; with only amdsmi redirected, the first H3 request
then faulted the GPU in hipBLASLt (torch built for 7.2.3 on ROCm 10 libraries).
A venv's own <venv>/rocm (scripts/install_rocm_userspace.sh) takes precedence;
without it, the venv's amdsmi library is the fallback.

Needs the runtime importable (i.e. `requests` installed).
"""
import sys
import tempfile
from pathlib import Path

from diffusion_bench.run_comparison import _isolate_rocm_userspace

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))
    fail |= not cond


HOST = {
    "ROCM_HOME": "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel",
    "ROCM_PATH": "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel",
    "LD_LIBRARY_PATH": "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib",
    "PATH": "/usr/bin",
}

with tempfile.TemporaryDirectory() as tmp:
    rocm_venv = Path(tmp, "vllm-omni")
    smi_dir = rocm_venv / "lib/python3.12/site-packages/amdsmi"
    smi_dir.mkdir(parents=True)
    (smi_dir / "libamd_smi.so").write_bytes(b"")
    env = dict(HOST)
    _isolate_rocm_userspace(env, rocm_venv)
    check("a ROCm venv drops the host SDK's ROCM_HOME", "ROCM_HOME" not in env)
    check("and its ROCM_PATH", "ROCM_PATH" not in env)
    check("and loads amdsmi's library from the venv", env.get("LD_LIBRARY_PATH") == str(smi_dir), env.get("LD_LIBRARY_PATH"))
    check("other variables are untouched", env["PATH"] == "/usr/bin")

    (rocm_venv / "rocm/lib").mkdir(parents=True)
    env = dict(HOST)
    _isolate_rocm_userspace(env, rocm_venv)
    prefix = str(rocm_venv / "rocm")
    check("a venv with its own ROCm runs on it", env.get("ROCM_PATH") == prefix and env.get("ROCM_HOME") == prefix, env.get("ROCM_PATH"))
    check("and loads only that ROCm's libraries", env.get("LD_LIBRARY_PATH") == prefix + "/lib", env.get("LD_LIBRARY_PATH"))

    cuda_venv = Path(tmp, "comfyui")
    (cuda_venv / "lib/python3.12/site-packages/torch").mkdir(parents=True)
    env = dict(HOST)
    _isolate_rocm_userspace(env, cuda_venv)
    check("a venv without a ROCm amdsmi keeps the environment as it was", env == HOST)

sys.exit(fail)
