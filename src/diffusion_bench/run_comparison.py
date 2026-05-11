"""Cross-framework comparison benchmark for diffusion serving.

Launches servers (SGLang, vLLM-Omni, LightX2V) for each test case, sends a
single request and/or bench_serving traffic, then writes comparison-results.json.

Usage:
    # Full run (requires GPU and installed serving frameworks)
    diffusion-bench-compare

    # Dry-run (config parsing + command preview only)
    diffusion-bench-compare --dry-run

    # Run only specific case(s)
    diffusion-bench-compare --case-ids flux1_dev_t2i_1024

    # Run only specific framework(s)
    diffusion-bench-compare --frameworks sglang

    # Run single-request E2E plus high-pressure throughput
    diffusion-bench-compare --modes single_e2e throughput
"""

import argparse
import base64
import copy
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_PATH = Path(__file__).parent / "comparison_configs.json"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_comparison_frameworks.sh"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
HEALTH_TIMEOUT = (
    2400  # seconds (40 min — large checkpoints can need long download/load time)
)
REQUEST_TIMEOUT = 1200  # seconds
GPU_CLEAR_WAIT = 15  # seconds between framework runs
MODE_SINGLE_E2E = "single_e2e"
MODE_THROUGHPUT = "throughput"
DEFAULT_BENCHMARK = {
    "warmup": {"num_requests": 2, "num_inference_steps": 3},
    "throughput": {"num_requests": 4, "max_concurrency": 2},
}
SGLANG_DEFAULT_WARMUP_STEPS = 3
DEFAULT_PROFILE = "default"
DEFAULT_SGLANG_PROFILE = DEFAULT_PROFILE
HARDWARE_PROFILE_ENV = "SGLANG_BENCH_HARDWARE_PROFILE"
SGLANG_EXTRA_SERVE_ARGS_ENV = "DIFFUSION_BENCH_SGLANG_EXTRA_SERVE_ARGS"
SKIP_FRAMEWORK_INSTALL_ENV = "SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL"
FORCED_BENCHMARK_ENV = {"TORCH_COMPILE_DISABLE": "1"}
VLLM_DISABLE_TORCH_COMPILE_ARGS = [
    "--enforce-eager",
    "--compilation-config",
    '{"mode":0}',
]
LIGHTX2V_DISABLE_TORCH_COMPILE_CONFIG = {"compile": False, "compile_shapes": []}

# Frameworks that need separate installation (conflict with sglang's deps)
INSTALLABLE_FRAMEWORKS = {"vllm-omni", "lightx2v"}
FRAMEWORK_ORDER = ["sglang", "vllm-omni", "lightx2v"]
SGLANG_PROFILE_RUNTIME_KEYS = {
    "serve_args",
    "num_gpus",
    "extra_env",
    "benchmark",
    "model",
    "model_path",
}
FRAMEWORK_PROFILE_RUNTIME_KEYS = SGLANG_PROFILE_RUNTIME_KEYS | {"lightx2v_config"}

# Cached reference image (downloaded once)
_cached_ref_image: bytes | None = None
_cached_ref_image_path: str | None = None


# ---------------------------------------------------------------------------
# Server lifecycle — command builders
# ---------------------------------------------------------------------------


def _build_sglang_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    model_path = _server_model_path(case, fw_cfg)
    num_gpus = fw_cfg.get("num_gpus", case["num_gpus"])
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if num_gpus > 1:
        cmd += ["--num-gpus", str(num_gpus)]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    if os.environ.get(SGLANG_EXTRA_SERVE_ARGS_ENV, "").strip():
        cmd += shlex.split(os.environ[SGLANG_EXTRA_SERVE_ARGS_ENV])
    if "--warmup" in cmd and "--warmup-steps" not in cmd:
        cmd += ["--warmup-steps", str(SGLANG_DEFAULT_WARMUP_STEPS)]
    return cmd


def _build_vllm_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    model_path = _server_model_path(case, fw_cfg)
    num_gpus = fw_cfg.get("num_gpus", case["num_gpus"])
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--omni",
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    parallel_args = {
        "--tensor-parallel-size",
        "--num-gpus",
        "--cfg-parallel-size",
        "--ulysses-degree",
        "--ring-degree",
        "--vae-patch-parallel-size",
    }
    has_parallel_arg = any(
        arg in parallel_args or any(arg.startswith(f"{name}=") for name in parallel_args)
        for arg in cmd
    )
    if num_gpus > 1 and not has_parallel_arg:
        cmd += ["--tensor-parallel-size", str(num_gpus)]
    cmd += VLLM_DISABLE_TORCH_COMPILE_ARGS
    return cmd


def _resolve_hf_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local cache path, or return as-is."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id)
        print(f"  Resolved {model_id} -> {path}")
        return path
    except Exception:
        return model_id


def _write_lightx2v_config(case: dict, fw_cfg: dict) -> str:
    """Write a minimal LightX2V config JSON and return its path."""
    cfg = {
        "infer_steps": case.get("num_inference_steps", 50),
        "seed": case.get("seed", 42),
    }
    if "guidance_scale" in case:
        cfg["guidance_scale"] = case["guidance_scale"]
    if (
        str(fw_cfg.get("model_cls", "")).startswith("wan")
        and "guidance_scale" in case
    ):
        cfg["sample_guide_scale"] = (
            [case["guidance_scale"], case["guidance_scale_2"]]
            if "guidance_scale_2" in case
            else case["guidance_scale"]
        )
    if "num_frames" in case:
        cfg["target_video_length"] = case["num_frames"]
    if "fps" in case:
        cfg["fps"] = case["fps"]
    if "height" in case:
        cfg["height"] = case["height"]
        cfg["target_height"] = case["height"]
    if "width" in case:
        cfg["width"] = case["width"]
        cfg["target_width"] = case["width"]
    cfg.update(fw_cfg.get("lightx2v_config", {}))
    cfg.update(LIGHTX2V_DISABLE_TORCH_COMPILE_CONFIG)

    config_path = os.path.join(
        tempfile.gettempdir(), f"lightx2v_config_{case['id']}.json"
    )
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    return config_path


def _build_lightx2v_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    """Build LightX2V server launch command.

    Single GPU:  python -m lightx2v.server --model_path ... --model_cls ... --task ... --port ...
    Multi GPU:   torchrun --nproc_per_node=N -m lightx2v.server ...

    LightX2V requires a local model path and a config JSON with infer params.
    """
    model_cls = fw_cfg["model_cls"]
    task = fw_cfg["lightx2v_task"]
    num_gpus = fw_cfg.get("num_gpus", case["num_gpus"])
    model_path = _server_model_path(case, fw_cfg)
    if not fw_cfg.get("_skip_model_path_resolution"):
        model_path = _resolve_hf_model_path(model_path)
    config_path = _write_lightx2v_config(case, fw_cfg)

    server_args = [
        "--model_path",
        model_path,
        "--model_cls",
        model_cls,
        "--task",
        task,
        "--config_json",
        config_path,
        "--host",
        DEFAULT_HOST,
        "--port",
        str(port),
    ]
    if fw_cfg.get("serve_args", "").strip():
        server_args += fw_cfg["serve_args"].strip().split()

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "lightx2v.server",
        ] + server_args
    else:
        cmd = ["python3", "-m", "lightx2v.server"] + server_args

    return cmd


def _server_model_path(case: dict, fw_cfg: dict) -> str:
    return os.path.expandvars(str(fw_cfg.get("model_path") or case["model"]))


def build_server_cmd(framework: str, case: dict, fw_cfg: dict, port: int) -> list[str]:
    builders = {
        "sglang": _build_sglang_cmd,
        "vllm-omni": _build_vllm_cmd,
        "lightx2v": _build_lightx2v_cmd,
    }
    builder = builders.get(framework)
    if builder is None:
        raise ValueError(f"Unknown framework: {framework}")
    return builder(case, fw_cfg, port)


def _explicit_sglang_profile(profile: str | None) -> str | None:
    return (
        profile
        or os.environ.get("SGLANG_BENCH_SGLANG_PROFILE")
        or os.environ.get("SGLANG_BENCH_PROFILE")
    )


def _explicit_framework_profile(framework: str, sglang_profile: str | None) -> str | None:
    if framework == "sglang":
        return _explicit_sglang_profile(sglang_profile)
    env_key = f"DIFFUSION_BENCH_{framework.upper().replace('-', '_')}_PROFILE"
    return os.environ.get(env_key) or os.environ.get("DIFFUSION_BENCH_FRAMEWORK_PROFILE")


def _requested_sglang_profile(profile: str | None) -> str:
    return _explicit_sglang_profile(profile) or "auto"


def _requested_framework_profile(framework: str, sglang_profile: str | None) -> str:
    return _explicit_framework_profile(framework, sglang_profile) or "auto"


def _hardware_profile_candidates(hardware_metadata: dict | None) -> list[str]:
    metadata = hardware_metadata or {}
    values = [
        metadata.get("hardware_profile_override"),
        os.environ.get(HARDWARE_PROFILE_ENV),
        metadata.get("gpu_config"),
        metadata.get("runner_labels"),
        *(metadata.get("gpus") or []),
    ]
    text = " ".join(str(value).lower() for value in values if value)
    candidates = []
    for profile in ("h200", "h100", "a100", "l40", "l4", "rtx4090", "rtx3090"):
        if profile in text or profile.replace("rtx", "rtx ") in text:
            candidates.append(profile)
    return candidates


def _profile_hardware_values(profile_cfg: dict) -> list[str]:
    hardware = (
        profile_cfg.get("hardware")
        or profile_cfg.get("hardware_profile")
        or profile_cfg.get("hardware_profiles")
    )
    if not hardware:
        return []
    if isinstance(hardware, str):
        values = [hardware]
    elif isinstance(hardware, list):
        values = hardware
    else:
        values = []
    return [str(value).lower() for value in values]


def _profile_matches_hardware(
    profile_name: str, profile_cfg: dict, candidates: list[str]
) -> bool:
    if not candidates:
        return False
    values = [profile_name.lower(), *_profile_hardware_values(profile_cfg)]
    return any(candidate in value for candidate in candidates for value in values)


def _select_command_profile(
    framework: str,
    profiles: dict,
    explicit_profile: str | None,
    hardware_metadata: dict | None,
) -> tuple[str | None, str, list[str]]:
    candidates = _hardware_profile_candidates(hardware_metadata)
    if explicit_profile:
        if explicit_profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown {framework} command profile '{explicit_profile}'. "
                f"Available profiles: {available}"
            )
        return explicit_profile, "explicit", candidates

    for profile_name, profile_cfg in profiles.items():
        if profile_name == DEFAULT_PROFILE:
            continue
        if _profile_matches_hardware(profile_name, profile_cfg, candidates):
            return profile_name, "hardware", candidates

    if DEFAULT_PROFILE in profiles:
        return DEFAULT_PROFILE, "default", candidates
    if profiles:
        return next(iter(profiles)), "first", candidates
    return None, "inline", candidates


def _resolve_framework_config(
    framework: str,
    fw_cfg: dict,
    sglang_profile: str | None = None,
    hardware_metadata: dict | None = None,
) -> dict:
    resolved = {
        key: value
        for key, value in fw_cfg.items()
        if key not in ("command_profiles", "_benchmark_metadata")
    }
    profiles = fw_cfg.get("command_profiles") or {}
    profile_name = None
    profile_cfg = {}
    profile_source = "inline"
    hardware_candidates: list[str] = []
    if profiles:
        profile_name, profile_source, hardware_candidates = _select_command_profile(
            framework,
            profiles,
            _explicit_framework_profile(framework, sglang_profile),
            hardware_metadata,
        )
        profile_cfg = profiles[profile_name]
        runtime_cfg = {
            key: value
            for key, value in profile_cfg.items()
            if key in FRAMEWORK_PROFILE_RUNTIME_KEYS
        }
        resolved = _merge_nested(resolved, runtime_cfg)

    metadata = {
        "profile": profile_name,
        "profile_request": _requested_framework_profile(framework, sglang_profile),
        "profile_source": profile_source,
        "framework_ref": profile_cfg.get("framework_ref"),
        "hardware": profile_cfg.get("hardware"),
        "hardware_candidates": hardware_candidates,
        "description": profile_cfg.get("description"),
        "notes": profile_cfg.get("notes"),
        "serve_args": resolved.get("serve_args", ""),
        "num_gpus": resolved.get("num_gpus"),
        "extra_env_keys": sorted((resolved.get("extra_env") or {}).keys()),
    }
    if framework == "sglang":
        metadata.update(
            {
                "sglang_profile": profile_name,
                "sglang_profile_request": _requested_sglang_profile(sglang_profile),
                "sglang_profile_source": profile_source,
                "sglang_ref": profile_cfg.get("sglang_ref"),
            }
        )
    if "lightx2v_config" in resolved:
        metadata["lightx2v_config_keys"] = sorted(resolved["lightx2v_config"])
    resolved["_benchmark_metadata"] = metadata
    return resolved


# ---------------------------------------------------------------------------
# Server lifecycle — health check & cleanup
# ---------------------------------------------------------------------------

# Health check endpoints per framework
HEALTH_ENDPOINTS = {
    "sglang": "/health",
    "vllm-omni": "/health",
    "lightx2v": "/v1/service/status",
}


def wait_for_health(
    base_url: str,
    framework: str = "sglang",
    timeout: int = HEALTH_TIMEOUT,
    proc: subprocess.Popen | None = None,
) -> None:
    """Poll health endpoint until 200, then verify model is loaded."""
    endpoint = HEALTH_ENDPOINTS.get(framework, "/health")
    health_url = f"{base_url}{endpoint}"
    print(f"  Waiting for server at {health_url} ...")
    start = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"{framework} server exited before health check passed "
                f"(exit {proc.returncode})"
            )
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Server at {health_url} did not start within {timeout}s"
            )
        time.sleep(2)

    # For SGLang, /health can return 200 before model routes are registered.
    # Poll /v1/models to confirm the model is fully loaded.
    if framework == "sglang":
        models_url = f"{base_url}/v1/models"
        while True:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"{framework} server exited before model routes were ready "
                    f"(exit {proc.returncode})"
                )
            try:
                resp = requests.get(models_url, timeout=5)
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Model at {models_url} not ready within {timeout}s")
            time.sleep(2)

    elapsed = time.time() - start
    print(f"  Server ready in {elapsed:.1f}s")


def kill_server(proc: subprocess.Popen) -> None:
    """Kill the server process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Reference image helpers
# ---------------------------------------------------------------------------


def _get_ref_image_bytes(config: dict) -> bytes:
    """Download and cache the shared test reference image."""
    global _cached_ref_image
    if _cached_ref_image is not None:
        return _cached_ref_image
    url = config.get("test_image_url", "")
    if not url:
        raise RuntimeError("No test_image_url in config for image-conditioned case")
    print(f"  Downloading reference image from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    _cached_ref_image = resp.content
    return _cached_ref_image


def _get_ref_image_b64(config: dict) -> str:
    """Get reference image as base64 string."""
    return base64.b64encode(_get_ref_image_bytes(config)).decode("utf-8")


def _get_ref_image_path(config: dict) -> str:
    """Save reference image to a temp file and return path."""
    global _cached_ref_image_path
    if _cached_ref_image_path and os.path.exists(_cached_ref_image_path):
        return _cached_ref_image_path
    data = _get_ref_image_bytes(config)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    _cached_ref_image_path = path
    return path


# ---------------------------------------------------------------------------
# Request helpers — SGLang (OpenAI-compatible)
# ---------------------------------------------------------------------------


def _build_sglang_payload(case: dict) -> dict:
    """Build common SGLang request payload."""
    payload = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": 1,
        "response_format": "b64_json",
    }
    for key in (
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "seed",
        "num_frames",
        "fps",
        "negative_prompt",
    ):
        if key in case:
            payload[key] = case[key]
    return payload


def _read_perf_dump(perf_dump_path: str, timeout: float = 10.0) -> float | None:
    """Read total_duration_ms from a perf dump JSON written by the server.

    The server writes the file asynchronously after the HTTP response,
    so we poll briefly.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(perf_dump_path) as f:
                data = json.load(f)
            total_ms = data.get("total_duration_ms")
            if total_ms is not None:
                return total_ms / 1000.0
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return None


def send_image_request_sglang(
    base_url: str, case: dict, perf_dump_path: str | None = None
) -> float:
    """Send a single T2I request via SGLang's /v1/images/generations."""
    payload = _build_sglang_payload(case)
    if perf_dump_path:
        payload["perf_dump_path"] = perf_dump_path

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/images/generations",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    client_latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or len(data["data"]) == 0:
        raise RuntimeError(f"Image request returned no data: {data}")

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Image generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Image generated in {client_latency:.2f}s")
    return client_latency


def send_video_request_sglang(
    base_url: str, case: dict, perf_dump_path: str | None = None
) -> float:
    """Send a single T2V request via SGLang's /v1/videos (async)."""
    payload = _build_sglang_payload(case)
    if perf_dump_path:
        payload["perf_dump_path"] = perf_dump_path

    start = time.time()

    # Submit job
    resp = requests.post(
        f"{base_url}/v1/videos",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"Video submit returned no job id: {job}")

    # Poll for completion
    poll_url = f"{base_url}/v1/videos/{job_id}"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        if status == "completed":
            break
        elif status == "failed":
            raise RuntimeError(f"Video generation failed: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"Video generation timed out after {REQUEST_TIMEOUT}s")

    client_latency = time.time() - start

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Video generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Video generated in {client_latency:.2f}s")
    return client_latency


def send_image_conditioned_request_sglang(
    base_url: str, case: dict, config: dict, perf_dump_path: str | None = None
) -> float:
    """Send an image-conditioned request (edit/I2V/TI2V) via SGLang multipart API."""
    task = case["task"]
    ref_bytes = _get_ref_image_bytes(config)

    # Build multipart form — field name depends on endpoint:
    # image edits use "image", video (I2V/TI2V) uses "input_reference"
    if task in ("image-to-video", "text-image-to-video"):
        file_field = "input_reference"
    else:
        file_field = "image"
    files = {file_field: ("ref.png", io.BytesIO(ref_bytes), "image/png")}
    data = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": "1",
        "response_format": "b64_json",
    }
    for key in (
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "seed",
        "num_frames",
        "fps",
        "negative_prompt",
    ):
        if key in case:
            data[key] = str(case[key])
    if perf_dump_path:
        data["perf_dump_path"] = perf_dump_path
    # Choose endpoint based on task
    if task in ("image-edit", "image-to-image"):
        endpoint = "/v1/images/edits"
    elif task in ("image-to-video", "text-image-to-video"):
        endpoint = "/v1/videos"
    else:
        endpoint = "/v1/images/generations"

    start = time.time()
    resp = requests.post(
        f"{base_url}{endpoint}",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    # For video endpoints, need to poll
    if task in ("image-to-video", "text-image-to-video"):
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Video submit returned no job id: {job}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            time.sleep(1)
            poll_resp = requests.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            if status == "completed":
                break
            elif status == "failed":
                raise RuntimeError(f"Video generation failed: {poll_data}")
            if time.time() - start > REQUEST_TIMEOUT:
                raise TimeoutError(f"Timed out after {REQUEST_TIMEOUT}s")
    else:
        resp.raise_for_status()

    client_latency = time.time() - start

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Generated in {client_latency:.2f}s (sglang, image-conditioned)")
    return client_latency


# ---------------------------------------------------------------------------
# Request helpers — vLLM-Omni
# ---------------------------------------------------------------------------


def send_request_vllm_omni(base_url: str, case: dict, config: dict) -> float:
    """Send request via vLLM-Omni's OpenAI-compatible diffusion endpoints."""
    task = case["task"]
    if task in ("text-to-video", "image-to-video", "text-image-to-video"):
        data = {
            "prompt": case["prompt"],
            "size": f"{case['width']}x{case['height']}",
            "width": str(case["width"]),
            "height": str(case["height"]),
        }
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "seed",
            "num_frames",
            "fps",
            "negative_prompt",
        ):
            if key in case:
                data[key] = str(case[key])
        files = None
        if case.get("reference_image"):
            files = {
                "input_reference": (
                    "ref.png",
                    io.BytesIO(_get_ref_image_bytes(config)),
                    "image/png",
                )
            }

        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/videos",
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"vLLM-Omni video submit returned no id: {job}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            time.sleep(1)
            poll_resp = requests.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = str(poll_data.get("status") or "").lower()
            if status in ("completed", "succeeded", "success"):
                break
            if status in ("failed", "error", "cancelled", "canceled") or poll_data.get(
                "error"
            ):
                raise RuntimeError(f"vLLM-Omni video generation failed: {poll_data}")
            if time.time() - start > REQUEST_TIMEOUT:
                raise TimeoutError(
                    f"vLLM-Omni video timed out after {REQUEST_TIMEOUT}s"
                )
        latency = time.time() - start
        print(f"  Generated in {latency:.2f}s (vllm-omni)")
        return latency

    if task == "text-to-image":
        payload = _build_sglang_payload(case)
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/images/generations",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        latency = time.time() - start
        resp.raise_for_status()
        return latency
    if task in ("image-edit", "image-to-image"):
        files = {
            "image": ("ref.png", io.BytesIO(_get_ref_image_bytes(config)), "image/png")
        }
        data = {
            "model": case["model"],
            "prompt": case["prompt"],
            "size": f"{case['width']}x{case['height']}",
            "n": "1",
            "response_format": "b64_json",
        }
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "true_cfg_scale",
            "seed",
            "negative_prompt",
        ):
            if key in case:
                data[key] = str(case[key])
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/images/edits",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        latency = time.time() - start
        resp.raise_for_status()
        return latency

    extra_body = {
        "height": case["height"],
        "width": case["width"],
        "num_inference_steps": case.get("num_inference_steps", 50),
        "guidance_scale": case.get("guidance_scale", 4.0),
        "seed": case.get("seed", 42),
    }
    if "num_frames" in case:
        extra_body["num_frames"] = case["num_frames"]
    if "fps" in case:
        extra_body["fps"] = case["fps"]
    if "negative_prompt" in case:
        extra_body["negative_prompt"] = case["negative_prompt"]
    if "guidance_scale_2" in case:
        extra_body["guidance_scale_2"] = case["guidance_scale_2"]
    if "true_cfg_scale" in case:
        extra_body["true_cfg_scale"] = case["true_cfg_scale"]

    # Build message content (text or text+image)
    content: list[dict] | str = case["prompt"]
    if case.get("reference_image"):
        ref_b64 = _get_ref_image_b64(config)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            },
            {"type": "text", "text": case["prompt"]},
        ]

    payload = {
        "model": case["model"],
        "messages": [{"role": "user", "content": content}],
        "extra_body": extra_body,
    }

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"vLLM-Omni request returned no choices: {data}")
    print(f"  Generated in {latency:.2f}s (vllm-omni)")
    return latency


# ---------------------------------------------------------------------------
# Request helpers — LightX2V
# ---------------------------------------------------------------------------


def send_request_lightx2v(base_url: str, case: dict, config: dict) -> float:
    """Send request via LightX2V's async task API."""
    endpoint = "/v1/tasks/"
    suffix = (
        ".png"
        if case["task"] in ("text-to-image", "image-edit", "image-to-image")
        else ".mp4"
    )

    payload = {
        "prompt": case["prompt"],
        "seed": case.get("seed", 42),
        "infer_steps": case.get("num_inference_steps", 50),
        "save_result_path": os.path.join(
            tempfile.gettempdir(),
            f"lightx2v_{case['id']}_{int(time.time() * 1000)}{suffix}",
        ),
    }
    # LightX2V uses target_video_length for frames, height/width directly
    if "num_frames" in case:
        payload["target_video_length"] = case["num_frames"]
    if "height" in case:
        payload["height"] = case["height"]
    if "width" in case:
        payload["width"] = case["width"]
    if "height" in case and "width" in case:
        payload["target_shape"] = [case["height"], case["width"]]
    if "guidance_scale" in case:
        payload["guidance_scale"] = case["guidance_scale"]
    if "fps" in case:
        payload["fps"] = case["fps"]
    if "negative_prompt" in case:
        payload["negative_prompt"] = case["negative_prompt"]
    if case.get("reference_image"):
        payload["image_path"] = _get_ref_image_path(config)

    start = time.time()

    # Submit task
    resp = requests.post(
        f"{base_url}{endpoint}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data.get("task_id")
    if not task_id:
        raise RuntimeError(f"LightX2V submit returned no task_id: {task_data}")

    # Poll for completion
    poll_url = f"{base_url}/v1/tasks/{task_id}/status"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = (
            poll_data.get("task_status") or poll_data.get("status") or ""
        ).upper()
        if status == "COMPLETED":
            break
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"LightX2V task {status}: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"LightX2V task timed out after {REQUEST_TIMEOUT}s")

    latency = time.time() - start
    print(f"  Generated in {latency:.2f}s (lightx2v)")
    return latency


# ---------------------------------------------------------------------------
# Unified request dispatcher
# ---------------------------------------------------------------------------


def send_request(
    base_url: str,
    case: dict,
    framework: str = "sglang",
    config: dict | None = None,
    perf_dump_path: str | None = None,
) -> float:
    config = config or {}
    if framework == "vllm-omni":
        return send_request_vllm_omni(base_url, case, config)
    elif framework == "lightx2v":
        return send_request_lightx2v(base_url, case, config)
    # SGLang — use OpenAI-compatible endpoints with optional perf log
    task = case["task"]
    if case.get("reference_image"):
        return send_image_conditioned_request_sglang(
            base_url, case, config, perf_dump_path
        )
    elif task == "text-to-image":
        return send_image_request_sglang(base_url, case, perf_dump_path)
    elif task == "text-to-video":
        return send_video_request_sglang(base_url, case, perf_dump_path)
    else:
        raise ValueError(f"Unknown task type: {task}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _merge_nested(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _benchmark_config(config: dict | None, case: dict) -> dict:
    cfg = _merge_nested(DEFAULT_BENCHMARK, (config or {}).get("benchmark_defaults", {}))
    return _merge_nested(cfg, case.get("benchmark", {}))


def _base_result(case: dict, framework: str, mode: str) -> dict:
    result = {
        "case_id": case["id"],
        "framework": framework,
        "mode": mode,
        "model": case["model"],
        "task": case["task"],
        "width": case.get("width"),
        "height": case.get("height"),
        "num_frames": case.get("num_frames"),
        "fps": case.get("fps"),
        "num_inference_steps": case.get("num_inference_steps"),
        "guidance_scale": case.get("guidance_scale"),
        "guidance_scale_2": case.get("guidance_scale_2"),
        "true_cfg_scale": case.get("true_cfg_scale"),
        "negative_prompt_set": "negative_prompt" in case,
        "num_gpus": case.get("num_gpus"),
        "latency_s": None,
        "error": None,
    }
    metadata = case.get("_framework_metadata")
    if metadata:
        result["framework_metadata"] = metadata
    command = case.get("_server_command")
    if command:
        result["server_command"] = command
    return result


def _case_for_framework(case: dict, fw_cfg: dict) -> dict:
    overrides = {key: fw_cfg[key] for key in ("model", "num_gpus") if key in fw_cfg}
    case_for_fw = {**case, **overrides} if overrides else dict(case)
    metadata = fw_cfg.get("_benchmark_metadata")
    if metadata:
        case_for_fw["_framework_metadata"] = metadata
    return case_for_fw


def _current_commit_sha() -> str:
    try:
        ret = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ret.returncode == 0:
            return ret.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("GITHUB_SHA", "unknown")


def _collect_hardware_metadata() -> dict:
    metadata = {
        "runner_labels": os.environ.get("RUNNER_LABELS"),
        "gpu_config": os.environ.get("GPU_CONFIG"),
    }
    try:
        query = "name,memory.total,driver_version"
        ret = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if ret.returncode == 0:
            metadata["gpus"] = [
                line.strip() for line in ret.stdout.splitlines() if line.strip()
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return metadata


def _collect_sglang_runtime_metadata() -> dict:
    metadata = {}
    try:
        from importlib.metadata import version

        metadata["package_version"] = version("sglang")
    except Exception:
        pass
    try:
        import sglang  # type: ignore

        module_path = Path(sglang.__file__).resolve()
        metadata["module_path"] = str(module_path)
        for parent in module_path.parents:
            if (parent / ".git").exists():
                metadata["git_root"] = str(parent)
                ret = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if ret.returncode == 0:
                    metadata["git_commit"] = ret.stdout.strip()
                break
    except Exception:
        pass
    return metadata


def _parse_pip_show(output: str) -> dict:
    packages = {}
    current = {}
    for line in output.splitlines():
        if line.strip() == "---":
            if current.get("Name"):
                packages[current["Name"]] = current
            current = {}
            continue
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in {"Name", "Version", "Location", "Home-page"}:
            current[key] = value
    if current.get("Name"):
        packages[current["Name"]] = current
    return packages


def _collect_framework_runtime_metadata() -> dict:
    metadata = {
        "venv_root": str(_framework_venv_root()),
        "install_specs": {
            "vllm": os.environ.get("VLLM_INSTALL_SPEC", "vllm==0.18.0"),
            "vllm_omni": os.environ.get(
                "VLLM_OMNI_INSTALL_SPEC", "vllm-omni==0.18.0"
            ),
            "lightx2v": os.environ.get(
                "LIGHTX2V_INSTALL_SPEC",
                "git+https://github.com/ModelTC/LightX2V.git@573b9613adb0c1d33894b0920b5e12c87e42d280",
            ),
            "lightx2v_flash_attn": os.environ.get(
                "LIGHTX2V_FLASH_ATTN_INSTALL_SPEC", "flash-attn==2.8.3"
            ),
            "lightx2v_flash_attn3": os.environ.get(
                "LIGHTX2V_FLASH_ATTN3_INSTALL_SPEC", ""
            ),
            "lightx2v_fa3_hf_repo": os.environ.get(
                "LIGHTX2V_FA3_HF_REPO", "varunneal/flash-attention-3"
            ),
            "lightx2v_fa3_hf_revision": os.environ.get(
                "LIGHTX2V_FA3_HF_REVISION",
                "de87b9b5af06dd9984df595bef90b2eba44b181a",
            ),
            "lightx2v_fa3_hf_subdir": os.environ.get(
                "LIGHTX2V_FA3_HF_SUBDIR",
                "build/torch28-cxx11-cu128-x86_64-linux/flash_attention_3",
            ),
            "lightx2v_flashinfer": os.environ.get(
                "LIGHTX2V_FLASHINFER_INSTALL_SPEC", "flashinfer-python==0.6.11"
            ),
        },
    }
    packages_by_framework = {
        "vllm-omni": ["vllm", "vllm-omni"],
        "lightx2v": ["lightx2v", "flash-attn", "flash-attn-3", "flashinfer-python"],
    }
    for framework, packages in packages_by_framework.items():
        venv_path = _framework_venv_path(framework)
        framework_metadata = {"venv_path": str(venv_path)}
        python_bin = venv_path / "bin" / "python3"
        if python_bin.exists():
            try:
                ret = subprocess.run(
                    [str(python_bin), "-m", "pip", "show", *packages],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if ret.stdout:
                    framework_metadata["packages"] = _parse_pip_show(ret.stdout)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        site_packages = next(venv_path.glob("lib/python*/site-packages"), None)
        if site_packages is not None:
            direct_urls = {}
            for direct_url in site_packages.glob("*.dist-info/direct_url.json"):
                try:
                    direct_urls[direct_url.parent.name] = json.loads(
                        direct_url.read_text()
                    )
                except Exception:
                    pass
            if direct_urls:
                framework_metadata["direct_urls"] = direct_urls
        metadata[framework] = framework_metadata
    return metadata


def _run_warmups(
    base_url: str,
    case: dict,
    framework: str,
    config: dict | None,
    bench_cfg: dict,
) -> None:
    warmup_cfg = bench_cfg.get("warmup", {})
    warmup_requests = int(warmup_cfg.get("num_requests", 0) or 0)
    warmup_steps = warmup_cfg.get("num_inference_steps")
    if framework == "lightx2v" and warmup_requests > 0:
        # lightx2v takes infer_steps from launch config; per-request overrides corrupt server reuse
        warmup_requests = min(warmup_requests, 1)
        warmup_steps = None
    if warmup_requests <= 0:
        return
    warmup_case = dict(case)
    if warmup_steps is not None:
        warmup_case["num_inference_steps"] = int(warmup_steps)
    for wi in range(1, warmup_requests + 1):
        print(f"  Sending warmup request ({wi}/{warmup_requests})...")
        try:
            send_request(base_url, warmup_case, framework, config)
        except Exception as e:
            print(f"  Warmup request {wi} failed (non-fatal): {e}")


def run_single_request(
    base_url: str,
    case: dict,
    framework: str,
    log_dir: Path,
    config: dict | None = None,
) -> dict:
    result = _base_result(case, framework, MODE_SINGLE_E2E)

    perf_dump_path = None
    if framework == "sglang":
        perf_dump_path = os.path.join(str(log_dir), f"perf_{case['id']}_measured.json")
    if perf_dump_path and os.path.exists(perf_dump_path):
        os.remove(perf_dump_path)
    print("  Sending measured single request...")
    latency = send_request(
        base_url, case, framework, config, perf_dump_path=perf_dump_path
    )
    result["latency_s"] = round(latency, 3)
    return result


def _bench_serving_task(task: str) -> str:
    return {
        "image-edit": "image-to-image",
        "text-image-to-video": "image-to-video",
    }.get(task, task)


def _bench_extra_body(case: dict) -> dict:
    extra_body = {}
    for key in (
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "negative_prompt",
        "seed",
    ):
        if key in case:
            extra_body[key] = case[key]
    return extra_body


def run_throughput(
    base_url: str,
    case: dict,
    framework: str,
    config: dict | None,
    bench_cfg: dict,
    log_dir: Path,
) -> dict:
    result = _base_result(case, framework, MODE_THROUGHPUT)
    throughput_cfg = bench_cfg.get("throughput", {})
    num_requests = int(throughput_cfg.get("num_requests", 4) or 4)
    max_concurrency = int(throughput_cfg.get("max_concurrency", 2) or 2)
    max_concurrency = max(1, min(max_concurrency, num_requests))

    metrics_path = log_dir / f"bench_serving_{case['id']}_{framework}.json"
    cmd = [
        sys.executable,
        "-m",
        "diffusion_bench.bench_serving",
        "--backend",
        framework,
        "--base-url",
        base_url,
        "--dataset",
        "fixed",
        "--prompt",
        case["prompt"],
        "--model",
        case["model"],
        "--task",
        _bench_serving_task(case["task"]),
        "--num-prompts",
        str(num_requests),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        str(throughput_cfg.get("request_rate", "inf")),
        "--request-timeout",
        str(REQUEST_TIMEOUT * max(1, max_concurrency)),
        "--output-file",
        str(metrics_path),
        "--disable-tqdm",
    ]
    for key in ("width", "height", "num_frames", "fps", "num_inference_steps"):
        if key in case:
            cmd.extend([f"--{key.replace('_', '-')}", str(case[key])])
    extra_body = _bench_extra_body(case)
    if extra_body:
        cmd.extend(["--extra-body", json.dumps(extra_body)])
    if case.get("reference_image"):
        cmd.extend(["--image-path", _get_ref_image_path(config or {})])

    print(
        f"  Running bench_serving throughput: requests={num_requests}, "
        f"max_concurrency={max_concurrency}"
    )
    ret = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=REQUEST_TIMEOUT * max(1, num_requests),
    )
    bench_log_path = log_dir / f"bench_serving_{case['id']}_{framework}.log"
    with open(bench_log_path, "w", encoding="utf-8") as f:
        f.write(ret.stdout)
    if ret.returncode != 0:
        result["error"] = f"bench_serving failed (exit {ret.returncode})"
        return result

    with open(metrics_path) as f:
        metrics = json.load(f)

    completed_requests = int(metrics.get("completed_requests", 0) or 0)
    failed_requests = int(metrics.get("failed_requests", 0) or 0)
    observed_requests = completed_requests + failed_requests

    result.update(
        {
            "latency_s": round(metrics.get("latency_p50", 0), 3),
            "metrics": {
                "duration_s": round(metrics.get("duration", 0), 3),
                "num_requests": observed_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "max_concurrency": max_concurrency,
                "throughput_rps": round(metrics.get("throughput_qps", 0), 4),
                "output_throughput_ops": round(
                    metrics.get("output_throughput_ops", 0), 4
                ),
                "latency_mean_s": round(metrics.get("latency_mean", 0), 3),
                "latency_p50_s": round(metrics.get("latency_p50", 0), 3),
                "latency_p90_s": round(metrics.get("latency_p90", 0), 3),
                "latency_p95_s": round(metrics.get("latency_p95", 0), 3),
                "latency_p99_s": round(metrics.get("latency_p99", 0), 3),
                "errors": metrics.get("errors", []),
            },
        }
    )
    if failed_requests or completed_requests != num_requests:
        result["error"] = (
            "bench_serving partial failure "
            f"({completed_requests}/{num_requests} completed, "
            f"{failed_requests} failed)"
        )
    return result


def run_case_framework(
    case: dict,
    framework: str,
    fw_cfg: dict,
    modes: list[str],
    port: int,
    log_dir: Path,
    config: dict | None = None,
) -> tuple[dict | None, dict | None]:
    """Run one server lifecycle and collect requested benchmark modes."""
    case = _case_for_framework(case, fw_cfg)
    single_result = None
    throughput_result = None
    cmd = build_server_cmd(framework, case, fw_cfg, port)
    case["_server_command"] = shlex.join(cmd)
    print(f"\n  Command: {shlex.join(cmd)}")

    env = os.environ.copy()
    env.update(fw_cfg.get("extra_env", {}))
    env.update(FORCED_BENCHMARK_ENV)
    env = _framework_env(framework, env)

    log_file = log_dir / f"{case['id']}_{framework}.log"
    log_fh = open(log_file, "w", encoding="utf-8", buffering=1)
    log_thread = None

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )

        # Tee server output to both log file and stdout (like test_server_utils)
        def _log_pipe(pipe, fh):
            try:
                for line in iter(pipe.readline, ""):
                    sys.stdout.write(f"  [server] {line}")
                    sys.stdout.flush()
                    fh.write(line)
            except ValueError:
                pass  # pipe closed

        log_thread = threading.Thread(target=_log_pipe, args=(proc.stdout, log_fh))
        log_thread.daemon = True
        log_thread.start()

        base_url = f"http://{DEFAULT_HOST}:{port}"
        wait_for_health(base_url, framework, proc=proc)
        bench_cfg = _merge_nested(
            _benchmark_config(config, case), fw_cfg.get("benchmark", {})
        )
        _run_warmups(base_url, case, framework, config, bench_cfg)

        if MODE_SINGLE_E2E in modes:
            single_result = run_single_request(
                base_url, case, framework, log_dir, config
            )
        if MODE_THROUGHPUT in modes:
            throughput_result = run_throughput(
                base_url, case, framework, config, bench_cfg, log_dir
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        if MODE_SINGLE_E2E in modes:
            single_result = _base_result(case, framework, MODE_SINGLE_E2E)
            single_result["error"] = str(e)
        if MODE_THROUGHPUT in modes:
            throughput_result = _base_result(case, framework, MODE_THROUGHPUT)
            throughput_result["error"] = str(e)
    finally:
        if proc:
            kill_server(proc)
        if log_thread:
            log_thread.join(timeout=5)
        log_fh.close()

    return single_result, throughput_result


def _install_framework(fw_name: str, dry_run: bool = False) -> bool:
    """Install a comparison framework via the install script. Returns True on success."""
    if fw_name not in INSTALLABLE_FRAMEWORKS:
        return True
    if os.environ.get(SKIP_FRAMEWORK_INSTALL_ENV) == "1":
        print(f"  Skipping {fw_name} installation ({SKIP_FRAMEWORK_INSTALL_ENV}=1)")
        return True
    if not INSTALL_SCRIPT.exists():
        print(f"  WARNING: Install script not found at {INSTALL_SCRIPT}")
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would install: bash {INSTALL_SCRIPT} {fw_name}")
        return True
    print(f"\n{'='*60}")
    print(f"Installing framework: {fw_name}")
    print(f"{'='*60}")
    ret = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), fw_name],
        timeout=600,
    )
    if ret.returncode != 0:
        print(f"  WARNING: {fw_name} installation failed (exit {ret.returncode})")
        return False
    return True


def _framework_venv_path(fw_name: str) -> Path:
    return _framework_venv_root() / fw_name


def _framework_venv_root() -> Path:
    return Path(
        os.environ.get(
            "SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT",
            "/tmp/sglang-diffusion-framework-venvs",
        )
    )


def _framework_env(fw_name: str, env: dict[str, str]) -> dict[str, str]:
    if fw_name not in INSTALLABLE_FRAMEWORKS:
        return env
    venv_path = _framework_venv_path(fw_name)
    bin_path = os.path.join(venv_path, "bin")
    framework_env = dict(env)
    framework_env["VIRTUAL_ENV"] = str(venv_path)
    framework_env["PATH"] = f"{bin_path}:{framework_env.get('PATH', '')}"
    return framework_env


def run_comparison(
    config: dict,
    case_ids: list[str] | None = None,
    frameworks: list[str] | None = None,
    modes: list[str] | None = None,
    sglang_profile: str | None = None,
    hardware_profile: str | None = None,
    run_id: str | None = None,
    port: int = DEFAULT_PORT,
    output: str = "comparison-results.json",
    dry_run: bool = False,
) -> dict:
    """Run all comparison cases, grouped by framework to minimize installs.

    Order: sglang first (already installed), then vllm-omni, then lightx2v.
    Each non-sglang framework is installed right before its cases run.
    """
    os.environ.update(FORCED_BENCHMARK_ENV)
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = _current_commit_sha()
    run_id = (
        run_id
        or os.environ.get("DIFFUSION_BENCH_RUN_ID")
        or os.environ.get("GITHUB_RUN_ID")
        or "local"
    )
    hardware_metadata = _collect_hardware_metadata()
    if hardware_profile:
        hardware_metadata["hardware_profile_override"] = hardware_profile

    log_dir = Path("comparison-logs")
    log_dir.mkdir(exist_ok=True)

    modes = modes or [MODE_SINGLE_E2E]
    invalid_modes = sorted(set(modes) - {MODE_SINGLE_E2E, MODE_THROUGHPUT})
    if invalid_modes:
        raise ValueError(f"Unknown benchmark mode(s): {invalid_modes}")

    fw_cases: dict[str, list[tuple[dict, dict]]] = {fw: [] for fw in FRAMEWORK_ORDER}

    for case in config["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        for fw_name, fw_cfg in case["frameworks"].items():
            if frameworks and fw_name not in frameworks:
                continue
            resolved_fw_cfg = _resolve_framework_config(
                fw_name, fw_cfg, sglang_profile, hardware_metadata
            )
            if fw_name not in fw_cases:
                fw_cases[fw_name] = []
            fw_cases[fw_name].append((case, resolved_fw_cfg))

    results = []
    throughput_results = []
    installed_fws: set[str] = set()

    for fw_name in FRAMEWORK_ORDER:
        pairs = fw_cases.get(fw_name, [])
        if not pairs:
            continue

        # Install framework if needed (once per framework)
        if fw_name not in installed_fws and fw_name in INSTALLABLE_FRAMEWORKS:
            if not _install_framework(fw_name, dry_run):
                # Skip all cases for this framework
                for case, pair_fw_cfg in pairs:
                    case_for_fw = _case_for_framework(case, pair_fw_cfg)
                    if MODE_SINGLE_E2E in modes:
                        result = _base_result(case_for_fw, fw_name, MODE_SINGLE_E2E)
                        result["error"] = f"{fw_name} installation failed"
                        results.append(result)
                    if MODE_THROUGHPUT in modes:
                        result = _base_result(case_for_fw, fw_name, MODE_THROUGHPUT)
                        result["error"] = f"{fw_name} installation failed"
                        throughput_results.append(result)
                continue
            installed_fws.add(fw_name)

        for case, fw_cfg in pairs:
            print(f"\n{'='*60}")
            print(f"Case: {case['id']} | Model: {case['model']} | Framework: {fw_name}")
            print(f"{'='*60}")

            if dry_run:
                case_for_fw = _case_for_framework(case, fw_cfg)
                dry_fw_cfg = dict(fw_cfg)
                dry_fw_cfg["_skip_model_path_resolution"] = True
                cmd = build_server_cmd(fw_name, case_for_fw, dry_fw_cfg, port)
                case_for_fw["_server_command"] = shlex.join(cmd)
                print(f"  [DRY-RUN] Would run: {shlex.join(cmd)}")
                if fw_name in INSTALLABLE_FRAMEWORKS:
                    print(f"  [DRY-RUN] venv: {_framework_venv_path(fw_name)}")
                if MODE_SINGLE_E2E in modes:
                    result = _base_result(case_for_fw, fw_name, MODE_SINGLE_E2E)
                    result["error"] = "dry-run"
                    results.append(result)
                if MODE_THROUGHPUT in modes:
                    result = _base_result(case_for_fw, fw_name, MODE_THROUGHPUT)
                    result["error"] = "dry-run"
                    throughput_results.append(result)
                continue

            single_result, throughput_result = run_case_framework(
                case, fw_name, fw_cfg, modes, port, log_dir, config
            )
            if single_result is not None:
                results.append(single_result)
            if throughput_result is not None:
                throughput_results.append(throughput_result)

            # Wait for GPU memory to clear
            print(f"  Waiting {GPU_CLEAR_WAIT}s for GPU memory to clear...")
            time.sleep(GPU_CLEAR_WAIT)

    output_data = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "run_id": run_id,
        "hardware": hardware_metadata,
        "sglang_runtime": _collect_sglang_runtime_metadata(),
        "framework_runtime": _collect_framework_runtime_metadata(),
        "benchmark_env": FORCED_BENCHMARK_ENV,
        "benchmark_framework_args": {
            "vllm-omni": VLLM_DISABLE_TORCH_COMPILE_ARGS,
            "lightx2v": {"config": LIGHTX2V_DISABLE_TORCH_COMPILE_CONFIG},
        },
        "benchmark_modes": modes,
        "requested_sglang_profile": _requested_sglang_profile(sglang_profile),
        "results": results,
        "throughput_results": throughput_results,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {output}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        lat = f"{r['latency_s']:.2f}s" if r["latency_s"] else r.get("error", "N/A")
        print(f"  {r['case_id']:30s} | {r['framework']:12s} | {lat}")
    if throughput_results:
        print("\nTHROUGHPUT")
        for r in throughput_results:
            metrics = r.get("metrics", {})
            throughput = metrics.get("throughput_rps")
            value = (
                f"{throughput:.4f} req/s"
                if isinstance(throughput, (float, int))
                else r.get("error", "N/A")
            )
            print(f"  {r['case_id']:30s} | {r['framework']:12s} | {value}")

    return output_data


def _apply_throughput_overrides(
    config: dict,
    num_requests: int | None,
    max_concurrency: int | None,
    request_rate: str | None,
) -> dict:
    if num_requests is None and max_concurrency is None and request_rate is None:
        return config
    config = copy.deepcopy(config)
    throughput_cfg = config.setdefault("benchmark_defaults", {}).setdefault(
        "throughput", {}
    )
    if num_requests is not None:
        throughput_cfg["num_requests"] = num_requests
    if max_concurrency is not None:
        throughput_cfg["max_concurrency"] = max_concurrency
    if request_rate is not None:
        throughput_cfg["request_rate"] = request_rate
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-framework diffusion serving comparison benchmark"
    )
    parser.add_argument(
        "--config",
        default=str(CONFIGS_PATH),
        help="Path to comparison_configs.json",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Only run specific case IDs",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="Only run specific frameworks (sglang, vllm-omni, lightx2v)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[MODE_SINGLE_E2E],
        choices=[MODE_SINGLE_E2E, MODE_THROUGHPUT],
        help="Benchmark modes to run",
    )
    parser.add_argument(
        "--sglang-profile",
        default=None,
        help=(
            "Command profile to use from each case's command_profiles. "
            "Defaults to SGLANG_BENCH_SGLANG_PROFILE, SGLANG_BENCH_PROFILE, "
            "then hardware auto-selection and 'default'. Other frameworks can "
            "use DIFFUSION_BENCH_<FRAMEWORK>_PROFILE or "
            "DIFFUSION_BENCH_FRAMEWORK_PROFILE."
        ),
    )
    parser.add_argument(
        "--hardware-profile",
        default=None,
        help=(
            "Hardware class override for SGLang profile auto-selection "
            "(for example h100 or h200). Defaults to "
            f"{HARDWARE_PROFILE_ENV}, GPU_CONFIG, RUNNER_LABELS, then nvidia-smi."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable identifier to record in the output JSON",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    parser.add_argument(
        "--output",
        default="comparison-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print commands without launching servers",
    )
    parser.add_argument(
        "--throughput-num-requests",
        type=int,
        default=None,
        help="Override benchmark_defaults.throughput.num_requests",
    )
    parser.add_argument(
        "--throughput-max-concurrency",
        type=int,
        default=None,
        help="Override benchmark_defaults.throughput.max_concurrency",
    )
    parser.add_argument(
        "--throughput-request-rate",
        default=None,
        help="Override benchmark_defaults.throughput.request_rate",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config = _apply_throughput_overrides(
        config,
        args.throughput_num_requests,
        args.throughput_max_concurrency,
        args.throughput_request_rate,
    )

    print(f"Loaded {len(config['cases'])} comparison case(s) from {args.config}")

    output_data = run_comparison(
        config=config,
        case_ids=args.case_ids,
        frameworks=args.frameworks,
        modes=args.modes,
        sglang_profile=args.sglang_profile,
        hardware_profile=args.hardware_profile,
        run_id=args.run_id,
        port=args.port,
        output=args.output,
        dry_run=args.dry_run,
    )

    # Exit with non-zero if any case had an error
    errors = [
        r
        for r in output_data.get("results", [])
        + output_data.get("throughput_results", [])
        if r.get("error")
    ]
    if errors and not args.dry_run:
        print(f"\n{len(errors)} case(s) had errors:")
        for e in errors:
            print(f"  {e['case_id']} ({e['framework']}): {e['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
