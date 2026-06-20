#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import http.client
import itertools
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path(os.environ.get("SGLANG_REPO", ".")).resolve()
DEFAULT_OUT_ROOT = Path(os.environ.get("DIFFUSION_BENCH_PROBE_OUT", "sglang-probe-runs"))
DEFAULT_TIMEOUT_S = 600
DEFAULT_POLL_S = 1.0
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "zimage_t2i",
        "family": "t2i",
        "model_path": "Tongyi-MAI/Z-Image-Turbo",
        "modality": "image",
        "task_type": "T2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "A clean product photo of a red cube on a white desk.",
        "tags": ["native", "small", "layerwise_candidate", "lora_candidate"],
        "lora_path": "reverentelusarca/elusarca-anime-style-lora-z-image-turbo",
        "second_lora_path": "tarn59/pixel_art_style_lora_z_image_turbo",
        "required_gpus": 1,
    },
    {
        "id": "qwen_image_t2i",
        "family": "t2i",
        "model_path": "Qwen/Qwen-Image",
        "modality": "image",
        "task_type": "T2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "A sharp studio photo of a ceramic teapot.",
        "tags": ["native", "diffusers_ok", "cache_dit_candidate"],
        "required_gpus": 1,
    },
    {
        "id": "flux1_t2i",
        "family": "t2i",
        "model_path": "black-forest-labs/FLUX.1-dev",
        "modality": "image",
        "task_type": "T2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "A minimal architectural interior with afternoon light.",
        "tags": ["native", "diffusers_ok"],
        "required_gpus": 1,
    },
    {
        "id": "flux2_t2i",
        "family": "t2i",
        "model_path": "black-forest-labs/FLUX.2-dev",
        "modality": "image",
        "task_type": "T2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "A detailed editorial photo of a blue glass vase.",
        "tags": ["native", "diffusers_ok", "postprocess_candidate"],
        "required_gpus": 1,
    },
    {
        "id": "qwen_image_edit_ti2i",
        "family": "ti2i",
        "model_path": "Qwen/Qwen-Image-Edit",
        "modality": "image",
        "task_type": "TI2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "Turn the object into a watercolor illustration.",
        "tags": ["native", "image_input", "lora_candidate"],
        "lora_path": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "required_gpus": 1,
    },
    {
        "id": "qwen_image_layered_i2i",
        "family": "i2i",
        "model_path": "Qwen/Qwen-Image-Layered",
        "modality": "image",
        "task_type": "I2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "Blend the input layers into a coherent product image.",
        "tags": ["native", "image_input", "multi_image"],
        "required_gpus": 1,
    },
    {
        "id": "joyai_image_edit_ti2i",
        "family": "ti2i",
        "model_path": "jdopensource/JoyAI-Image-Edit-Diffusers",
        "modality": "image",
        "task_type": "TI2I",
        "size": "512x512",
        "steps": 4,
        "prompt": "Change the background to a calm beach scene.",
        "tags": ["diffusers_ok", "image_input"],
        "required_gpus": 1,
    },
    {
        "id": "wan21_t2v_1_3b",
        "family": "t2v",
        "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "modality": "video",
        "task_type": "T2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A blue box slides across a clean warehouse floor.",
        "tags": ["native", "diffusers_ok", "layerwise_candidate", "lora_candidate"],
        "dynamic_lora_path": "Cseti/Wan-LoRA-Arcane-Jinx-v1",
        "required_gpus": 1,
    },
    {
        "id": "turbo_wan21_t2v",
        "family": "t2v",
        "model_path": "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        "modality": "video",
        "task_type": "T2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A close-up of paper lanterns moving in the wind.",
        "tags": ["native", "video_fast"],
        "required_gpus": 1,
    },
    {
        "id": "fast_hunyuan_video",
        "family": "t2v",
        "model_path": "FastVideo/FastHunyuan-diffusers",
        "modality": "video",
        "task_type": "T2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A camera moves slowly through a modern gallery.",
        "tags": ["diffusers_ok", "video_fast"],
        "required_gpus": 1,
    },
    {
        "id": "cosmos3_nano_t2v",
        "family": "t2v",
        "model_path": "nvidia/Cosmos3-Nano",
        "modality": "video",
        "task_type": "T2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A yellow robot arm sorts small boxes on a table.",
        "tags": ["native", "cosmos3"],
        "env": {"SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1"},
        "required_gpus": 1,
    },
    {
        "id": "wan21_i2v_480p",
        "family": "i2v",
        "model_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "modality": "video",
        "task_type": "I2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "Animate the still image with gentle camera motion.",
        "tags": ["native", "image_input", "large"],
        "required_gpus": 2,
    },
    {
        "id": "wan22_i2v_a14b",
        "family": "i2v",
        "model_path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "modality": "video",
        "task_type": "I2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "Animate the object with a slow dolly-in.",
        "tags": ["native", "image_input", "large", "dual_dit"],
        "required_gpus": 2,
    },
    {
        "id": "wan22_ti2v_5b",
        "family": "ti2v",
        "model_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "modality": "video",
        "task_type": "TI2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "Use the input image as the first frame, then pan right.",
        "tags": ["native", "image_input"],
        "required_gpus": 1,
    },
    {
        "id": "sana_wm_ti2v",
        "family": "ti2v",
        "model_path": "Efficient-Large-Model/SANA-WM_streaming",
        "modality": "video",
        "task_type": "TI2V",
        "size": "512x288",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "Move forward through a simple corridor.",
        "tags": ["native", "image_input", "sana_wm"],
        "required_gpus": 1,
    },
    {
        "id": "lingbot_world_realtime",
        "family": "realtime",
        "model_path": "robbyant/lingbot-world-fast-diffusers",
        "modality": "video",
        "task_type": "REALTIME",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A plastic beach scene with smooth camera motion.",
        "tags": ["native", "realtime", "image_input"],
        "server_args": [
            "--pipeline-class-name",
            "LingBotWorldCausalDMDPipeline",
            "--warmup",
            "false",
            "--text-encoder-cpu-offload",
            "true",
        ],
        "required_gpus": 1,
    },
    {
        "id": "hunyuan3d_shape",
        "family": "i2m",
        "model_path": "tencent/Hunyuan3D-2",
        "modality": "3d",
        "task_type": "I2M",
        "size": "512x512",
        "steps": 20,
        "prompt": "generate 3d mesh",
        "tags": ["native", "mesh", "image_input"],
        "required_gpus": 1,
    },
    {
        "id": "ltx23_two_stage",
        "family": "t2v",
        "model_path": "Lightricks/LTX-2.3",
        "modality": "video",
        "task_type": "T2V",
        "size": "768x512",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A cinematic shot of a small boat crossing a lake.",
        "tags": ["native", "ltx2", "two_stage"],
        "server_args": [
            "--pipeline-class-name",
            "LTX2TwoStageHQPipeline",
            "--ltx2-two-stage-device-mode",
            "original",
        ],
        "required_gpus": 1,
    },
    {
        "id": "ideogram4_nvfp4_b200",
        "family": "t2i",
        "model_path": "Comfy-Org/Ideogram-4",
        "modality": "image",
        "task_type": "T2I",
        "size": "1024x1024",
        "steps": 20,
        "prompt": "A crisp poster with the word SGLang.",
        "tags": ["b200", "nvfp4", "modelopt", "special"],
        "env": {"SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": "flashinfer_trtllm"},
        "required_gpus": 1,
    },
    {
        "id": "wan22_nvfp4_b200",
        "family": "t2v",
        "model_path": "nvidia/Wan2.2-T2V-A14B-nvfp4",
        "modality": "video",
        "task_type": "T2V",
        "size": "832x480",
        "steps": 4,
        "num_frames": 9,
        "fps": 8,
        "prompt": "A soft light sweeps across a studio wall.",
        "tags": ["b200", "nvfp4", "modelopt", "dual_dit", "special"],
        "env": {"SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": "flashinfer_trtllm"},
        "required_gpus": 1,
    },
]


ENV_GROUPS: dict[str, dict[str, str]] = {
    "baseline": {},
    "stage_logging": {"SGLANG_DIFFUSION_STAGE_LOGGING": "1"},
    "dev_mode": {"SGLANG_DIFFUSION_SERVER_DEV_MODE": "1"},
    "runai_off": {"SGLANG_USE_RUNAI_MODEL_STREAMER": "0"},
    "cache_dit_basic": {
        "SGLANG_CACHE_DIT_ENABLED": "true",
        "SGLANG_CACHE_DIT_WARMUP": "1",
        "SGLANG_CACHE_DIT_RDT": "0.24",
        "SGLANG_CACHE_DIT_MC": "2",
    },
    "cache_dit_scm_fast": {
        "SGLANG_CACHE_DIT_ENABLED": "true",
        "SGLANG_CACHE_DIT_SCM_PRESET": "fast",
        "SGLANG_CACHE_DIT_SCM_POLICY": "dynamic",
    },
    "cache_dit_secondary": {
        "SGLANG_CACHE_DIT_ENABLED": "true",
        "SGLANG_CACHE_DIT_SECONDARY_WARMUP": "1",
        "SGLANG_CACHE_DIT_SECONDARY_RDT": "0.24",
    },
    "profiler": {"SGLANG_DIFFUSION_TORCH_PROFILER_DIR": "{run_dir}/profiler"},
    "kernel_api_log": {
        "SGLANG_KERNEL_API_LOGLEVEL": "1",
        "SGLANG_KERNEL_API_LOGDEST": "stdout",
    },
    "misconfigured_s3": {
        "SGLANG_CLOUD_STORAGE_TYPE": "s3",
        "SGLANG_S3_BUCKET_NAME": "",
    },
}


SERVER_ARG_GROUPS: dict[str, list[str]] = {
    "baseline": [],
    "backend_auto": ["--backend", "auto"],
    "backend_sglang": ["--backend", "sglang"],
    "backend_diffusers": ["--backend", "diffusers"],
    "warmup_off": ["--warmup-mode", "off"],
    "warmup_request": ["--warmup-mode", "request"],
    "warmup_server": ["--warmup-mode", "server"],
    "persistent_io": ["--output-path", "{run_dir}/outputs", "--input-save-path", "{run_dir}/inputs"],
    "disabled_io": ["--output-path", "", "--input-save-path", ""],
    "batching": ["--batching-max-size", "4", "--batching-delay-ms", "20"],
    "batching_metrics": [
        "--batching-max-size",
        "4",
        "--batching-delay-ms",
        "20",
        "--enable-batching-metrics",
    ],
    "text_offload": ["--text-encoder-cpu-offload", "true"],
    "component_layerwise_offload": [
        "--layerwise-offload-components",
        "text_encoder",
        "image_encoder",
        "vae",
    ],
    "dit_layerwise_offload": [
        "--dit-layerwise-offload",
        "true",
        "--dit-offload-prefetch-size",
        "2",
    ],
    "fsdp_inference": ["--use-fsdp-inference", "true"],
    "cfg_parallel": ["--num-gpus", "2", "--enable-cfg-parallel", "true"],
    "ulysses2": ["--num-gpus", "2", "--ulysses-degree", "2"],
    "ring2": ["--num-gpus", "2", "--ulysses-degree", "1", "--ring-degree", "2"],
    "lora_static": ["--lora-path", "{lora_path}", "--lora-merge-mode", "auto"],
    "lora_dynamic": ["--lora-merge-mode", "dynamic"],
}


ENTRYPOINTS = [
    "serve_raw_http",
    "serve_openai_sdk",
    "cli_generate",
    "python_diffgenerator",
    "python_server_args",
    "python_lora",
]

REQUEST_FAMILIES = [
    "control",
    "image_generation",
    "image_edit",
    "video_generation",
    "mesh",
    "realtime",
    "invalid",
]

STRESS_PROFILES: dict[str, dict[str, int | float]] = {
    "smoke": {"requests": 20, "concurrency": 1, "video_jobs": 1},
    "stability": {"requests": 120, "concurrency": 4, "video_jobs": 2},
    "stress": {"requests": 500, "concurrency": 8, "video_jobs": 4},
}


SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABE0lEQVR4nO3aPU7DUBBF4ZMjr4iN0F"
    "C7omU1tFSvpmEjWQkNDTUFDRJBEZ7n2DcvX+mfpzsea2xZPrx/fJJMwkk4CSfhJJyEk3DTXzsen4/s"
    "zMvT3RV2QMJJOAkn4SSchJNwcq1P4qJ2//p74/z2wP4LaKei/9zVt4zpMtHXK8MLp198/LoFLEvTpY"
    "YOBVRytHIN8WPU4vn1S9hqK4zdgdZpkrTCOmN3YA8knISTcDJyAXOnN8q5sM7YHaBHE+baCsN3gNol"
    "nMsN7NOBZTm6zIBut9B/0/SaYBP9fGc6+2q5368SZ8vI+C60XtaTbmN0axJOwkk4CSfhJJyEk3ASTs"
    "JJuMPtx9eNSTgJJ+EknFsHqPoCsB49lg8iEdEAAAAASUVORK5CYII="
)


def now_ms() -> int:
    return int(time.time() * 1000)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def load_profile(name: str) -> dict[str, Any]:
    profile_path = PROFILE_DIR / f"{name}.json"
    if not profile_path.exists():
        available = ", ".join(sorted(path.stem for path in PROFILE_DIR.glob("*.json")))
        raise SystemExit(f"unknown profile {name!r}; available profiles: {available}")
    return json.loads(profile_path.read_text(encoding="utf-8"))


def apply_profile(args: argparse.Namespace) -> None:
    if not args.profile:
        return
    profile = load_profile(args.profile)
    if not args.phase:
        args.phase = list(profile.get("phases", []))
    if args.limit is None and profile.get("limit") is not None:
        args.limit = int(profile["limit"])
    if args.max_pairwise_cells == 180 and profile.get("max_pairwise_cells") is not None:
        args.max_pairwise_cells = int(profile["max_pairwise_cells"])


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def safe_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_")[:120] or "item"


def flatten_args(args: list[str], cell: dict[str, Any], run_dir: Path) -> list[str]:
    out: list[str] = []
    replacements = {
        "run_dir": str(run_dir),
        "lora_path": cell["model"].get("lora_path", ""),
        "second_lora_path": cell["model"].get("second_lora_path", ""),
        "dynamic_lora_path": cell["model"].get("dynamic_lora_path", ""),
    }
    for item in args:
        formatted = item.format(**replacements)
        if formatted != "":
            out.append(formatted)
    return out


def merge_env(cell: dict[str, Any], run_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    env.update(ENV_GROUPS[cell["env_group"]])
    env.update(cell["model"].get("env", {}))
    return {k: v.format(run_dir=str(run_dir)) for k, v in env.items()}


def arg_list_has(args: list[str], name: str) -> bool:
    return name in args


def get_num_gpus_from_args(args: list[str], fallback: int) -> int:
    for name in ("--num-gpus",):
        if name in args:
            idx = args.index(name)
            if idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except ValueError:
                    return fallback
    return fallback


def model_supports_request(model: dict[str, Any], request_family: str) -> bool:
    modality = model["modality"]
    tags = set(model.get("tags", []))
    if request_family in {"control", "invalid"}:
        return True
    if request_family == "image_generation":
        return modality == "image" and model["task_type"] == "T2I"
    if request_family == "image_edit":
        return modality == "image" and "image_input" in tags
    if request_family == "video_generation":
        return modality == "video" and "realtime" not in tags
    if request_family == "mesh":
        return modality == "3d"
    if request_family == "realtime":
        return "realtime" in tags
    return False


def compatible(model: dict[str, Any], entrypoint: str, arg_group: str, env_group: str, request_family: str) -> bool:
    tags = set(model.get("tags", []))
    if not model_supports_request(model, request_family):
        return False
    if entrypoint == "python_lora" and not (
        model.get("lora_path") or model.get("dynamic_lora_path")
    ):
        return False
    if entrypoint in {"cli_generate", "python_diffgenerator", "python_server_args", "python_lora"}:
        if request_family in {"control", "invalid", "realtime"}:
            return False
    if arg_group == "backend_diffusers" and "diffusers_ok" not in tags:
        return False
    if arg_group in {"lora_static", "lora_dynamic"} and not (
        model.get("lora_path") or model.get("dynamic_lora_path")
    ):
        return False
    if arg_group == "cfg_parallel" and model["modality"] == "3d":
        return False
    if arg_group in {"ulysses2", "ring2"} and model.get("required_gpus", 1) > 2:
        return False
    if arg_group == "ring2" and model["modality"] != "video":
        return False
    if arg_group in {"dit_layerwise_offload", "fsdp_inference"} and model["modality"] == "3d":
        return False
    if env_group == "cache_dit_secondary" and "dual_dit" not in tags:
        return False
    if env_group in {"cache_dit_basic", "cache_dit_scm_fast"} and model["modality"] == "3d":
        return False
    if env_group == "profiler" and request_family in {"control", "invalid"}:
        return False
    return True


def make_cell(
    *,
    phase: str,
    model: dict[str, Any],
    entrypoint: str,
    arg_group: str,
    env_group: str,
    request_family: str,
    stress_profile: str,
    suffix: str = "",
) -> dict[str, Any]:
    cell_id = "_".join(
        part
        for part in [
            phase,
            model["id"],
            entrypoint,
            request_family,
            arg_group,
            env_group,
            stress_profile,
            suffix,
        ]
        if part
    )
    return {
        "id": cell_id,
        "phase": phase,
        "model": model,
        "entrypoint": entrypoint,
        "arg_group": arg_group,
        "env_group": env_group,
        "request_family": request_family,
        "stress_profile": stress_profile,
    }


def extract_cli_args(repo: Path) -> dict[str, list[str]]:
    files = {
        "server_args": repo / "python/sglang/multimodal_gen/runtime/server_args.py",
        "sampling_params": repo / "python/sglang/multimodal_gen/configs/sample/sampling_params.py",
        "cli_generate": repo / "python/sglang/multimodal_gen/runtime/entrypoints/cli/generate.py",
        "cli_serve": repo / "python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py",
    }
    result: dict[str, list[str]] = {}
    for name, path in files.items():
        text = read_text(path)
        args = sorted(set(re.findall(r"""["'](--[a-zA-Z0-9][a-zA-Z0-9_-]*)["']""", text)))
        result[name] = args
    return result


def extract_env_vars(repo: Path) -> list[str]:
    docs = read_text(repo / "docs/diffusion/environment_variables.md")
    envs_py = read_text(repo / "python/sglang/multimodal_gen/envs.py")
    return sorted(set(re.findall(r"\bSGLANG_[A-Z0-9_]+\b", docs + "\n" + envs_py)))


def extract_protocol_fields(repo: Path) -> dict[str, list[str]]:
    text = read_text(
        repo / "python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py"
    )
    fields: dict[str, list[str]] = {}
    for class_name in [
        "ImageGenerationsRequest",
        "VideoGenerationsRequest",
        "RealtimeVideoGenerationsRequest",
        "MeshGenerationsRequest",
    ]:
        match = re.search(rf"class {class_name}\(.*?\):\n(?P<body>.*?)(?=\nclass |\n@dataclass|\Z)", text, re.S)
        if not match:
            fields[class_name] = []
            continue
        body = match.group("body")
        fields[class_name] = sorted(
            set(re.findall(r"^\s{4}([a-zA-Z_][a-zA-Z0-9_]*)\s*:", body, re.M))
        )
    return fields


def all_pair_keys(cell: dict[str, Any]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    axes = {
        "model": cell["model"]["id"],
        "entrypoint": cell["entrypoint"],
        "arg_group": cell["arg_group"],
        "env_group": cell["env_group"],
        "request_family": cell["request_family"],
        "stress_profile": cell["stress_profile"],
    }
    pairs = []
    items = sorted(axes.items())
    for (ak, av), (bk, bv) in itertools.combinations(items, 2):
        pairs.append(((ak, av), (bk, bv)))
    return pairs


def build_pairwise_cells(max_cells: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    candidates: list[dict[str, Any]] = []
    for model, entrypoint, arg_group, env_group, request_family, stress_profile in itertools.product(
        MODEL_CATALOG,
        ENTRYPOINTS,
        SERVER_ARG_GROUPS,
        ENV_GROUPS,
        REQUEST_FAMILIES,
        ("smoke", "stability"),
    ):
        if compatible(model, entrypoint, arg_group, env_group, request_family):
            candidates.append(
                make_cell(
                    phase="pairwise",
                    model=model,
                    entrypoint=entrypoint,
                    arg_group=arg_group,
                    env_group=env_group,
                    request_family=request_family,
                    stress_profile=stress_profile,
                )
            )

    random.shuffle(candidates)
    uncovered: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    candidate_pairs: dict[str, set[tuple[tuple[str, str], tuple[str, str]]]] = {}
    for cell in candidates:
        pairs = set(all_pair_keys(cell))
        candidate_pairs[cell["id"]] = pairs
        uncovered.update(pairs)

    selected: list[dict[str, Any]] = []
    remaining = candidates[:]
    while uncovered and remaining and len(selected) < max_cells:
        best_idx = 0
        best_score = -1
        for idx, cell in enumerate(remaining[:5000]):
            score = len(candidate_pairs[cell["id"]] & uncovered)
            if score > best_score:
                best_score = score
                best_idx = idx
        cell = remaining.pop(best_idx)
        selected.append(cell)
        uncovered -= candidate_pairs[cell["id"]]
        if best_score <= 0:
            break
    return selected


def build_smoke_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for model in MODEL_CATALOG:
        for request_family in REQUEST_FAMILIES:
            if not model_supports_request(model, request_family):
                continue
            entrypoint = "serve_raw_http"
            if request_family in {"image_generation", "image_edit", "video_generation"}:
                cells.append(
                    make_cell(
                        phase="smoke",
                        model=model,
                        entrypoint=entrypoint,
                        arg_group="baseline",
                        env_group="baseline",
                        request_family=request_family,
                        stress_profile="smoke",
                    )
                )
                cells.append(
                    make_cell(
                        phase="smoke",
                        model=model,
                        entrypoint="serve_openai_sdk",
                        arg_group="baseline",
                        env_group="baseline",
                        request_family=request_family,
                        stress_profile="smoke",
                    )
                )
                cells.append(
                    make_cell(
                        phase="smoke",
                        model=model,
                        entrypoint="cli_generate",
                        arg_group="baseline",
                        env_group="baseline",
                        request_family=request_family,
                        stress_profile="smoke",
                    )
                )
            elif request_family in {"mesh", "realtime", "control", "invalid"}:
                cells.append(
                    make_cell(
                        phase="smoke",
                        model=model,
                        entrypoint=entrypoint,
                        arg_group="baseline",
                        env_group="baseline",
                        request_family=request_family,
                        stress_profile="smoke",
                    )
                )
    return cells


def build_targeted_cells() -> list[dict[str, Any]]:
    by_id = {m["id"]: m for m in MODEL_CATALOG}
    specs = [
        ("zimage_t2i", "python_lora", "lora_static", "baseline", "image_generation", "stability", "layerwise_lora_zimage"),
        ("wan21_t2v_1_3b", "python_lora", "lora_dynamic", "baseline", "video_generation", "stability", "layerwise_lora_wan"),
        ("wan21_t2v_1_3b", "serve_raw_http", "dit_layerwise_offload", "stage_logging", "video_generation", "stability", "dit_layerwise_server"),
        ("qwen_image_t2i", "serve_raw_http", "backend_sglang", "cache_dit_basic", "image_generation", "stability", "cache_warmup_backend"),
        ("qwen_image_t2i", "serve_raw_http", "backend_diffusers", "cache_dit_scm_fast", "image_generation", "stability", "diffusers_cache"),
        ("wan22_i2v_a14b", "serve_raw_http", "ulysses2", "cache_dit_secondary", "video_generation", "stability", "sp_secondary_cache"),
        ("wan22_ti2v_5b", "serve_raw_http", "ring2", "stage_logging", "video_generation", "stability", "ring_frame_adjust"),
        ("wan21_t2v_1_3b", "serve_raw_http", "batching_metrics", "stage_logging", "video_generation", "stability", "batching_async_video"),
        ("hunyuan3d_shape", "serve_raw_http", "batching_metrics", "stage_logging", "mesh", "stability", "batching_async_mesh"),
        ("zimage_t2i", "serve_raw_http", "disabled_io", "baseline", "image_generation", "stability", "url_no_persistence"),
        ("flux2_t2i", "serve_raw_http", "backend_diffusers", "baseline", "image_generation", "stability", "diffusers_postprocess"),
        ("lingbot_world_realtime", "serve_raw_http", "baseline", "stage_logging", "realtime", "smoke", "realtime_session"),
    ]
    cells = []
    for model_id, entrypoint, arg_group, env_group, request_family, stress_profile, suffix in specs:
        model = by_id[model_id]
        if compatible(model, entrypoint, arg_group, env_group, request_family):
            cells.append(
                make_cell(
                    phase="targeted",
                    model=model,
                    entrypoint=entrypoint,
                    arg_group=arg_group,
                    env_group=env_group,
                    request_family=request_family,
                    stress_profile=stress_profile,
                    suffix=suffix,
                )
            )
    return cells


def build_stress_cells() -> list[dict[str, Any]]:
    ids = [
        ("zimage_t2i", "image_generation"),
        ("qwen_image_edit_ti2i", "image_edit"),
        ("wan21_t2v_1_3b", "video_generation"),
        ("wan22_ti2v_5b", "video_generation"),
        ("hunyuan3d_shape", "mesh"),
    ]
    by_id = {m["id"]: m for m in MODEL_CATALOG}
    cells = []
    for model_id, family in ids:
        model = by_id[model_id]
        cells.append(
            make_cell(
                phase="stress",
                model=model,
                entrypoint="serve_raw_http",
                arg_group="batching_metrics",
                env_group="stage_logging",
                request_family=family,
                stress_profile="stress",
            )
        )
    return cells


def build_matrix(repo: Path, max_pairwise_cells: int, seed: int) -> dict[str, Any]:
    catalog = {
        "cli_args": extract_cli_args(repo),
        "env_vars": extract_env_vars(repo),
        "protocol_fields": extract_protocol_fields(repo),
        "models": MODEL_CATALOG,
        "entrypoints": ENTRYPOINTS,
        "env_groups": ENV_GROUPS,
        "server_arg_groups": SERVER_ARG_GROUPS,
        "request_families": REQUEST_FAMILIES,
        "stress_profiles": STRESS_PROFILES,
    }
    phases = {
        "smoke": build_smoke_cells(),
        "pairwise": build_pairwise_cells(max_pairwise_cells, seed),
        "targeted": build_targeted_cells(),
        "stress": build_stress_cells(),
    }
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "repo": str(repo),
        "catalog": catalog,
        "phases": phases,
        "notes": [
            "Pairwise cells are compatibility-filtered and bounded; this is intentionally not a cartesian product.",
            "Stress cells should run on a GPU devbox or CI runner with matching hardware.",
        ],
    }


@dataclasses.dataclass
class HttpResult:
    method: str
    url: str
    status: int | None
    latency_ms: int
    ok: bool
    body_preview: str = ""
    error: str | None = None
    content_type: str | None = None
    body: bytes = b""


class HttpClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        form: dict[str, Any] | None = None,
        expected: set[int] | None = None,
    ) -> HttpResult:
        expected = expected or {200}
        url = self.base_url + path
        started = now_ms()
        try:
            if files is not None:
                return self._request_multipart(method, url, files, form or {}, expected, started)

            body = data
            req_headers = dict(headers or {})
            if json_body is not None:
                body = json.dumps(json_body).encode("utf-8")
                req_headers.setdefault("Content-Type", "application/json")
            req = urllib.request.Request(url, data=body, method=method, headers=req_headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
                status = int(resp.status)
                return HttpResult(
                    method=method,
                    url=url,
                    status=status,
                    latency_ms=now_ms() - started,
                    ok=status in expected,
                    body_preview=content[:500].decode("utf-8", errors="replace"),
                    content_type=resp.headers.get("content-type"),
                    body=content,
                )
        except urllib.error.HTTPError as e:
            content = e.read()
            status = int(e.code)
            return HttpResult(
                method=method,
                url=url,
                status=status,
                latency_ms=now_ms() - started,
                ok=status in expected,
                body_preview=content[:500].decode("utf-8", errors="replace"),
                content_type=e.headers.get("content-type") if e.headers else None,
                body=content,
            )
        except Exception as e:
            return HttpResult(
                method=method,
                url=url,
                status=None,
                latency_ms=now_ms() - started,
                ok=False,
                error=repr(e),
            )

    def _request_multipart(
        self,
        method: str,
        url: str,
        files: dict[str, tuple[str, bytes, str]],
        form: dict[str, Any],
        expected: set[int],
        started: int,
    ) -> HttpResult:
        boundary = f"----sglang-probe-{random.randint(0, 1_000_000_000)}"
        chunks: list[bytes] = []
        for key, value in form.items():
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            chunks.append(str(value).encode())
            chunks.append(b"\r\n")
        for key, (filename, content, content_type) in files.items():
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
            )
            chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
            chunks.append(content)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        body = b"".join(chunks)
        return self.request(
            method,
            urllib.parse.urlparse(url).path,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            expected=expected,
        )


class Recorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.runs_path = run_dir / "runs.jsonl"

    def event(self, event: str, **data: Any) -> None:
        payload = {"ts_ms": now_ms(), "event": event, **data}
        append_jsonl(self.runs_path, payload)


class ServerProcess:
    def __init__(self, cell: dict[str, Any], run_dir: Path, repo: Path, recorder: Recorder, python: str):
        self.cell = cell
        self.run_dir = run_dir
        self.repo = repo
        self.recorder = recorder
        self.python = python
        self.process: subprocess.Popen | None = None
        self.log_file = run_dir / "server.log"
        self._log_fh: Any | None = None
        self._assign_ports()

    def _assign_ports(self) -> None:
        ports = find_free_ports(3)
        self.port = ports[0]
        self.master_port = ports[1]
        self.scheduler_port = ports[2]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float) -> str:
        env_delta = merge_env(self.cell, self.run_dir)
        env = os.environ.copy()
        env.update(env_delta)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        last_error: RuntimeError | None = None
        for attempt in range(1, 6):
            if attempt > 1:
                self._assign_ports()
            cmd = self.command()
            log_fh = self.log_file.open("a", encoding="utf-8", buffering=1)
            log_fh.write(f"\n--- server start attempt {attempt} port {self.port} ---\n")
            self._log_fh = log_fh
            self.recorder.event(
                "server_start",
                cell_id=self.cell["id"],
                cmd=cmd,
                env=env_delta,
                attempt=attempt,
            )
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=env,
            )
            try:
                wait_for_health(self.base_url, timeout_s, self.process, self.log_file)
                return self.base_url
            except RuntimeError as exc:
                last_error = exc
                self.stop()
                msg = str(exc)
                if (
                    attempt < 5
                    and "--strict-ports" in msg
                    and "port" in msg
                    and "unavailable" in msg
                ):
                    self.recorder.event(
                        "server_start_retry",
                        cell_id=self.cell["id"],
                        attempt=attempt,
                        reason=msg[:2000],
                    )
                    continue
                raise
        assert last_error is not None
        raise last_error

    def command(self) -> list[str]:
        model = self.cell["model"]
        return [
            "sglang",
            "serve",
            "--model-path",
            model["model_path"],
            "--model-type",
            "diffusion",
            "--port",
            str(self.port),
            "--master-port",
            str(self.master_port),
            "--scheduler-port",
            str(self.scheduler_port),
            "--strict-ports",
            "--log-level",
            "debug",
        ] + self.server_args()

    def server_args(self) -> list[str]:
        model = self.cell["model"]
        arg_group = self.cell["arg_group"]
        args = list(model.get("server_args", []))
        args += flatten_args(SERVER_ARG_GROUPS[arg_group], self.cell, self.run_dir)
        if not arg_list_has(args, "--num-gpus"):
            args += ["--num-gpus", str(model.get("required_gpus", 1))]
        return args

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is not None:
                self.recorder.event("server_exit", returncode=self.process.returncode)
                return
            self.recorder.event("server_stop")
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=30)
            except Exception:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                except Exception:
                    pass
        finally:
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None
            self.process = None


def find_free_port() -> int:
    return find_free_ports(1)[0]


def find_free_ports(count: int) -> list[int]:
    sockets = []
    try:
        ports: list[int] = []
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
            ports.append(int(s.getsockname()[1]))
        return ports
    finally:
        for s in sockets:
            s.close()


def wait_for_health(base_url: str, timeout_s: float, process: subprocess.Popen, log_file: Path) -> None:
    client = HttpClient(base_url, timeout=5)
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {process.returncode}\n{tail(log_file)}"
            )
        res = client.request("GET", "/health", expected={200})
        if res.ok:
            return
        last_error = res.error or res.body_preview
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {last_error}\n{tail(log_file)}")


def tail(path: Path, lines: int = 120) -> str:
    text = read_text(path)
    return "\n".join(text.splitlines()[-lines:])


def decode_json(data: bytes) -> Any | None:
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def sample_image_bytes(run_dir: Path) -> bytes:
    path = run_dir / "sample_input.png"
    if not path.exists():
        path.write_bytes(base64.b64decode(SAMPLE_PNG_B64))
    return path.read_bytes()


def record_http(recorder: Recorder, cell: dict[str, Any], name: str, result: HttpResult, payload: Any | None = None) -> None:
    failure_file = None
    if not result.ok:
        failure_path = (
            recorder.run_dir
            / "failures"
            / f"{safe_slug(cell['id'])}_{safe_slug(name)}_{now_ms()}.json"
        )
        write_json(
            failure_path,
            {
                "cell_id": cell["id"],
                "name": name,
                "method": result.method,
                "url": result.url,
                "status": result.status,
                "latency_ms": result.latency_ms,
                "content_type": result.content_type,
                "error": result.error,
                "body_preview": result.body_preview,
                "payload": payload,
                "cell": cell,
            },
        )
        failure_file = str(failure_path)
    recorder.event(
        "request",
        cell_id=cell["id"],
        name=name,
        method=result.method,
        url=result.url,
        status=result.status,
        ok=result.ok,
        latency_ms=result.latency_ms,
        content_type=result.content_type,
        error=result.error,
        body_preview=result.body_preview,
        payload=payload,
        failure_file=failure_file,
    )


def run_control_requests(client: HttpClient, recorder: Recorder, cell: dict[str, Any]) -> list[bool]:
    ok = []
    for path, expected in [
        ("/health", {200}),
        ("/stats", {200}),
        ("/server_info", {200}),
        ("/v1/model_info", {200}),
        ("/v1/models", {200}),
    ]:
        res = client.request("GET", path, expected=expected)
        record_http(recorder, cell, f"control {path}", res)
        ok.append(res.ok)
    model_path = urllib.parse.quote(cell["model"]["model_path"], safe="")
    res = client.request("GET", f"/v1/models/{model_path}", expected={200, 404})
    record_http(recorder, cell, "control model retrieve", res)
    ok.append(res.ok)
    res = client.request("GET", "/v1/models/non_existent_model", expected={404})
    record_http(recorder, cell, "control missing model", res)
    ok.append(res.ok)
    return ok


def image_generation_payload(model: dict[str, Any], idx: int, *, response_format: str = "b64_json") -> dict[str, Any]:
    formats = [None, "png", "jpeg", "webp"]
    payload = {
        "model": model["model_path"],
        "prompt": model["prompt"],
        "size": model["size"],
        "n": 2 if idx > 0 and idx % 5 == 0 else 1,
        "response_format": response_format,
        "seed": idx,
        "num_inference_steps": model.get("steps", 4),
        "negative_prompt": "low quality, blurry",
    }
    fmt = formats[idx % len(formats)]
    if fmt:
        payload["output_format"] = fmt
    if idx % 7 == 0:
        width, height = model["size"].split("x")
        payload["width"] = int(width)
        payload["height"] = int(height)
    return payload


def run_image_generation(client: HttpClient, recorder: Recorder, cell: dict[str, Any], count: int) -> list[bool]:
    ok = []
    model = cell["model"]
    for idx in range(count):
        response_format = "url" if idx % 6 == 5 else "b64_json"
        payload = image_generation_payload(model, idx, response_format=response_format)
        expected = {200}
        if response_format == "url" and cell["arg_group"] == "disabled_io":
            expected = {400}
        res = client.request("POST", "/v1/images/generations", json_body=payload, expected=expected)
        record_http(recorder, cell, "image generation", res, payload)
        ok.append(res.ok)
        data = decode_json(res.body)
        if res.ok and response_format == "url" and isinstance(data, dict):
            url = (((data.get("data") or [{}])[0] or {}).get("url") or "")
            if url.startswith("/"):
                dl = client.request("GET", url, expected={200, 400, 404})
                record_http(recorder, cell, "image content", dl)
                ok.append(dl.ok)
    return ok


def run_image_edit(client: HttpClient, recorder: Recorder, cell: dict[str, Any], run_dir: Path, count: int) -> list[bool]:
    ok = []
    model = cell["model"]
    img = sample_image_bytes(run_dir)
    for idx in range(count):
        form = {
            "model": model["model_path"],
            "prompt": model["prompt"],
            "size": model["size"],
            "response_format": "b64_json",
            "n": 1,
            "seed": idx,
            "num_inference_steps": model.get("steps", 4),
        }
        files = {"image": ("sample_input.png", img, "image/png")}
        if "multi_image" in model.get("tags", []):
            files["image[]"] = ("sample_input_2.png", img, "image/png")
        res = client.request("POST", "/v1/images/edits", files=files, form=form, expected={200})
        record_http(recorder, cell, "image edit multipart", res, form)
        ok.append(res.ok)
    return ok


def video_payload(model: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "model": model["model_path"],
        "prompt": model["prompt"],
        "size": model["size"],
        "seconds": 1,
        "fps": model.get("fps", 8),
        "num_frames": model.get("num_frames", 9),
        "seed": idx,
        "num_inference_steps": model.get("steps", 4),
    }


def poll_async_job(
    client: HttpClient,
    recorder: Recorder,
    cell: dict[str, Any],
    prefix: str,
    job_id: str,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        res = client.request("GET", f"/v1/{prefix}/{job_id}", expected={200, 404})
        record_http(recorder, cell, f"{prefix} poll", res)
        data = decode_json(res.body)
        if isinstance(data, dict):
            status = data.get("status")
            if status == "completed":
                dl = client.request("GET", f"/v1/{prefix}/{job_id}/content", expected={200, 400, 404})
                record_http(recorder, cell, f"{prefix} content", dl)
                return dl.ok
            if status in {"failed", "cancelled", "deleted"}:
                return False
        time.sleep(DEFAULT_POLL_S)
    return False


def poll_expected_failed_job(
    client: HttpClient,
    recorder: Recorder,
    cell: dict[str, Any],
    prefix: str,
    job_id: str,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        res = client.request("GET", f"/v1/{prefix}/{job_id}", expected={200, 404})
        record_http(recorder, cell, f"{prefix} invalid poll", res)
        data = decode_json(res.body)
        if isinstance(data, dict):
            status = data.get("status")
            if status == "failed":
                return True
            if status == "completed":
                return False
        time.sleep(DEFAULT_POLL_S)
    return False


def run_video_generation(client: HttpClient, recorder: Recorder, cell: dict[str, Any], run_dir: Path, count: int, timeout_s: float) -> list[bool]:
    ok = []
    model = cell["model"]
    image_needed = model["task_type"] in {"I2V", "TI2V"} or "image_input" in model.get("tags", [])
    img = sample_image_bytes(run_dir)
    for idx in range(count):
        if image_needed:
            form = video_payload(model, idx)
            files = {"input_reference": ("sample_input.png", img, "image/png")}
            res = client.request("POST", "/v1/videos", files=files, form=form, expected={200})
        else:
            payload = video_payload(model, idx)
            res = client.request("POST", "/v1/videos", json_body=payload, expected={200})
        record_http(recorder, cell, "video create", res)
        ok.append(res.ok)
        data = decode_json(res.body)
        job_id = data.get("id") if isinstance(data, dict) else None
        if job_id:
            early = client.request("GET", f"/v1/videos/{job_id}/content", expected={200, 404})
            record_http(recorder, cell, "video early content", early)
            ok.append(early.ok)
            ok.append(poll_async_job(client, recorder, cell, "videos", job_id, timeout_s))
        list_res = client.request("GET", "/v1/videos?limit=10&order=desc", expected={200})
        record_http(recorder, cell, "video list", list_res)
        ok.append(list_res.ok)
    return ok


def run_mesh(client: HttpClient, recorder: Recorder, cell: dict[str, Any], run_dir: Path, count: int, timeout_s: float) -> list[bool]:
    ok = []
    model = cell["model"]
    img = sample_image_bytes(run_dir)
    for idx in range(count):
        form = {
            "model": model["model_path"],
            "prompt": model["prompt"],
            "seed": idx,
            "num_inference_steps": model.get("steps", 20),
            "output_format": "glb",
        }
        files = {"image": ("sample_input.png", img, "image/png")}
        res = client.request("POST", "/v1/meshes", files=files, form=form, expected={200})
        record_http(recorder, cell, "mesh create", res, form)
        ok.append(res.ok)
        data = decode_json(res.body)
        job_id = data.get("id") if isinstance(data, dict) else None
        if job_id:
            ok.append(poll_async_job(client, recorder, cell, "meshes", job_id, timeout_s))
            delete_res = client.request("DELETE", f"/v1/meshes/{job_id}", expected={200, 404})
            record_http(recorder, cell, "mesh delete", delete_res)
            ok.append(delete_res.ok)
    return ok


def run_invalid_requests(client: HttpClient, recorder: Recorder, cell: dict[str, Any]) -> list[bool]:
    bad_cases = [
        ("bad json image", "POST", "/v1/images/generations", b"{bad-json", {"Content-Type": "application/json"}, {400, 422}, None),
        ("missing prompt image", "POST", "/v1/images/generations", json.dumps({"size": "512x512"}).encode(), {"Content-Type": "application/json"}, {400, 422}, None),
        ("bad size image", "POST", "/v1/images/generations", json.dumps({"prompt": "x", "size": "-1xabc"}).encode(), {"Content-Type": "application/json"}, {400, 422}, None),
        ("missing input edit", "POST", "/v1/images/edits", None, {}, {400, 422}, None),
        ("missing video prompt", "POST", "/v1/videos", json.dumps({"size": "832x480"}).encode(), {"Content-Type": "application/json"}, {400, 422}, None),
        ("bad video frames", "POST", "/v1/videos", json.dumps({"prompt": "x", "num_frames": -1}).encode(), {"Content-Type": "application/json"}, {400, 422, 200}, "videos"),
        ("missing mesh image", "POST", "/v1/meshes", json.dumps({"prompt": "mesh"}).encode(), {"Content-Type": "application/json"}, {400, 422}, None),
    ]
    ok = []
    for name, method, path, body, headers, expected, async_prefix in bad_cases:
        res = client.request(method, path, data=body, headers=headers, expected=expected)
        record_http(recorder, cell, name, res)
        case_ok = res.ok
        data = decode_json(res.body)
        if res.status == 200 and async_prefix and isinstance(data, dict) and data.get("id"):
            case_ok = poll_expected_failed_job(
                client, recorder, cell, async_prefix, data["id"], timeout_s=120
            )
        ok.append(case_ok)
        health = client.request("GET", "/health", expected={200})
        record_http(recorder, cell, "post-invalid health", health)
        ok.append(health.ok)
    return ok


def run_realtime(base_url: str, recorder: Recorder, cell: dict[str, Any], run_dir: Path, timeout_s: float) -> list[bool]:
    try:
        import asyncio
        import msgspec
        import websockets
    except Exception as e:
        recorder.event("dependency_missing", cell_id=cell["id"], dependency="websockets/msgspec", error=repr(e))
        return [False]

    async def _run() -> bool:
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/") + "/v1/realtime_video/generate"
        model = cell["model"]
        payload = {
            "type": "init",
            "model": model["model_path"],
            "prompt": model["prompt"],
            "size": model["size"],
            "max_chunks": 1,
            "num_inference_steps": model.get("steps", 4),
            "fps": model.get("fps", 8),
            "num_frames": model.get("num_frames", 9),
            "seed": 0,
        }
        if "image_input" in model.get("tags", []) or model.get("task_type") in {"I2V", "TI2V"}:
            payload["first_frame"] = sample_image_bytes(run_dir)
        started = now_ms()
        try:
            async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as ws:
                await ws.send(msgspec.msgpack.encode({"bad": "init"}))
                first = await asyncio.wait_for(ws.recv(), timeout=30)
                recorder.event("realtime_invalid_init", cell_id=cell["id"], latency_ms=now_ms() - started, bytes=len(first) if isinstance(first, bytes) else 0)
                await ws.send(msgspec.msgpack.encode(payload))
                got_frame = False
                deadline = time.time() + timeout_s
                while time.time() < deadline:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    if isinstance(msg, bytes):
                        decoded = None
                        try:
                            decoded = msgspec.msgpack.decode(msg)
                        except Exception:
                            pass
                        recorder.event("realtime_message", cell_id=cell["id"], payload_preview=str(decoded)[:300], bytes=len(msg))
                        if isinstance(decoded, dict) and decoded.get("type") in {"chunk_stats", "complete"}:
                            got_frame = True
                        elif decoded is None:
                            got_frame = True
                    if got_frame:
                        return True
        except Exception as e:
            recorder.event("realtime_error", cell_id=cell["id"], error=repr(e), traceback=traceback.format_exc())
            return False
        return False

    return [asyncio.run(_run())]


def repeat_count(cell: dict[str, Any]) -> int:
    profile = STRESS_PROFILES[cell["stress_profile"]]
    family = cell["request_family"]
    if cell["phase"] == "smoke":
        return 1
    if family in {"video_generation", "mesh", "realtime"}:
        return int(profile.get("video_jobs", 1))
    if family in {"control", "invalid"}:
        return 1
    return max(1, int(profile["requests"]) // 20)


def run_http_cell(base_url: str, recorder: Recorder, cell: dict[str, Any], run_dir: Path, timeout_s: float) -> bool:
    client = HttpClient(base_url, timeout=min(timeout_s, 600))
    family = cell["request_family"]
    count = repeat_count(cell)
    if family == "control":
        results = run_control_requests(client, recorder, cell)
    elif family == "image_generation":
        results = run_image_generation(client, recorder, cell, count)
    elif family == "image_edit":
        results = run_image_edit(client, recorder, cell, run_dir, count)
    elif family == "video_generation":
        results = run_video_generation(client, recorder, cell, run_dir, count, timeout_s)
    elif family == "mesh":
        results = run_mesh(client, recorder, cell, run_dir, count, timeout_s)
    elif family == "realtime":
        results = run_realtime(base_url, recorder, cell, run_dir, timeout_s)
    elif family == "invalid":
        results = run_invalid_requests(client, recorder, cell)
    else:
        raise ValueError(f"unknown request family: {family}")
    final_health = client.request("GET", "/health", expected={200})
    record_http(recorder, cell, "final health", final_health)
    results.append(final_health.ok)
    return all(results)


def run_openai_sdk_cell(base_url: str, recorder: Recorder, cell: dict[str, Any], run_dir: Path, timeout_s: float) -> bool:
    try:
        from openai import OpenAI
    except Exception as e:
        recorder.event("dependency_missing", cell_id=cell["id"], dependency="openai", error=repr(e))
        return False
    client = OpenAI(api_key="sglang-anything", base_url=base_url.rstrip("/") + "/v1", timeout=timeout_s, max_retries=0)
    model = cell["model"]
    family = cell["request_family"]
    started = now_ms()
    try:
        if family == "image_generation":
            response = client.images.generate(
                model=model["model_path"],
                prompt=model["prompt"],
                size=model["size"],
                n=1,
                response_format="b64_json",
                extra_body={"num_inference_steps": model.get("steps", 4), "seed": 0},
            )
            ok = bool(response.data and response.data[0].b64_json)
        elif family == "image_edit":
            image_path = run_dir / "sample_input.png"
            image_path.write_bytes(sample_image_bytes(run_dir))
            with image_path.open("rb") as f:
                response = client.images.edit(
                    model=model["model_path"],
                    image=f,
                    prompt=model["prompt"],
                    size=model["size"],
                    n=1,
                    response_format="b64_json",
                    extra_body={"num_inference_steps": model.get("steps", 4), "seed": 0},
                )
            ok = bool(response.data and response.data[0].b64_json)
        elif family == "video_generation":
            kwargs = video_payload(model, 0)
            if model["task_type"] in {"I2V", "TI2V"} or "image_input" in model.get("tags", []):
                image_path = run_dir / "sample_input.png"
                image_path.write_bytes(sample_image_bytes(run_dir))
                with image_path.open("rb") as f:
                    job = client.post(
                        "/videos",
                        cast_to=object,
                        body=kwargs,
                        files={"input_reference": ("sample_input.png", f, "image/png")},
                        options={"headers": {"Content-Type": "multipart/form-data"}},
                    )
            else:
                job = client.post("/videos", cast_to=object, body=kwargs)
            video_id = job["id"] if isinstance(job, dict) else job.id
            ok = poll_openai_video(client, recorder, cell, video_id, timeout_s)
        else:
            return run_http_cell(base_url, recorder, cell, run_dir, timeout_s)
        recorder.event("openai_sdk_request", cell_id=cell["id"], ok=ok, latency_ms=now_ms() - started, family=family)
        return ok
    except Exception as e:
        recorder.event("openai_sdk_error", cell_id=cell["id"], error=repr(e), traceback=traceback.format_exc())
        return False


def poll_openai_video(client: Any, recorder: Recorder, cell: dict[str, Any], video_id: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        page = client.videos.list()  # type: ignore[attr-defined]
        item = next((v for v in page.data if v.id == video_id), None)
        status = getattr(item, "status", None) if item else None
        recorder.event("openai_video_poll", cell_id=cell["id"], video_id=video_id, status=status)
        if status == "completed":
            content = client.videos.download_content(video_id=video_id).read()  # type: ignore[attr-defined]
            recorder.event("openai_video_content", cell_id=cell["id"], video_id=video_id, bytes=len(content), ok=bool(content))
            return bool(content)
        if status == "failed":
            return False
        time.sleep(DEFAULT_POLL_S)
    return False


def sampling_kwargs(model: dict[str, Any], run_dir: Path, family: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": model["prompt"],
        "num_inference_steps": model.get("steps", 4),
        "seed": 0,
        "save_output": True,
        "output_path": str(run_dir / "outputs"),
    }
    if "x" in model["size"]:
        width, height = model["size"].split("x")
        kwargs["width"] = int(width)
        kwargs["height"] = int(height)
    if model["modality"] == "video":
        kwargs["num_frames"] = model.get("num_frames", 9)
        kwargs["fps"] = model.get("fps", 8)
    if model["modality"] == "3d" or family in {"image_edit", "video_generation"} and "image_input" in model.get("tags", []):
        image_path = run_dir / "sample_input.png"
        image_path.write_bytes(sample_image_bytes(run_dir))
        kwargs["image_path"] = str(image_path)
    if model["modality"] == "3d":
        kwargs["data_type"] = "mesh"
    return kwargs


def server_kwargs_from_cell(cell: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    model = cell["model"]
    args = list(model.get("server_args", [])) + flatten_args(
        SERVER_ARG_GROUPS[cell["arg_group"]], cell, run_dir
    )
    kwargs: dict[str, Any] = {
        "model_path": model["model_path"],
        "num_gpus": get_num_gpus_from_args(args, model.get("required_gpus", 1)),
    }
    key_map = {
        "--backend": "backend",
        "--warmup-mode": "warmup_mode",
        "--output-path": "output_path",
        "--input-save-path": "input_save_path",
        "--pipeline-class-name": "pipeline_class_name",
        "--ltx2-two-stage-device-mode": "ltx2_two_stage_device_mode",
        "--lora-path": "lora_path",
        "--lora-merge-mode": "lora_merge_mode",
        "--dit-offload-prefetch-size": "dit_offload_prefetch_size",
    }
    bool_map = {
        "--text-encoder-cpu-offload": "text_encoder_cpu_offload",
        "--image-encoder-cpu-offload": "image_encoder_cpu_offload",
        "--vae-cpu-offload": "vae_cpu_offload",
        "--dit-layerwise-offload": "dit_layerwise_offload",
        "--use-fsdp-inference": "use_fsdp_inference",
        "--enable-cfg-parallel": "enable_cfg_parallel",
    }
    idx = 0
    while idx < len(args):
        item = args[idx]
        if item in key_map and idx + 1 < len(args):
            kwargs[key_map[item]] = args[idx + 1]
            idx += 2
            continue
        if item in bool_map:
            value = True
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                value = args[idx + 1].lower() not in {"0", "false", "no", "off"}
                idx += 2
            else:
                idx += 1
            kwargs[bool_map[item]] = value
            continue
        if item == "--layerwise-offload-components":
            values = []
            idx += 1
            while idx < len(args) and not args[idx].startswith("--"):
                values.append(args[idx])
                idx += 1
            kwargs["layerwise_offload_components"] = values
            continue
        idx += 1
    return kwargs


def run_cli_generate(cell: dict[str, Any], run_dir: Path, repo: Path, recorder: Recorder, python: str, timeout_s: float, dry_run: bool) -> bool:
    model = cell["model"]
    kwargs = sampling_kwargs(model, run_dir, cell["request_family"])
    cmd = ["sglang", "generate", "--model-path", model["model_path"]]
    cmd += flatten_args(SERVER_ARG_GROUPS[cell["arg_group"]], cell, run_dir)
    cmd += ["--prompt", kwargs["prompt"], "--num-inference-steps", str(kwargs["num_inference_steps"])]
    if "width" in kwargs:
        cmd += ["--width", str(kwargs["width"]), "--height", str(kwargs["height"])]
    if "num_frames" in kwargs:
        cmd += ["--num-frames", str(kwargs["num_frames"]), "--fps", str(kwargs["fps"])]
    if "image_path" in kwargs:
        image_arg = kwargs["image_path"]
        cmd += ["--image-path", image_arg]
    if model["modality"] == "3d":
        cmd += ["--data-type", "mesh"]
    cmd += ["--save-output", "--output-path", str(run_dir / "outputs")]
    return run_subprocess_cell(cell, cmd, run_dir, repo, recorder, timeout_s, dry_run, merge_env(cell, run_dir))


def write_python_snippet(path: Path, cell: dict[str, Any], run_dir: Path, mode: str) -> None:
    server_kwargs = server_kwargs_from_cell(cell, run_dir)
    sample_kwargs = sampling_kwargs(cell["model"], run_dir, cell["request_family"])
    lora_path = cell["model"].get("dynamic_lora_path") or cell["model"].get("lora_path")
    second_lora_path = cell["model"].get("second_lora_path")
    code = f"""
import json
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.runtime.server_args import ServerArgs

def main():
    server_kwargs = {server_kwargs!r}
    sampling_kwargs = {sample_kwargs!r}
    mode = {mode!r}

    if mode == "from_pretrained":
        generator = DiffGenerator.from_pretrained(local_mode=True, **server_kwargs)
    else:
        server_args = ServerArgs.from_kwargs(**server_kwargs)
        generator = DiffGenerator.from_server_args(server_args, local_mode=True)

    if mode == "lora":
        print("initial_loras", json.dumps(generator.list_loras(), default=str))
        lora_path = {lora_path!r}
        second_lora_path = {second_lora_path!r}
        if lora_path:
            generator.set_lora("probe", lora_path=lora_path, merge_mode=server_kwargs.get("lora_merge_mode"))
            print("after_set_lora", json.dumps(generator.list_loras(), default=str))
            generator.merge_lora_weights()
            generator.unmerge_lora_weights()
        if second_lora_path:
            generator.set_lora(["probe", "probe2"], lora_path=[lora_path, second_lora_path], strength=[0.5, 0.5])
            print("after_multi_lora", json.dumps(generator.list_loras(), default=str))

    result = generator.generate(sampling_params_kwargs=sampling_kwargs)
    print("result_type", type(result).__name__)
    if isinstance(result, list):
        print("result_count", len(result))
    else:
        print("result_count", 1 if result is not None else 0)


if __name__ == "__main__":
    main()
"""
    path.write_text(code.strip() + "\n", encoding="utf-8")


def run_python_api_cell(cell: dict[str, Any], run_dir: Path, repo: Path, recorder: Recorder, python: str, timeout_s: float, dry_run: bool, mode: str) -> bool:
    snippet = run_dir / f"{mode}.py"
    write_python_snippet(snippet, cell, run_dir, mode)
    cmd = [python, str(snippet)]
    env = merge_env(cell, run_dir)
    return run_subprocess_cell(cell, cmd, run_dir, repo, recorder, timeout_s, dry_run, env)


def run_subprocess_cell(
    cell: dict[str, Any],
    cmd: list[str],
    run_dir: Path,
    repo: Path,
    recorder: Recorder,
    timeout_s: float,
    dry_run: bool,
    env_update: dict[str, str],
) -> bool:
    log_path = run_dir / "command.log"
    recorder.event("command_plan", cell_id=cell["id"], cmd=cmd, env=env_update)
    if dry_run:
        return True
    env = os.environ.copy()
    env.update(env_update)
    started = now_ms()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                env=env,
            )
            ok = proc.returncode == 0
            recorder.event("command_result", cell_id=cell["id"], ok=ok, returncode=proc.returncode, latency_ms=now_ms() - started, log=str(log_path))
            return ok
        except Exception as e:
            recorder.event("command_error", cell_id=cell["id"], ok=False, error=repr(e), traceback=traceback.format_exc(), log=str(log_path))
            return False


def run_one_cell(
    cell: dict[str, Any],
    *,
    repo: Path,
    out_root: Path,
    base_url: str | None,
    timeout_s: float,
    dry_run: bool,
    python: str,
) -> bool:
    run_dir = out_root / "runs" / cell["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(out_root)
    write_json(run_dir / "cell.json", cell)
    recorder.event("cell_start", cell_id=cell["id"], dry_run=dry_run)
    ok = False
    try:
        if cell["entrypoint"] == "cli_generate":
            ok = run_cli_generate(cell, run_dir, repo, recorder, python, timeout_s, dry_run)
        elif cell["entrypoint"] == "python_diffgenerator":
            ok = run_python_api_cell(cell, run_dir, repo, recorder, python, timeout_s, dry_run, "from_pretrained")
        elif cell["entrypoint"] == "python_server_args":
            ok = run_python_api_cell(cell, run_dir, repo, recorder, python, timeout_s, dry_run, "from_server_args")
        elif cell["entrypoint"] == "python_lora":
            ok = run_python_api_cell(cell, run_dir, repo, recorder, python, timeout_s, dry_run, "lora")
        else:
            server: ServerProcess | None = None
            if base_url:
                url = base_url.rstrip("/")
            else:
                server = ServerProcess(cell, run_dir, repo, recorder, python)
                if dry_run:
                    recorder.event("server_plan", cell_id=cell["id"], cmd=server.command(), env=merge_env(cell, run_dir))
                    recorder.event("cell_end", cell_id=cell["id"], ok=True, dry_run=True)
                    return True
                url = server.start(timeout_s)
            try:
                if cell["entrypoint"] == "serve_openai_sdk":
                    ok = run_openai_sdk_cell(url, recorder, cell, run_dir, timeout_s)
                else:
                    ok = run_http_cell(url, recorder, cell, run_dir, timeout_s)
            finally:
                if server is not None:
                    server.stop()
    except Exception as e:
        recorder.event("cell_error", cell_id=cell["id"], error=repr(e), traceback=traceback.format_exc())
        ok = False
    recorder.event("cell_end", cell_id=cell["id"], ok=ok, dry_run=dry_run)
    return ok


def collect_cells(matrix: dict[str, Any], phases: list[str], cell_ids: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for phase in phases:
        cells.extend(matrix["phases"].get(phase, []))
    if cell_ids:
        cells = [c for c in cells if c["id"] in cell_ids]
    if limit is not None:
        cells = cells[:limit]
    return cells


def load_or_build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = Path(args.matrix) if args.matrix else Path(args.out_dir) / "matrix.json"
    if matrix_path.exists() and not args.rebuild_matrix:
        return json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix = build_matrix(Path(args.repo), args.max_pairwise_cells, args.seed)
    write_json(matrix_path, matrix)
    return matrix


def summarize_runs(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "runs.jsonl"
    if not path.exists():
        return {"error": f"{path} does not exist"}
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cells: dict[str, dict[str, Any]] = {}
    requests = []
    for event in events:
        cell_id = event.get("cell_id")
        if cell_id:
            cells.setdefault(cell_id, {"ok": None, "requests": 0, "failures": 0})
        if event.get("event") == "request":
            requests.append(event)
            if cell_id:
                cells[cell_id]["requests"] += 1
                if not event.get("ok"):
                    cells[cell_id]["failures"] += 1
        if event.get("event") == "cell_end" and cell_id:
            cells[cell_id]["ok"] = event.get("ok")
    latencies = [r["latency_ms"] for r in requests if isinstance(r.get("latency_ms"), int)]
    latencies.sort()
    def pct(p: float) -> int | None:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round((len(latencies) - 1) * p)))
        return latencies[idx]
    summary = {
        "cells_total": len(cells),
        "cells_failed": sum(1 for c in cells.values() if c["ok"] is False),
        "requests_total": len(requests),
        "requests_failed": sum(1 for r in requests if not r.get("ok")),
        "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)},
        "failed_cells": [cid for cid, c in cells.items() if c["ok"] is False],
        "failed_requests": [
            {
                "cell_id": r.get("cell_id"),
                "name": r.get("name"),
                "status": r.get("status"),
                "url": r.get("url"),
                "payload": r.get("payload"),
                "body_preview": r.get("body_preview"),
                "error": r.get("error"),
            }
            for r in requests
            if not r.get("ok")
        ][:100],
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGLang-Diffusion server/API stability probe harness")
    parser.add_argument("command", choices=["catalog", "list", "run", "summarize"])
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT / "latest"))
    parser.add_argument("--matrix", default="")
    parser.add_argument("--rebuild-matrix", action="store_true")
    parser.add_argument(
        "--profile",
        default="",
        help="Named profile from diffusion_bench.sglang_probe.profiles: daily_smoke, weekly_pairwise, release_full, stress",
    )
    parser.add_argument("--max-pairwise-cells", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--phase", action="append", choices=["smoke", "pairwise", "targeted", "stress"], default=[])
    parser.add_argument("--cell-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default="", help="Use an already-running server instead of starting sglang serve")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true", help="Plan commands/requests without launching sglang or sending traffic")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_profile(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_or_build_matrix(args)
    matrix_path = Path(args.matrix) if args.matrix else out_dir / "matrix.json"
    if args.command == "catalog":
        write_json(matrix_path, matrix)
        print(matrix_path)
        return 0
    phases = args.phase or (["smoke", "pairwise", "targeted", "stress"] if args.cell_id else ["smoke"])
    cells = collect_cells(matrix, phases, set(args.cell_id) if args.cell_id else None, args.limit)
    if args.command == "list":
        for cell in cells:
            print(
                json.dumps(
                    {
                        "id": cell["id"],
                        "phase": cell["phase"],
                        "model": cell["model"]["id"],
                        "entrypoint": cell["entrypoint"],
                        "request_family": cell["request_family"],
                        "arg_group": cell["arg_group"],
                        "env_group": cell["env_group"],
                        "stress_profile": cell["stress_profile"],
                    },
                    sort_keys=True,
                )
            )
        return 0
    if args.command == "summarize":
        print(json.dumps(summarize_runs(out_dir), indent=2, sort_keys=True))
        return 0
    failures = 0
    for cell in cells:
        ok = run_one_cell(
            cell,
            repo=Path(args.repo),
            out_root=out_dir,
            base_url=args.base_url or None,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            python=args.python,
        )
        failures += 0 if ok else 1
    summary = summarize_runs(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
