"""FastVideo as a benchmarked framework: its serve command and its video request.

`fastvideo serve` reads a nested serve config (`generator` / `server` /
`default_request`) and applies `--dotted.key value` overrides on top of it. The
harness hands it a stub config and passes every setting as an override, so the
recorded server command is the whole configuration. FastVideo's offload switches
all default to true, so a resident profile has to say `false` for each of them.

Video goes to POST /v1/videos/sync, which blocks and answers with the MP4: the
client clock stops at the last MP4 byte, with no poll quantum. The request model
is `extra="forbid"` (an unknown field is a 400), and MiniMax-H3 keeps its
video/audio shifts in the checkpoint's scheduler configs, so no shift is sent.
"""

import struct

SERVE_STUB = {"generator": {}}
SYNC_VIDEO_PATH = "/v1/videos/sync"
HEALTH_PATH = "/health"
VIDEO_TASKS = ("text-to-video", "image-to-video", "text-image-to-video")

# Overrides the harness writes itself, from the case and the launch.
HARNESS_KEYS = (
    "generator.model_path",
    "generator.engine.num_gpus",
    "server.host",
    "server.port",
    "server.output_dir",
)
OFFLOAD_PARTS = ("dit", "dit_layerwise", "text_encoder", "image_encoder", "vae")
# Load a component per request (re-read from disk) instead of keeping it resident.
DEFERRED_LOAD_KEYS = (
    "generator.engine.offload.lazy_module_load",
    "generator.pipeline.experimental.h3_sequential_load",
)
# Regional per-block compile of the DiT, or the whole-DiT torch.compile.
COMPILE_KEYS = (
    "generator.pipeline.experimental.inference_torch_compile",
    "generator.engine.compile.enabled",
)
# Appended in the harness's compile-off mode; overrides apply last-wins.
COMPILE_OFF_ARGS = [
    "--generator.pipeline.experimental.inference_torch_compile",
    "false",
    "--generator.engine.compile.enabled",
    "false",
    "--generator.engine.compile.vae_enabled",
    "false",
]

# The track containers on the path moov/trak/mdia/hdlr.
_TRACK_CONTAINERS = {b"moov", b"trak", b"mdia"}


def serve_overrides(args: list[str]) -> dict[str, str]:
    """`--dotted.key value` / `--dotted.key=value` pairs as `fastvideo serve` reads them:
    `-` in a key becomes `_`, and a later key wins. Values stay strings."""
    parsed: dict[str, str] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--") or len(token) == 2:
            raise ValueError(f"expected --dotted.key, got {token!r}")
        key, sep, value = token[2:].partition("=")
        if not sep:
            index += 1
            if index >= len(args):
                raise ValueError(f"missing value for {token}")
            value = args[index]
        parsed[key.replace("-", "_")] = value
        index += 1
    return parsed


def video_payload(
    prompt: str,
    *,
    width: int,
    height: int,
    num_frames: int | None,
    fps: int | None,
    num_inference_steps: int | None,
    extra: dict,
) -> dict:
    """The JSON body for /v1/videos/sync: shape and steps, then `extra` (seed,
    guidance, task, references), which wins."""
    payload = {"prompt": prompt, "size": f"{width}x{height}"}
    for key, value in (
        ("num_frames", num_frames),
        ("fps", fps),
        ("num_inference_steps", num_inference_steps),
    ):
        if value is not None:
            payload[key] = value
    payload.update(extra)
    return payload


def mp4_handler_types(data: bytes) -> list[str]:
    """The handler type of every track (`vide`, `soun`, ...), from moov/trak/mdia/hdlr."""
    found: list[str] = []

    def walk(start: int, end: int) -> None:
        pos = start
        while pos + 8 <= end:
            size, kind = struct.unpack_from(">I4s", data, pos)
            header = 8
            if size == 1:
                if pos + 16 > end:
                    raise ValueError(f"truncated box {kind!r} at byte {pos}")
                size = struct.unpack_from(">Q", data, pos + 8)[0]
                header = 16
            elif size == 0:
                size = end - pos
            if size < header or pos + size > end:
                raise ValueError(f"box {kind!r} at byte {pos} claims {size} bytes")
            if kind in _TRACK_CONTAINERS:
                walk(pos + header, pos + size)
            elif kind == b"hdlr" and size >= header + 12:
                # FullBox: version+flags, pre_defined, then handler_type.
                found.append(data[pos + header + 8 : pos + header + 12].decode("latin-1"))
            pos += size

    walk(0, len(data))
    return found


def check_mp4(data: bytes, *, audio: bool) -> None:
    """Raise unless `data` is an MP4 with a video track, and an audio track when `audio`.

    FastVideo saves the video without its soundtrack when muxing fails, which is
    a cheaper output than the case asks for, not a latency of it.
    """
    try:
        tracks = mp4_handler_types(data)
    except (ValueError, struct.error) as exc:
        raise RuntimeError(f"FastVideo answered {len(data)} bytes that do not parse as MP4: {exc}") from exc
    missing = [kind for kind, wanted in (("vide", True), ("soun", audio)) if wanted and kind not in tracks]
    if missing:
        raise RuntimeError(
            f"FastVideo's MP4 ({len(data)} bytes) has tracks {tracks or 'none'}; missing {missing}"
        )
