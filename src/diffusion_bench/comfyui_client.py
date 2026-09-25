"""ComfyUI as a benchmarked framework: workflow rendering and the request client.

ComfyUI has no generation endpoint. A request is a workflow graph in its API
format POSTed to /prompt; completion arrives on the client's websocket, and the
outputs are listed in /history. Two properties of that design decide how it has
to be measured:

* ComfyUI caches every node's output keyed on the node's inputs and its
  ancestors'. The harness sends the same prompt and seed on every request, so a
  second identical graph is answered from cache -- the sampler never runs. A
  benchmark request has to do the work a new generation does, so templates add
  an undeclared ``__bench_nonce`` input to the node(s) that consume the request
  (the text encoders). Undeclared inputs are part of the cache key but are not
  passed to the node, so that node and everything downstream re-executes while
  the model loaders stay cached, which is what a resident server does. Every
  request is then checked: a sampler served from cache is a failure, not a
  latency.
* Completion is the ``executing`` message with ``node: null`` for the prompt.
  ``execution_success`` is sent earlier, before /history is written, so reading
  outputs right after it can race.
"""

import copy
import json
import re
import time
import uuid

# A placeholder that is the whole string keeps the value's type (an int stays an
# int); one embedded in text is interpolated.
_FULL_TOKEN = re.compile(r"^\{\{(\w+)\}\}$")
_PART_TOKEN = re.compile(r"\{\{(\w+)\}\}")

NONCE_INPUT = "__bench_nonce"
SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
REF_IMAGE_NAME = "bench_reference.png"
HEALTH_PATH = "/system_stats"
_READ_CHUNK = 1 << 18


def render_workflow(template: dict, params: dict) -> dict:
    """Substitute ``{{name}}`` placeholders; an unknown name is an error."""

    def lookup(name: str):
        if name not in params or params[name] is None:
            raise KeyError(f"workflow placeholder {{{{{name}}}}} has no value")
        return params[name]

    def sub(node):
        if isinstance(node, dict):
            return {key: sub(value) for key, value in node.items()}
        if isinstance(node, list):
            return [sub(value) for value in node]
        if isinstance(node, str):
            full = _FULL_TOKEN.match(node)
            if full:
                return lookup(full.group(1))
            return _PART_TOKEN.sub(lambda m: str(lookup(m.group(1))), node)
        return node

    return sub(copy.deepcopy(template))


def workflow_params(case: dict, spec: dict) -> dict:
    """Placeholder values for one request of `case` under ComfyUI spec `spec`."""
    steps = int(case["num_inference_steps"])
    params = {
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative_prompt", ""),
        "seed": int(case.get("seed", 0)),
        "width": case.get("width"),
        "height": case.get("height"),
        "steps": steps,
        # Two-expert models (Wan2.2) hand over between samplers mid-schedule.
        "half_steps": max(1, steps // 2),
        # Two-stage pipelines (LTX-2) run their first stage at half resolution.
        "half_width": case["width"] // 2 if case.get("width") else None,
        "half_height": case["height"] // 2 if case.get("height") else None,
        # GPUs the MultiGPU CFG Split node spreads guidance branches over; 1 is a no-op.
        "cfg_gpus": 1,
        "num_frames": case.get("num_frames"),
        "fps": case.get("fps"),
        "guidance": case.get("guidance_scale"),
        "guidance_2": case.get("guidance_scale_2"),
        "true_cfg": case.get("true_cfg_scale"),
        "ref_image": REF_IMAGE_NAME,
        "filename_prefix": f"bench_{case['id']}",
    }
    params.update(spec.get("params") or {})
    params["nonce"] = uuid.uuid4().hex
    return params


def _output_files(history_entry: dict) -> list[dict]:
    files = []
    for node_output in (history_entry.get("outputs") or {}).values():
        for value in node_output.values():
            if isinstance(value, list):
                files += [f for f in value if isinstance(f, dict) and f.get("filename")]
    return files


def _fetch(base_url: str, entry: dict, first_chunk_only: bool) -> int:
    import requests

    params = {
        "filename": entry["filename"],
        "subfolder": entry.get("subfolder", ""),
        "type": entry.get("type", "output"),
    }
    with requests.get(f"{base_url}/view", params=params, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        size = 0
        for chunk in resp.iter_content(_READ_CHUNK):
            size += len(chunk)
            if first_chunk_only and size:
                break
    return size


def run_prompt(
    base_url: str,
    graph: dict,
    *,
    timeout_s: float,
    fetch_outputs: bool,
) -> tuple[float, dict]:
    """Queue `graph` and wait for it; return (seconds, info).

    The clock starts at the POST. With `fetch_outputs` it stops once every
    output file has been read back (images: the client has the picture, as it
    does from the other frameworks' responses); otherwise it stops at
    completion and the outputs are only checked to exist (video: the other
    frameworks' video timings also end at job completion).
    """
    # Imported here so the config builder can render workflows with a bare python3.
    import requests
    import websocket  # websocket-client

    client_id = uuid.uuid4().hex
    ws_url = "ws://" + base_url.split("://", 1)[-1] + f"/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=timeout_s)
    cached: list[str] = []
    try:
        start = time.time()
        deadline = start + timeout_s
        resp = requests.post(
            f"{base_url}/prompt",
            json={"prompt": graph, "client_id": client_id},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow (HTTP {resp.status_code}): {resp.text[:2000]}")
        prompt_id = resp.json()["prompt_id"]
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {timeout_s:.0f}s")
            ws.settimeout(remaining)
            message = ws.recv()
            if not isinstance(message, str):
                continue  # binary frames are previews
            event = json.loads(message)
            kind, data = event.get("type"), event.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue
            if kind == "execution_cached":
                cached = [str(n) for n in data.get("nodes") or []]
            elif kind == "execution_error":
                raise RuntimeError(
                    f"ComfyUI execution failed in node {data.get('node_id')} "
                    f"({data.get('node_type')}): {data.get('exception_type')}: "
                    f"{str(data.get('exception_message', ''))[:1000]}"
                )
            elif kind == "execution_interrupted":
                raise RuntimeError(f"ComfyUI execution interrupted: {data}")
            elif kind == "executing" and data.get("node") is None:
                break
        finished = time.time()
    finally:
        ws.close()

    stale = [n for n in cached if (graph.get(n) or {}).get("class_type") in SAMPLER_CLASSES]
    if stale:
        raise RuntimeError(
            f"ComfyUI answered sampler node(s) {stale} from its cache, so no generation "
            f"ran; the template must carry {NONCE_INPUT} on the node that consumes the request"
        )
    history = requests.get(f"{base_url}/history/{prompt_id}", timeout=60).json().get(prompt_id) or {}
    status = (history.get("status") or {}).get("status_str")
    if status != "success":
        raise RuntimeError(f"ComfyUI prompt {prompt_id} ended with status {status!r}")
    files = _output_files(history)
    if not files:
        raise RuntimeError(f"ComfyUI prompt {prompt_id} completed without producing an output")
    sizes = [_fetch(base_url, entry, first_chunk_only=not fetch_outputs) for entry in files]
    if not all(sizes):
        raise RuntimeError(f"ComfyUI prompt {prompt_id} produced an empty output: {files}")
    end = time.time() if fetch_outputs else finished
    return end - start, {"prompt_id": prompt_id, "outputs": files, "cached_nodes": cached}
