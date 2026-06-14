# Running the `trtllm-visual` (TensorRT-LLM VisualGen) benchmark on H200

Reproduction + hard-won environment notes for benchmarking TensorRT-LLM
VisualGen (`trtllm-serve`) against SGLang-Diffusion / vLLM-Omni / LightX2V.
Verified on a 2×H200 rx devbox, 2026-06.

## TL;DR

```bash
# 0. Acquire a fresh GPU box (rx devbox example)
rx devbox acquire --gpu h200 --count 2 --ttl 6h --image lmsysorg/sglang:dev --name trtllm-visual-bench
rx devbox ssh-config trtllm-visual-bench           # sets up the `ssh <name>` alias

# 1. Sync this repo to the box (no rsync on the image; use tar-over-ssh)
tar czf - --exclude=__pycache__ src configs scripts pyproject.toml \
  | ssh <name> 'mkdir -p ~/diffusion-bench-framework && tar xzf - -C ~/diffusion-bench-framework'

# 2. Install trtllm-visual into its isolated venv (see env notes below)
ssh <name> 'export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/scratch/fw-venvs TMPDIR=/scratch/tmp FORCE_FRAMEWORK_REINSTALL=1
            cd ~/diffusion-bench-framework && bash scripts/install_comparison_frameworks.sh trtllm-visual'

# 3. Run (compile-on is auto-applied for trtllm-visual; 1 GPU per image case)
ssh <name> 'cd ~/diffusion-bench-framework
            export SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT=/scratch/fw-venvs SGLANG_DIFFUSION_SKIP_FRAMEWORK_INSTALL=1
            export HF_HOME=/cluster-storage/models   # shared model cache
            PYTHONPATH=src python3 -m diffusion_bench.run_comparison \
              --config configs/comparison_configs.json --frameworks trtllm-visual \
              --case-ids flux1_dev_t2i_1024 flux2_dev_t2i_1024 qwen_image_2512_t2i_1024 cosmos3_nano_t2i_720p \
              --modes single_e2e --hardware-profile h200 --port 8100 --output /scratch/trtllm.json'
```

Drive long server launches / runs through `tmux new-session -d` (the image's
sshd + background `&` are flaky over the proxy); poll the log file.

## Environment notes (the parts that bite)

1. **VisualGen needs `tensorrt-llm` >= 1.3.0rc** — it is NOT in the 1.2.x
   stable line. With 1.2.x, `trtllm-serve serve <flux>` routes the model
   through the LLM engine and crashes (no `model_type`). `1.3.0rc18` ships
   `tensorrt_llm/_torch/visual_gen/` and auto-detects diffusion models via
   `get_is_diffusion_model`.

2. **rc18 has an unresolvable pip conflict on a clean index.** It requires
   `cuda-python>=13` (→ cuda-bindings 13.x) *and* `torch>=2.10`, but PyPI's
   torch 2.10.0 pins `cuda-bindings==12.9.4`. The installer works around it by
   installing torch from PyTorch's **cu130** index first, then tensorrt-llm
   with `--upgrade-strategy only-if-needed`. (See the `trtllm-visual` branch in
   `scripts/install_comparison_frameworks.sh`; override with
   `TRTLLM_TORCH_INDEX_URL` / `TRTLLM_INSTALL_SPEC`.)

3. **VisualGen requires torch.compile.** In eager mode the warmup hits
   `layer_norm: expected scalar type Float but found BFloat16`. The harness's
   cache-free `TORCH_COMPILE_DISABLE=1` policy is therefore skipped for
   `trtllm-visual` (`REQUIRES_TORCH_COMPILE` in `run_comparison.py`). Compile
   cost is paid during warmup and excluded from the measured latency, so the
   comparison stays fair.

4. **OpenAI image endpoint:** `/v1/images/generations`, but it rejects
   `response_format: "url"` ("URL mode is not supported") — send `b64_json`.
   The generic HTTP request path already does.

5. **Parallelism:** VisualGen sets CFG/Ulysses sizes via a
   `--extra_visual_gen_options` YAML, not `--tp_size`. The default image cases
   are run on 1 GPU, matching the existing 1-GPU SGLang/vLLM rows.

## Fairness

Measure **only** trtllm-visual and align it to the existing dashboard cases —
do not re-measure SGLang each time (it adds variance and its own env breakage
for no gain). The one caveat to disclose in the row note: trtllm-visual runs
compile-on because that is its only functional mode; SGLang/vLLM figures are
the existing committed measurements. Same H200 SKU, same case params (prompt,
resolution, steps, seed, guidance), cache-free.

Measured (H200, 1 GPU, compile-on, cache-free, client latency):
- FLUX.1-dev T2I 1024² · 50 steps · cfg 3.5 → **7.32s**
- FLUX.2-dev T2I 1024² · 50 steps · cfg 4 → **23.77s**

## SGLang-side gotchas seen on a fresh image

- The gated FLUX repos 403'd on `model_index.json` even though cached →
  upstream fix in PR sgl-project/sglang#28177 (Hub-first, local-cache
  fallback). Workaround until merged: `export HF_HUB_OFFLINE=1`.
- `lmsysorg/sglang:dev` shipped without `accelerate` (a declared dep); the
  diffusers-fallback path needs it. `pip install accelerate` if you re-measure
  SGLang on that image.
