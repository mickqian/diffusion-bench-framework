# 4xB200 cross-framework, ComfyUI added (2026-09-25)

Latest-vs-latest on one 4x NVIDIA B200 183GB devbox (driver 580.126.09): 16 cases,
5 frameworks, 49 measured single-request cells and 7 throughput cells. First run
with ComfyUI and with Qwen-Image-2.1.

| framework | version |
|---|---|
| SGLang-Diffusion | main `434c2e3a` |
| vLLM-Omni | main `69de153f6` (0.30.0rc2.dev69) on vLLM 0.30.0 |
| LightX2V | main `a4b8ce30` (torch 2.11) |
| TRT-LLM VisualGen | 1.3.0rc28 |
| ComfyUI | master `88ab4a06` (torch 2.14) |

## Single request

Steady-state client latency in seconds, median of the measured repeats after
warmup; GPU count in parentheses; fastest per case in bold. `—` = unsupported or
no validated profile (per-cell reasons in [issue.md](issue.md)).

| case | SGLang-Diffusion | vLLM-Omni | LightX2V | TRT-LLM VisualGen | ComfyUI |
|---|---:|---:|---:|---:|---:|
| `flux1_dev_t2i_1024` | **2.39 (2)** | 3.35 (2) | — | 3.55 (2) | 4.73 (1) |
| `flux2_dev_t2i_1024` | **6.95 (2)** | 9.81 (2) | 16.55 (2) | 11.02 (2) | 14.56 (1) |
| `qwen_image_2512_t2i_1024` | 2.78 (2) | **2.64 (2)** | — | 3.48 (2) | 4.93 (1) |
| `qwen_image_2512_t2i_1024_truecfg` | **3.82 (2)** | 5.12 (2) | — | — | 7.07 (2) |
| `qwen_image_edit_2511` | **4.25 (2)** | 4.39 (2) | — | — | 12.02 (1) |
| `qwen_image_21_t2i_1024` | **2.46 (1)** | — | 3.01 (1) | — | 3.47 (1) |
| `zimage_turbo_t2i_1024` | **0.47 (2)** | 0.49 (2) | 1.01 (2) | — | 0.82 (1) |
| `ideogram4_t2i_1024_2gpu_tp` | 3.43 (2) | — | — | — | **3.18 (1)** |
| `wan22_t2v_a14b_720p` | **106.39 (4)** | 129.06 (4) | 258.34 (4) | — | 541.09 (2) |
| `ltx2_twostage_t2v` | **6.05 (2)** | 15.57 (2) | 34.57 (2) | — | 24.51 (2) |
| `ltx2.3_twostage_t2v_2gpus` | 7.75 (2) | — | — | — | — |
| `cosmos3_nano_t2i_720p` | **0.57 (2)** | 0.63 (2) | — | — | — |
| `cosmos3_nano_t2v_720p_189f` | **31.71 (4)** | 33.04 (4) | — | — | — |
| `cosmos3_nano_i2v_720p_189f` | **31.82 (4)** | 33.40 (4) | — | — | — |
| `minimax_h3_t2va_5s` | **72.88 (2)** | 78.64 (2) | — | — | 319.13 (1) |
| `minimax_h3_ref2va_5s` | **111.63 (2)** | 118.99 (2) | — | — | 534.70 (1) |

## Throughput

Four requests at concurrency 2, one representative image and one video case.

| case | framework | GPUs | done/requests | QPS | p50 (s) | p95 (s) |
|---|---|---:|---:|---:|---:|---:|
| `flux1_dev_t2i_1024` | SGLang-Diffusion | 2 | 4/4 | 0.4187 | 4.78 | 4.79 |
| `flux1_dev_t2i_1024` | vLLM-Omni | 2 | 4/4 | 0.3054 | 6.50 | 6.58 |
| `flux1_dev_t2i_1024` | TRT-LLM VisualGen | 2 | 4/4 | 0.2880 | 6.90 | 6.98 |
| `flux1_dev_t2i_1024` | ComfyUI | 1 | 4/4 | 0.2067 | 9.68 | 9.90 |
| `minimax_h3_t2va_5s` | SGLang-Diffusion | 2 | 4/4 | 0.0137 | 145.84 | 146.36 |
| `minimax_h3_t2va_5s` | vLLM-Omni | 2 | 4/4 | 0.0128 | 155.83 | 156.34 |
| `minimax_h3_t2va_5s` | ComfyUI | 1 | 4/4 | 0.0031 | 638.37 | 638.43 |

## Notes

- **qwen_image_2512 is a tie.** The main pass put sglang 5.2% behind vLLM-Omni;
  two ABAB-interleaved paired reruns on the same box put it 0.7% and 0.8% ahead
  ([evidence/](evidence/)).
- **ideogram4 is an sglang gap.** ComfyUI on one GPU beats sglang on two by 7.8%.
  Both run the fp8 weight-only checkpoint (the only one published) with bf16
  compute and two transformer forwards per step.
- **ComfyUI** runs graphs built from its official workflow templates, with
  full-precision weights in place of the templates' fp8/int8/nvfp4 files,
  `--gpu-only` on Blackwell, and `TorchCompileModel` in 7 cells. qwen-image-2.1,
  ideogram4 and the three cells that split CFG across GPUs (qwen-image-2512
  true-CFG, LTX-2, Wan2.2) run eager because compile fails there. Splitting CFG
  is its only multi-GPU path, so single-branch cells run on one GPU.
- **MiniMax-H3 ref2va:** sglang and vLLM-Omni resize the reference image to a
  2048-pixel short edge; ComfyUI's nodes never upscale, so the harness
  pre-resizes the reference by the same rule for it.
- **Replaced cells.** LTX-2 (every framework) was re-run because the main pass
  read a `text_encoder/config.json` that sglang's LTX-2.3 overlay materializer
  had rewritten inside the shared model cache; LTX-2.3 sglang was re-run because
  the same materializer could not write to the read-only cache and the server
  never started; ComfyUI ref2va was re-run after the reference fix. Discarded
  attempts are in [failures/](failures/).

## Reproduce

`scripts/run_b200_full_20260925.sh` (harness `bc439d0`; the three replaced
cells ran `c201f19`, which adds only the ComfyUI reference resize). Merge and
publish with `scripts/merge_and_publish_run.sh`, inputs in the order listed in
[manifest.json](manifest.json).
