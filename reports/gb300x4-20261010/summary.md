# 4xGB300 cross-framework (2026-10-10)

Latest-vs-latest on two 4x NVIDIA GB300 288GB pods (aarch64 Grace, driver
580.159.04; image cases on one pod, video cases on the other): 17 cases,
6 frameworks, 55 measured single-request cells and 9 throughput cells.
Qwen-Image-2512 is retired for Qwen-Image-2.1 (1-GPU, 2-GPU and edit), and
MiniMax-H3 gains a 4-GPU case and FastH3-8-Step-V2.

| framework | version |
|---|---|
| SGLang-Diffusion | main `6576e50cd5` (2026-10-11; every fix below merged) |
| vLLM-Omni | main `f69b1f2b1` (0.31.0rc2.dev98) on vLLM 0.31.0 |
| LightX2V | main `b6d38283` (0.5.0, torch 2.11; FA3 skipped on aarch64) |
| TRT-LLM VisualGen | 1.3.0rc29 |
| ComfyUI | master `0df64eb2` (2026-10-09, torch 2.14.1) |
| FastVideo | main `d5253fd7` (0.2.1) |

**The sglang column is main `6576e50cd5`, re-run on 2026-10-11 after every fix
from this round had merged.** The first pass, on main `3831e7e09`, lost cells
on this box to placement and compile-path bugs, not kernels. The fixes went
upstream and were measured on a pinned tree carrying them (sha `cf05d511` /
`284d91e7`). Once they had all landed, the whole sglang column was run again
on main (via `DBF_SGLANG_PYTHONPATH`). The pinned-main and fix-tree cells stay
in [raw/](raw/).

| PR | fix |
|---|---|
| #43565 | bind each diffusion GPU worker to its GPU's NUMA node (srt already does) |
| #43568 | BCG outside the validated list: warn and enable instead of silently disabling; warn after warmup when nothing was captured |
| #43577 | torch.compile keeps the fused custom ops; JIT qk-norm registered as an opaque op (Cosmos3 compile 1.43 → 0.88 s) |
| #43575 | FLUX regional compile (`_compile_conditions`) |
| #43576 | Qwen-Image-2.1 compile with offload-during-compile no longer crashes on the 1-D placeholder |
| #43584 | IPC all-to-all buffers allocated outside expandable segments (GB fabric handles crashed `empty_cache`) |
| #43586 | Qwen / Ulysses compile: no graph breaks from the packed-QKV check, async A2A or IPC A2A |
| #43601 | Z-Image compile: skip inductor's saved multi-kernel pick (every warm start crashed, or hung under TP) |
| #43602 | TP all-reduce as an opaque op under torch.compile (FLUX.1 TP2 compile 3.65 → 2.59 s) |
| #43631 | offload-during-compile: serve whole graphs after the warmup (Qwen-Image-2.1 TP2 compile 2.15 → 1.56–1.67 s) |
| #43632 / #43633 | Qwen-Image BCG padding attended the pad / Qwen-Image-2.1 BCG replays across prompt lengths |
| #43634 | MiniMax-H3: the attention section runs eager under torch.compile (compile 61.4 → 58.7 s) |

## Single request

Steady-state client latency in seconds, median of the measured repeats after
warmup; GPU count in parentheses; fastest per case in bold. `—` = unsupported or
no validated profile (per-cell reasons in [issue.md](issue.md)).

| case | SGLang-Diffusion | vLLM-Omni | LightX2V | TRT-LLM VisualGen | ComfyUI | FastVideo |
|---|---:|---:|---:|---:|---:|---:|
| `flux1_dev_t2i_1024` | 4.00 (2) | 5.31 (2) | — | **3.10 (2)** | 4.41 (1) | — |
| `flux2_dev_t2i_1024` | **5.13 (2)** | 8.37 (2) | 14.54 (2) | 9.29 (2) | 12.38 (1) | — |
| `qwen_image_21_t2i_1024` | **2.00 (1)** | 2.15 (1) | 3.01 (1) | — | 3.22 (1) | — |
| `qwen_image_21_t2i_1024_2gpu` | **1.67 (2)** | 3.08 (2) | — | — | — | — |
| `qwen_image_21_edit_1024` | **2.43 (1)** | 2.60 (1) | — | — | — | — |
| `zimage_turbo_t2i_1024` | **0.52 (2)** | 0.73 (2) | 1.50 (2) ~ | — | 0.79 (1) | — |
| `ideogram4_t2i_1024_2gpu_tp` | **2.60 (2)** | — | — | — | 3.62 (1) | — |
| `wan22_t2v_a14b_720p` | **87.03 (4)** | 108.99 (4) | 225.85 (4) | — | 484.03 (2) | — |
| `ltx2_twostage_t2v` | **9.07 (2)** | 12.76 (2) | 20.07 (2) | — | 29.74 (2) | — |
| `ltx2.3_twostage_t2v_2gpus` | 11.02 (2) | — | — | — | — | — |
| `cosmos3_nano_t2i_720p` | **0.90 (2)** | 0.94 (2) | — | — | — | — |
| `cosmos3_nano_t2v_720p_189f` | **22.46 (4)** | 26.05 (4) | — | — | — | — |
| `cosmos3_nano_i2v_720p_189f` | **22.63 (4)** | 26.39 (4) | — | — | — | — |
| `minimax_h3_t2va_5s` | **57.46 (2)** | 60.95 (2) | 141.04 (2) | — | 283.48 (1) | 61.15 (2) |
| `minimax_h3_t2va_5s_4gpu` | **31.76 (4)** | 34.08 (4) | 73.29 (4) | — | — | 34.08 (4) |
| `minimax_h3_ref2va_5s` | **88.36 (2)** | 91.91 (2) | 229.21 (2) | — | 479.48 (1) | 90.46 (2) |
| `minimax_h3_fasth3_v2_t2va_5s` | **6.44 (4)** | 7.44 (4) | — | — | — | 7.17 (4) |

`~` LightX2V Z-Image: the first repeat ran 1.01 s and the other four 1.50 s;
the median is published with its samples listed.

## Throughput

Four requests at concurrency 2, one representative image and one video case.

| case | framework | GPUs | done/requests | QPS | p50 (s) | p95 (s) |
|---|---|---:|---:|---:|---:|---:|
| `flux1_dev_t2i_1024` | SGLang-Diffusion | 2 | 4/4 | 0.2535 | 7.89 | 7.96 |
| `flux1_dev_t2i_1024` | vLLM-Omni | 2 | 4/4 | 0.1909 | 10.40 | 10.52 |
| `flux1_dev_t2i_1024` | TRT-LLM VisualGen | 2 | 4/4 | 0.3312 | 5.97 | 6.08 |
| `flux1_dev_t2i_1024` | ComfyUI | 1 | 4/4 | 0.1940 | 10.27 | 10.34 |
| `minimax_h3_t2va_5s` | SGLang-Diffusion | 2 | 4/4 | 0.0173 | 114.83 | 115.35 |
| `minimax_h3_t2va_5s` | vLLM-Omni | 2 | 4/4 | 0.0164 | 122.30 | 122.31 |
| `minimax_h3_t2va_5s` | LightX2V | 2 | 4/4 | 0.0071 | 281.81 | 282.61 |
| `minimax_h3_t2va_5s` | ComfyUI | 1 | 4/4 | 0.0035 | 566.77 | 567.04 |
| `minimax_h3_t2va_5s` | FastVideo | 2 | 4/4 | 0.0163 | 122.32 | 122.43 |

## Notes

- **sglang leads every cell except FLUX.1, where TRT-LLM VisualGen leads**
  (3.10 against 4.00 s single, 0.33 against 0.25 QPS). sglang's FLUX.1 row is
  TP2 eager. With breakable CUDA graphs sglang measured 1.88 s on this box,
  before the NUMA fix. BCG is kept out of the table because a graph captured at
  concurrency 1 is not production-realistic. sglang is ahead of vLLM-Omni in
  every cell.
- **The sglang rows are eager** except Cosmos3 T2I and LTX-2 (compiled
  CFG-parallel); vLLM-Omni runs its default, compiled. Measured afterwards with
  torch.compile on (rerun tree plus sgl-project/sglang#43601 and #43602,
  `--offload-during-compile false` except FLUX.1): FLUX.1 TP2 2.59 s, Qwen-2.1
  2-GPU 1.50 s, Z-Image 0.57 s, FLUX.2 5.23 s, Qwen-2.1 1-GPU 2.10 s, edit 2.50
  s, Cosmos3 T2V 24.5 s, Wan2.2 91.3 s, ahead of vLLM-Omni's compiled rows, as
  is MiniMax-H3 t2va 58.6–58.7 s with #43634 (61.5 s before it) against its 60.9
  s. These are not in the table.
- **What the pinned-main pass got wrong.** sglang main `3831e7e09`, same
  commands: FLUX.1 6.17 s, Z-Image 0.79, ideogram4 4.15, Qwen-2.1 2-GPU 4.36 (Ulysses),
  LTX-2 18.02, LTX-2.3 15.57, Cosmos3 T2I 1.54. Main 6576e50cd5: 4.00, 0.52,
  2.60, 1.67 (TP2), 9.07 (compiled CFG-parallel), 11.02, 0.90 (compiled
  CFG-parallel).
  Unbound workers drifted to the other Grace socket and paid remote memory on
  every host-bound op, so the same command landed at ~4.0 or ~6.5 s run to run.
  vLLM-Omni moves 3–8% when pinned.
- **Profiles promoted for GB300** (harness b8c7b41, 45e55dd): Qwen-2.1 2-GPU
  runs TP2 (Ulysses was 4.36 s); Cosmos3 T2I and LTX-2 run compiled
  CFG-parallel, and those two carry a `policy_exception` note for compile on.
- **vLLM-Omni Qwen-2.1 2-GPU** comes from a rerun: before harness 4cac45e
  vLLM-Omni's `--usp` was not recognised, so it also got TP2 and ran on all
  four cards (3.50 s there, 3.08 s on its two).
- **FastH3-8-Step-V2**: sglang's native support counts DiT forwards and
  rejects 9 steps with 400. Harness 1d771ec sends it 8. vLLM-Omni is sent 8 and
  FastVideo 9 sigma points: the same eight forwards everywhere.
- **Not here**: the single-RTX-5090 desktop cells (Qwen-2.1 t2i/edit, H3 480p)
  are still on that host, which dropped off the network before they were
  copied. The MI355X and 4x5090 H3 attempts produced sglang-only cells. They are
  in [../h3-partial-20261010/](../h3-partial-20261010/).

## Reproduce

`scripts/run_gb300_full_20261010.sh` (main pass at harness `b32615f`; reruns
listed per tag in [manifest.json](manifest.json)). Merge and publish with
`scripts/merge_and_publish_run.sh`, inputs in the order listed in the manifest.
