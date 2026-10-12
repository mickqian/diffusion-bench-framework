# One RTX 5090 desktop (2026-10-10 / 10-11)

A home desktop, not a datacenter pod: one GeForce RTX 5090 32 GB, Ryzen 9
9950X, 60 GB RAM, PCIe 5.0 NVMe, Ubuntu 26.04, driver 595.99.02 (open kernel
module), CUDA 13.1, bare metal (`rdxa-5090-pa`). It is the machine class the
sglang cookbook's consumer recipes target, so it stands for what a single-GPU
user gets. Three cases fit on one card: Qwen-Image-2.1 t2i and edit at 1024²,
and MiniMax-H3 T2VA at 864×480, 124 frames, 20 steps.

| framework | version |
|---|---|
| SGLang-Diffusion | main `6576e50cd5` (codeload tarball `6576e50cd5e5817b10e12245104603eb1b0d8a4f`, run 2026-10-11, the same commit as the GB300 / H200 report) |
| vLLM-Omni | main `f69b1f2b` on vLLM 0.31.0 |
| ComfyUI | master `0df64eb2` |
| LightX2V | main `b6d38283` (0.5.0) |

The competitor cells ran on 2026-10-10; the sglang column was re-run on
2026-10-11 on main `6576e50cd5` (the first pass used the round's earlier pin
`3831e7e09`: 13.26 / 15.75 / 103.22 s, the same within 0.3%).

## Single request

Client wall clock, median of five requests (two for H3) after warmup, seconds.

| case | SGLang-Diffusion | vLLM-Omni | ComfyUI | LightX2V |
|---|---:|---:|---:|---:|
| `qwen_image_21_t2i_1024` | **13.30** | 14.71 | 13.83 | 14.04 |
| `qwen_image_21_edit_1024` | **15.74** | 17.22 | — | — |
| `minimax_h3_t2va_480p_1gpu` | **103.28** | — | — † | — |

`—` = no path for that framework on this case. sglang leads every cell. Its
thinnest lead is over ComfyUI on Qwen t2i (1.04×). The two were run
interleaved twice, and every sglang sample was below every ComfyUI sample:
13.29 / 13.30 s against 13.82 / 13.83 s ([evidence/pa_ab_comfy.log](evidence/pa_ab_comfy.log)).

† ComfyUI cannot run MiniMax-H3 on this host. With its `--fast-disk
--cache-none` recipe and torch.compile on, the sampler failed
(`SamplerCustomAdvanced`: AttributeError). The eager rerun, same flags, staged
63 GB for dynamic VRAM loading on a 60 GB host and ran out of RAM. The kernel's OOM killer took `tailscaled`,
and the box was off the network for a day. sglang serves the same case in
103 s with the cookbook's 60 GB-host recipe (layerwise offload, 14 resident DiT
layers).

## Eager against eager

sglang's cells are eager (no torch.compile). vLLM-Omni's table cells use its
default compiled path; with `--enforce-eager` it runs 15.89 s (t2i) and 18.47 s
(edit), so sglang eager leads vLLM-Omni eager by 1.20× and 1.17×
([evidence/veager1011_*.json](evidence/)).

## Commands

- sglang, Qwen: `--performance-mode manual --component-residency dit=resident
  text_encoder=layerwise-offload vae=resident` (the 20B DiT stays on the card;
  the Qwen2.5-VL text encoder streams).
- sglang, H3: `--model-variant fl2va --performance-mode memory
  --layerwise-offload-components dit,text_encoder,vae
  --dit-layerwise-resident-layers 14 --layerwise-resident-layers video_vae=36`.
- vLLM-Omni, Qwen: `--enable-cpu-offload --no-enable-cuda-graph-decode`, its
  recipe's low-memory path. Resident, it OOMs at the dummy run (36.9 GB BF16
  peak); with CUDA-graph decode on, the first request OOMs swapping the text
  encoder in.
- ComfyUI and LightX2V: the case profiles' rtx5090 commands, recorded per cell
  in [merged.json](merged.json).

## Published to the history, not as the latest run

The run is in `docs/data/historical-cross-framework.json` as
`rtx5090-desktop-20261011`. It went in with `--keep-latest`
(mickqian/diffusion-bench-framework#12), so `latest-cross-framework.json`
stays on the 4xGB300 round. The merge ran with the publish gates (no command
drift, no partial sources, no native fallbacks): [merged.json](merged.json),
[issue.md](issue.md).

## Reproduce

`scripts/run_rtx5090_pa_20261010.sh` (the 10-10 cells),
[evidence/pa_rerun_1011.sh](evidence/pa_rerun_1011.sh) (sglang main and
vLLM-Omni eager, with a memory watchdog) and
[evidence/pa_ab_comfy.sh](evidence/pa_ab_comfy.sh) (the interleave). Merge:

```
DBF_REPORT_DIR=reports/rtx5090-desktop-20261011 scripts/merge_and_publish_run.sh \
  "<raw/run_pa5090qwen1010_*> <qwen1010b> <qwen1010c> <h31010b> <main1011> <ab1011comfy1> <ab1011sgl1> <ab1011comfy2> <ab1011sgl2>" \
  rtx5090-desktop-20261011 '1x RTX 5090 desktop' '1x NVIDIA GeForce RTX 5090 32GB (desktop, 60 GB RAM)' \
  scripts/run_rtx5090_pa_20261010.sh --live --keep-latest
```

Later files win per (case, framework), so the sglang row is the last
interleaved round and the competitors' rows are their 10-10 runs.
