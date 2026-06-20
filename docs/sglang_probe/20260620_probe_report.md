# SGLang-Diffusion Temporary Probe Final Report

Generated: 2026-06-21 00:20 CST

## Scope

Temporary experiment only. No repo edits, no pytest changes, no PR.

Coverage matrix:
- total cells: 293
- smoke: 96
- pairwise: 180
- targeted triples: 12
- stress: 5

## Result

- complete: 293/293
- ok: 170
- trusted failures: 123
- missing: 0

Phase breakdown:
- smoke: ok 65, failed 31
- pairwise: ok 97, failed 83
- targeted: ok 5, failed 7
- stress: ok 3, failed 2

Ignored false results were from known harness/download pollution, not selected for final status:
{'sglang-probe-out-full': 7, 'sglang-probe-out-resume-fixed': 7, 'sglang-probe-out-resume3-fixed': 14, 'incomplete_qwen_layered_cache': 6}

## Machines

The run used multiple remote GPU devboxes across Blackwell-class systems. Some reruns were needed because one environment had an incomplete Qwen-Image-Layered cache; those cache-polluted cells were excluded from the final selected status. The final Qwen-Image-Layered rerun used a fresh Blackwell devbox, downloaded a complete snapshot, and closed the last 6 missing cells.

## Main Failure Clusters

- Invalid video frames frequently return 500 instead of a clean 4xx: 27 request failures.
- Invalid image size is inconsistent: 20 cases returned 200, 19 returned 500.
- Qwen-Image-Layered raw HTTP image edit returns 500 for the valid multipart image_edit smoke. The invalid image smoke also returns 500 for bad size.
- Qwen-Image-Layered Python server_args can report cell success while command logs include `IsADirectoryError: '/'` in `QwenImageLayeredBeforeDenoisingStage`; this looks like an image_path normalization/API adapter issue.
- Ideogram4 NVFP4/B200 combinations hit the historical probe bug `num_inference_steps=4` versus `V4_DEFAULT_20` and can hang during shutdown/GC. The harness catalog now uses the preset-aligned default of 20 steps for future runs.
- LingBot realtime/OpenAI SDK combinations remain fragile around image_path/init semantics and invalid session handling.
- Dynamic multi-LoRA still fails for Z-Image because dynamic LoRA supports one adapter per target unless merge mode is used.

## Positive Coverage

- Wan I2V/TI2V and several video paths completed in raw HTTP, OpenAI SDK, and Python server_args paths.
- Hunyuan3D mesh create/list/retrieve/delete/content paths completed in multiple entrypoints, though one stress mesh cell failed.
- Flux2 raw image generation and Python DiffGenerator + Cache-DiT passed in selected cells.
- Stress phase completed all 5 selected cells: 3 OK, 2 trusted failures.

## Local Artifacts

- `matrix.json`: complete generated parameter matrix.
- `final_merge_summary.json`: final aggregate counts and top failure buckets.
- `final_selected_runs.jsonl`: one selected status per matrix cell.
- `final_trusted_failed_cells.txt`: final trusted failed cell IDs.
- `final_missing_cells.txt`: empty after probe6 rerun.
- `final_merge_inputs/`: all fetched remote `runs.jsonl` files used for merge.

The final run status data is complete in the committed summaries. Raw media outputs and credentials are intentionally excluded from this repository.
