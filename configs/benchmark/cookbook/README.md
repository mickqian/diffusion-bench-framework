# Cookbook track

The headline matrix (`../cases/`) runs **our** case shapes with each framework's best
command. We tune that comparison, so it is also the one where our own setup hides our
problems. On 2xH200 (2026-10-30), running the competitors' published workloads found
three things no headline row could have shown:

- LTX-2.5 two-stage cannot serve vLLM-Omni's 960x544 single-stage shape.
- The MiniMax-H3 server warmup cannot reach the 544 short edge that LightX2V and TRT-LLM ship as their default.
- The sglang cookbook's own Wan2.2 builder emits `--dit-layerwise-offload true` on a 141 GB H200.

A **cookbook cell** takes one framework's published workload **and** command, verbatim,
and runs every other framework against that workload. The results are a separate track;
they are never merged into headline rows.

```
cookbook/
├── README.md       # this file
├── COOKBOOK.md     # generated: every cell, its source, deviations and resolved arm commands
└── cells/<id>.json # one cell per source workload
```

Build with `python3 scripts/build_cookbook_config.py`. It writes `configs/cookbook_configs.json`
and `COOKBOOK.md`; `scripts/tests/test_cookbook_cells.py` fails while either is stale.
Run one cell's arms on a devbox:

```bash
CB_SGLANG_COMMIT=<pin> DBF_SGLANG_PYTHONPATH=<sglang tree at the pin>/python \
  bash scripts/run_cookbook_cells.sh <tag> 0,1 39600 cb_h3_t2va_544_lightx2v_tp2
```

`scripts/check_competitor_recipes.py` checks every cell's source and every cited arm recipe
for upstream drift, the same way it checks the headline commands.

## Cell schema

```jsonc
{
  "id": "cb_...",                         // always cb_*: never collides with a headline case
  "base_case": "minimax_h3_t2va_5s",      // headline case to reuse for request plumbing
  "set":   { "width": 960, "framework_request_extra.sglang.num_inference_steps": 29 },
  "unset": [ "frameworks.trtllm-visual.visual_gen_args.torch_compile_config" ],
  "cookbook": {
    "framework": "lightx2v",              // whose cookbook this is
    "repo": "...", "path": "...", "commit": "<last commit to the file>", "commit_date": "YYYY-MM-DD",
    "section": "...", "source_hardware": "...", "source_metric": "...",
    "sglang_verbatim_gap": "...",         // only when the sglang cookbook has no matching command
    "workload": { "width": 960, "height": 544, "num_gpus": 2, "dit_forwards": 29, "metric": "..." }
  },
  "forwards": { "lightx2v": 29, "sglang": 29, "trtllm-visual": null },  // null = unverified
  "deviations": [ { "field": "metric", "source": "...", "run": "client_e2e_median", "reason": "..." } ],
  "arms": { "<framework>": { "cb-<name>": { "tier": "verbatim", "num_gpus": 2, "serve_args": "",
                                            "upstream_recipe": { ... }, "lightx2v_config": { ... } } } }
}
```

`set` and `unset` take dotted paths into the base case. Only the arm frameworks are kept.
They keep none of the base case's profiles or `serve_args`, and every arm states its own
`serve_args` and `num_gpus`. Arm names start with `cb-` and never contain a hardware
token, because arms are only ever selected explicitly.

### Tiers

| tier | what it is | required |
|---|---|---|
| `verbatim` | a command copied from a published cookbook, recipe or config. The harness adds `--model-path`, `--port`, `--host` and `--num-gpus`; do not repeat them. | `upstream_recipe`. The source framework's verbatim arm cites the cell's source file. |
| `matched-lossy` | turns on the lossy features the source ships with (SageAttention, AdaLN cache, FP8, Cache-DiT) | `lossy` lists them |
| `tuned` | anything we chose ourselves | `notes` |

A cell's sglang number is its **verbatim** sglang arm. Tuned rows are reported beside it,
never in its place. When the sglang cookbook has no command for the workload (for example,
Z-Image or Wan2.2 on 2 NVIDIA GPUs), the cell says so in `cookbook.sglang_verbatim_gap`.
That gap is itself a finding for the cookbook.

### Workload and deviations

`cookbook.workload` copies the source's values. Keys are case fields or dotted paths
(`request_extra.extra_params.guardrails`). Two keys are special:

- `dit_forwards` is compared against each arm framework's `forwards`.
- `metric` is compared against `client_e2e_median`, the only timer the harness has.

Every difference needs a `deviations` entry whose `source` and `run` match what the build
sees, and an entry for a field that does not differ fails as stale. Typical entries are a
4-GPU recipe on 2 GPUs, 80 frames rounded to 81, or LightX2V's offline `RUN pipeline cost`.

Forward counts are stated per framework because "N steps" means different things. vLLM-Omni,
LightX2V and sglang read it as N forwards; TRT-LLM and FastVideo read it as N sigma grid
points, which is N-1 forwards.

### Inherited configuration

A profile's `lightx2v_config`, `visual_gen_args` and `extra_env` deep-merge over the base
case's, so a headline setting could reach a cookbook arm unseen. In the H200 round, the
4-GPU Cosmos case's `parallel_vae_size: 4` and TRT-LLM `torch_compile_config` sat under
"verbatim" 1-GPU arms this way. The build therefore does three things:

- resolves each arm's config in full;
- lists what came from the base case in COOKBOOK.md;
- fails a verbatim arm unless every inherited key is either `unset` in the cell or named in
  its `allow_inherited`. Use `allow_inherited` only for harness plumbing such as MP4 encode
  alignment or allocator settings.

A verbatim LightX2V arm must set `use_compile`. `devbox_run_cases.sh` exports
`DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0`, so the harness setdefaults `use_compile: true`,
while LightX2V's own default is false.

## Adding a cell

1. Pick a source and pin it: the last commit to the file at the framework version you will
   install (`git log -1 <version> -- <path>`). Check that the file exists at that version.
   TRT-LLM's `minimax-h3-bf16-2gpu.yaml` exists only on main, not at v1.4.0rc1.
2. Copy the workload into `cookbook.workload` and the command into the source's verbatim arm.
3. Add the sglang verbatim arm from `sgl-project/sglang` `docs/cookbook/diffusion/` at the
   sglang pin. Many pages generate commands through a builder
   (`docs/src/snippets/configs/<Org>/<model>.jsx`); cite that file.
4. Add the other frameworks' arms, then run the build until it passes. Read COOKBOOK.md's
   inherited column before trusting a verbatim arm.
