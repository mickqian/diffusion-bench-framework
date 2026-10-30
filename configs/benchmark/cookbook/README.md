# Cookbook track

The headline matrix (`../cases/`) runs **our** case shapes with each framework's best
command. We tune that comparison, so it is also the one where our own setup hides our
problems. On 2xH200 (2026-10-30), running the competitors' published workloads found
three things no headline row could have shown:

- LTX-2.5 two-stage with sglang's own 2-GPU cookbook preset does not start on >=130 GiB GPUs: breakable CUDA graphs come on, and the default warmup shape (960x544) cannot be halved to a 64-aligned size. `cb_ltx25_t2v_1088_2gpu_vllm` keeps that preset as an arm.
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

The runner's free-space floor does not protect against ephemeral-storage eviction. The 2xH200 box was evicted for exceeding its pod limit (250Gi) while `df` showed terabytes free, and two 2xB200 boxes when their node ran low on ephemeral storage. Keep weights, HF cache and logs on a volume that does not count as ephemeral.

`scripts/check_competitor_recipes.py --cookbook` checks every cell's source and every cited arm
recipe for upstream drift, the same way it checks the headline commands. Without `--cookbook`
it checks only the headline commands, so a headline round's pre-flight is unchanged.

The runner prints `HARDWARE DEVIATION` when `DBF_HARDWARE_PROFILE` is not among the cell's
`cookbook.source_gpu`. That is not an error, because running a source's workload on another
GPU class is often the point, but the report has to carry it.

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
    "source_gpu": ["h200"],               // runtime hardware tokens; [] if the source names none
    "sglang_verbatim_gap": "...",         // only when the sglang cookbook has no matching command
    "workload": { "width": 960, "height": 544, "num_gpus": 2, "dit_forwards": 29,
                  "metric": "...", "warmup": "...", "repeats": "..." }   // last four required
  },
  "forwards": { "lightx2v": 29, "sglang": 29, "trtllm-visual": null },  // null = unverified
  "deviations": [ { "field": "metric", "source": "...", "run": "client_e2e_median", "reason": "..." } ],
  "arms": { "<framework>": { "cb-<name>": { "tier": "verbatim", "num_gpus": 2, "serve_args": "",
                                            "upstream_recipe": { ... }, "lightx2v_config": { ... },
                                            "requires": [ "scripts/prepare_cookbook_h3.sh sgl-adaln" ] } } }
}
```

`set` and `unset` take dotted paths into the base case. Only the arm frameworks are kept.
They keep none of the base case's profiles or `serve_args`, and every arm states its own
`serve_args` and `num_gpus`. Arm names start with `cb-` and never contain a hardware
token, because arms are only ever selected explicitly.

### Tiers

| tier | what it is | required |
|---|---|---|
| `verbatim` | a command copied from a published cookbook, recipe or config. The harness adds `--model-path`, `--port`, `--host` and `--num-gpus`, and for sglang also `--backend sglang` (so a native-pipeline miss fails instead of benchmarking the Diffusers fallback); do not repeat them. | `upstream_recipe`. The source framework's verbatim arm cites the cell's source file. |
| `matched` | turns on a lossless feature the source runs by default, through a switch its own cookbook documents (vLLM-Omni's exact AdaLN reuse is on by default; sglang matches it with `--minimax-h3-adaln-online true`, bit-exact per its cookbook) | `matches` names the source default; `upstream_recipe` cites the switch's documentation |
| `matched-lossy` | turns on the lossy features the source ships with (SageAttention, AdaLN cache, FP8, Cache-DiT) | `lossy` lists them |
| `tuned` | anything we chose ourselves | `notes` |

A cell's sglang number is its **verbatim** sglang arm. Tuned rows are reported beside it,
never in its place. When the sglang cookbook has no command for the workload (for example,
Z-Image or Wan2.2 on 2 NVIDIA GPUs), the cell says so in `cookbook.sglang_verbatim_gap`.
That gap is itself a finding for the cookbook.

### Groups

`cookbook.group` is `lossless` (default), `lossy` (the source ships a lossy default:
SageAttention, FP8, an AdaLN cache sidecar) or `distilled` (FastH3, FastH3-Omni, VDN).
Each group is reported on its own and never ranked against the others.

### Scaling to fewer (or more) GPUs

A box with fewer GPUs than a source command was written for is not a reason to skip the
section, but the command is re-derived one way only, never decided per cell
(`scripts/cookbook_scaling.py`):

1. If the source documents a section for the target GPU count, use that section and
   scale nothing.
2. Otherwise re-derive the parallel layout for M GPUs. A source that runs CFG parallel
   keeps CFG 2 when M >= 2 and gives the rest to sequence parallel (Ulysses / USP /
   ring): SP = M / 2. A source without CFG parallel runs pure SP: SP = M. Tensor
   parallelism of the DiT is not used. Per-component degrees that follow the GPU count
   (VAE patch parallel, text-encoder TP) are capped at M. Every other flag stays as the
   source wrote it. sglang without an SP flag gives its GPUs to Ulysses, so the
   cookbook's `--num-gpus 8` commands scale through `--num-gpus` alone.
3. On datacenter GPUs an arm may also drop memory-saving switches (offload, layerwise
   streaming) and lists each in `scaled_from.removed`.
4. Anything else is listed in `scaled_from.added` as `{flag, reason}`. For example,
   vLLM-Omni's four-GPU combined H3 service gets `--task-type fl2va`, because both DiTs
   plus Qwen3-VL do not fit on one rank.

Such an arm carries `scaled_from` (`num_gpus` plus the source command:
`serve_args`, `lightx2v_config` or `visual_gen_args`). The build re-derives the arm from
it and fails on any difference, so a scaled command cannot be hand-written. The cell
records the GPU count as a `num_gpus` deviation.

### Source inventory

`sources.json` lists every section of every tracked recipe file. Each section either
names the cells built from it (or running it as an arm) or says in `skip` why none is
built: AMD only, RTX only, a lossy flag variant, prose without a command, and so on.
The build fails in three cases: a cell whose source section is not listed, a section
with neither `cells` nor `skip`, or a listed cell that does not exist. Without this,
the first cut covered three of the twenty-odd sections of vLLM-Omni's MiniMax-H3
recipe, and nothing said so.

### Workload and deviations

`cookbook.workload` copies the source's values. Keys are case fields or dotted paths
(`request_extra.extra_params.guardrails`). Four are required and compared specially:

- `dit_forwards` is compared against each arm framework's `forwards`.
- `metric` is compared against `client_e2e_median`, the only timer the harness has.
- `warmup` and `repeats` are compared against what the harness does for the case, e.g.
  `3 request(s) at 3 steps` and `median of 2` for video (`benchmark_defaults`, overridden by
  the case's `benchmark.warmup`). Write `not stated` when the source gives none.

How a number is measured belongs to the workload: vLLM-Omni's Qwen-Image-2.1 recipe warms up
with one full request and takes the median of three, TRT-LLM precompiles its shapes at startup
and times three prompts. Any of these that differ from the harness are deviations.

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

### Prerequisites

An arm whose command reads an absolute path (an AdaLN cache sidecar, a patched kernel's site
directory), and every `matched-lossy` arm, lists in `requires` the steps that create what it
reads. Hand-built artifacts are lost with the box: the H200 round's AdaLN caches and its
patched SageAttention went with its eviction. `scripts/prepare_cookbook_h3.sh` builds the
MiniMax-H3 ones: `sgl-adaln`, `lx-adaln`, and `sage`. The `sage` step applies the SM90 fix
from thu-ml/SageAttention@d9704247 and fails unless the result matches SDPA.

### LightX2V compile

A verbatim LightX2V arm must set `use_compile`. `devbox_run_cases.sh` exports
`DIFFUSION_BENCH_DISABLE_TORCH_COMPILE=0`, so the harness setdefaults `use_compile: true`,
while LightX2V's own default is false.

## Adding a cell

Start from `sources.json`: list the file's sections, then build the ones that run.

1. Pick a source and pin it: the last commit to the file at the framework version you will
   install (`git log -1 <version> -- <path>`). Check that the file exists at that version.
   TRT-LLM's `minimax-h3-bf16-2gpu.yaml` exists only on main, not at v1.4.0rc1.
2. Copy the workload into `cookbook.workload` and the command into the source's verbatim arm.
3. Add the sglang verbatim arm from `sgl-project/sglang` `docs/cookbook/diffusion/` at the
   sglang pin. Many pages generate commands through a builder
   (`docs/src/snippets/configs/<Org>/<model>.jsx`); cite that file.
4. Add the other frameworks' arms, then run the build until it passes. Read COOKBOOK.md's
   inherited column before trusting a verbatim arm.
