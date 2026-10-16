"""diffusers reference latency for FLUX.3 Klein Base, one GPU, prompt encoded as given.

python diffusers_ref2.py <caption file> <out dir>
Median of 5 timed calls after 2 warmups, CUDA-synchronized around each call.
"""
import json, os, statistics, sys, time

import diffusers
import torch
from diffusers.modular_pipelines import Flux3AutoBlocks

caption = open(sys.argv[1]).read().strip()
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
# The case prompt is already the upsampled caption; the default blocks would rewrite it again.
# (pipe.blocks is a deepcopy, so the step has to go before the pipeline is built.)
blocks = Flux3AutoBlocks()
blocks.sub_blocks.pop("prompt_upsampler")
pipe = blocks.init_pipeline("black-forest-labs/flux-3-image-klein-base")
print("R blocks:", list(pipe.blocks.sub_blocks), flush=True)
pipe.load_components(names=["text_encoder", "tokenizer", "transformer", "vae"], torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("R dtypes:", {n: str(pipe.get_component(n).dtype) for n in ("text_encoder", "transformer", "vae")} if hasattr(pipe, "get_component") else
      {n: str(getattr(pipe, n).dtype) for n in ("text_encoder", "transformer", "vae")}, flush=True)
print("R guider:", type(pipe.guider).__name__, dict(pipe.guider.config), flush=True)
kwargs = {"prompt": caption, "height": 1024, "width": 1024, "num_inference_steps": 50, "output": "images"}


def run():
    torch.cuda.synchronize()
    t = time.perf_counter()
    image = pipe(**kwargs, generator=torch.Generator().manual_seed(42))[0]
    torch.cuda.synchronize()
    return time.perf_counter() - t, image


for _ in range(2):
    run()
samples = []
for _ in range(5):
    dt, image = run()
    samples.append(round(dt, 3))
image.save(f"{out}/diffusers_flux3_base.png")
result = {"framework": "diffusers", "version": diffusers.__version__, "latency_s": statistics.median(samples),
          "samples_s": samples, "kwargs": {k: v for k, v in kwargs.items() if k != "prompt"}, "image_size": image.size}
json.dump(result, open(f"{out}/diffusers_result.json", "w"), indent=1)
print("R", json.dumps(result), flush=True)
