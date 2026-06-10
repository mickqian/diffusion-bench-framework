import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

const [reportDirArg, outputArg] = process.argv.slice(2);

if (!reportDirArg) {
  console.error("usage: node scripts/export_onepager_data.mjs <report-dir> [output-json]");
  process.exit(1);
}

const reportDir = resolve(reportDirArg);
const outputPath = resolve(outputArg || "docs/data/latest-cross-framework.json");
const manifestPath = join(reportDir, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

if (!manifest.onepager?.rows?.length) {
  throw new Error(`missing onepager.rows in ${manifestPath}`);
}

const frameworks = manifest.frameworks || {};
const data = {
  id: manifest.id,
  title: manifest.title,
  updated_at: manifest.created_at,
  source_report: manifest.id,
  source_script: manifest.benchmark_repo?.script,
  hardware: {
    label: "2x NVIDIA H200",
    gpu_name: manifest.hardware?.h200?.gpu_name || "NVIDIA H200",
    gpus: manifest.hardware?.h200?.gpus
  },
  policy: {
    latency_source: manifest.policy?.latency_source,
    selection: manifest.policy?.selection,
    cache: manifest.policy?.cache,
    torch_compile: manifest.policy?.torch_compile,
    hardware: "2x NVIDIA H200 devbox, not CI/runner/build/GitHub Actions infrastructure",
    client_server_breakdown: manifest.policy?.client_server_breakdown
  },
  frameworks: {
    sglang_diffusion_now: {
      label: frameworks.sglang_diffusion_now?.label || "SGLang-Diffusion",
      commit: frameworks.sglang_diffusion_now?.commit
    },
    vllm_omni_now: {
      label: frameworks.vllm_omni_now?.label || "vLLM-Omni",
      vllm: frameworks.vllm_omni_now?.vllm,
      vllm_omni: frameworks.vllm_omni_now?.vllm_omni,
      vllm_omni_commit: frameworks.vllm_omni_now?.vllm_omni_commit
    }
  },
  summary: manifest.onepager.summary,
  rows: manifest.onepager.rows
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(data, null, 2)}\n`);
console.log(`wrote ${outputPath}`);
