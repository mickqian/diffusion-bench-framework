"""A framework's `status` is one global verdict; `status_by_hardware` scopes it.

LTX-2.3 two-stage classified lightx2v `failed` for an OOM on an 80GB H100, and
because the status was global that verdict also skipped the cell on a 179GB
B200 — a competitor silently missing from a published matrix for a reason that
expired with the hardware. Flipping it to `supported` would instead make every
H100 run spend ~40 minutes re-discovering the OOM.

So this covers both halves: a status that differs by hardware must resolve
correctly for EACH hardware, and the build's "every framework classified" check
must still refuse a case that leaves one out.

The build runs against a throwaway repo-shaped tree (a copied script + a stub
`diffusion_bench` package), so it writes its output there and never near the
real configs.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.config_guard import (  # noqa: E402
    HARDWARE_TOKENS,
    apply_hardware_statuses,
    resolve_status,
)

fail = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global fail
    if ok:
        print(f"  ok   {label}")
    else:
        fail = 1
        print(f"  FAIL {label}" + (f"\n       {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Resolution: one case, two hardware classes, two different verdicts.
# --------------------------------------------------------------------------
BUILT = {
    "cases": [
        {
            "id": "demo",
            "frameworks": {"sglang": {"serve_args": "-a"}, "lightx2v": {"serve_args": "-b"}},
            "report_framework_statuses": {"vllm-omni": "no_profile"},
            "report_framework_reasons": {"vllm-omni": "no serving path"},
            "framework_status_by_hardware": {
                "lightx2v": {"h100": {"status": "failed", "reason": "OOM on 80GB"}},
                # the other direction: unsupported everywhere but this box
                "vllm-omni": {"b200": {"status": "supported"}},
            },
        }
    ]
}
BUILT["cases"][0]["frameworks"]["vllm-omni"] = {"serve_args": "-c"}

h100, h100_notes = apply_hardware_statuses(BUILT, ["h100"])
b200, _ = apply_hardware_statuses(BUILT, ["b200"])
unknown, unknown_notes = apply_hardware_statuses(BUILT, [])
h100_case, b200_case, unknown_case = (c["cases"][0] for c in (h100, b200, unknown))

check(
    sorted(h100_case["frameworks"]) == ["sglang"],
    "h100: the failed framework leaves `frameworks` (never attempted)",
    f"got {sorted(h100_case['frameworks'])}",
)
check(
    h100_case["report_framework_statuses"].get("lightx2v") == "failed"
    and "OOM" in h100_case["report_framework_reasons"].get("lightx2v", ""),
    "h100: it is still classified `failed`, with its reason, for the report",
    f"got {h100_case.get('report_framework_statuses')}",
)
check(
    sorted(b200_case["frameworks"]) == ["lightx2v", "sglang", "vllm-omni"],
    "b200: the same cell runs, and a per-hardware `supported` promotes vllm-omni",
    f"got {sorted(b200_case['frameworks'])}",
)
check(
    "vllm-omni" not in b200_case.get("report_framework_statuses", {})
    and h100_case["report_framework_statuses"]["vllm-omni"] == "no_profile",
    "b200: the promoted framework drops its baseline status; h100 keeps it",
)
check(
    sorted(unknown_case["frameworks"]) == ["lightx2v", "sglang", "vllm-omni"]
    and unknown_notes == [],
    "unknown hardware falls back to the baseline status, unchanged",
)
check(
    any("failed on h100" in n for n in h100_notes),
    "the resolution is announced, so a skipped cell is never silent",
    f"got {h100_notes}",
)
check(
    sorted(BUILT["cases"][0]["frameworks"]) == ["lightx2v", "sglang", "vllm-omni"],
    "resolution does not mutate the caller's config",
)
entry = {"status": "supported",
         "status_by_hardware": {"h100": {"status": "failed", "reason": "x"}}}
check(
    resolve_status(entry, ["b200", "h100"]) == ("failed", "x")
    and resolve_status(entry, ["b200"]) == ("supported", ""),
    "source-shape resolution scans every candidate, like profile selection does",
    f"got {resolve_status(entry, ['b200', 'h100'])}",
)

# --------------------------------------------------------------------------
# Build: the same schema, end to end, including what it must reject.
# --------------------------------------------------------------------------
CASE = {
    "id": "demo_case",
    "model": "acme/demo",
    "task": "text-to-image",
    "width": 1024, "height": 1024, "seed": 42, "num_gpus": 1,
    "frameworks": {
        "sglang": {"status": "supported", "serve_args": "--model-type diffusion"},
        "vllm-omni": {"status": "unsupported", "reason": "no serving path"},
        "lightx2v": {
            "status": "supported",
            "status_by_hardware": {"h100": {"status": "failed", "reason": "OOM on 80GB"}},
            "serve_args": "",
        },
        "trtllm-visual": {"status": "not_run", "reason": "pending"},
        "comfyui": {"status": "unsupported", "reason": "no workflow"},
        "fastvideo": {"status": "unsupported", "reason": "no serving path"},
    },
}


def run_build(case: dict, tmp: Path) -> subprocess.CompletedProcess:
    """Build `case` alone in a throwaway repo-shaped tree."""
    shutil.rmtree(tmp, ignore_errors=True)
    bench = tmp / "configs" / "benchmark" / "cases" / "image"
    bench.mkdir(parents=True)
    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "build_benchmark_config.py", tmp / "scripts")
    # The build imports more of the package than config_guard (the ComfyUI and
    # FastVideo clients), so ship the whole package rather than a hand-picked list.
    shutil.copytree(
        ROOT / "src" / "diffusion_bench",
        tmp / "src" / "diffusion_bench",
        ignore=shutil.ignore_patterns("__pycache__", "*.json"),
    )
    for name in ("frameworks.json", "workloads.json", "meta.json"):
        shutil.copy(ROOT / "configs" / "benchmark" / name, tmp / "configs" / "benchmark")
    (bench / f"{case['id']}.json").write_text(json.dumps(case))
    (bench.parent / "_order.json").write_text(json.dumps({"order": [case["id"]]}))
    return subprocess.run(
        [sys.executable, str(tmp / "scripts" / "build_benchmark_config.py")],
        capture_output=True, text=True,
    )


def mutate(**frameworks) -> dict:
    case = json.loads(json.dumps(CASE))
    for name, entry in frameworks.items():
        name = name.replace("_", "-")
        if entry is None:
            case["frameworks"].pop(name)
        else:
            case["frameworks"][name] = entry
    return case


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "repo"

    ok = run_build(CASE, tmp)
    check(ok.returncode == 0, "build accepts a case using status_by_hardware", ok.stderr)
    if ok.returncode == 0:
        built = json.loads((tmp / "configs" / "comparison_configs.json").read_text())
        case = built["cases"][0]
        check(
            "lightx2v" in case["frameworks"]
            and case["framework_status_by_hardware"]["lightx2v"]["h100"]["status"] == "failed",
            "the command body is published AND the per-hardware verdict travels with it",
        )
        for hardware, expected in (("h100", ["sglang"]), ("b200", ["lightx2v", "sglang"])):
            resolved, _ = apply_hardware_statuses(built, [hardware])
            got = sorted(resolved["cases"][0]["frameworks"])
            check(got == expected, f"built config runs {expected} on {hardware}", f"got {got}")
        packaged = tmp / "src" / "diffusion_bench" / "comparison_configs.json"
        check(
            json.loads(packaged.read_text()) == built
            and "git add" in ok.stdout,
            "both config copies are written, and the build says how to stage them",
        )

    # The pre-existing total-classification guard must still bite.
    missing = run_build(mutate(trtllm_visual=None), tmp)
    check(
        missing.returncode != 0 and "not classified" in missing.stderr,
        "a case that leaves a framework unclassified still fails the build",
        f"rc={missing.returncode} stderr={missing.stderr.strip()[:200]}",
    )

    # One hardware vocabulary: keys must be tokens the runtime actually derives.
    bad_token = run_build(
        mutate(lightx2v={"status": "supported", "serve_args": "",
                         "status_by_hardware": {"hopper": {"status": "failed", "reason": "x"}}}),
        tmp,
    )
    check(
        bad_token.returncode != 0 and "not a hardware token" in bad_token.stderr,
        "a hardware key outside HARDWARE_TOKENS fails the build",
        f"rc={bad_token.returncode} stderr={bad_token.stderr.strip()[:200]}",
    )

    bad_status = run_build(
        mutate(lightx2v={"status": "supported", "serve_args": "",
                         "status_by_hardware": {"h100": {"status": "oom", "reason": "x"}}}),
        tmp,
    )
    check(
        bad_status.returncode != 0 and "bad status" in bad_status.stderr,
        "an invalid per-hardware status fails the build",
        f"rc={bad_status.returncode} stderr={bad_status.stderr.strip()[:200]}",
    )

    no_reason = run_build(
        mutate(lightx2v={"status": "supported", "serve_args": "",
                         "status_by_hardware": {"h100": {"status": "failed"}}}),
        tmp,
    )
    check(
        no_reason.returncode != 0 and "no `reason`" in no_reason.stderr,
        "a per-hardware failure without a reason fails the build",
        f"rc={no_reason.returncode} stderr={no_reason.stderr.strip()[:200]}",
    )

# --------------------------------------------------------------------------
# The real config: the case this schema exists for.
# --------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "scripts"))
real = json.loads((ROOT / "configs" / "comparison_configs.json").read_text())
# Every per-hardware verdict in the repo's own matrix resolves as declared. (The
# LTX-2.3 LightX2V OOM this schema was built for left on 2026-10-22, when that
# cell became globally unsupported.)
declared = [
    (case["id"], fw, hw, verdict["status"])
    for case in real["cases"]
    for fw, by_hw in (case.get("framework_status_by_hardware") or {}).items()
    for hw, verdict in by_hw.items()
    if verdict.get("status") != "supported"
]
for case_id, fw, hw, status in declared:
    resolved, _ = apply_hardware_statuses(real, [hw])
    case = next(c for c in resolved["cases"] if c["id"] == case_id)
    check(
        fw not in case["frameworks"] and (case.get("report_framework_statuses") or {}).get(fw) == status,
        f"{case_id}: {fw} is classified {status} on {hw}, not run",
    )
print(f"  (real matrix: {len(declared)} per-hardware verdict(s) checked)")

import build_benchmark_config as bbc  # noqa: E402

check(
    set(bbc.POLICY_HARDWARE) <= set(HARDWARE_TOKENS),
    "POLICY_HARDWARE draws from the same vocabulary as status_by_hardware",
    f"stray: {sorted(set(bbc.POLICY_HARDWARE) - set(HARDWARE_TOKENS))}",
)

sys.exit(fail)
