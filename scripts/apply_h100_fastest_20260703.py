#!/usr/bin/env python3
"""Write the 2026-07-03 measured-fastest H100 sglang commands into the case files.

Evidence (H100, steady-state single_e2e via the harness, A/B pairs, logs on the
run devbox under /persistent/logs/sab_*.runlog):

- zimage:  auto-offload default 0.793/0.799s; speed+compile 0.987/0.997s;
           speed eager (resident, no compile) 0.701/0.699s  -> speed-eager wins
- wan21:   default 26.556s -> speed+compile 25.046s (-5.7%)
- wan22:   default(+textenc offload) 35.07s -> speed+compile 31.068s (-11.4%)
- cosmos:  default 55.115s -> speed+compile 53.614s (-2.7%)
- ltx2.3:  speed+compile OOMs (79.1/79.2 GiB on both GPUs) -> keep snapshot,
           record a policy_exception

Un-run cases (not in the published matrix) get policy-aligned candidate
commands (offload flags stripped, compile on) marked "validate before
publishing" — they publish nothing until actually run.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(REPO, "configs", "benchmark", "cases")

SPEED_TRIO = " --dit-layerwise-offload false --performance-mode speed --enable-torch-compile"

MEASURED = "Measured H100 A/B 2026-07-03 (steady-state single_e2e)"


def load(rel):
    path = os.path.join(CASES, rel)
    with open(path) as f:
        return path, json.load(f)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("updated", os.path.relpath(path, REPO))


def sglang_profiles(case):
    return case["frameworks"]["sglang"]["command_profiles"]


# --- zimage: speed-eager wins; compile measured slower twice (B200-era and now) ---
path, case = load("image/zimage_turbo_t2i_1024.json")
profiles = sglang_profiles(case)
profiles["h100-h200-2gpu-tp-resident-eager"] = {
    "description": "Hopper 2-GPU TP, resident components, eager DiT — measured fastest.",
    "hardware": ["h100", "h200"],
    "sglang_ref": "current-main",
    "serve_args": (
        "--model-type diffusion --warmup --tp-size 2 "
        "--dit-layerwise-offload false --performance-mode speed "
        "--enable-torch-compile false"
    ),
    "policy_exception": (
        f"{MEASURED}: speed+compile 0.987/0.997s vs speed-eager 0.701/0.699s vs "
        "auto-offload default 0.793/0.799s — compile loses on this 9-step turbo "
        "model. Explicit --enable-torch-compile false also future-proofs against "
        "speed-mode auto-compile (sglang PR #30016)."
    ),
    "notes": "Validated on H100; hardware list includes h200 for selection parity — re-validate there before publishing h200 numbers.",
}
save(path, case)

# --- wan21 t2v: speed+compile wins ---
path, case = load("video/wan21_t2v_1_3b_480p.json")
profiles = sglang_profiles(case)
base = profiles["default"]["serve_args"]
profiles["h100-2gpu-cfg-speed-compile"] = {
    "description": "Hopper 2-GPU CFG-parallel + resident + torch.compile — measured fastest.",
    "hardware": ["h100"],
    "sglang_ref": "current-main",
    "serve_args": base + SPEED_TRIO,
    "notes": f"{MEASURED}: 26.556s (auto default) -> 25.046s (-5.7%).",
}
save(path, case)

# --- wan22 ti2v: drop textenc offload, add speed trio ---
path, case = load("video/wan22_ti2v_5b_704p.json")
profiles = sglang_profiles(case)
p = profiles["h100-h200-2gpu"]
p["serve_args"] = (
    p["serve_args"].replace(" --text-encoder-cpu-offload", "") + SPEED_TRIO
)
p["description"] = (
    "Hopper 2-GPU CFG-parallel, all components resident, torch.compile — measured fastest."
)
p["notes"] = (
    f"{MEASURED}: 35.07s (textenc-offload default) -> 31.068s (-11.4%). "
    "Measured on H100; re-validate before publishing new h200 numbers."
)
save(path, case)

# --- cosmos t2v: speed+compile wins ---
path, case = load("video/cosmos3_nano_t2v_720p_189f.json")
profiles = sglang_profiles(case)
base = profiles["default"]["serve_args"]
profiles["h100-4gpu-cfg-ulysses-speed-compile"] = {
    "description": "Hopper 4-GPU CFG+Ulysses, resident + torch.compile — measured fastest.",
    "hardware": ["h100"],
    "sglang_ref": "current-main",
    "serve_args": base + SPEED_TRIO,
    "notes": f"{MEASURED}: 55.115s (auto default) -> 53.614s (-2.7%).",
}
save(path, case)

# --- ltx2.3: speed+compile OOMs; snapshot default stays, with evidence ---
path, case = load("video/ltx2.3_twostage_t2v_2gpus.json")
profiles = sglang_profiles(case)
profiles["h100-80gb-2gpu"]["policy_exception"] = (
    f"{MEASURED}: speed+compile OOMs on 2x80GB (79.1/79.2 GiB used, 30MiB alloc "
    "failure during request) — two-stage residency + compile workspace does not "
    "fit. Snapshot device mode is the fastest configuration that fits (12.03s)."
)
save(path, case)

# --- un-run cases: policy-aligned candidates (nothing published from these yet) ---
CANDIDATE_NOTE = (
    "Policy-aligned candidate (compile on, offload off) — NOT yet validated on "
    "H100; validate (incl. OOM risk) before publishing."
)
for rel, profile_name in (
    ("video/wan21_i2v_14b_480p.json", "default"),
    ("video/wan21_i2v_14b_720p.json", "default"),
    ("video/wan22_i2v_a14b_720p.json", "default"),
    ("video/wan22_t2v_a14b_720p.json", "default"),
    ("video/ltx2_twostage_t2v.json", "default"),
    ("video/cosmos3_nano_i2v_720p_189f.json", "default"),
    ("image/cosmos3_nano_t2i_720p.json", "default"),
):
    path, case = load(rel)
    sg = case["frameworks"]["sglang"]
    if sg.get("status") != "supported":
        print("skip (not supported):", rel)
        continue
    profiles = sglang_profiles(case)
    p = profiles[profile_name]
    args = p["serve_args"].replace(" --text-encoder-cpu-offload", "")
    if "--enable-torch-compile" not in args:
        args += SPEED_TRIO
    p["serve_args"] = args
    p["notes"] = (p.get("notes", "") + " " + CANDIDATE_NOTE).strip()
    save(path, case)

print("done")
