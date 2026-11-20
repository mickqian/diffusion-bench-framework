"""The cookbook track's GPU-count scaling rule, as code the build checks arms against.

A cookbook command written for N GPUs runs on M GPUs only one way, fixed here so
it is never decided per cell:

  1. If the source documents a section for M GPUs, the cell uses that section
     and nothing is scaled (this module is not involved).
  2. Otherwise the parallel layout is re-derived for M GPUs:
       - a source that runs CFG parallel keeps CFG 2 when M >= 2 and gives the
         rest to sequence parallel (Ulysses/USP/ring): SP = M / 2;
       - a source without CFG parallel runs pure sequence parallel: SP = M;
       - tensor parallelism of the DiT is not used (TP = 1);
       - per-component degrees that follow the GPU count (VAE patch parallel,
         text-encoder TP) are capped at M.
     Every other flag stays as the source wrote it.
  3. On datacenter GPUs an arm may additionally drop memory-saving switches
     (offload, layerwise streaming); it lists them in `scaled_from.removed`.
  4. Anything else an arm adds is listed in `scaled_from.added` with a reason,
     and the cell records the scaling and each addition as deviations.

`check_scaled_arm` re-derives the arm from `scaled_from` and reports every
difference, so a scaled command cannot be hand-written.
"""
from __future__ import annotations

import re

# Command-line parallel flags -> role, per framework whose command is a flag list.
_FLAG_ROLES = {
    "vllm-omni": {
        "--num-gpus": "n",
        "--cfg-parallel-size": "cfg",
        "--usp": "sp",
        "--ulysses-degree": "sp",
        "--ring": "ring",
        "--ring-degree": "ring",
        "--tensor-parallel-size": "tp",
        "--text-encoder-tp-size": "aux_te",
        "--vae-patch-parallel-size": "aux_vae",
    },
    "sglang": {
        "--num-gpus": "n",
        "--cfg-parallel-size": "cfg",
        "--ulysses-degree": "sp",
        "--ring-degree": "ring",
        "--sp-degree": "sp_total",
        "--tp-size": "tp",
    },
    "fastvideo": {
        "--generator.engine.parallelism.sp_size": "sp",
        "--generator.engine.parallelism.tp_size": "tp",
    },
}
_BOOL_CFG = {"sglang": "--enable-cfg-parallel"}
# A flag or config key that only trades memory for time.
OFFLOAD_RE = re.compile(r"offload|dlo|lazy_?load|unload_modules|layerwise|residency", re.I)


def parse_flags(args: str) -> list[tuple[str, str | None]]:
    """`--a 1 --b --c=x` -> [("--a","1"), ("--b",None), ("--c","x")], in order."""
    tokens = (args or "").split()
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            flag, value = tok.split("=", 1)
            out.append((flag, value))
        elif tok.startswith("--"):
            value = None
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                i += 1
            out.append((tok, value))
        else:
            out.append(("", tok))  # a stray positional; kept so it is compared too
        i += 1
    return out


def _layout_from_flags(fw: str, flags: list[tuple[str, str | None]], gpus: int) -> dict[str, int]:
    roles = _FLAG_ROLES[fw]
    lay = {"cfg": 1, "sp": 1, "ring": 1, "tp": 1, "aux_te": 1, "aux_vae": 1}
    sp_total = None
    if fw == "sglang" and not any(roles.get(f) in ("sp", "ring", "sp_total") for f, _ in flags):
        # sglang gives the GPUs no CFG or TP uses to Ulysses when no SP flag is
        # set: the cookbook's `--num-gpus 8` VDN and FastH3 commands rely on it.
        n = next((int(v) for f, v in flags if roles.get(f) == "n" and v is not None), gpus)
        tp = next((int(v) for f, v in flags if roles.get(f) == "tp" and v is not None), 1)
        cfg = 2 if any(f == _BOOL_CFG["sglang"] for f, _ in flags) else next(
            (int(v) for f, v in flags if roles.get(f) == "cfg" and v is not None), 1)
        sp_total = max(1, n // (tp * cfg))
    for flag, value in flags:
        role = roles.get(flag)
        if flag == _BOOL_CFG.get(fw):
            lay["cfg"] = 2
        elif role == "sp_total":
            sp_total = int(value)
        elif role and role != "n" and value is not None:
            lay[role] = int(value)
    lay["sp"] = sp_total if sp_total is not None else lay["sp"] * lay.pop("ring")
    lay.pop("ring", None)
    return lay


def _layout_from_config(fw: str, cfg: dict) -> dict[str, int]:
    lay = {"cfg": 1, "sp": 1, "tp": 1, "aux_te": 1, "aux_vae": 1}
    if fw == "lightx2v":
        par = cfg.get("parallel")
        if isinstance(par, dict):
            lay.update(cfg=par.get("cfg_p_size", 1), sp=par.get("seq_p_size", 1), tp=par.get("tensor_p_size", 1))
    elif fw == "trtllm-visual":
        par = cfg.get("parallel_config") or {}
        lay.update(cfg=par.get("cfg_size", 1), sp=par.get("ulysses_size", 1) * par.get("ring_size", 1),
                   tp=par.get("tp_size", 1), aux_vae=par.get("parallel_vae_size", 1))
    return {k: int(v) for k, v in lay.items()}


def scaled_layout(source: dict[str, int], gpus: int, source_gpus: int) -> dict[str, int]:
    """The rule: CFG 2 if the source runs CFG parallel and gpus >= 2, SP fills the rest, no TP.

    At the source's own GPU count nothing is re-derived: the layout is the source's.
    """
    if gpus == source_gpus:
        return dict(source)
    cfg = 2 if source["cfg"] > 1 and gpus >= 2 else 1
    return {
        "cfg": cfg,
        "sp": gpus // cfg,
        "tp": 1,
        "aux_te": min(source["aux_te"], gpus),
        "aux_vae": min(source["aux_vae"], gpus),
    }


def _diff_rest(src: dict, arm: dict, removed: list, added: list, where: str) -> list[str]:
    problems = []
    for key in sorted(set(src) | set(arm), key=str):
        s, a = src.get(key, "<absent>"), arm.get(key, "<absent>")
        if s == a:
            continue
        if key in removed:
            if not OFFLOAD_RE.search(str(key)):
                problems.append(f"{where}: `{key}` is listed as removed but is not a memory-saving switch")
            elif a not in ("<absent>", None, False, "false", "False", "none", "0", 0):
                problems.append(f"{where}: removed `{key}` still enables it ({a!r})")
            continue
        if key in added:
            continue
        problems.append(f"{where}: `{key}` is {a!r} where the rule gives the source's {s!r}")
    for key in removed:
        if src.get(key, "<absent>") == "<absent>":
            problems.append(f"{where}: `{key}` is listed as removed but the source does not set it")
    for key in added:
        if key in src and src[key] == arm.get(key):
            problems.append(f"{where}: `{key}` is listed as added but the source already has it")
        if key not in arm:
            problems.append(f"{where}: `{key}` is listed as added but the arm does not set it")
    return problems


def check_scaled_arm(fw: str, arm: dict, gpus: int, where: str) -> list[str]:
    """Problems with an arm that claims to be its `scaled_from` source under the rule."""
    src = arm.get("scaled_from") or {}
    removed = list(src.get("removed") or [])
    added = [a["flag"] if isinstance(a, dict) else a for a in src.get("added") or []]
    if any(isinstance(a, dict) and not a.get("reason") for a in src.get("added") or []) or any(
        not isinstance(a, dict) for a in src.get("added") or []
    ):
        return [f"{where}: every scaled_from.added entry is {{flag, reason}}"]
    if not isinstance(src.get("num_gpus"), int):
        return [f"{where}: scaled_from.num_gpus must be the source command's GPU count"]
    if src["num_gpus"] == gpus and not (removed or added):
        return [f"{where}: scaled_from at the source's own GPU count changes nothing; drop it"]

    if fw in _FLAG_ROLES:
        if "serve_args" not in src:
            return [f"{where}: scaled_from.serve_args missing (the source command, verbatim)"]
        s_flags, a_flags = parse_flags(src["serve_args"]), parse_flags(arm.get("serve_args", ""))
        want = scaled_layout(_layout_from_flags(fw, s_flags, src["num_gpus"]), gpus, src["num_gpus"])
        got = _layout_from_flags(fw, a_flags, gpus)
        par = set(_FLAG_ROLES[fw]) | {_BOOL_CFG.get(fw)}
        rest_src = {f: v for f, v in s_flags if f not in par}
        rest_arm = {f: v for f, v in a_flags if f not in par}
        n_flag = next((v for f, v in a_flags if _FLAG_ROLES[fw].get(f) == "n"), None)
        problems = [] if n_flag in (None, str(gpus)) else [f"{where}: --num-gpus {n_flag} != {gpus}"]
    elif fw in ("lightx2v", "trtllm-visual"):
        key = "lightx2v_config" if fw == "lightx2v" else "visual_gen_args"
        if key not in src:
            return [f"{where}: scaled_from.{key} missing (the source config, verbatim)"]
        want = scaled_layout(_layout_from_config(fw, src[key]), gpus, src["num_gpus"])
        got = _layout_from_config(fw, arm.get(key) or {})
        par_key = "parallel" if fw == "lightx2v" else "parallel_config"
        rest_src = {k: v for k, v in src[key].items() if k != par_key}
        rest_arm = {k: v for k, v in (arm.get(key) or {}).items() if k != par_key}
        problems = []
    else:
        return [f"{where}: no scaling rule for framework {fw!r}; run the source's own GPU count or make the arm `tuned`"]

    for role in ("cfg", "sp", "tp", "aux_te", "aux_vae"):
        if got[role] != want[role]:
            problems.append(f"{where}: {role} {got[role]} where the scaling rule gives {want[role]}")
    return problems + _diff_rest(rest_src, rest_arm, removed, added, where)
