#!/usr/bin/env python3
"""The profile-vs-default summary must not invent findings.

It printed `zimage_turbo_t2i_1024 0.677 0.576 -14.9% THE PIN IS HARMING` for a
case whose two arms ran the SAME command -- `--only residency` had nothing to
strip there, so the run measured the box twice and the summary read the gap as a
verdict about a pin that was never removed. The 14.9% was a bimodal flip: that
case sits at either ~0.47 or ~0.68, and one arm landed in each mode.

Two rules come out of it, both checked here:

* arms that ran the same command are a CONTROL -- their spread is the noise
  floor, and reporting it as an effect is backwards;
* under 5%, the paired per-round diffs must agree in sign before the summary is
  allowed to call anything an effect, which is the repo's standing rule for
  small latency claims.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_profile_vs_default.sh"

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


def summary_source() -> str:
    """The SUMMARY heredoc, not the per-case one that also uses <<'PY'."""
    text = RUNNER.read_text()
    after = text[text.index("=== SUMMARY ==="):]
    m = re.search(r"<<'PY'\n(.*?)\nPY\n", after, re.S)
    assert m, "the summary block moved; this test extracts it by its heredoc"
    return m.group(1)


def write_case(d: Path, prefix, case, arm, latencies, command):
    for i, lat in enumerate(latencies, start=1):
        (d / f"{prefix}_{case}_{arm}_{i}.json").write_text(
            json.dumps({"results": [{"framework": "sglang", "latency_s": lat}]})
        )
    (d / f"{prefix}_{case}_{arm}_1.runlog").write_text(
        f"some preamble\nsglang serve {command}\nmore output\n"
    )


def run(tmp: Path, cases) -> str:
    src = tmp / "summary.py"
    src.write_text(summary_source())
    out = subprocess.run(
        [sys.executable, str(src), str(tmp), "2", "pvd", *cases],
        capture_output=True, text=True,
    )
    return out.stdout + out.stderr


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)

    # 1. Same command in both arms: a control, whatever the numbers say.
    same = "--model-path M --port 1 --model-type diffusion"
    write_case(tmp, "pvd", "control_case", "profile", [0.677, 0.678], same)
    write_case(tmp, "pvd", "control_case", "default", [0.473, 0.679],
               same.replace("--port 1", "--port 3"))

    # 2. A real strip, small delta, paired diffs disagreeing in sign.
    write_case(tmp, "pvd", "noisy_case", "profile", [2.392, 2.409],
               "--model-path M --port 5 --dit-layerwise-offload false")
    write_case(tmp, "pvd", "noisy_case", "default", [2.393, 2.390],
               "--model-path M --port 7")

    # 3. A real strip, large delta, consistent sign.
    write_case(tmp, "pvd", "real_case", "profile", [3.42, 3.43],
               "--model-path M --port 9 --tp-size 2")
    write_case(tmp, "pvd", "real_case", "default", [5.76, 5.77],
               "--model-path M --port 11")

    # 4. A default arm that FAILED is the study's strongest finding -- the
    #    runtime's own choice does not work -- and "missing an arm" buries it.
    #    Five of fifteen cases were exactly that.
    write_case(tmp, "pvd", "crash_case", "profile", [6.878, 6.9],
               "--model-path M --port 13 --tp-size 2")
    (tmp / "pvd_crash_case_default_1.runlog").write_text(
        "sglang serve --model-path M --port 15\n"
        "ValueError: Tensors must be contiguous\n"
    )
    write_case(tmp, "pvd", "reject_case", "profile", [2.681, 2.7],
               "--model-path M --port 17 --tp-size 2")
    (tmp / "pvd_reject_case_default_1.runlog").write_text(
        "sglang serve --model-path M --port 19\n"
        "request does not use classifier-free guidance\n"
    )
    # 5. Our own harness limit must not read as an sglang finding.
    write_case(tmp, "pvd", "ourfault_case", "profile", [1.0, 1.0],
               "--model-path M --port 21 --tp-size 2")
    (tmp / "pvd_ourfault_case_default_1.runlog").write_text(
        "sglang serve --model-path M --port 23\nAssertionError: Invalid device id\n"
    )

    out = run(tmp, ["control_case", "noisy_case", "real_case",
                    "crash_case", "reject_case", "ourfault_case"])
    print(out.rstrip())
    print("  ---")

    ctl = next((ln for ln in out.splitlines() if "control_case" in ln), "")
    noisy = next((ln for ln in out.splitlines() if "noisy_case" in ln), "")
    real = next((ln for ln in out.splitlines() if "real_case" in ln), "")

    check("an identical-command pair is called a CONTROL", "CONTROL" in ctl, ctl.strip())
    check(
        "and is never called a harming pin",
        "THE PIN IS HARMING" not in ctl,
        "a pin that was never stripped cannot be harming",
    )
    check(
        "mixed-sign paired diffs read as noise",
        "within noise" in noisy,
        noisy.strip(),
    )
    check(
        "a large consistent gap is still reported",
        "a real sglang gap" in real,
        real.strip(),
    )
    crash = next((ln for ln in out.splitlines() if "crash_case" in ln), "")
    reject = next((ln for ln in out.splitlines() if "reject_case" in ln), "")
    ours = next((ln for ln in out.splitlines() if "ourfault_case" in ln), "")
    check("a crashing default arm says so", "CRASHES" in crash, crash.strip())
    # The profile arm still has a median worth showing (6.878 and 6.9 -> 6.889);
    # a failing default must not blank out the half that did run.
    check(
        "and keeps the profile arm's number",
        bool(re.search(r"crash_case\s+\d+\.\d+\s+FAILS", crash)),
        crash.strip(),
    )
    check("a rejecting default arm says so", "REJECTS every request" in reject, reject.strip())
    check(
        "our own GPU-count limit is not reported as an sglang result",
        "needs more GPUs than were exposed" in ours,
        ours.strip(),
    )
    check(
        "no failing arm is reported as merely missing",
        all("missing an arm" not in ln for ln in (crash, reject, ours)),
    )
    check("the port is not mistaken for a stripped flag", "CONTROL" not in noisy, noisy.strip())

sys.exit(fail)
