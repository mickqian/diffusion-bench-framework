#!/usr/bin/env python3
"""scripts/check_competitor_recipes.py reports stale, fresh and unknown correctly.

It exists because nobody re-read a competitor's recipe: the MiniMax-H3 vLLM-Omni
cell ran the recipe's consumer-GPU command on B200 for seven weeks while the
same recipe documented a high-memory one. The failure it must never have is the
quiet one -- reporting "fresh" because the question could not be asked -- so the
API is faked here, offline, including the ways `gh` itself fails.
"""
import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_competitor_recipes as ccr  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(detail)[:400]) if detail and not cond else ''}")
    if not cond:
        fail = 1


RECIPE = {"repo": "vllm-project/vllm-omni", "path": "recipes/MiniMaxAI/MiniMax-H3.md",
          "commit": "7266fc613", "commit_date": "2026-09-28", "section": "Four GPUs"}
RECORDED = {"sha": "7266fc6138b0" + "0" * 28, "date": "2026-09-28T20:12:53Z", "subject": "[Bugfix] recorded"}
SAME_DAY_OLDER = {"sha": "1111111" + "0" * 33, "date": "2026-09-28T08:00:00Z", "subject": "older, same day"}
NEWER = {"sha": "8a97c7591d" + "0" * 30, "date": "2026-09-29T17:58:09Z", "subject": "[Perf] a newer recipe"}


def write_cases(td, recipes):
    """One case per recipe, each on a vllm-omni profile; returns the cases dir."""
    cases = pathlib.Path(td) / "cases"
    (cases / "video").mkdir(parents=True)
    for i, recipe in enumerate(recipes):
        case = {"id": f"case{i}", "frameworks": {"vllm-omni": {"status": "supported", "command_profiles": {
            "default": {"upstream_recipe": recipe}, "rtx5090-2gpu": {"hardware": ["rtx5090"]}}}}}
        (cases / "video" / f"case{i}.json").write_text(json.dumps(case))
    (cases / "_order.json").write_text("{}")
    return str(cases)


def run(recipes, list_commits):
    with tempfile.TemporaryDirectory() as td:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ccr.main(["--cases-dir", write_cases(td, recipes)], list_commits=list_commits)
        return code, out.getvalue()


calls = []


def api(commits):
    def list_commits(repo, path, since):
        calls.append((repo, path, since))
        return commits
    return list_commits


def broken(repo, path, since):
    raise RuntimeError("HTTP 502 from api.github.com")


code, out = run([RECIPE, RECIPE], api([RECORDED, SAME_DAY_OLDER]))
check("fresh: exit 0", code == 0, out)
check("fresh: both profiles say fresh", out.count("fresh") >= 2, out)
check("the query is the recipe's path since its recorded date",
      calls[-1] == (RECIPE["repo"], RECIPE["path"], "2026-09-28T00:00:00Z"), calls)
check("two profiles on one recipe cost one query", len(calls) == 1, calls)

code, out = run([RECIPE], api([NEWER, RECORDED, SAME_DAY_OLDER]))
check("stale: exit 1", code == 1, out)
check("stale: names the newer commit with its date and subject",
      "8a97c7591d00 2026-09-29 [Perf] a newer recipe" in out and "STALE: 1 newer" in out, out)
check("stale: an older commit from the same day is not counted", "older, same day" not in out, out)

code, out = run([RECIPE], broken)
check("API failure: exit 2", code == 2, out)
check("API failure: reported UNKNOWN, never fresh",
      "UNKNOWN" in out and not any(line.endswith("  fresh") for line in out.splitlines()), out)

code, out = run([RECIPE], api([NEWER]))
check("recorded commit not found: exit 2", code == 2 and "not among" in out, out)

code, out = run([RECIPE, {**RECIPE, "path": "recipes/other.md"}],
                lambda repo, path, since: [NEWER, RECORDED] if path == RECIPE["path"] else broken(repo, path, since))
check("stale plus a failed query: exit 2 (unknown outranks stale)", code == 2 and "STALE" in out, out)

code, out = run([{**RECIPE, "commit_date": "Sep 28"}], api([RECORDED]))
check("malformed upstream_recipe: exit 2", code == 2 and "YYYY-MM-DD" in out, out)

code, out = run([], api([]))
check("no recorded recipes: exit 0 and says so", code == 0 and "no profile records" in out, out)


# The real gh path, with subprocess stubbed: its failures must surface as UNKNOWN.
def stub_run(returncode=0, stdout="", stderr="", raises=None):
    def fake(cmd, **kwargs):
        stub_run.cmd = cmd
        if raises:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return fake


real_run = ccr.subprocess.run
try:
    tsv = f"{NEWER['sha']}\t{NEWER['date']}\t{NEWER['subject']}\n{RECORDED['sha']}\t{RECORDED['date']}\t{RECORDED['subject']}\n"
    ccr.subprocess.run = stub_run(stdout=tsv)
    code, out = run([RECIPE], ccr.gh_commits)
    check("gh: parses the paginated TSV into a stale verdict", code == 1 and "a newer recipe" in out, out)
    check("gh: asks for the path since the recorded date, paginated",
          "--paginate" in stub_run.cmd and any("since=2026-09-28T00%3A00%3A00Z" in a and "path=recipes%2FMiniMaxAI" in a
                                               for a in stub_run.cmd), stub_run.cmd)
    for label, fake in (
        ("non-zero exit", stub_run(returncode=1, stderr="HTTP 403: API rate limit exceeded")),
        ("not installed", stub_run(raises=FileNotFoundError("gh"))),
        ("timeout", stub_run(raises=subprocess.TimeoutExpired("gh", 120))),
    ):
        ccr.subprocess.run = fake
        code, out = run([RECIPE], ccr.gh_commits)
        check(f"gh {label}: exit 2, UNKNOWN", code == 2 and "UNKNOWN" in out, out)
finally:
    ccr.subprocess.run = real_run

sys.exit(fail)
