#!/usr/bin/env python3
"""Which competitor commands were derived from a recipe that has since changed?

A profile's `upstream_recipe` ({repo, path, commit, commit_date, section}) names
the upstream recipe its command was derived from. This asks GitHub for commits
to that path newer than the recorded one. Run it before a round, and re-derive a
stale command from the current recipe before measuring: MiniMax-H3's vLLM-Omni
cell ran the recipe's 2x24/32 GB consumer command on 183 GB B200s from
2026-08-07 to 2026-09-25, while the same recipe had documented a high-memory
no-offload, compile-on configuration since its first version (2026-08-03).

Exit 0: every recorded recipe is current. 1: at least one is stale.
2: a query failed or a recorded commit could not be found -- unknown is not
fresh, so this outranks 1.

    python3 scripts/check_competitor_recipes.py
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from diffusion_bench.config_guard import upstream_recipe_problems  # noqa: E402

CASES = os.path.join(REPO, "configs", "benchmark", "cases")


def gh_commits(repo: str, path: str, since: str) -> list[dict]:
    """Commits to `path` on the default branch since `since`, newest first."""
    query = urllib.parse.urlencode({"path": path, "since": since, "per_page": 100})
    cmd = [
        "gh", "api", "--paginate", f"repos/{repo}/commits?{query}",
        "--jq", r'.[] | [.sha, .commit.committer.date, (.commit.message | split("\n")[0])] | @tsv',
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gh api failed: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"gh api exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    commits = []
    for line in proc.stdout.splitlines():
        sha, date, subject = (line.split("\t", 2) + ["", ""])[:3]
        commits.append({"sha": sha, "date": date, "subject": subject})
    return commits


def recipe_profiles(cases_dir: str) -> list[tuple[str, str, str, dict]]:
    """(case, framework, profile, upstream_recipe) for every profile that records one."""
    found = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(path).startswith("_"):
            continue
        case = json.load(open(path))
        for fw, entry in (case.get("frameworks") or {}).items():
            for name, source in [("inline", entry), *(entry.get("command_profiles") or {}).items()]:
                if "upstream_recipe" in source:
                    found.append((case["id"], fw, name, source["upstream_recipe"]))
    return found


def newer_commits(recipe: dict, list_commits) -> list[dict]:
    """Commits to the recipe's path after its recorded commit; raises if that
    commit is not among the path's commits since its recorded date."""
    commits = list_commits(recipe["repo"], recipe["path"], f"{recipe['commit_date']}T00:00:00Z")
    for index, commit in enumerate(commits):
        if commit["sha"].startswith(recipe["commit"]):
            return commits[:index]
    raise RuntimeError(
        f"recorded commit {recipe['commit']} is not among the {len(commits)} commit(s) to "
        f"{recipe['path']} since {recipe['commit_date']} -- fix upstream_recipe.commit/commit_date"
    )


def main(argv=None, list_commits=gh_commits) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases-dir", default=CASES)
    args = parser.parse_args(argv)

    rows, stale, failed = [], 0, 0
    answers = {}  # (repo, path, commit, commit_date) -> newer commits, or the error
    for cid, fw, name, recipe in recipe_profiles(args.cases_dir):
        problems = upstream_recipe_problems(recipe)
        if problems:
            failed += 1
            rows.append((cid, fw, name, "?", f"UNKNOWN: {'; '.join(problems)}", []))
            continue
        key = (recipe["repo"], recipe["path"], recipe["commit"], recipe["commit_date"])
        if key not in answers:
            try:
                answers[key] = newer_commits(recipe, list_commits)
            except Exception as exc:  # any failure to ask is "unknown", never "fresh"
                answers[key] = exc
        recorded = f"{recipe['commit']} {recipe['commit_date']}"
        answer = answers[key]
        if isinstance(answer, Exception):
            failed += 1
            rows.append((cid, fw, name, recorded, f"UNKNOWN: {answer}", []))
        elif answer:
            stale += 1
            rows.append((cid, fw, name, recorded, f"STALE: {len(answer)} newer commit(s) to {recipe['path']}", answer))
        else:
            rows.append((cid, fw, name, recorded, "fresh", []))

    if not rows:
        print("no profile records an upstream_recipe")
        return 0
    widths = [max(len(str(r[i])) for r in rows + [("case", "framework", "profile", "recorded")]) for i in range(4)]
    print("  ".join(h.ljust(w) for h, w in zip(("case", "framework", "profile", "recorded"), widths)) + "  result")
    for cid, fw, name, recorded, result, newer in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip((cid, fw, name, recorded), widths)) + "  " + result)
        for commit in newer:
            print(f"      {commit['sha'][:12]} {commit['date'][:10]} {commit['subject']}")
    print(f"\n{len(rows)} profile(s): {len(rows) - stale - failed} fresh, {stale} stale, {failed} unknown")
    if stale:
        print("Re-derive each stale command from its current recipe for the target hardware "
              "before measuring, then update the profile's upstream_recipe.")
    return 2 if failed else (1 if stale else 0)


if __name__ == "__main__":
    sys.exit(main())
