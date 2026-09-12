#!/usr/bin/env python3
"""A source-installed framework must publish the commit it was built from.

LightX2V publishes version 0.5.0 for every main-HEAD build, so "lightx2v 0.5.0"
identifies nothing and a latest-vs-latest claim cannot be checked later. pip
records the truth in each distribution's direct_url.json, which the harness
already collects; `install_specs` does not -- it stores what was *asked* for and
falls back to a stale default whenever the run process lacks the *_INSTALL_SPEC
env vars, which it normally does.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_bench.page_export import _vcs_commit, framework_versions  # noqa: E402

fail = 0


def check(name, got, want):
    global fail
    ok = got == want
    if not ok:
        fail = 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" != {want!r}"))


GIT = {
    "direct_urls": {
        "lightx2v-0.5.0.dist-info": {
            "url": "https://github.com/ModelTC/LightX2V.git",
            "vcs_info": {"commit_id": "9bf4d39ac048a0d922b9a2139da9ed795f78f4e8"},
        }
    }
}
check("finds the commit", _vcs_commit(GIT, "lightx2v"), "9bf4d39ac048a0d922b9a2139da9ed795f78f4e8")
check("underscore/hyphen insensitive", _vcs_commit(GIT, "light_x2v") is None, True)
check("no match for another package", _vcs_commit(GIT, "flash_attn"), None)
check("empty block", _vcs_commit({}, "lightx2v"), None)

# a local editable install has direct_url.json but no vcs_info -- must not crash
LOCAL = {"direct_urls": {"vllm_omni-0.29.0.dist-info": {"url": "file:///src/vllm-omni"}}}
check("local install has no commit", _vcs_commit(LOCAL, "vllm-omni"), None)

# name normalisation the other way: dist-info uses underscores
UNDERSCORE = {
    "direct_urls": {
        "vllm_omni-0.29.0.dist-info": {
            "url": "https://github.com/vllm-project/vllm-omni.git",
            "vcs_info": {"commit_id": "7e520f9cc47e6884520ea8f3f9ab94ab97b34780"},
        }
    }
}
check("dist-info underscores match a hyphenated package",
      _vcs_commit(UNDERSCORE, "vllm-omni"), "7e520f9cc47e6884520ea8f3f9ab94ab97b34780")

# end to end through framework_versions
merged = {
    "framework_runtime": {
        "lightx2v": {"packages": {"lightx2v": {"Version": "0.5.0"}}, **GIT},
        "trtllm-visual": {"packages": {"tensorrt_llm": {"Version": "1.3.0rc26"}}},
    }
}
versions = framework_versions(merged)
check("git install shows version @ commit", versions.get("lightx2v"), "lightx2v 0.5.0 @ 9bf4d39ac")
check("released package unchanged", versions.get("trtllm-visual"), "tensorrt_llm 1.3.0rc26")

sys.exit(fail)
