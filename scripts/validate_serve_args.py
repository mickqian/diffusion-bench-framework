"""Parse every sglang serve_args in the benchmark config with sglang's own CLI
parser. A retired or renamed flag fails here instead of burning a 40-minute
nightly slot per case waiting on a health check that will never pass.

Usage: python3 validate_serve_args.py <comparison_configs.json>
"""

import argparse
import json
import sys

from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs


def collect(cfg):
    """(case_id, profile, serve_args) for every sglang entry in the config."""
    out = []
    for case in cfg.get("cases", []):
        cid = case.get("id", "?")
        fw = (case.get("frameworks") or {}).get("sglang")
        if not isinstance(fw, dict):
            continue
        if isinstance(fw.get("serve_args"), str):
            out.append((cid, "<inline>", fw["serve_args"]))
        for pname, prof in (fw.get("command_profiles") or {}).items():
            if isinstance(prof, dict) and isinstance(prof.get("serve_args"), str):
                out.append((cid, pname, prof["serve_args"]))
    return out


def main() -> int:
    cfg = json.load(open(sys.argv[1]))
    entries = collect(cfg)
    parser = argparse.ArgumentParser(prog="sglang serve", add_help=False)
    ServerArgs.add_cli_args(parser)

    bad = []
    for cid, prof, args in entries:
        try:
            parser.parse_known_args(
                ["--model-path", "dummy"] + args.split(), namespace=None
            )
        except SystemExit:
            bad.append((cid, prof, args))
        except Exception as exc:  # a parser bug is not a config bug
            bad.append((cid, prof, f"{args}   [{type(exc).__name__}: {exc}]"))

    print(f"checked {len(entries)} sglang serve_args entries")
    if bad:
        print(f"\n{len(bad)} REJECTED by the sglang parser:")
        for cid, prof, args in bad:
            print(f"  {cid} / {prof}\n     {args}")
        return 1
    print("all accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
