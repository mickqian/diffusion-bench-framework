#!/usr/bin/env python3
"""Describe how MP4s were encoded: geometry, pix_fmt, audio, and the x264 preset/CRF read from the stream.

x264 writes its full options string into every stream it encodes, so the preset
(identified by me/subme/ref/bframes) and the CRF can be read off any output,
whatever wrote it. That is how every video row's encode is checked against its
case (the MP4 encode rule in the benchmarking SKILL's fairness checklist).

    scripts/probe_mp4_encode.py [--since EPOCH] [--group] [--label TEXT] <file-or-dir> ...

One JSON line per file, or per distinct (directory, geometry, encode) with --group.
Needs ffprobe on PATH.
"""
import json
import os
import re
import subprocess
import sys

# (me, subme, ref, bframes) of each x264 preset
PRESETS = {
    ("dia", 0, 1, 0): "ultrafast",
    ("dia", 1, 1, 0): "superfast",
    ("hex", 2, 1, 3): "veryfast",
    ("hex", 4, 2, 3): "faster",
    ("hex", 6, 2, 3): "fast",
    ("hex", 7, 3, 3): "medium",
    ("umh", 8, 5, 3): "slow",
}


def x264_options(data: bytes) -> dict | None:
    m = re.search(rb"x264 - core (\d+)[^\x00]*?options: ([^\x00]+)", data)
    if not m:
        return None
    opts = dict(kv.split("=", 1) for kv in m.group(2).decode(errors="replace").split() if "=" in kv)
    key = (opts.get("me"), int(opts.get("subme", -1)), int(opts.get("ref", -1)), int(opts.get("bframes", -1)))
    return {
        "core": int(m.group(1)),
        "preset": PRESETS.get(key, f"unknown{key}"),
        "crf": opts.get("crf"),
        "rc": opts.get("rc"),
        "threads": opts.get("threads"),
    }


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True,
    )
    info = json.loads(out.stdout or "{}")
    rec = {"file": path, "bytes": os.path.getsize(path), "mtime": int(os.path.getmtime(path)),
           "container": (info.get("format") or {}).get("format_name")}
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            rec.update(v_codec=s.get("codec_name"), pix_fmt=s.get("pix_fmt"), size=f"{s.get('width')}x{s.get('height')}",
                       frames=s.get("nb_frames"), fps=s.get("r_frame_rate"), v_kbps=round(int(s.get("bit_rate", 0)) / 1000))
        elif s.get("codec_type") == "audio":
            rec.update(a_codec=s.get("codec_name"), a_rate=s.get("sample_rate"), a_ch=s.get("channels"),
                       a_kbps=round(int(s.get("bit_rate", 0)) / 1000))
    with open(path, "rb") as f:
        rec["x264"] = x264_options(f.read())
    return rec


def main(argv: list[str]) -> None:
    since, group, label, paths = 0, False, None, []
    it = iter(argv)
    for arg in it:
        if arg == "--since":
            since = int(next(it))
        elif arg == "--label":
            label = next(it)
        elif arg == "--group":
            group = True
        else:
            paths.append(arg)
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(root, n) for root, _, names in os.walk(p) for n in names if n.endswith(".mp4")]
        elif os.path.isfile(p):
            files.append(p)
    groups: dict = {}
    for f in sorted(f for f in files if os.path.getmtime(f) >= since):
        try:
            rec = probe(f)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            rec = {"file": f, "error": str(exc)}
        if label:
            rec["label"] = label
        if not group:
            print(json.dumps(rec))
            continue
        x = rec.get("x264") or {}
        sig = (os.path.dirname(f), rec.get("size"), rec.get("frames"), rec.get("pix_fmt"), x.get("preset"),
               x.get("crf"), rec.get("a_codec"), rec.get("a_rate"))
        groups.setdefault(sig, dict(rec, count=0))["count"] += 1
    for rec in groups.values():
        print(json.dumps(rec))


if __name__ == "__main__":
    main(sys.argv[1:])
