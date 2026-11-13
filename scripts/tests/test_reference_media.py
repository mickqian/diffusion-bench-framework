#!/usr/bin/env python3
"""Cells with several references, or a reference video, name their files in `reference_media`.

`__MEDIA:<name>__` in a framework's request extras becomes the file inline as a data URI,
`__MEDIA_PATH:<name>__` its host path (LightX2V's video API takes only local paths), and a
vLLM-Omni entry's `reference_uploads` sends each file as one `input_references` part. A
`{variant}` in a path takes the request's input variant, so every request conditions on
files of its own, as the marked reference image does.
"""
import base64
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from diffusion_bench import run_comparison as rc  # noqa: E402


class ReferenceMediaTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for variant in (0, 1):
            for name, ext, body in (("img1", "png", b"png-%d"), ("vid1", "mp4", b"mp4-%d")):
                with open(os.path.join(self.dir.name, f"{name}_{variant:02d}.{ext}"), "wb") as f:
                    f.write(body % variant)
        d = self.dir.name
        self.case = {
            "id": "mr",
            "input_variant": 1,
            "reference_media": {"img1": f"{d}/img1_{{variant}}.png", "vid1": f"{d}/vid1_{{variant}}.mp4"},
            "framework_request_extra": {
                "sglang": {
                    "conditions": [
                        {"type": "image", "uri": "__MEDIA:img1__", "role": "reference"},
                        {"type": "video", "uri": "__MEDIA:vid1__", "role": "reference"},
                    ],
                    "_note": "comments are dropped, tokens and all: __MEDIA:absent__",
                },
                "lightx2v": {"task": "ref2av", "image_path": "__MEDIA_PATH:img1__", "video_path": "__MEDIA_PATH:vid1__"},
            },
            "frameworks": {"vllm-omni": {"reference_uploads": ["img1", "vid1"]}},
        }

    def tearDown(self):
        self.dir.cleanup()

    def test_data_uri_per_request_variant(self):
        conds = rc._request_extra(self.case, "sglang")["conditions"]
        self.assertEqual(conds[0]["uri"], "data:image/png;base64," + base64.b64encode(b"png-1").decode())
        self.assertEqual(conds[1]["uri"], "data:video/mp4;base64," + base64.b64encode(b"mp4-1").decode())
        first = rc._request_extra({**self.case, "input_variant": 0}, "sglang")["conditions"][0]["uri"]
        self.assertNotEqual(first, conds[0]["uri"])

    def test_host_paths_for_lightx2v(self):
        payload = rc._lightx2v_payload({**self.case, "prompt": "p"}, {}, is_image=False)
        self.assertEqual(payload["image_path"], f"{self.dir.name}/img1_01.png")
        self.assertEqual(payload["video_path"], f"{self.dir.name}/vid1_01.mp4")

    def test_vllm_parts_in_order(self):
        parts = rc._media_uploads(self.case, "input_references", ["img1", "vid1"])
        self.assertEqual([p[0] for p in parts], ["input_references"] * 2)
        self.assertEqual([(p[1][0], p[1][2]) for p in parts], [("img1_01.png", "image/png"), ("vid1_01.mp4", "video/mp4")])
        self.assertEqual(parts[1][1][1].read(), b"mp4-1")

    def test_missing_name_or_file_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "absent from reference_media"):
            rc._media_uploads(self.case, "input_references", ["img2"])
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            rc._media_uploads({**self.case, "input_variant": 7}, "input_references", ["img1"])


if __name__ == "__main__":
    unittest.main()
