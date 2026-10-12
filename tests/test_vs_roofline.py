"""tools/bench/vs_roofline.py on a SYNTHETIC run dir (format of tools/bench/run_matrix.py)."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vs_roofline as vr  # noqa: E402
import roofline  # noqa: E402


class VsRoofline(unittest.TestCase):
    def test_ratio_against_direct_roofline(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            os.makedirs(run)
            cells = [{"concurrency": 1, "input": 32768, "output": 512, "state": "cold",
                      "metrics": {"decode_tok_s": {"p50": 20.0}, "prompt_tokens": {"p50": 32768},
                                  "completion_tokens": {"p50": 512}}},
                     {"concurrency": 4, "input": 32768, "output": 512, "state": "cold", "metrics": {}}]
            with open(os.path.join(run, "summary.json"), "w") as f:
                json.dump({"cells": cells}, f)
            with open(os.path.join(run, "meta.json"), "w") as f:
                json.dump({"is_mock_server": False, "engine_desc": "SYNTHETIC"}, f)
            spec_path = os.path.join(ROOT, "models", "specs", "glm-5.3-flash.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                vr.main([run, "--spec", spec_path, "--hit", "0.7", "--json"])
            rows = json.loads(buf.getvalue())
            self.assertEqual(len(rows), 1)                   # concurrency 4 is not single-stream
            with open(spec_path) as f:
                spec = json.load(f)
            with open(os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json")) as f:
                hw = json.load(f)
            q = {"expert_bpw": 4.5, "dense_bpw": 8.5, "kv_bytes_per_elem": 2.0}
            bound = roofline.decode(spec, hw, q, 32768 + 256, 0.7)["tok_s"]
            self.assertAlmostEqual(rows[0]["bound"], bound)
            self.assertAlmostEqual(rows[0]["ratio"], 20.0 / bound)

    def test_strategy_miss_and_mock_flag_are_passed_through(self):
        # mutations V3, V4 of docs/evidence/2026-10-09-tool-mutations.md survived the default-only test above
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            os.makedirs(run)
            cells = [{"concurrency": 1, "input": 32768, "output": 512, "state": "cold",
                      "metrics": {"decode_tok_s": {"p50": 10.0}, "prompt_tokens": {"p50": 32768},
                                  "completion_tokens": {"p50": 512}}}]
            with open(os.path.join(run, "summary.json"), "w") as f:
                json.dump({"cells": cells}, f)
            with open(os.path.join(run, "meta.json"), "w") as f:
                json.dump({"is_mock_server": True, "engine_desc": "SYNTHETIC"}, f)
            spec_path = os.path.join(ROOT, "models", "specs", "glm-5.3-flash.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                vr.main([run, "--spec", spec_path, "--hit", "0.7", "--strategy", "tp", "--miss", "pcie", "--json"])
            rows = json.loads(buf.getvalue())
            with open(spec_path) as f:
                spec = json.load(f)
            with open(os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json")) as f:
                hw = json.load(f)
            q = {"expert_bpw": 4.5, "dense_bpw": 8.5, "kv_bytes_per_elem": 2.0}
            want = roofline.decode(spec, hw, q, 32768 + 256, 0.7, strategy="tp", miss="pcie")["tok_s"]
            default = roofline.decode(spec, hw, q, 32768 + 256, 0.7)["tok_s"]
            self.assertGreater(abs(want - default), 1e-6 * default)   # the case distinguishes tp/pcie from pp/cpu
            self.assertAlmostEqual(rows[0]["bound"], want)
            self.assertTrue(rows[0]["mock"])

    def test_engine_mode_flags_envelope_and_inputs(self):
        # 2026-10-10 evidence audit F1/F2: compare with the engine as it ran (speculation, overlap), bound = envelope
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            os.makedirs(run)
            cells = [{"concurrency": 1, "input": 261888, "output": 512, "state": "cold",
                      "metrics": {"decode_tok_s": {"p50": 30.0}, "prompt_tokens": {"p50": 261888},
                                  "completion_tokens": {"p50": 512}}}]
            with open(os.path.join(run, "summary.json"), "w") as f:
                json.dump({"cells": cells}, f)
            with open(os.path.join(run, "meta.json"), "w") as f:
                json.dump({"is_mock_server": False, "engine_desc": "SYNTHETIC"}, f)
            spec_path = os.path.join(ROOT, "models", "specs", "glm-5.3-flash.json")
            base = [run, "--spec", spec_path, "--hit", "0.9", "--expert-bpw", "4.89", "--dense-bpw", "8.04", "--json"]

            def go(*extra):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    vr.main(base + list(extra))
                return json.loads(buf.getvalue())[0]
            plain, spec2, serial = go(), go("--spec-k", "2", "--spec-accept", "0.7"), go("--overlap", "no")
            self.assertGreater(spec2["bound"], plain["bound"] + 1.0)
            with open(spec_path) as f:
                spec = json.load(f)
            with open(os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json")) as f:
                hw = json.load(f)
            q = {"expert_bpw": 4.89, "dense_bpw": 8.04, "kv_bytes_per_elem": 2.0}
            want = roofline.decode(spec, hw, q, 261888 + 256, 0.9, spec_k=2, spec_accept=0.7)["tok_s"]
            self.assertAlmostEqual(spec2["bound_at_hit"], want)                      # the value at the hit, with drafts
            self.assertLess(serial["bound"], plain["bound"])
            self.assertAlmostEqual(serial["bound"], serial["bound_at_hit"])          # serial bound is monotone
            self.assertGreater(spec2["bound"], spec2["bound_at_hit"])                # overlap peak below hit 0.9
            self.assertLess(spec2["bound_hit"], 0.9)
            self.assertEqual(spec2["inputs"]["spec_k"], 2)
            self.assertEqual(serial["inputs"]["overlap"], "no")
            self.assertAlmostEqual(spec2["ratio"], 30.0 / spec2["bound"])


if __name__ == "__main__":
    unittest.main()
