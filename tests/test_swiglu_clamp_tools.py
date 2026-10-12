"""N24 SwiGLU clamp experiment tools: tools/quality/swiglu_gate_stats.py on SYNTHETIC dumps, the experiment driver's
usage path, and the patch's own consistency (stdlib only).

EVERY DUMP HERE IS SYNTHETIC TEST DATA, NOT EVIDENCE.
Run: python3 -m unittest tests.test_swiglu_clamp_tools -v   (from the repo root)
"""
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "quality"))
import swiglu_gate_stats as gs  # noqa: E402

SCRIPT = os.path.join(ROOT, "tools", "quality", "swiglu_clamp_experiment.sh")
PATCH = os.path.join(ROOT, "patches", "llama.cpp", "n24-swiglu-clamp-after-silu.patch")


def write_dump(d, name, il, k, values, ne=None):
    a = array.array("f", values)
    with open(os.path.join(d, "%s-%d__%d.bin" % (name, il, k)), "wb") as f:
        f.write(a.tobytes())
    with open(os.path.join(d, "%s-%d__%d.json" % (name, il, k)), "w") as f:
        json.dump({"type": "f32", "ggml_type": "f32", "ne": ne or [len(values), 1, 1, 1]}, f)


def pre(g, u, L):
    return gs.silu(min(g, L)) * max(-L, min(L, u))


def post(g, u, L):
    return min(gs.silu(g), L) * max(-L, min(L, u))


class GateStats(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="swiglu-SYNTHETIC-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_counts_and_largest_difference(self):
        g = [-3.0, 0.5, 9.99, 10.5, 12.0, 30.0]
        u = [1.0, -2.0, 4.0, 3.0, -50.0, 0.25]
        write_dump(self.d, "ffn_moe_gate", 5, 0, g)
        write_dump(self.d, "ffn_moe_up", 5, 0, u)
        write_dump(self.d, "ffn_gate", 1, 0, [0.1, 0.2])            # no up: counted, no output difference
        r = gs.stats(self.d, 10.0)
        s = r["kinds"]["ffn_moe_gate"]
        self.assertEqual((s["tensors"], s["elements"], s["above_limit"]), (1, 6, 3))
        self.assertAlmostEqual(s["share_above"], 0.5)
        self.assertAlmostEqual(s["max_gate"], 30.0)
        want = max(abs(post(x, y, 10.0) - pre(x, y, 10.0)) for x, y in zip(g, u))
        self.assertAlmostEqual(s["max_abs_output_diff"], want, places=6)
        self.assertGreater(want, 0)
        self.assertLessEqual(want, 10.0 * r["bound_gate_factor"] + 1e-9)   # |clamp(u)| <= L times the gate bound
        self.assertEqual(s["layers_above"], [{"layer": 5, "k": 0, "above": 3}])
        self.assertEqual(r["kinds"]["ffn_gate"]["above_limit"], 0)
        self.assertAlmostEqual(r["bound_gate_factor"], 10.0 / (1 + math.exp(10.0)))

    def test_semantics_equal_below_the_limit(self):
        for x in (-20.0, -1.0, 0.0, 1.0, 5.0, 9.999, 10.0):
            for y in (-12.0, 0.3, 11.0):
                self.assertAlmostEqual(pre(x, y, 10.0), post(x, y, 10.0), places=12, msg=(x, y))

    def test_cli_and_wrong_type(self):
        write_dump(self.d, "ffn_moe_gate", 0, 0, [11.0])
        out = os.path.join(self.d, "s.json")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "quality", "swiglu_gate_stats.py"), self.d,
                            "--limit", "10", "--out", out], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gate > 10: 1", p.stdout)
        with open(os.path.join(self.d, "ffn_moe_gate-0__0.json"), "w") as f:
            json.dump({"type": "i32", "ne": [1, 1, 1, 1]}, f)
        with self.assertRaises(ValueError):
            gs.stats(self.d, 10.0)
        empty = tempfile.mkdtemp(prefix="swiglu-empty-SYNTHETIC-", dir=self.d)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "quality", "swiglu_gate_stats.py"), empty],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)


class Driver(unittest.TestCase):
    def test_syntax_and_usage(self):
        self.assertEqual(subprocess.run(["bash", "-n", SCRIPT]).returncode, 0)
        p = subprocess.run(["bash", SCRIPT], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("SwiGLU clamp semantics", p.stderr)

    def test_patch_switches_only_the_clamp_order(self):
        with open(PATCH) as f:
            text = f.read()
        body = text[text.index("diff -ru"):]
        minus = [l[1:].strip() for l in body.splitlines() if l.startswith("-") and not l.startswith("---")]
        plus = [l[1:].strip() for l in body.splitlines() if l.startswith("+") and not l.startswith("+++")]
        self.assertEqual(minus, [
            "const float gate = std::min(src0_p[k], limit);",
            "dst_p[k]         = gate / (1.f + expf(-gate)) * up;",
            "const float gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);",
            "dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);",
            "gate = fminf(gate, limit);",
            "return ggml_cuda_op_silu_single(gate) * up;"])
        self.assertEqual(sum("N24 experiment" in l for l in plus), 3)
        self.assertEqual(sorted({l.split()[1] for l in body.splitlines() if l.startswith("--- ")}),
                         ["a/ggml/src/ggml-cpu/ops.cpp", "a/ggml/src/ggml-cuda/unary.cuh"])


if __name__ == "__main__":
    unittest.main()
