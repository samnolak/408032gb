"""tools/g0/analyze.py on synthetic routing with known answers."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "g0"))
sys.path.insert(0, str(ROOT / "tools"))
import analyze  # noqa: E402
import fitplan  # noqa: E402


def write_trace(path, layers, n_experts, k, n_tok, weights, rng):
    with open(path, "w") as f:
        f.write(f"# trace={path.name} tokens={n_tok}\nlayer,expert,count\n")
        for layer in layers:
            counts = np.zeros(n_experts, dtype=int)
            for _ in range(n_tok):
                counts[rng.choice(n_experts, size=k, replace=False, p=weights)] += 1
            for e, c in enumerate(counts):
                if c:
                    f.write(f"{layer},{e},{c}\n")


class Analyze(unittest.TestCase):
    def make(self, d, weights, n_files=8, n_experts=64, k=4, n_tok=400, seed=1):
        rng = np.random.default_rng(seed)
        for i in range(n_files):
            write_trace(pathlib.Path(d) / f"t{i:02d}.csv", [3, 4], n_experts, k, n_tok, weights, rng)

    def test_uniform_routing_gives_linear_curve(self):
        n = 64
        with tempfile.TemporaryDirectory() as d:
            self.make(d, np.full(n, 1 / n), n_experts=n)
            res = analyze.analyze(d, n)
        for f, h in res["global_curve"]:
            self.assertAlmostEqual(h, f, delta=0.06)

    def test_skewed_routing_beats_uniform(self):
        n = 64
        w = 1.0 / np.arange(1, n + 1) ** 1.2
        w /= w.sum()
        with tempfile.TemporaryDirectory() as d:
            self.make(d, w, n_experts=n)
            res = analyze.analyze(d, n)
        h25 = dict((round(f, 2), h) for f, h in res["global_curve"])[0.25]
        self.assertGreater(h25, 0.45)
        self.assertLess(res["layers"]["3"]["k80"], 0.8 * n)

    def test_curve_monotone_and_bounded(self):
        n = 32
        with tempfile.TemporaryDirectory() as d:
            self.make(d, np.random.default_rng(3).dirichlet(np.ones(n) * 0.3), n_experts=n, k=2)
            res = analyze.analyze(d, n)
        hs = [h for _, h in res["global_curve"]]
        self.assertEqual(hs, sorted(hs))
        self.assertEqual(hs[0], 0.0)
        self.assertAlmostEqual(hs[-1], 1.0)

    def test_output_feeds_fitplan(self):
        n = 288
        w = 1.0 / np.arange(1, n + 1) ** 1.1
        w /= w.sum()
        with tempfile.TemporaryDirectory() as d:
            self.make(d, w, n_experts=n, k=8, n_tok=200)
            out = pathlib.Path(d) / "curve.json"
            subprocess.run([sys.executable, str(ROOT / "tools/g0/analyze.py"), d, "--n-experts", str(n),
                            "-o", str(out)], check=True, capture_output=True)
            curve = json.loads(out.read_text())["global_curve"]
        m = json.loads((ROOT / "models/glm-5.3-flash.json").read_text())
        hw = json.loads((ROOT / "hardware/4x4080s-32g.json").read_text())
        p_u = fitplan.plan(m, hw, "Q4_K", 32768)
        p_c = fitplan.plan(m, hw, "Q4_K", 32768, hit_curve=curve)
        self.assertGreater(p_c["hit_rate"], p_u["hit_rate"])


if __name__ == "__main__":
    unittest.main()
