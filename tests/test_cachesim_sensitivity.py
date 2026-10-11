"""scripts/traces/cachesim_sensitivity.py on synthetic traces (SYNTHETIC: properties of the wrapper, not of any model)."""
import io
import json
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "g0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "traces"))
import cachesim as cs  # noqa: E402
import cachesim_sensitivity as sens  # noqa: E402


def rand_tokens(rng, n, n_layers, E, k, pool=None):
    pool = pool or list(range(E))
    return [[tuple(sorted(rng.sample(pool, k))) for _ in range(n_layers)] for _ in range(n)]


class Sensitivity(unittest.TestCase):
    E, K, L = 16, 2, 3

    def write(self, d, name, tokens, n_prompt):
        cs.write_trace(os.path.join(d, name + ".trace"), list(range(1, self.L + 1)), self.K, tokens, n_prompt)

    def corpus(self, d, n_files=8, seed=3):
        rng = random.Random(seed)
        for i in range(n_files):
            n_p, n_g = 30 + 5 * i, 10 + i
            self.write(d, f"c{i:02d}", rand_tokens(rng, n_p + n_g, self.L, self.E, self.K, pool=list(range(i % 3, 12))),
                       n_p)

    def test_last_fold_reproduces_cachesim_cli(self):
        with tempfile.TemporaryDirectory() as d:
            self.corpus(d)
            out = os.path.join(d, "cs.json")
            with redirect_stdout(io.StringIO()):
                cs.main([d, "--n-experts", str(self.E), "--capacity", "0.25,0.5", "--holdout-every", "4", "-o", out])
            with open(out) as f:
                ref = json.load(f)["hit_rate_by_capacity"]
            rows = sens.run(sens.load(d), self.E, [0.25, 0.5], list(cs.POLICIES), fold="last", every=4)
            for r in rows:
                self.assertEqual(r["rates"], ref[str(r["capacity"])])
                self.assertEqual(r["test"], ["c03", "c07"])

    def test_all_folds_test_every_trace_once(self):
        traces = [{"name": f"t{i}"} for i in range(10)]
        seen = []
        for r, train, test in sens.folds(traces, 4, "all"):
            self.assertFalse({t["name"] for t in train} & {t["name"] for t in test})
            self.assertEqual(len(train) + len(test), 10)
            seen += [t["name"] for t in test]
        self.assertEqual(sorted(seen), sorted(t["name"] for t in traces))

    def test_gen_only_starts_cold_full_view_is_warmed_by_the_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            same = [[(0, 1)] * self.L for _ in range(20)]          # every token uses experts 0 and 1
            for i in range(4):
                self.write(d, f"c{i:02d}", same, 15)               # 15 prompt + 5 generated tokens
            tr = sens.load(d)
            full = sens.run(tr, self.E, [0.125], ["lru"], every=4)[0]["rates"]["lru"]["gen"]
            cold = sens.run(tr, self.E, [0.125], ["lru"], every=4, view="gen_only")[0]["rates"]["lru"]["gen"]
            self.assertEqual(full, 1.0)
            self.assertAlmostEqual(cold, 4 / 5)                    # the first generated token misses both experts
            warm = sens.run(tr, self.E, [0.125], ["lru"], every=4, view="gen_only", warm=True)[0]["rates"]["lru"]["gen"]
            self.assertEqual(warm, 1.0)                            # the static set (from train) is {0, 1}

    def test_gen_cap_and_slices(self):
        with tempfile.TemporaryDirectory() as d:
            rng = random.Random(1)
            self.write(d, "c00", rand_tokens(rng, 50, self.L, self.E, self.K), 40)
            tr = sens.load(d)[0]
            v = sens.test_view(tr, "full", gen_cap=4)
            self.assertEqual((v["n_tokens"], v["n_prompt"]), (44, 40))
            g = sens.test_view(tr, "gen_only", gen_cap=100)
            self.assertEqual((g["n_tokens"], g["n_prompt"]), (10, 0))
            self.assertEqual(g["per_layer"][2], tr["per_layer"][2][40:])
            p = sens.train_view(tr, "prompt")
            self.assertEqual((p["n_tokens"], p["n_prompt"], len(p["per_layer"][1])), (40, 40, 40))

    def test_train_part_changes_the_static_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            prompt, gen = [[(0, 1)] * self.L] * 30, [[(5, 6)] * self.L] * 10
            for i in range(4):
                self.write(d, f"c{i:02d}", prompt + gen, 30)
            tr = sens.load(d)
            hit = lambda part: sens.run(tr, self.E, [0.125], ["static"], train_part=part)[0]["rates"]["static"]["gen"]  # noqa: E731
            self.assertEqual(hit("all"), 0.0)       # 30 prompt tokens outvote 10 generated ones
            self.assertEqual(hit("prompt"), 0.0)
            self.assertEqual(hit("gen"), 1.0)

    def test_per_trace_pooled_equals_the_joint_run_for_per_layer_policies(self):
        with tempfile.TemporaryDirectory() as d:
            self.corpus(d, n_files=8)
            tr = sens.load(d)
            pols = list(cs.LAYER_POLICIES)
            joint = sens.run(tr, self.E, [0.25], pols, fold="all")
            per = sens.run(tr, self.E, [0.25], pols, fold="all", per_trace=True)
            a, b = sens.pooled(joint, pols)[0.25], sens.pooled(per, pols)[0.25]
            for p in pols:
                self.assertAlmostEqual(a[p]["pooled"], b[p]["pooled"], places=12)
                self.assertLessEqual(b[p]["min"], a[p]["min"] + 1e-12)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            self.corpus(d)
            out = os.path.join(d, "s.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = sens.main([d, "--n-experts", str(self.E), "--capacity", "0.5", "--fold", "all", "--policies",
                                "static,lru", "--test-view", "gen_only", "-o", out])
            self.assertEqual(rc, 0)
            self.assertIn("pooled", buf.getvalue())
            with open(out) as f:
                j = json.load(f)
            self.assertEqual(len(j["rows"]), 4)
            self.assertIn("0.5", j["summary"])


if __name__ == "__main__":
    unittest.main()
