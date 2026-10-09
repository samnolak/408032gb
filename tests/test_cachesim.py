"""tools/g0/cachesim.py on synthetic traces (SYNTHETIC: properties of the simulator, not of any model)."""
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "g0"))
import cachesim as cs  # noqa: E402


def zipf_trace(rng, n_tokens, n_experts, k, s=1.1, perm=None):
    w = [1.0 / (i + 1) ** s for i in range(n_experts)]
    perm = perm or list(range(n_experts))
    out = []
    for _ in range(n_tokens):
        chosen = set()
        while len(chosen) < k:
            chosen.add(perm[rng.choices(range(n_experts), weights=w)[0]])
        out.append(tuple(sorted(chosen)))
    return out


class Policies(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.E, self.k = 32, 4
        self.seq = zipf_trace(self.rng, 600, self.E, self.k)
        self.ranked = sorted(range(self.E))   # zipf_trace without perm: expert 0 is the hottest

    def hit(self, pol, cap, warm=False, seq=None, ranked=None):
        seq = seq or self.seq
        h = cs.simulate_layer(seq, pol, cap, self.E, static=ranked or self.ranked, warm=warm)
        return sum(h) / (len(seq) * self.k)

    def test_opt_is_an_upper_bound_for_demand_policies(self):
        for cap in (4, 8, 16):
            opt = self.hit("opt", cap)
            for pol in ("static", "lru", "lfu", "tinylfu", "hybrid"):
                self.assertGreaterEqual(opt + 1e-12, self.hit(pol, cap), (pol, cap))
                self.assertGreaterEqual(opt + 1e-12, self.hit(pol, cap, warm=True), (pol, cap, "warm"))

    def test_monotone_in_capacity(self):
        for pol in cs.POLICIES:
            rates = [self.hit(pol, c) for c in (4, 8, 16, 24)]
            self.assertEqual(rates, sorted(rates), pol)

    def test_full_capacity(self):
        self.assertEqual(self.hit("static", self.E), 1.0)
        cold = self.hit("lru", self.E)
        self.assertLess(cold, 1.0)                       # compulsory misses
        self.assertEqual(self.hit("opt", self.E), 1.0)   # the bound preloads for free
        self.assertEqual(self.hit("lru", self.E, warm=True), 1.0)

    def test_static_good_on_stationary_zipf(self):
        self.assertGreater(self.hit("static", 8), self.hit("lru", 8))

    def test_lru_thrashes_on_a_cycle_static_does_not(self):
        cyc = [tuple(range(i % 9, i % 9 + 1)) for i in range(900)]   # k = 1, cycle over 9 experts
        lru = sum(cs.simulate_layer(cyc, "lru", 8, 16, static=list(range(16)))) / 900
        static = sum(cs.simulate_layer(cyc, "static", 8, 16, static=list(range(16)))) / 900
        opt = sum(cs.simulate_layer(cyc, "opt", 8, 16, static=list(range(16)))) / 900
        self.assertEqual(lru, 0.0)
        self.assertAlmostEqual(static, 8 / 9, places=6)
        self.assertGreaterEqual(opt, static)

    def test_drift_favours_adaptive_policies(self):
        # the hot set moves after the first half; a static set chosen on the first half goes stale
        perm = list(range(self.E))
        self.rng.shuffle(perm)
        drifted = self.seq[:300] + zipf_trace(self.rng, 1200, self.E, self.k, perm=perm)
        static = self.hit("static", 8, seq=drifted)
        lru = self.hit("lru", 8, seq=drifted)
        self.assertGreater(lru, static)


class Windows(unittest.TestCase):
    def test_identical_tokens_touch_k_experts(self):
        tr = {"n_prompt": 0, "n_tokens": 10, "k": 2, "per_layer": {0: [(1, 2)] * 10}}
        self.assertEqual(cs.window_distinct([tr], windows=(1, 4, 8)), {1: 2.0, 4: 2.0, 8: 2.0})

    def test_independent_routing_matches_formula(self):
        rng = random.Random(11)
        E, k = 64, 4
        seq = [tuple(rng.sample(range(E), k)) for _ in range(4000)]
        tr = {"n_prompt": 0, "n_tokens": len(seq), "k": k, "per_layer": {0: seq}}
        wd = cs.window_distinct([tr], windows=(4,))
        self.assertAlmostEqual(wd[4], cs.independent_distinct(E, k, 4), delta=0.15)


class TraceFormat(unittest.TestCase):
    def test_roundtrip_and_cli(self):
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as d:
            for i in range(4):
                toks = []
                per_layer = {lid: zipf_trace(rng, 120, 16, 2) for lid in (3, 4, 5)}
                for t in range(120):
                    toks.append([per_layer[lid][t] for lid in (3, 4, 5)])
                cs.write_trace(os.path.join(d, f"t{i}.trace"), [3, 4, 5], 2, toks, 100)
            tr = cs.read_trace(os.path.join(d, "t0.trace"))
            self.assertEqual(tr["layer_ids"], [3, 4, 5])
            self.assertEqual(tr["n_prompt"], 100)
            self.assertEqual(len(tr["per_layer"][4]), 120)
            out = os.path.join(d, "res.json")
            cs.main([d, "--n-experts", "16", "--capacity", "0.25,0.5,1.0", "-o", out])
            import json
            with open(out) as f:
                r = json.load(f)
            self.assertEqual(r["split"], "held-out every 4th file: 3 train / 1 test")
            full = r["hit_rate_by_capacity"]["1.0"]
            self.assertEqual(full["static"]["all"], 1.0)
            for frac in ("0.25", "0.5", "1.0"):
                row = r["hit_rate_by_capacity"][frac]
                for pol in ("static", "lru", "lfu", "tinylfu", "hybrid"):
                    self.assertGreaterEqual(row["opt"]["all"] + 1e-12, row[pol]["all"])
                self.assertIsNotNone(row["static"]["gen"])

    def test_rejects_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.trace")
            cs.write_trace(p, [0], 2, [[(0, 1)], [(1, 2)]], 1)
            with open(p, "r+b") as f:
                f.truncate(os.path.getsize(p) - 2)
            with self.assertRaises(ValueError):
                cs.read_trace(p)


if __name__ == "__main__":
    unittest.main()
