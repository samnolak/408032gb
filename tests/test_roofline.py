"""tools/roofline.py: parameter totals against published sizes, formula properties, physical sanity."""
import glob
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import roofline as rl  # noqa: E402

def _load(path):
    with open(path) as f:
        return json.load(f)


HW = _load(os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json"))
Q4 = {"expert_bpw": 4.5, "dense_bpw": 8.5, "kv_bytes_per_elem": 2.0}


def spec(name):
    return _load(os.path.join(ROOT, "models", "specs", name + ".json"))


class PublishedTotals(unittest.TestCase):
    def test_glm_total_matches_pr_27754(self):
        s = spec("glm-5.3-flash")
        total = rl.dense_stored_params(s) + rl.routed_stored_params(s) + s["dense_params"]["vision"]
        self.assertAlmostEqual(total / 321.3e9, 1.0, delta=0.005)          # 321.3B quoted in GLM53-TZ.md

    def test_glm_active_is_17_to_18b_with_embedding_row_counted_as_weights(self):
        s = spec("glm-5.3-flash")
        active = rl.dense_active_params(s) + rl.routed_active_params(s)
        self.assertGreater(active, 16.5e9)
        self.assertLess(active + s["dense_params"]["embedding"], 18.0e9)

    def test_glm_expert_count_and_size(self):
        s = spec("glm-5.3-flash")
        self.assertEqual(rl.expert_params(s), 25165824)
        self.assertEqual(rl.routed_stored_params(s) / rl.expert_params(s), 43 * 288)

    def test_mimo_total(self):
        s = spec("mimo-v2.6-flash")
        total = rl.dense_stored_params(s) + rl.routed_stored_params(s)
        self.assertAlmostEqual(total / 309e9, 1.0, delta=0.01)

    def test_deepseek_routed(self):
        s = spec("deepseek-v4.1-flash")
        routed = rl.routed_stored_params(s) - s["moe"]["extra_stored_params"]
        self.assertAlmostEqual(routed / 543.6e9, 1.0, delta=0.001)


class Formulas(unittest.TestCase):
    def test_ep_imbalance(self):
        self.assertEqual(rl.ep_imbalance(8, 1), 1.0)
        self.assertAlmostEqual(rl.ep_imbalance(1, 4), 4.0)               # one expert: all load on one GPU
        v = rl.ep_imbalance(8, 4)
        self.assertGreater(v, 1.0)
        self.assertLess(v, 2.0)
        self.assertLess(rl.ep_imbalance(8, 2), v)

    def test_distinct_experts(self):
        self.assertAlmostEqual(rl.distinct_experts(288, 8, 1), 8.0)
        prev = 8.0
        for n in range(2, 10):
            d = rl.distinct_experts(288, 8, n)
            self.assertGreater(d, prev)
            self.assertLessEqual(d, 8 * n)
            prev = d

    def test_spec_tokens_per_step(self):
        self.assertEqual(rl.spec_tokens_per_step(0.0, 3), 1.0)
        self.assertEqual(rl.spec_tokens_per_step(1.0, 3), 4)
        self.assertAlmostEqual(rl.spec_tokens_per_step(0.5, 1), 1.5)

    def test_kv_read_scaling(self):
        mimo = spec("mimo-v2.6-flash")
        a, b = rl.kv_read_bytes(mimo, 65536), rl.kv_read_bytes(mimo, 131072)
        full = 9 * 4 * 320 * 2.0
        self.assertAlmostEqual(b - a, full * 65536, delta=1.0)           # sliding part saturated, full grows linearly
        glm = spec("glm-5.3-flash")
        state = 2 * 34 * glm["attention"][0]["state_bytes"]
        self.assertGreater(rl.kv_read_bytes(glm, 1), state)               # the recurrent state is read and written
        g1, g2 = rl.kv_read_bytes(glm, 262144), rl.kv_read_bytes(glm, 131072)
        self.assertAlmostEqual(g1 - g2, 11 * 256 * 131072 / 4, delta=1.0)  # only the pooled-key scan grows

    def test_kv_bytes_scale(self):
        mimo = spec("mimo-v2.6-flash")
        self.assertLess(rl.kv_store_bytes(mimo, 262144, 1.0), rl.kv_store_bytes(mimo, 262144, 2.0))


class Physics(unittest.TestCase):
    def test_longer_context_never_faster(self):
        for name in ("glm-5.3-flash", "mimo-v2.6-flash", "deepseek-v4.1-flash"):
            s = spec(name)
            t = [rl.decode(s, HW, Q4, c, 0.8)["tok_s"] for c in (4096, 65536, 262144)]
            self.assertGreaterEqual(t[0], t[1], name)
            self.assertGreaterEqual(t[1], t[2], name)

    def test_tp_beats_pp_when_resident(self):
        s = spec("glm-5.3-flash")
        self.assertGreater(rl.decode(s, HW, Q4, 262144, 1.0, strategy="tp")["tok_s"],
                           rl.decode(s, HW, Q4, 262144, 1.0, strategy="pp")["tok_s"])

    def test_cpu_compute_beats_pcie_copy_for_batch_one(self):
        s = spec("glm-5.3-flash")
        self.assertGreater(rl.decode(s, HW, Q4, 32768, 0.6, miss="cpu")["tok_s"],
                           rl.decode(s, HW, Q4, 32768, 0.6, miss="pcie")["tok_s"])

    def test_all_experts_from_ram_priced_at_ram_bandwidth(self):
        s = spec("glm-5.3-flash")
        d = rl.decode(s, HW, Q4, 4096, 0.0, strategy="pp", miss="cpu", ram_channels=8)
        gbw = HW["gpus"][0]["hbm_bw_gbs"] * 1e9 * HW["gpus"][0]["bw_eff"]
        rbw = HW["ram"]["bw_gbs_by_channels"]["8"] * 1e9 * HW["ram"]["bw_eff"]
        lat = HW["pcie"]["latency_us"] * 1e-6
        w_dense = rl.dense_active_params(s) * 8.5 / 8
        w_routed = rl.routed_active_params(s) * 4.5 / 8
        expect = (w_dense + rl.kv_read_bytes(s, 4096)) / gbw + w_routed / rbw + 42 * 2 * lat + 3 * lat
        self.assertAlmostEqual(d["ms_per_step"], expect * 1e3, delta=expect * 1e3 * 1e-9)

    def test_more_bits_less_resident(self):
        s = spec("glm-5.3-flash")
        lo = rl.placement(s, HW, dict(Q4, expert_bpw=4.25), 262144)["resident_frac"]
        hi = rl.placement(s, HW, dict(Q4, expert_bpw=5.5), 262144)["resident_frac"]
        self.assertGreater(lo, hi)

    def test_ssd_spill_costs(self):
        s = spec("deepseek-v4.1-flash")
        with_ssd = rl.decode(s, HW, Q4, 32768, 0.4, hit_ram=0.4)["tok_s"]
        no_ssd = rl.decode(s, HW, Q4, 32768, 0.4, hit_ram=0.6)["tok_s"]
        self.assertGreater(no_ssd, with_ssd)

    def test_speculation_with_full_acceptance_helps_when_resident(self):
        s = spec("glm-5.3-flash")
        base = rl.decode(s, HW, Q4, 32768, 1.0, strategy="tp")["tok_s"]
        sp = rl.decode(s, HW, Q4, 32768, 1.0, strategy="tp", spec_k=3, spec_accept=0.8)["tok_s"]
        self.assertGreater(sp, base)


class Cli(unittest.TestCase):
    def test_cli_json_for_every_spec(self):
        files = sorted(glob.glob(os.path.join(ROOT, "models", "specs", "*.json")))
        self.assertGreaterEqual(len(files), 3)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rl.main(files + ["--hw", os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json"),
                             "--json", "--sweep-hit", "auto,1.0"])
        out = json.loads(buf.getvalue())
        self.assertEqual(len(out), len(files))
        for r in out:
            for d in r["decode"]:
                self.assertGreater(d["tok_s"], 0)


if __name__ == "__main__":
    unittest.main()
