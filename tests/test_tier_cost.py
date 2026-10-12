"""tools/ceilings/tier_cost.py: closed-form cases, the properties R8 section 8.3 asks for, and the fixes against the
R8 scratch implementation."""
import io
import json
import os
import random
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "ceilings"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import roofline as rl  # noqa: E402
import tier_cost as tc  # noqa: E402

HW, LY = tc.HW(), tc.Layer()


def tok(po, hw=HW, ly=LY):
    return tc.step(hw, ly, po, 42, 1.6e-3)["tok_s"]


def random_policies(n=300, seed=7):
    r = random.Random(seed)
    for _ in range(n):
        h_v = r.uniform(0.2, 1.0)
        h_r = r.uniform(0.0, 1.0 - h_v)
        yield tc.Policy(h_v=h_v, h_r=h_r, miss_path=r.choice(["cpu", "pcie"]), overlap=r.choice([True, False]),
                        p_recall=r.choice([0.0, r.uniform(0, 1)]), pi_prec=r.uniform(0.05, 1.0),
                        W=r.choice([0.0, r.uniform(0.1e-3, 2e-3)]), prefetch_to=r.choice(["gpu", "ram"]),
                        n_tok=r.choice([1, 2, 4]), accept=r.uniform(0.0, 0.95))


class ClosedForm(unittest.TestCase):
    def test_all_vram_layer(self):
        t, info = tc.layer_time(HW, LY, tc.Policy(h_v=1.0, h_r=0.0))
        self.assertAlmostEqual(t, LY.t_pre + LY.t_shared + LY.K * LY.S_e / HW.B_v + HW.t_kern, places=15)
        self.assertEqual(info["t_cpu"], 0.0)

    def test_cpu_miss_layer_is_dram_bound_at_defaults(self):
        po = tc.Policy()
        t, info = tc.layer_time(HW, LY, po)
        n_cpu = LY.K * po.h_r
        expect_cpu = max(n_cpu * LY.S_e / (HW.n_cpu * HW.b_core),
                         n_cpu * LY.S_e / (HW.B_dram * HW.eta_dram)) + HW.t_sync
        self.assertAlmostEqual(info["t_cpu"], expect_cpu, places=15)
        self.assertAlmostEqual(t, LY.t_pre + max(info["t_gpu"], expect_cpu), places=15)

    def test_tokens_per_step_matches_roofline(self):
        for a in (0.0, 0.3, 0.7, 0.99, 1.0):
            for k in (0, 1, 3, 7):
                self.assertAlmostEqual(tc.tokens_per_step(a, k), rl.spec_tokens_per_step(a, k) if k else 1.0)

    def test_distinct_matches_roofline(self):
        self.assertAlmostEqual(tc.distinct(288, 8, 4), rl.distinct_experts(288, 8, 4))

    def test_r8_rows_reproduced_with_r8_constants(self):
        # R8 section 8.3 (literal output, B_s_drive 4.2e9): rows the fixes do not change, to R8's one-decimal print; the
        # 5%-NVMe row with prediction moves 26.55 -> 26.64 (optimal staging), still within that print
        # R8's hardware row (R8-memory-tiers-expert-scheduling.md:328): 7.0e9 x 0.6 per drive and 4 drives; the tool's
        # default drive count is 3 since the customer correction of 2026-10-12, so R8's count is pinned here explicitly
        hw8 = replace(HW, B_s_drive=4.2e9, n_drives=4)
        for po, want in ((tc.Policy(h_v=1.0, h_r=0.0), 39.0), (tc.Policy(), 34.4), (tc.Policy(overlap=False), 27.1),
                         (tc.Policy(miss_path="pcie"), 10.4), (tc.Policy(h_v=0.82, h_r=0.18), 41.8),
                         (tc.Policy(p_recall=0.85, pi_prec=0.5, W=0.44e-3, prefetch_opt=False), 32.6),
                         (tc.Policy(n_tok=2, accept=0.7), 39.8),
                         (tc.Policy(h_r=0.34, p_recall=0.5, pi_prec=0.5, W=0.44e-3, prefetch_to="ram"), 26.6)):
            self.assertAlmostEqual(tok(po, hw8), want, delta=0.051, msg=po)

    def test_default_drive_count_is_the_customer_3x1tb(self):
        # customer correction 2026-10-12: 3 x 1 TB NVMe (planned; count/models UNKNOWN until R0)
        self.assertEqual(tc.HW().n_drives, 3)
        self.assertAlmostEqual(tc.HW().B_s_drive * tc.HW().n_drives, 3 * 7.45e9 * 0.6, places=3)
        self.assertEqual([(lo, hi) for p, lo, hi in tc.TORNADO if p == "n_drives"], [(2, 3)])

    def test_fewer_drives_never_faster(self):
        for po in random_policies():
            self.assertLessEqual(tok(po, replace(HW, n_drives=2)), tok(po, replace(HW, n_drives=3)) + 1e-9, po)

    def test_glm_preset_matches_r8_constants(self):
        ly, n_moe, t_other, name = tc.preset("glm")
        self.assertEqual((n_moe, ly.K, ly.E), (42, 8, 288))
        self.assertAlmostEqual(ly.S_e / 15.38e6, 1.0, delta=0.002)
        self.assertAlmostEqual(ly.t_pre / 0.30e-3, 1.0, delta=0.05)
        self.assertAlmostEqual(t_other / 1.6e-3, 1.0, delta=0.02)


class Fixes(unittest.TestCase):
    def test_no_prediction_costs_no_background_latency(self):
        # R8 scratch charged max(0, 0 + L_s - W) = L_s even with p = 0
        po = tc.Policy(h_r=0.38)                       # 1% on NVMe
        _, info = tc.layer_time(HW, LY, po)
        n_s = LY.K * 0.01
        per = LY.S_e / (HW.B_s_drive * HW.n_drives)
        expect = n_s * per + HW.L_s * (1 - (1 - n_s / LY.K) ** LY.K)
        self.assertAlmostEqual(info["t_ssd"], expect, places=15)

    def test_gpu_prefetch_takes_only_ram_resident_experts(self):
        po = tc.Policy(h_v=0.6, h_r=0.0, p_recall=1.0, pi_prec=1.0, W=5e-3)   # everything off-VRAM is on NVMe
        _, info = tc.layer_time(HW, LY, po)
        self.assertAlmostEqual(info["n_gpu"], LY.K * 0.6, places=12)
        self.assertEqual(info["dma_bytes"], 0.0)


class Properties(unittest.TestCase):
    def test_prediction_off_is_exactly_no_prediction(self):
        for po in random_policies():
            off = replace(po, p_recall=0.0)
            self.assertEqual(tok(replace(off, W=0.0)), tok(replace(off, W=1e-3)))

    def test_more_nvme_is_never_faster(self):
        for po in random_policies():
            h_s = 1 - po.h_v - po.h_r
            more = replace(po, h_r=max(0.0, po.h_r - 0.05))
            if 1 - more.h_v - more.h_r > h_s:
                self.assertLessEqual(tok(more), tok(po) + 1e-9, po)

    def test_overlap_is_never_slower(self):
        for po in random_policies():
            self.assertGreaterEqual(tok(replace(po, overlap=True)) + 1e-9, tok(replace(po, overlap=False)), po)

    def test_x8_link_is_never_faster_than_x16(self):
        for po in random_policies():
            self.assertLessEqual(tok(po, replace(HW, B_p=12.5e9)), tok(po, replace(HW, B_p=25e9)) + 1e-9, po)

    def test_more_dram_bandwidth_is_never_slower(self):
        for po in random_policies():
            self.assertLessEqual(tok(po, replace(HW, B_dram=102.4e9)), tok(po, replace(HW, B_dram=204.8e9)) + 1e-9)

    def test_link_rate_prefetch_can_lose_on_a_faster_link(self):
        # why prefetch_opt exists: low-precision copies at link rate waste DRAM bandwidth, more on x16 than on x8
        po = tc.Policy(h_v=0.6245810839858167, h_r=0.18113475171879803, p_recall=0.24849432104309,
                       pi_prec=0.0900889499757293, W=0.00028571601930749846, n_tok=1, prefetch_opt=False)
        self.assertGreater(tok(po, replace(HW, B_p=12.5e9)), tok(po, replace(HW, B_p=25e9)))
        po = replace(po, prefetch_opt=True)
        self.assertLessEqual(tok(po, replace(HW, B_p=12.5e9)), tok(po, replace(HW, B_p=25e9)) + 1e-9)

    def test_optimal_prefetch_is_never_slower_than_none_or_link_rate(self):
        for po in random_policies():
            po = replace(po, prefetch_to="gpu", W=max(po.W, 0.2e-3), p_recall=max(po.p_recall, 0.1))
            best = tok(replace(po, prefetch_opt=True))
            self.assertGreaterEqual(best + 1e-9, tok(replace(po, prefetch_to="ram")), po)     # same NVMe staging
            self.assertGreaterEqual(best + 1e-9, tok(replace(po, prefetch_opt=False)), po)
            self.assertGreaterEqual(best + 1e-9, tok(replace(po, p_recall=0.0)), po)

    def test_better_prediction_is_never_slower(self):
        for po in random_policies():
            po = replace(po, prefetch_opt=True, W=max(po.W, 0.1e-3))
            for key in ("p_recall", "pi_prec"):
                lo = replace(po, **{key: 0.3})
                hi = replace(po, **{key: 0.95})
                self.assertGreaterEqual(tok(hi) + 1e-9, tok(lo), (key, po))

    def test_nvme_staging_optimum_matches_a_fine_grid(self):
        r = random.Random(3)
        for _ in range(200):
            po = tc.Policy(h_v=r.uniform(0.3, 0.7), h_r=0.0, p_recall=r.uniform(0.1, 1), pi_prec=r.uniform(0.1, 1),
                           W=r.uniform(0, 2e-3), prefetch_to="ram")
            po = replace(po, h_r=r.uniform(0, 1 - po.h_v))
            t_best = tc.layer_time(HW, LY, po)[1]["t_ssd"]
            # brute force over the staged share, by calling the model with a reduced recall (same precision)
            grid = min(tc.layer_time(HW, LY, replace(po, p_recall=po.p_recall * f / 400, prefetch_opt=False))[1]["t_ssd"]
                       for f in range(0, 401))
            self.assertLessEqual(t_best, grid + 1e-12, po)

    def test_useless_gpu_prefetch_is_never_faster_than_none(self):
        for po in random_policies():
            po = replace(po, prefetch_to="gpu", miss_path="cpu", W=max(po.W, 0.2e-3))
            bad = replace(po, p_recall=1.0, pi_prec=1e-6, prefetch_opt=False)
            self.assertLessEqual(tok(bad), tok(replace(po, p_recall=0.0)) + 1e-9, po)

    def test_zero_acceptance_drafts_never_pay(self):
        for po in random_policies():
            one = replace(po, n_tok=1)
            self.assertLessEqual(tok(replace(po, n_tok=3, accept=0.0)), tok(one) + 1e-9, po)

    def test_invalid_shares_are_rejected(self):
        with self.assertRaises(ValueError):
            tc.layer_time(HW, LY, tc.Policy(h_v=0.8, h_r=0.3))


class Lookups(unittest.TestCase):
    def test_lead_time_hides_the_read(self):
        self.assertEqual(tc.lookup_exposed(HW, 48, 1.0), 0.0)
        self.assertGreater(tc.lookup_exposed(HW, 48, 0.0), HW.L_s)

    def test_more_rows_never_cheaper_and_cache_helps(self):
        a = [tc.lookup_exposed(HW, r, 0.0) for r in (1, 16, 48, 200)]
        self.assertEqual(a, sorted(a))
        self.assertLess(tc.lookup_exposed(HW, 48, 0.0, cache_hit=0.9), a[2])
        self.assertEqual(tc.lookup_exposed(HW, 48, 0.0, cache_hit=1.0), 0.0)


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tc.main(list(args))
        return rc, buf.getvalue()

    def test_sweep_grid(self):
        rc, out = self.run_cli("--model", "glm", "--sweep", "h_v=0.5,0.7", "--sweep", "B_dram=102.4e9,204.8e9", "--json")
        self.assertEqual(rc, 0)
        rows = json.loads(out)["rows"]
        self.assertEqual(len(rows), 4)
        self.assertLess(rows[0]["tok_s"], rows[3]["tok_s"])

    def test_tornado_sorted_by_swing(self):
        rc, out = self.run_cli("--model", "glm", "--tornado", "--json")
        rows = json.loads(out)["rows"]
        self.assertEqual([r["swing"] for r in rows], sorted((r["swing"] for r in rows), reverse=True))
        self.assertEqual({r["param"] for r in rows}, {p for p, _, _ in tc.TORNADO})

    def test_r8_cases_and_models(self):
        for m in tc.MODELS:
            rc, out = self.run_cli("--model", m, "--cases", "r8", "--json")
            self.assertEqual(len(json.loads(out)["rows"]), 2 * len(tc.R8_CASES))

    def test_unknown_parameter(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--set", "nope=1")


if __name__ == "__main__":
    unittest.main()
