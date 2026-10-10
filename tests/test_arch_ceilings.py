"""tools/ceilings/arch_ceilings.py: consistency with tools/roofline.py, monotonicity, limits, physics bounds, CLI."""
import contextlib
import copy
import io
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "ceilings"))
import arch_ceilings as ac  # noqa: E402
import roofline as rl  # noqa: E402

with open(os.path.join(ROOT, "hardware", "target-4x4080-32g.unverified.json")) as _f:
    HW = json.load(_f)
CTX = 262144
ALL_ARCHS = list(ac.ARCHS)


def model(name):
    return ac.load_model(name)


def tok_s(name, arch, preset="base", **over):
    spec, sq = model(name)
    p = ac.resolve_params(preset, over)
    return ac.step(spec, sq, p, arch)["tok_s"]


class RooflineConsistency(unittest.TestCase):
    """Preset 'roofline' must reproduce tools/roofline.py decode() exactly for the strategies both tools model."""

    def check(self, name, q, strat, arch, hit, k, a, overlap, draft_gb=None):
        spec, sq = model(name)
        ps = rl.placement(spec, HW, q, CTX, strat)
        if hit == "auto":
            hv, hr = ps["resident_frac"], ps["ram_frac"]
        else:
            hv, hr = hit, rl.miss_split(ps, hit)
        ref = rl.decode(spec, HW, q, CTX, hv, hr, strat, "cpu", k, a, None, overlap, draft_gb)["tok_s"]
        over = {"overlap": overlap, "hit": "uniform" if hit == "auto" else hit,
                "spec": "none" if k == 0 else f"mtp:{k}:{a}", "quant": {"expert_bpw": q["expert_bpw"],
                                                                         "dense_bpw": q["dense_bpw"]}}
        if draft_gb is not None:
            over["draft_gb"] = draft_gb
        mine = ac.step(spec, sq, ac.resolve_params("roofline", over), arch)["tok_s"]
        self.assertAlmostEqual(mine / ref, 1.0, places=9, msg=f"{name} {strat} hit {hit} k {k} overlap {overlap}")

    def test_glm_and_qwen_all_strategies(self):
        cases = (("glm", {"expert_bpw": 4.89, "dense_bpw": 8.04, "kv_bytes_per_elem": 2.0}),
                 ("qwen", {"expert_bpw": 5.10, "dense_bpw": 10.0, "kv_bytes_per_elem": 2.0}))
        for name, q in cases:
            for strat, arch in (("pp", "pp"), ("tp", "tp_repl"), ("ep", "ep")):
                for overlap in (True, False):
                    for hit in ("auto", 0.8, 0.9, 1.0):
                        for k, a in ((0, 0.0), (2, 0.7)):
                            self.check(name, q, strat, arch, hit, k, a, overlap)

    def test_mimo_with_explicit_draft_cost(self):
        q = {"expert_bpw": 4.25, "dense_bpw": 8.5, "kv_bytes_per_elem": 2.0}
        for strat, arch in (("pp", "pp"), ("tp", "tp_repl")):
            self.check("mimo", q, strat, arch, 0.9, 2, 0.6, True, draft_gb=0.5)

    def test_roofline_md_headline_numbers(self):
        """ROOFLINE.md rev. 3 section 2-3 values, reproduced through this tool."""
        self.assertAlmostEqual(tok_s("glm", "pp", "roofline", hit=1.0), 39.4, delta=0.05)
        self.assertAlmostEqual(tok_s("glm", "tp_repl", "roofline", hit="uniform"), 41.3, delta=0.05)
        self.assertAlmostEqual(tok_s("glm", "tp_repl", "roofline", hit=0.9), 87.5, delta=0.05)
        self.assertAlmostEqual(tok_s("qwen", "pp", "roofline", hit=1.0), 84.3, delta=0.05)
        self.assertAlmostEqual(tok_s("qwen", "tp_repl", "roofline", hit=1.0), 168.0, delta=0.05)

    def test_placement_matches(self):
        spec, sq = model("glm")
        q = {"expert_bpw": 4.89, "dense_bpw": 8.04, "kv_bytes_per_elem": 2.0}
        p = ac.resolve_params("roofline", {})
        for strat, rep in (("pp", 1), ("tp", 4)):
            ref = rl.placement(spec, HW, q, CTX, strat)
            mine = ac.placement(spec, p, q, rep)
            for key in ("resident_frac", "ram_frac", "ssd_frac", "kv_state_gib"):
                self.assertAlmostEqual(mine[key], ref[key], places=12, msg=f"{strat} {key}")


class Monotonicity(unittest.TestCase):
    def assert_monotone(self, name, arch, key, values, increasing, **base):
        prev = None
        for v in values:
            t = tok_s(name, arch, **dict(base, **{key: v}))
            if prev is not None:
                if increasing:
                    self.assertGreaterEqual(t, prev - 1e-9, f"{arch} {key}={v}")
                else:
                    self.assertLessEqual(t, prev + 1e-9, f"{arch} {key}={v}")
            prev = t

    def test_bandwidth_efficiencies_help(self):
        for arch in ALL_ARCHS:
            for hit in ("uniform", "static"):
                self.assert_monotone("glm", arch, "vram_eff", [0.4, 0.55, 0.75, 0.9, 1.0], True, hit=hit)
                self.assert_monotone("glm", arch, "dram_eff", [0.3, 0.5, 0.7, 0.85, 1.0], True, hit=hit)
                self.assert_monotone("glm", arch, "dram_channels", [2, 4, 6, 8], True, hit=hit)
                self.assert_monotone("glm", arch, "cpu_compute_gbs", [20, 40, 80, 160, 400], True, hit=hit)

    def test_latencies_and_fixed_costs_hurt(self):
        for arch in ALL_ARCHS:
            for key in ("allreduce_us", "gpu_layer_fixed_us", "handoff_us", "step_fixed_us", "cpu_layer_fixed_us",
                        "ep_xfer_us", "stage_hop_us"):
                self.assert_monotone("glm", arch, key, [0, 5, 20, 60, 200], False, hit="mid")

    def test_hit_share_helps(self):
        hits = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        for arch in ALL_ARCHS:                          # serial CPU path: always monotone
            self.assert_monotone("glm", arch, "hit", hits, True, overlap=False, miss_union="additive")
        for arch in ("tp_repl", "tp_owner", "tp_owner_hostcpu", "pp2_tp2"):   # overlap, while the CPU path binds
            self.assert_monotone("glm", arch, "hit", hits[:5], True, overlap=True, miss_union="additive")

    def test_overlapped_bound_peaks_below_full_hit(self):
        """Same property as ROOFLINE.md rev. 3 (envelope): with overlap, once the GPU side binds, a higher hit share
        moves work from the parallel CPU onto the GPU, so the bound peaks below hit 1.0. The tool takes hit as an
        input and prints no envelope (an envelope needs RAM copies of resident experts; GLM's do not all fit)."""
        for arch in ("pp", "ep", "tp_repl"):
            grid = [0.80 + 0.01 * i for i in range(20)]
            best = max(tok_s("glm", arch, hit=h, miss_union="additive") for h in grid)
            self.assertGreater(best, tok_s("glm", arch, hit=1.0, miss_union="additive"), arch)

    def test_acceptance_helps(self):
        spec, sq = model("glm")
        for arch in ("pp", "tp_owner"):
            prev = 0.0
            for a in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
                t = ac.step(spec, sq, ac.resolve_params("base", {"hit": "static", "spec": f"mtp:2:{a}"}), arch)["tok_s"]
                self.assertGreaterEqual(t, prev - 1e-9)
                prev = t

    def test_longer_context_never_faster(self):
        for arch in ALL_ARCHS:
            self.assert_monotone("glm", arch, "ctx", [4096, 65536, 262144], False, hit="static")

    def test_union_properties(self):
        spec, _ = model("glm")
        prev = 0.0
        for n in range(1, 17):
            u = ac.union_experts(spec, n, "independent")
            self.assertGreaterEqual(u, 8.0 - 1e-12)
            self.assertLessEqual(u, 8.0 * n + 1e-9)
            self.assertGreater(u, prev)
            prev = u
            self.assertLessEqual(ac.union_experts(spec, n, 0.246), u + 1e-12)
            self.assertLessEqual(ac.union_experts(spec, n, 0.5), ac.union_experts(spec, n, 0.246) + 1e-12)

    def test_expected_max_load(self):
        self.assertEqual(ac.expected_max_load(8, 1), 8.0)
        prev = 0.0
        for m in (1, 2, 4, 8, 16, 32, 64):
            v = ac.expected_max_load(m, 4)
            self.assertGreaterEqual(v, m / 4 - 1e-9)
            self.assertLessEqual(v, m + 1e-9)
            self.assertGreaterEqual(v, prev - 1e-9)
            prev = v
        self.assertAlmostEqual(ac.expected_max_load(1, 4), 1.0, places=9)
        # 8 picks over 4 GPUs: same expectation as tools/roofline.py's enumeration
        self.assertAlmostEqual(ac.expected_max_load(8, 4) / 2.0, rl.ep_imbalance(8, 4), places=9)

    def test_static_curve_shape(self):
        for curve in (ac.GLM_STATIC_CURVE, ac.GLM_STATIC_CURVE_GLOBAL):
            self.assertEqual(curve[0], (0.0, 0.0))
            self.assertEqual(curve[-1], (1.0, 1.0))
            for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
                self.assertGreater(x1, x0)
                self.assertGreaterEqual(y1, y0)
                self.assertGreaterEqual(y1, x1)          # a frequency-sorted placement beats its capacity share


class Limits(unittest.TestCase):
    def test_tp_of_one_gpu_is_the_pipeline(self):
        spec, sq = model("glm")
        ac.ARCHS["_tp1"] = ("tp", {"tp_kv": "replicate", "tp_size": 1})
        try:
            for hit in ("uniform", "static", 1.0):
                for s in ("none", "mtp:2:0.745"):
                    p = ac.resolve_params("roofline", {"hit": hit, "spec": s, "allreduce_us": 0.0})
                    pp = ac.step(spec, sq, copy.deepcopy(p), "pp")["tok_s"]
                    tp1 = ac.step(spec, sq, copy.deepcopy(p), "_tp1")["tok_s"]
                    self.assertAlmostEqual(tp1 / pp, 1.0, places=9, msg=f"hit {hit} {s}")
        finally:
            del ac.ARCHS["_tp1"]

    def test_huge_collective_latency_kills_tp(self):
        self.assertLess(tok_s("glm", "tp_owner", allreduce_us=5000.0, hit="static"), tok_s("glm", "pp", hit="static"))
        self.assertLess(tok_s("qwen", "tp_owner", allreduce_us=5000.0, hit=1.0), 5.0)

    def test_no_architecture_beats_the_physics_bounds(self):
        spec, sq = model("glm")
        for preset in ("roofline", "base", "optimistic", "pessimistic"):
            for hit in ("uniform", "mid", "static", 1.0):
                for s in ("none", "mtp:3:0.745", "dflash2:7"):
                    for ch in (4, 8):
                        p = ac.resolve_params(preset, {"hit": hit, "spec": s, "dram_channels": ch})
                        b = ac.physics_bounds(spec, sq, p)
                        for arch in ALL_ARCHS:
                            r = ac.step(spec, sq, p, arch)
                            self.assertLessEqual(r["tok_s"], b["dram_bound_tok_s"] * (1 + 1e-9), arch)
                            self.assertLessEqual(r["tok_s"], b["aggregate_vram_bound_tok_s"] * 1.0001, arch)
                            if arch in ("pp", "pp_serial"):
                                self.assertLessEqual(r["tok_s"], b["single_gpu_pp_bound_tok_s"] * (1 + 1e-9))

    def test_zero_acceptance_never_helps(self):
        for arch in ALL_ARCHS:
            for hit in ("uniform", "static", 1.0):
                base = tok_s("glm", arch, hit=hit)
                for k in (1, 3):
                    self.assertLessEqual(tok_s("glm", arch, hit=hit, spec=f"mtp:{k}:0.0"), base + 1e-9)

    def test_full_hit_means_no_cpu_work(self):
        spec, sq = model("glm")
        r = ac.step(spec, sq, ac.resolve_params("base", {"hit": 1.0}), "tp_owner")
        self.assertEqual(r["ms_parts"]["moe_cpu"], 0.0)
        self.assertEqual(r["cpu_cap_tok_s"], float("inf"))

    def test_serial_never_faster_than_overlap(self):
        for hit in ("uniform", "mid", "static"):
            self.assertLessEqual(tok_s("glm", "pp_serial", hit=hit), tok_s("glm", "pp", hit=hit) + 1e-9)

    def test_prefetch_off_is_neutral_and_cannot_beat_dram(self):
        spec, sq = model("glm")
        p0 = ac.resolve_params("base", {"hit": "uniform"})
        p1 = ac.resolve_params("base", {"hit": "uniform", "prefetch_recall": 0.0, "prefetch_precision": 0.3})
        for arch in ALL_ARCHS:
            self.assertAlmostEqual(ac.step(spec, sq, p0, arch)["tok_s"], ac.step(spec, sq, p1, arch)["tok_s"], places=9)
        # a DRAM-bound CPU (no compute cap): moving misses to PCIe cannot lower the DRAM time of the misses
        for arch in ("tp_owner", "ep"):
            for rec in (0.3, 0.6, 1.0):
                p = ac.resolve_params("optimistic", {"hit": "uniform", "prefetch_recall": rec, "allreduce_us": 0.0,
                                                     "gpu_layer_fixed_us": 0.0, "step_fixed_us": 0.0,
                                                     "cpu_layer_fixed_us": 0.0, "handoff_us": 0.0, "dram_eff": 1.0})
                b = ac.physics_bounds(spec, sq, p)
                self.assertLessEqual(ac.step(spec, sq, p, arch)["tok_s"], b["dram_bound_tok_s"] * (1 + 1e-9))

    def test_prefetch_helps_a_compute_bound_cpu(self):
        slow = tok_s("glm", "tp_owner", hit="static", cpu_compute_gbs=60.0)
        pf = tok_s("glm", "tp_owner", hit="static", cpu_compute_gbs=60.0, prefetch_recall=0.6, prefetch_precision=0.9)
        self.assertGreater(pf, slow)

    def test_pipelined_windows_limits(self):
        spec, sq = model("glm")
        for hit in ("uniform", "static", 1.0):
            pp = ac.step(spec, sq, ac.resolve_params("base", {"hit": hit, "spec": "mtp:1:0.0"}), "pp")["tok_s"]
            pw = ac.step(spec, sq, ac.resolve_params("base", {"hit": hit, "spec": "mtp:1:0.0"}), "pp_pipewin")
            self.assertAlmostEqual(pw["tok_s"], pp, places=6)            # nothing valid to run ahead
            for a in (0.5, 0.745, 0.95):
                p = ac.resolve_params("base", {"hit": hit, "spec": f"mtp:1:{a}"})
                one = ac.step(spec, sq, p, "pp")["tok_s"]
                four = ac.step(spec, sq, p, "pp_pipewin")["tok_s"]
                self.assertGreaterEqual(four, one - 1e-9)
                self.assertLessEqual(four, 4 * one + 1e-9)
        # closed form with two windows and no CPU limit: 2 E[N] / (E[N] + 1), pi = a^(k+1)
        p = ac.resolve_params("base", {"hit": 1.0, "spec": "mtp:1:0.745", "pipeline_windows": 2})
        r = ac.step(spec, sq, p, "pp")
        en = 1 / (1 - 0.745 ** 2)
        self.assertAlmostEqual(r["pipeline_window_mult"], 2 * en / (en + 1), places=9)

    def test_owner_kv_policy_stores_less(self):
        spec, sq = model("glm")
        p = ac.resolve_params("base", {})
        q = ac.quant_of(sq, p)
        self.assertGreater(ac.placement(spec, p, q, 1)["resident_frac"], ac.placement(spec, p, q, 4)["resident_frac"])


class ReportFacts(unittest.TestCase):
    """Numbers quoted in docs/reassessment/research/R9-adversarial-ceilings.md (regression guards)."""

    def test_static_hit_at_glm_residency(self):
        spec, sq = model("glm")
        p = ac.resolve_params("base", {})
        r = ac.placement(spec, p, ac.quant_of(sq, p), 1)["resident_frac"]
        self.assertAlmostEqual(r, 0.607, delta=0.001)
        self.assertAlmostEqual(ac.interp(ac.GLM_STATIC_CURVE, r), 0.907, delta=0.002)

    def test_dflash2_tau(self):
        spec, sq = model("glm")
        p = ac.resolve_params("base", {"spec": "dflash2:7"})
        self.assertAlmostEqual(ac.spec_plan(spec, p, ac.quant_of(sq, p))["tau"], 3.676, places=3)

    def test_dram_bounds_uniform_routing(self):
        spec, sq = model("glm")
        b4 = ac.physics_bounds(spec, sq, ac.resolve_params("base", {"hit": "uniform", "dram_channels": 4}))
        b8 = ac.physics_bounds(spec, sq, ac.resolve_params("base", {"hit": "uniform", "dram_channels": 8}))
        self.assertAlmostEqual(b4["dram_bound_tok_s"], 50.4, delta=0.1)
        self.assertAlmostEqual(b8["dram_bound_tok_s"], 100.9, delta=0.1)
        self.assertAlmostEqual(b8["single_gpu_pp_bound_tok_s"], 61.4, delta=0.1)

    def test_pp_without_speculation_stays_below_physics(self):
        spec, sq = model("glm")
        p = ac.resolve_params("base", {"hit": 1.0, "vram_eff": 1.0})
        self.assertLess(ac.step(spec, sq, p, "pp")["tok_s"], 52.6)


class Cli(unittest.TestCase):
    def run_cli(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ac.main(args)
        return buf.getvalue()

    def test_json_rows_for_every_model(self):
        out = json.loads(self.run_cli(["--model", "glm,qwen,mimo,dsv4f,dsv41", "--arch", "pp,tp_owner",
                                       "--hit", "uniform,1.0", "--json"]))
        self.assertEqual(len(out), 5 * 2 * 2)
        for r in out:
            self.assertNotIn("error", r)
            self.assertGreater(r["tok_s"], 0)

    def test_solver_handles_non_monotone_bounds(self):
        spec, sq = model("glm")
        p = ac.resolve_params("base", {"spec": "mtp:1:0.745", "miss_union": "additive"})
        v, how = ac.solve(spec, sq, p, "pp_pipewin", "hit", 80.0, 0.0, 1.0)
        self.assertEqual(how, "min")
        self.assertLess(v, 0.907)                                            # reached below the static share
        self.assertGreaterEqual(ac.step(spec, sq, dict(p, hit=v + 1e-3), "pp_pipewin")["tok_s"], 80.0 - 1e-6)
        v, how = ac.solve(spec, sq, ac.resolve_params("base", {"hit": "static"}), "tp_owner", "allreduce_us",
                          50.0, 0.0, 300.0)
        self.assertEqual(how, "max")
        self.assertAlmostEqual(tok_s("glm", "tp_owner", hit="static", allreduce_us=v), 50.0, delta=0.01)

    def test_sweep_solve_dump_and_report(self):
        out = self.run_cli(["--model", "glm", "--arch", "tp_owner", "--hit", "static", "--sweep",
                            "allreduce_us=5,60"])
        self.assertIn('"allreduce_us": 5', out)
        out = self.run_cli(["--model", "glm", "--arch", "tp_owner", "--hit", "static", "--solve", "allreduce_us",
                            "--target", "60"])
        self.assertIn("max", out)
        params = json.loads(self.run_cli(["--dump-params", "--preset", "optimistic"]))
        self.assertEqual(params["vram_eff"], 0.9)
        rep = self.run_cli(["--report", "r9"])
        for tag in ("## T1", "## T2", "## T3", "## T4", "## T5", "## T6"):
            self.assertIn(tag, rep)


if __name__ == "__main__":
    unittest.main()
