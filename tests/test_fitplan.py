"""fitplan must reproduce sizes published independently of us.

Each expected value cites where it was published. If one of these fails, the geometry in
models/*.json or the bpw table is wrong, and every placement number downstream is wrong too.
"""
import json
import os
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fitplan  # noqa: E402
from quant_sizes import GGML_BLOCKS, bpw  # noqa: E402

GIB = 1024 ** 3


def load(name):
    return json.loads((ROOT / "models" / f"{name}.json").read_text())


HW = json.loads((ROOT / "hardware" / "4x4080s-32g.json").read_text())


class PublishedSizes(unittest.TestCase):
    def test_deepseek_routed_experts_mxfp4(self):
        # LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4: routed experts 268.9 GiB as shipped (E2M1 + E8M0/32)
        m = load("deepseek-v4.1-flash")
        routed = m["moe"]["moe_layers"] * m["moe"]["n_routed"] * fitplan.expert_params(m)
        self.assertAlmostEqual(routed * bpw("MXFP4") / 8 / GIB, 268.9, delta=268.9 * 0.005)

    def test_deepseek_routed_param_count(self):
        # pipenetwork MLX build: routed experts 40 x 384 (w1/w2/w3) = 543.6B
        m = load("deepseek-v4.1-flash")
        routed = m["moe"]["moe_layers"] * m["moe"]["n_routed"] * fitplan.expert_params(m)
        self.assertAlmostEqual(routed / 1e9, 543.6, delta=0.1)

    def test_deepseek_engram(self):
        # LibertAIDAI: Engram tables 189.1 GiB (E4M3 + E8M0/32); 2 x 384,006,168 x 256 entries
        e = load("deepseek-v4.1-flash")["engram"]
        self.assertAlmostEqual(2 * e["rows_per_table"] * e["dim"] / 1e9, e["params_b"], delta=0.05)
        self.assertAlmostEqual(e["params_b"] * 1e9 * bpw("FP8_E8M0_32") / 8 / GIB, 189.1, delta=189.1 * 0.005)

    def test_mimo_mxfp4_experts(self):
        # MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks: 64 expert shards, ~150 GiB of MXFP4 routed experts
        p = fitplan.plan(load("mimo-v2.6-flash"), HW, "MXFP4", 4096)
        self.assertAlmostEqual(p["experts_gib"], 150.0, delta=3.0)

    def test_glm_expert_count(self):
        # arXiv 2609.09793 Table 1: expert down-projections, 288 experts x 43 sparse layers = 12,384
        p = fitplan.plan(load("glm-5.3-flash"), HW, "IQ2_XS", 4096)
        self.assertEqual(round(p["n_experts"]), 12384)

    def test_glm_param_budget(self):
        m = load("glm-5.3-flash")
        routed = fitplan.routed_expert_params_total(m) / 1e9
        self.assertLess(routed, m["params_b"]["total"])
        self.assertAlmostEqual(fitplan.dense_params(m) / 1e9, 7.8, delta=1.0)

    def test_qwen_strata_reference(self):
        # Strata README: 24,576 experts. Strata DETAILS.md: every extra GB of VRAM holds ~700 experts (IQ2_XS)
        m = load("qwen3.8-flash-next")
        p = fitplan.plan(m, HW, "IQ2_XS", 4096)
        self.assertEqual(round(p["n_experts"]), 24576)
        per_gb = 1e9 / (fitplan.expert_params(m) * bpw("IQ2_XS") / 8)
        self.assertAlmostEqual(per_gb, 700, delta=35)


class Behaviour(unittest.TestCase):
    def test_more_bits_less_resident(self):
        m = load("glm-5.3-flash")
        fr = [fitplan.plan(m, HW, q, 32768)["resident_frac"] for q in ("IQ2_XS", "IQ3_XXS", "Q4_K", "Q8_0")]
        self.assertEqual(fr, sorted(fr, reverse=True))
        self.assertEqual(fr[0], 1.0)

    def test_more_context_less_resident(self):
        m = load("glm-5.3-flash")
        a = fitplan.plan(m, HW, "Q4_K", 8192)["resident_frac"]
        b = fitplan.plan(m, HW, "Q4_K", 524288)["resident_frac"]
        self.assertGreater(a, b)

    def test_hit_curve_interpolation(self):
        curve = [[0.0, 0.0], [0.25, 0.6], [0.5, 0.85], [1.0, 1.0]]
        self.assertAlmostEqual(fitplan.interp_hit(curve, 0.375), 0.725)
        self.assertEqual(fitplan.interp_hit(curve, 1.0), 1.0)
        p_uniform = fitplan.plan(load("glm-5.3-flash"), HW, "Q4_K", 32768)
        p_curve = fitplan.plan(load("glm-5.3-flash"), HW, "Q4_K", 32768, hit_curve=curve)
        self.assertGreater(p_curve["hit_rate"], p_uniform["hit_rate"])
        self.assertLess(p_curve["decode_cpu_mib_per_tok"], p_uniform["decode_cpu_mib_per_tok"])

    def test_deepseek_native_does_not_fit(self):
        # ADR-001: FP4 routed experts exceed VRAM + RAM on this server
        p = fitplan.plan(load("deepseek-v4.1-flash"), HW, "MXFP4", 32768)
        self.assertFalse(p["disjoint_fits"])
        self.assertGreater(p["nvme_spill_gib"], 0)

    def test_cli_runs_for_every_model(self):
        for f in sorted((ROOT / "models").glob("*.json")):
            out = subprocess.run([sys.executable, str(ROOT / "tools/fitplan.py"), str(f), "--ctx", "8192"],
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn("VRAM-resident experts", out, f)


class QuantTable(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(bpw("IQ2_XS"), 2.3125)
        self.assertEqual(bpw("IQ3_XXS"), 3.0625)
        self.assertEqual(bpw("MXFP4"), 4.25)
        self.assertEqual(bpw("Q8_0"), 8.5)

    @unittest.skipUnless(os.environ.get("LLAMA_CPP_GGUF_PY"), "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py")
    def test_matches_pinned_gguf_py(self):
        sys.path.insert(0, os.environ["LLAMA_CPP_GGUF_PY"])
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
        for name, (block, size) in GGML_BLOCKS.items():
            self.assertEqual(tuple(GGML_QUANT_SIZES[getattr(T, name)]), (block, size), name)


if __name__ == "__main__":
    unittest.main()


class StrataGlmSizes(unittest.TestCase):
    def test_nvfp4_expert_and_pack(self):
        # sergqwer/strata-glm README @ed37419: 12,096 routed experts of 14.16 MB each, a ~171 GB pack
        m = load("glm-5.3-flash")
        per = fitplan.expert_params(m) * bpw("NVFP4") / 8
        self.assertAlmostEqual(per / 1e6, 14.16, delta=0.01)
        main_layers = m["moe"]["moe_layers"] * m["moe"]["n_routed"]
        self.assertEqual(main_layers, 12096)
        self.assertAlmostEqual(main_layers * per / 1e9, 171.0, delta=1.0)
