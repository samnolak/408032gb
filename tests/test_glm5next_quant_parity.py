"""N10 (gap 2 of docs/reassessment/CORRECTNESS.md): quantized weights, glm5-next. llama.cpp's real glm5-next graph on
Q8_0 and Q4_K_M tiny models against ref/glm5next.py on the dequantized weights with the CPU activation quantization
emulated (ref/quant.py), matrix product by matrix product, as tests/test_qwen4exp_quant_parity.py does for qwen4exp
(same metric, tolerances, checks and table; the reasons are written there).

tools/tinygen/run_parity_quant.sh writes the G1 / N9 tiny architecture (tools/tinygen/glm5next_tiny.py --layers 12
--quant-dims: 9 KDA layers, DSA layers 3, 7, 11, a leading dense FFN layer, MoE + shared expert elsewhere), quantizes
it with the pinned llama-quantize to Q8_0 and to Q4_K_M (no option needed: llama-quantize keeps hc_*, the indexer,
ssm_f_* / ssm_g_* / ssm_beta, attn_kv_a_mqa, attn_k_b and attn_v_b of glm5-next F32, src/llama-quant.cpp:333-346) and
dumps the N9 token sequence at -ctk f32 -ctv f32 -fa off: pf64, ub10, decode (and pf64-nr, decode-nr for Q4_K_M).

Products, per graph from llama.cpp's dumped tensors (cb() names of src/models/glm5-next.cpp and src/llama-graph.cpp
@ec7630a; "direct": dumped input -> dumped output, scaled error < TOL_OP):
  KDA layers  attn_q / attn_k / attn_v of attn_norm, through the short conv over [history | this graph] and silu, to
              kda_q_conv / kda_k_conv / kda_v_conv (G:211-233; the history is the projections of the earlier graphs,
              which the test computes from their dumped attn_norm the same way; normwise < TOL_OP);
              attn_output of gated = kda_normed * sigmoid(kda_g2) to kda_out (G:750-759): gated has no name, so the
              test recomputes it in f32 the way ggml does (op_sigmoid 1.f / (1.f + expf(-x)),
              ggml/src/ggml-cpu/unary-ops.cpp:31-33, then the f32 product). numpy's expf may differ from the C
              library's in the last place; an element whose scaled value then sits within 8 f32 spacings of a rounding
              boundary of the activation quantization may round to the neighbouring integer on llama.cpp's side, and
              for Q8_0 a block scale near a binary16 boundary may round the other way: the bound is one quantization
              step times |W| for those (ref/quant.py flip_bound) plus TOL_OP |W| |x_q|, and at least half of the rows
              within TOL_OP (measured: every row; a float64 gate rounded once to f32 instead flipped the f16 scale
              of one block and needed that term)
  DSA layers  attn_q_a of attn_norm, rms_norm * attn_q_a_norm = q_resid (G:937-939; normwise < TOL_OP);
              attn_q_b of q_resid, absorbed by the F32 attn_k_b = q_absorbed (G:941-953; normwise < TOL_OP);
              attn_output of kqv_out = attn_out (G:1007-1010; direct)
  FFN         layer 0 (dense): ffn_up, ffn_gate of ffn_norm (direct), ffn_down of ffn_swiglu_limited, named ffn_out
              (the later cb(), G:617-622; direct); MoE layers: ffn_moe_logits (F32 router), ffn_moe_gate / ffn_moe_up
              (mul_mat_id, ids from ffn_moe_topk), ffn_moe_down of ffn_moe_swiglu_limited, the shared expert's ffn_up,
              ffn_gate and its down projection (named ffn_shexp, G:637-642); all direct
  head        result_output of result_norm (direct)
  gathers     inp_embd = the token_embd rows, exact
Env: GLM5NEXT_QUANT_DUMP (root: <type>/model.gguf, <type>/tokens.txt, <type>/<run>/), LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT.
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from ref import glm5next as G  # noqa: E402
from ref import quant as QT  # noqa: E402
from tests import test_qwen4exp_quant_parity as base  # noqa: E402

DUMP = os.environ.get("GLM5NEXT_QUANT_DUMP")
SKIP = "set GLM5NEXT_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh"
TABLE = []
FLIP_ULPS = 8


def _same(a):
    return a


class _GlmQuantOps(base._QuantOpsBase):
    ROOT, SKIP, TABLE = DUMP, SKIP, TABLE
    HPARAMS = staticmethod(G.hparams)
    NOT_MATMUL = ("token_embd.weight",)

    @classmethod
    def _products(cls, gr):
        hp, n, D = cls.hp, gr["n"], cls._dump
        g = gr["g"]
        out = []

        def add(site, wnames, il, got, fn, bound=None):
            out.append((site, (wnames,) if isinstance(wnames, str) else wnames, il, g, got, fn, bound))

        def lw(W, il):
            return G.layer_weights(W, il)

        def rows(x, wname, il):
            def f(W, A=_same):
                return A(x) @ (lw(W, il)[wname] if il >= 0 else W[wname]).T
            return f

        def experts(x, wname, il, ids, per_slot):
            def f(W, A=_same):
                w = lw(W, il)[wname]
                return np.stack([[w[e] @ A(x[t, j] if per_slot else x[t]) for j, e in enumerate(ids[t])]
                                 for t in range(n)])
            return f

        def zero(W):
            return 0.0

        def first(*names):          # the swiglu output name depends on the layer's clamp
            return next(nm for nm in names if cls.R.has(g, nm))

        H, Dh, K = hp["n_head"], hp["kda_head_dim"], hp["d_conv"]
        for il in range(hp["n_layer"]):
            b = f"blk.{il}."
            x = D(gr, f"attn_norm-{il}")
            if hp["kda"][il]:
                # the projections of every earlier graph of the run feed this graph's conv history
                xs = np.vstack([D(p, f"attn_norm-{il}") for p in cls.R.graphs if p["g"] <= g])
                for c in "qkv":
                    def conv(W, A=_same, il=il, c=c, xs=xs):
                        w = lw(W, il)
                        proj = A(xs) @ w[f"attn_{c}.weight"].T
                        cw = np.asarray(w[f"ssm_conv1d_{c}.weight"], dtype=np.float64).reshape(H * Dh, -1)
                        raw, _ = G.conv_hist(proj, cw, np.zeros((K - 1, H * Dh)))
                        return G.silu(raw)[-n:]
                    add(f"kda_{c}_conv (attn_{c}, conv, silu)", f"{b}attn_{c}.weight", il,
                        D(gr, f"kda_{c}_conv-{il}"), conv, zero)
                # in f32 as ggml computes it (op_sigmoid 1.f / (1.f + expf(-x)), then the f32 product); numpy's expf
                # may differ from the C library's in the last place, which flip_bound covers
                g2 = D(gr, f"kda_g2-{il}").astype(np.float32)
                gated = D(gr, f"kda_normed-{il}").astype(np.float32) * (np.float32(1) / (np.float32(1) + np.exp(-g2)))
                add("kda_out (attn_output of the recomputed gate)", f"{b}attn_output.weight", il,
                    D(gr, f"kda_out-{il}"), rows(gated, "attn_output.weight", il),
                    lambda W, il=il, gated=gated: QT.flip_bound(gated, lw(W, il)["attn_output.weight"], FLIP_ULPS)
                    + base.TOL_OP * float(np.max(np.abs(gated) @ np.abs(np.asarray(
                        lw(W, il)["attn_output.weight"])).T)))
            else:
                def q_resid(W, A=_same, il=il, x=x):
                    w = lw(W, il)
                    return G.rms_norm(A(x) @ w["attn_q_a.weight"].T, hp["rms_eps"], w["attn_q_a_norm.weight"])
                add("q_resid (attn_q_a, rms_norm)", f"{b}attn_q_a.weight", il, D(gr, f"q_resid-{il}"), q_resid, zero)
                qr = D(gr, f"q_resid-{il}")

                def q_abs(W, A=_same, il=il, qr=qr):
                    w = lw(W, il)
                    q = (A(qr) @ w["attn_q_b.weight"].T).reshape(n, H, -1)
                    return np.einsum("hld,thd->thl", np.asarray(w["attn_k_b.weight"], dtype=np.float64), q)
                add("q_absorbed (attn_q_b, F32 attn_k_b)", f"{b}attn_q_b.weight", il,
                    D(gr, f"q_absorbed-{il}", rows=False), q_abs, zero)
                add("attn_out (attn_output of kqv_out)", f"{b}attn_output.weight", il, D(gr, f"attn_out-{il}"),
                    rows(D(gr, f"kqv_out-{il}"), "attn_output.weight", il))
            x2 = D(gr, f"ffn_norm-{il}")
            if il < hp["dense_lead"]:
                for name, wn in (("ffn_up (dense)", "ffn_up"), ("ffn_gate (dense)", "ffn_gate")):
                    add(name, f"{b}{wn}.weight", il, D(gr, f"{wn}-{il}"), rows(x2, f"{wn}.weight", il))
                sw = D(gr, first(f"ffn_swiglu_limited-{il}", f"ffn_swiglu-{il}"))
                add("ffn_out (dense down)", f"{b}ffn_down.weight", il, D(gr, f"ffn_out-{il}"),
                    rows(sw, "ffn_down.weight", il))
                continue
            ids = D(gr, f"ffn_moe_topk-{il}").astype(int)
            ke = ids.shape[1]
            add("ffn_moe_logits", f"{b}ffn_gate_inp.weight", il, D(gr, f"ffn_moe_logits-{il}"),
                rows(x2, "ffn_gate_inp.weight", il))
            for name, wn in (("ffn_moe_gate", "ffn_gate_exps"), ("ffn_moe_up", "ffn_up_exps")):
                add(name, f"{b}{wn}.weight", il, D(gr, f"{name}-{il}", rows=False).reshape(n, ke, -1),
                    experts(x2, f"{wn}.weight", il, ids, False))
            sw = D(gr, first(f"ffn_moe_swiglu_limited-{il}", f"ffn_moe_swiglu-{il}"), rows=False).reshape(n, ke, -1)
            add("ffn_moe_down", f"{b}ffn_down_exps.weight", il,
                D(gr, f"ffn_moe_down-{il}", rows=False).reshape(n, ke, -1),
                experts(sw, "ffn_down_exps.weight", il, ids, True))
            for name, wn in (("ffn_up (shared expert)", "ffn_up_shexp"),
                             ("ffn_gate (shared expert)", "ffn_gate_shexp")):
                add(name, f"{b}{wn}.weight", il, D(gr, f"{name.split()[0]}-{il}"), rows(x2, f"{wn}.weight", il))
            ss = D(gr, first(f"ffn_swiglu_limited-{il}", f"ffn_swiglu-{il}"))
            add("ffn_shexp (shared expert down)", f"{b}ffn_down_shexp.weight", il, D(gr, f"ffn_shexp-{il}"),
                rows(ss, "ffn_down_shexp.weight", il))
        add("result_output", "output.weight", -1, D(gr, "result_output"),
            rows(D(gr, "result_norm"), "output.weight", -1))
        return out

    def test_weight_types(self):
        """llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-...,
        glm5-next exclusions 333-346): F32 stays for 1-D tensors, *_norm.weight, ffn_gate_inp, ssm_conv1d_*, hc_*,
        the indexer, ssm_f_* / ssm_g_* / ssm_beta, attn_kv_a_mqa, attn_k_b, attn_v_b; everything else the target type
        (Q4_K_M: a k-quant mix with Q8_0 promotions)."""
        keep_sub = ("hc_", "indexer", "ssm_f_a", "ssm_f_b", "ssm_g_a", "ssm_g_b", "ssm_beta", "attn_kv_a_mqa",
                    "attn_k_b", "attn_v_b", "ssm_conv1d", "ffn_gate_inp")
        for name, ty in self.types.items():
            ne = self.shapes[name]
            if len(ne) < 2 or name.endswith("_norm.weight") or any(k in name for k in keep_sub):
                self.assertEqual(ty, "F32", name)
            elif ne[0] % 32:
                self.assertEqual(ty, "F16", name)
            elif self.QTYPE == "Q8_0":
                self.assertEqual(ty, "Q8_0", name)
            else:
                self.assertIn(ty, ("Q4_K", "Q5_K", "Q6_K", "Q8_0"), name)

    def test_every_type_checked(self):
        checked = self.checked_types()
        if self.QTYPE == "Q4_K_M":
            self.assertTrue({"Q4_K", "Q5_K", "Q6_K", "Q8_0"} <= checked, sorted(checked))
        else:
            self.assertEqual(checked, {"Q8_0", "F32"})

    def test_gathers_exact(self):
        for gr in self.R.graphs:
            want = self.W["token_embd.weight"][gr["toks"]]
            self.assertTrue(np.array_equal(self._dump(gr, "inp_embd"), want), f"inp_embd g{gr['g']}")
            self.TABLE.append((self.QTYPE, self.RUN, "inp_embd (gather, exact)", self.types["token_embd.weight"], -1,
                               gr["g"], 0.0, 0.0, 0.0, 0.0))


def _cls(qtype, run):
    name = f"Glm_{qtype}_{run.replace('-', '_')}"
    return type(name, (_GlmQuantOps, unittest.TestCase), dict(QTYPE=qtype, RUN=run))


Glm_Q8_0_pf64 = _cls("Q8_0", "pf64")
Glm_Q8_0_ub10 = _cls("Q8_0", "ub10")
Glm_Q8_0_decode = _cls("Q8_0", "decode")
Glm_Q4_K_M_pf64 = _cls("Q4_K_M", "pf64")
Glm_Q4_K_M_ub10 = _cls("Q4_K_M", "ub10")
Glm_Q4_K_M_decode = _cls("Q4_K_M", "decode")
Glm_Q4_K_M_pf64_nr = _cls("Q4_K_M", "pf64-nr")
Glm_Q4_K_M_decode_nr = _cls("Q4_K_M", "decode-nr")


def tearDownModule():
    base.print_table(TABLE, "glm5-next")


if __name__ == "__main__":
    unittest.main()
