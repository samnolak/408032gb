"""N10 (gap 2 of docs/reassessment/CORRECTNESS.md): quantized weights. llama.cpp's real qwen4exp graph on Q8_0 and
Q4_K_M tiny models against ref/qwen4exp.py on the dequantized weights, with the CPU activation quantization of
ggml's matrix products emulated (ref/quant.py), matrix product by matrix product.

tools/tinygen/run_parity_quant.sh writes the N2/N6 tiny architecture with every quantizable input width a multiple
of 256 (tools/tinygen/qwen4exp_tiny.py --quant-dims), quantizes it with the pinned llama.cpp's llama-quantize to Q8_0
and to Q4_K_M (the PLE gammas kept F32, see the script) and dumps, graph by graph, the N6 token sequence at the f32
cache / flash-attention-off contract of N2/N6: pf64 (one 64-token ubatch), ub10 (10 x 6 + 4), decode (48 + 16 x 1),
and for Q4_K_M also pf64-nr / decode-nr with --no-repack (Q4_K weights then stay in the plain layout: the activations
go through quantize_row_q8_K instead of the four-row ggml_quantize_mat_q8_K_4x8 of the repacked path).

Checks, per graph (llama.cpp's dumped input -> its dumped output, "teacher forcing": no oracle state is involved):
  test_matmuls          every matrix product of the graph (names from cb() in src/models/qwen4exp.cpp and
                        src/llama-graph.cpp @ec7630a): hc_inject = hc_norm W_inject (both mixers of every layer),
                        hc_gate = f16(silu(hc_norm W_down / hc)) W_up (F16 up projection), the attention projections
                        Qcur_full, Kcur, Vcur, indexer_k_raw (F32) from hc_mixed, attn_output from attn_gated, the GDN
                        projections linear_attn_qkv_mixed, z, beta, alpha from hc_mixed and linear_attn_out from
                        final_output, the router ffn_moe_logits (F32), the selected experts ffn_moe_gate / ffn_moe_up
                        (mul_mat_id, ids from ffn_moe_topk) and ffn_moe_down from ffn_moe_swiglu, the shared expert
                        ffn_up / ffn_gate and its down projection (named ffn_shexp, the later cb() name) from
                        ffn_swiglu, shared_expert_gate (F32), the head mixer and result_output from result_norm.
                        Each against ref/quant.py with the activations quantized (asserted) and without (recorded).
  test_ple_module       ple_gate, ple_gated_value, ple_conv_out from the dumped ple_embd and layer input (two
                        quantized products, ple_key and ple_value, inside)
  test_gathers_exact    hc_init (token_embd rows) and ple_embd (per_layer_token_embd rows): ggml_get_rows of the
                        quantized tables equals gguf.quants.dequantize of the pinned gguf-py, bit for bit
  test_weight_types     the per-tensor types llama-quantize chose (read with the pinned gguf-py), by the rules of
                        src/llama-quant.cpp (tensor_allows_quantization, tensor_type_fallback)
  test_every_type_checked  every type of a weight that enters a matrix product was checked in this run
  test_activation_quantization_visible  per product and weight type, the error without the activation quantization
                        exceeds TOL_OP in every graph and 20 TOL_OP in some: the check can tell the two apart

Metric and tolerances:
  * scaled error = max|got - want| / max(|W| |x_q|) over the output tensor, |x_q| the absolute quantized activations
    (act_quant(|x|) = |act_quant(x)|), i.e. the error relative to the size of the terms the product sums: f32
    rounding errors scale with that, not with the (possibly cancelled) result. TOL_OP = 1e-5. The emulation feeds the
    same f32 input through the same activation quantization (same integers and scales) against the same dequantized
    weights, so the integer block dots agree exactly; what remains is llama.cpp's f32 arithmetic on the block scales
    and the sum over the blocks: to first order <= (n_b + 3) 2^-24 times the summed magnitudes (n_b blocks; the
    int -> float conversion, the scale product and the accumulation). Q8_0 x Q8_0: n_b = K/32 <= 32 (K <= 1024),
    bound 2.1e-6. k-quant x Q8_K: n_b = K/256 <= 4, but the scale term d sc q and the minimum term dmin m are rounded
    separately and each is at most about 3 times the block's largest |w|, which is about 3 times the mean |w| of
    Gaussian weights: bound (4 + 3) 2^-24 x 9 = 3.8e-6 (PROVISIONAL derivation). TOL_OP covers both with a margin
    >= 2.6. Measured worst over both types, every run and both pins (docs/evidence/2026-10-10-quant-fa-parity.md):
    scaled 2.4e-7 (normwise 3.8e-6). The products with F32 weights (router, indexer projection, shared-expert gate)
    meet the same tolerance (worst scaled 4.9e-8) and their two oracles coincide (no activation conversion).
  * test_activation_quantization_visible: without the activation quantization the scaled error exceeds TOL_OP in
    every graph and VISIBLE = 20 TOL_OP in some graph of the run, for every product and weight type.
  * hc_gate (and the head's): the F16 up projection rounds silu(hc_norm W_down / hc) to f16, and llama.cpp's f32 value
    of that input and the oracle's float64 one can fall on two sides of a binary16 rounding boundary (measured: 0 to 6
    of the 1088 rows of a run have such an input; ref/fattn.py ulp16). Bound per element:
    sum_j ulp16(a_j) |W_up[i, j]| (one binary16 spacing per input) + TOL_OP max|want|; and at least CHAIN_AGREE =
    50 % of the rows (tokens) of a run within TOL_OP max|want| (rows whose inputs round alike; an oracle without the
    f16 rounding misses nearly all).
  * PLE module (ple_gate, ple_gated_value, ple_conv_out): normwise max|got - want| / max|want| < TOL_OP.
  * gathers: exact (tolerance 0).
Why no whole-model comparison from the token ids here: a difference of one f32 rounding (1e-7 relative) in a product's
input moves an activation that sits near a rounding boundary of the 8-bit activation quantization by a whole step
(max|x| / 127), and those steps grow through the random tiny model: the full oracle from token ids drifts to 4e-2 by
the last layer although every product agrees from its own input to a scaled error of 2.4e-7 (normwise 3.8e-6)
(measured; the probe and its output are in docs/evidence/2026-10-10-quant-fa-parity.md). A whole-model comparison of
quantized weights needs a statistical metric; the per-product check is the exact one.

The model-independent part (_QuantOpsBase: the evaluation of every product once per run, test_matmuls,
test_activation_quantization_visible, the coverage helper, print_table) is shared with
tests/test_glm5next_quant_parity.py.

Env: QWEN4EXP_QUANT_DUMP (root: <type>/model.gguf, <type>/tokens.txt, <type>/<run>/), LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT.
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from ref import fattn as F  # noqa: E402
from ref import quant as QT  # noqa: E402
from ref import qwen4exp as Q  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402
from tests.test_qwen4exp_tiny_parity import read_hparams  # noqa: E402

DUMP = os.environ.get("QWEN4EXP_QUANT_DUMP")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "UNKNOWN")
SKIP = "set QWEN4EXP_QUANT_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_quant.sh"
TOL_OP = 1e-5
VISIBLE = 20 * TOL_OP
CHAIN_AGREE = 0.5   # per bounded product: share of its rows (all layers and graphs of a run) within TOL_OP
TABLE = []          # (type, run, site, weight type, layer, graph, rel emulated, rel without activation quantization)
_MODELS = {}
# tensors that never enter a matrix product: gathered (get_rows) or cast to f32 (the PLE conv kernel)
NOT_MATMUL = ("token_embd.weight", "per_layer_token_embd.weight", "ple_conv1d.weight")


def model(path, act, hparams=read_hparams):
    key = (str(path), act)
    if key not in _MODELS:
        sys.path.append(GGUF_PY)
        try:
            from gguf import GGUFReader
            from gguf.quants import dequantize
        finally:
            sys.path.remove(GGUF_PY)
        r = GGUFReader(str(path))
        _MODELS[key] = (QT.load(r, dequantize, act), hparams({f.name: f for f in r.fields.values()}),
                        {t.name: t.tensor_type.name for t in r.tensors},
                        {t.name: [int(x) for x in t.shape] for t in r.tensors})
    return _MODELS[key]


def _same(a):
    return a


def absolute(W):
    """|W| with the same activation conversion: act_quant(|x|) = |act_quant(x)| (sign-symmetric rounding, one scale
    per block), so f(absolute(W), np.abs) is |W| |x_q|."""
    return {k: QT.QWeight(np.abs(v.w), v.qtype, v.act) if isinstance(v, QT.QWeight) else np.abs(v)
            for k, v in W.items()}


def scaled(got, want, scale):
    """max |got - want| / max (|W| |x_q|): the error relative to the size of the terms summed (module docstring)."""
    want = np.asarray(want, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64).reshape(want.shape)
    return float(np.abs(got - want).max() / (float(np.max(scale)) + 1e-30))


def rel(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    got = got.reshape(want.shape)
    return float(np.abs(got - want).max() / (float(np.abs(want).max()) + 1e-30))


class _QuantOpsBase:
    """Model-independent part: setUpClass evaluates every product of every graph of one run (QTYPE / RUN under ROOT)
    once; test_matmuls and test_activation_quantization_visible check them. A model's subclass gives _products,
    HPARAMS (GGUF fields -> hyper-parameters), NOT_MATMUL, SKIP, its own TABLE and the model-specific tests."""
    QTYPE = RUN = ROOT = SKIP = HPARAMS = None
    TABLE = None
    NOT_MATMUL = ()

    @classmethod
    def setUpClass(cls):
        if not (cls.ROOT and GGUF_PY):
            raise unittest.SkipTest(cls.SKIP)
        cls.dir = pathlib.Path(cls.ROOT, cls.QTYPE)
        if not (cls.dir / cls.RUN / "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.QTYPE}/{cls.RUN} under {cls.ROOT}")
        cls.W, cls.hp, cls.types, cls.shapes = model(cls.dir / "model.gguf", True, cls.HPARAMS)
        cls.W0 = model(cls.dir / "model.gguf", False, cls.HPARAMS)[0]
        cls.Wabs = absolute(cls.W)
        cls.R = Run(cls.RUN, cls.dir)
        # every product evaluated once: (site, weight names, layer, graph, got, want, without activation
        # quantization, scale |W| |x_q| or None, elementwise bound or None)
        cls.results = []
        for gr in cls.R.graphs:
            for site, wnames, il, g, got, fn, bound in cls._products(gr):
                cls.results.append((site, wnames, il, g, np.asarray(got, dtype=np.float64), fn(cls.W), fn(cls.W0),
                                    None if bound else fn(cls.Wabs, np.abs), bound(cls.W) if bound else None))

    @classmethod
    def _dump(cls, gr, name, k=0, rows=True):
        a, _ = cls.R.dump(gr["g"], name, k)
        return a.reshape(gr["n"], -1) if rows else a

    # ---- tests ----
    def ty(self, wnames):
        return "+".join(self.types[w] for w in wnames)

    def test_matmuls(self):
        agree = {}          # bounded site -> [rows within TOL_OP, rows]
        for site, wnames, il, g, got, want, plain, sc, bound in self.results:
            ty = self.ty(wnames)
            e, p = rel(got, want), rel(got, plain)
            if bound is not None:
                lim = bound + TOL_OP * np.abs(want).max()
                diff = np.abs(np.asarray(got).reshape(want.shape) - want)
                self.TABLE.append((self.QTYPE, self.RUN, site, ty, il, g, float((diff / lim).max()), e, np.nan, p))
                self.assertLessEqual(float((diff - lim).max()), 0.0,
                                     f"{self.QTYPE}/{self.RUN} {site} L{il} g{g}: beyond the bound (rel {e:.2e})")
                d2 = diff.reshape(diff.shape[0], -1)
                a = agree.setdefault(site, [0, 0])
                a[0] += int((d2.max(axis=1) <= TOL_OP * np.abs(want).max()).sum())
                a[1] += d2.shape[0]
                continue
            es, ps = scaled(got, want, sc), scaled(got, plain, sc)
            self.TABLE.append((self.QTYPE, self.RUN, site, ty, il, g, es, e, ps, p))
            self.assertLess(es, TOL_OP, f"{self.QTYPE}/{self.RUN} {site} ({ty}) L{il} g{g}: scaled error {es:.2e} "
                                        f"(normwise {e:.2e}; without activation quantization {ps:.2e})")
            if ty == "F32":     # no conversion of the activations (vec_dot_type F32): both oracles agree
                self.assertEqual(e, p)
        # bounded sites: a row (token) whose inputs all round alike matches within TOL_OP; only rows with an input on
        # a rounding boundary may differ (an oracle without the rounding there would miss nearly every row)
        for site, (ok, rows) in agree.items():
            self.TABLE.append((self.QTYPE, self.RUN, f"{site}: rows within TOL_OP {ok}/{rows}", "-", -1, -1,
                               np.nan, np.nan, np.nan, np.nan))
            self.assertGreaterEqual(ok / max(rows, 1), CHAIN_AGREE, f"{site}: rows within TOL_OP: {ok}/{rows}")

    def test_activation_quantization_visible(self):
        """Per product and weight type: without the activation quantization the scaled error exceeds TOL_OP in every
        graph and reaches VISIBLE = 20 TOL_OP in some graph of the run (the check tells the emulation from its
        absence)."""
        low, high = {}, {}
        for site, wnames, il, g, got, want, plain, sc, bound in self.results:
            ty = self.ty(wnames)
            if ty == "F32" or bound is not None:
                continue
            ps = scaled(got, plain, sc)
            low[(site, ty)] = min(low.get((site, ty), np.inf), ps)
            high[(site, ty)] = max(high.get((site, ty), 0.0), ps)
        self.assertTrue(low)
        for key in low:
            self.assertGreater(low[key], TOL_OP, f"{self.QTYPE}/{self.RUN} {key}: without activation quantization "
                                                 f"only {low[key]:.2e} in some graph")
            self.assertGreater(high[key], VISIBLE, f"{self.QTYPE}/{self.RUN} {key}: without activation quantization "
                                                   f"at most {high[key]:.2e}")

    def checked_types(self):
        """The weight types the products of this run used; every non-F32 type of a matrix-product weight of the model
        must be among them."""
        model_types = {ty for name, ty in self.types.items() if ty != "F32" and not name.endswith(self.NOT_MATMUL)}
        checked = {self.types[w] for _, wnames, *_ in self.results for w in wnames}
        self.assertTrue(model_types <= checked, f"weight types never checked: {sorted(model_types - checked)}")
        return checked


class _QuantOps(_QuantOpsBase):
    ROOT, SKIP, TABLE = DUMP, SKIP, TABLE
    HPARAMS = staticmethod(read_hparams)
    NOT_MATMUL = NOT_MATMUL

    @classmethod
    def _products(cls, gr):
        """(site, weight names, layer, graph, got, f(W, A) -> want, bound) for every matrix product of graph gr.
        A is applied to every activation input (np.abs for the scale |W| |x_q|, see scaled()); bound: None (TOL_OP
        on the scaled error) or f(W) -> elementwise bound (the hc_gate chains)."""
        hp, n, D = cls.hp, gr["n"], cls._dump
        out = []

        def add(site, wnames, il, got, fn, bound=None):
            out.append((site, (wnames,) if isinstance(wnames, str) else wnames, il, gr["g"], got, fn, bound))

        def rows(x, wname, il):           # x @ W.T, one activation row per token (il -1: a model-level weight)
            def f(W, A=_same):
                return A(x) @ (Q.layer_weights(W, il)[wname] if il >= 0 else W[wname]).T
            return f

        def experts(x, wname, il, ids, per_slot):   # mul_mat_id: W[e] @ x[t] (x[t, j] for the down projection)
            def f(W, A=_same):
                w = Q.layer_weights(W, il)[wname]
                return np.stack([[w[e] @ A(x[t, j] if per_slot else x[t]) for j, e in enumerate(ids[t])]
                                 for t in range(n)])
            return f

        for il in range(hp["n_layer"]):
            b = f"blk.{il}."
            for k, mix in ((0, "attn"), (1, "ffn")):
                xn = D(gr, f"hc_norm-{il}", k)
                add(f"hc_inject ({mix} mixer)", f"{b}hc_{mix}_inject.weight", il, D(gr, f"hc_inject-{il}", k),
                    rows(xn, f"hc_{mix}_inject.weight", il))
                wd, wu = f"hc_{mix}_down.weight", f"hc_{mix}_up.weight"
                add(f"hc_gate ({mix} mixer: down, f16 up)", (b + wd, b + wu), il, D(gr, f"hc_gate-{il}", k),
                    lambda W, A=_same, xn=xn, il=il, wd=wd, wu=wu: cls._gate(xn, Q.layer_weights(W, il)[wd],
                                                                           Q.layer_weights(W, il)[wu]),
                    lambda W, xn=xn, il=il, wd=wd, wu=wu: cls._gate_bound(xn, Q.layer_weights(W, il)[wd],
                                                                          Q.layer_weights(W, il)[wu]))
            x0 = D(gr, f"hc_mixed-{il}", 0)
            if hp["recurrent"][il]:
                for name, wn in (("linear_attn_qkv_mixed", "attn_qkv"), ("z", "attn_gate"), ("beta", "ssm_beta"),
                                 ("alpha", "ssm_alpha")):
                    add(name, f"{b}{wn}.weight", il, D(gr, f"{name}-{il}"), rows(x0, f"{wn}.weight", il))
                add("linear_attn_out", f"{b}ssm_out.weight", il, D(gr, f"linear_attn_out-{il}"),
                    rows(D(gr, f"final_output-{il}"), "ssm_out.weight", il))
            else:
                for name, wn in (("Qcur_full", "attn_q"), ("Kcur (projection)", "attn_k"),
                                 ("Vcur (projection)", "attn_v"), ("indexer_k_raw", "indexer.k_proj")):
                    if wn.startswith("indexer") and not hp["compress"][il]:
                        continue                                  # a dense layer: no QSA indexer in the graph
                    add(name, f"{b}{wn}.weight", il, D(gr, f"{name.split()[0]}-{il}", 0), rows(x0, f"{wn}.weight", il))
                add("attn_output", f"{b}attn_output.weight", il, D(gr, f"attn_output-{il}"),
                    rows(D(gr, f"attn_gated-{il}"), "attn_output.weight", il))
            x1 = D(gr, f"hc_mixed-{il}", 1)
            ids = D(gr, f"ffn_moe_topk-{il}").astype(int)
            ke = ids.shape[1]
            add("ffn_moe_logits", f"{b}ffn_gate_inp.weight", il, D(gr, f"ffn_moe_logits-{il}"),
                rows(x1, "ffn_gate_inp.weight", il))
            for name, wn in (("ffn_moe_gate", "ffn_gate_exps"), ("ffn_moe_up", "ffn_up_exps")):
                add(name, f"{b}{wn}.weight", il, D(gr, f"{name}-{il}", rows=False).reshape(n, ke, -1),
                    experts(x1, f"{wn}.weight", il, ids, False))
            sw = D(gr, f"ffn_moe_swiglu-{il}", rows=False).reshape(n, ke, -1)
            add("ffn_moe_down", f"{b}ffn_down_exps.weight", il,
                D(gr, f"ffn_moe_down-{il}", rows=False).reshape(n, ke, -1),
                experts(sw, "ffn_down_exps.weight", il, ids, True))
            for name, wn in (("ffn_up (shared expert)", "ffn_up_shexp"),
                             ("ffn_gate (shared expert)", "ffn_gate_shexp")):
                add(name, f"{b}{wn}.weight", il, D(gr, f"{name.split()[0]}-{il}"), rows(x1, f"{wn}.weight", il))
            add("ffn_shexp (shared expert down)", f"{b}ffn_down_shexp.weight", il, D(gr, f"ffn_shexp-{il}"),
                rows(D(gr, f"ffn_swiglu-{il}"), "ffn_down_shexp.weight", il))
            add("shared_expert_gate", f"{b}ffn_gate_inp_shexp.weight", il, D(gr, f"shared_expert_gate-{il}"),
                lambda W, A=_same, il=il, x1=x1: (A(x1) @ Q.layer_weights(W, il)["ffn_gate_inp_shexp.weight"])[:, None])
        hn = D(gr, "hc_norm")
        add("hc_gate (head mixer: down, f16 up)", ("output_hc_down.weight", "output_hc_up.weight"), -1,
            D(gr, "hc_gate"), lambda W, A=_same: cls._gate(hn, W["output_hc_down.weight"], W["output_hc_up.weight"]),
            lambda W: cls._gate_bound(hn, W["output_hc_down.weight"], W["output_hc_up.weight"]))
        add("result_output", "output.weight", -1, D(gr, "result_output"),
            rows(D(gr, "result_norm"), "output.weight", -1))
        return out

    @classmethod
    def _gate(cls, xn, w_down, w_up):
        return Q.silu((xn @ w_down.T) / cls.hp["hc"]) @ w_up.T

    @classmethod
    def _gate_bound(cls, xn, w_down, w_up):
        """One binary16 spacing per f16 input of the up projection, weighted by |W_up| (module docstring)."""
        a16 = QT.act_quant(Q.silu((xn @ w_down.T) / cls.hp["hc"]), "F16")
        return F.ulp16(a16) @ np.abs(np.asarray(w_up)).T

    def test_every_type_checked(self):
        checked = self.checked_types()
        if self.QTYPE == "Q4_K_M":
            self.assertTrue({"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16"} <= checked, sorted(checked))
        else:
            self.assertEqual(checked, {"Q8_0", "F16", "F32"})

    def test_weight_types(self):
        """llama-quantize's choice per tensor (src/llama-quant.cpp @ec7630a: tensor_allows_quantization 287-...,
        tensor_type_fallback 390-439): F32 stays for 1-D tensors, *_norm.weight, ffn_gate_inp, the indexer
        projections, ssm_conv1d and the PLE gammas (--tensor-type ple_norm=f32); input widths not divisible by 32
        fall back to F16 (hc_*_up: 8, ple_conv1d: 3); the PLE table (width 64) takes the 32-block fallback of the
        k-quant; everything else the target type (Q4_K_M: a k-quant mix, attn_qkv / attn_k / attn_v promoted)."""
        for name, ty in self.types.items():
            ne = self.shapes[name]
            keep = (len(ne) < 2 or name.endswith("_norm.weight") or "ffn_gate_inp" in name or "ple_norm" in name
                    or name.endswith(("indexer.k_proj.weight", "indexer.q_proj.weight", "ssm_conv1d.weight")))
            if keep:
                self.assertEqual(ty, "F32", name)
            elif ne[0] % 32:
                self.assertEqual(ty, "F16", name)
            elif self.QTYPE == "Q8_0":
                self.assertEqual(ty, "Q8_0", name)
            elif ne[0] % 256:
                self.assertIn(ty, ("Q5_0", "Q8_0"), name)       # Q4_K -> Q5_0, Q6_K -> Q8_0
            else:
                self.assertIn(ty, ("Q4_K", "Q5_K", "Q6_K", "Q8_0"), name)

    def test_gathers_exact(self):
        hp = self.hp
        rows_all = Q.ple_rows(self.R.final, hp["ple_multipliers"], hp["ple_offsets"], hp["ple_vocab_sizes"],
                              hp["ple_ngram"], hp["ple_heads_per_ngram"], hp["ple_eos"])
        for gr in self.R.graphs:
            emb = self.W["token_embd.weight"][gr["toks"]]
            got = self._dump(gr, "hc_init", rows=False).reshape(gr["n"], hp["hc"], -1)
            self.assertTrue(np.array_equal(got, Q.hc_init(emb, hp["hc"])), f"hc_init g{gr['g']}")
            want = Q.ple_embed(self.W["per_layer_token_embd.weight"], rows_all[np.array(gr["pos"])])
            self.assertTrue(np.array_equal(self._dump(gr, "ple_embd"), want), f"ple_embd g{gr['g']}")
            self.TABLE.append((self.QTYPE, self.RUN, "hc_init, ple_embd (gathers, exact)",
                          f"{self.types['token_embd.weight']},{self.types['per_layer_token_embd.weight']}", -1,
                          gr["g"], 0.0, 0.0, 0.0, 0.0))

    def test_ple_module(self):
        hp = self.hp
        for gr in self.R.graphs:
            if gr["p0"] != 0:     # the conv reads the history of earlier graphs: the first graph only
                continue
            emb = self._dump(gr, "ple_embd")
            for il in hp["ple_layers"]:
                res = self._dump(gr, "hc_init" if il == 0 else f"l_last-{il - 1}", rows=False).reshape(
                    gr["n"], hp["hc"], -1)
                outs = {}
                for act, W in ((True, self.W), (False, self.W0)):
                    w = Q.layer_weights(W, il)
                    outs[act] = Q.ple_forward(emb, res, w["ple_key.weight"], w["ple_value.weight"],
                                              w["ple_norm_key.weight"], w["ple_norm_query.weight"],
                                              w["ple_norm_conv.weight"], w["ple_conv1d.weight"], hp["eps"],
                                              hp["ple_ngram"])
                ty = self.types[f"blk.{il}.ple_key.weight"]
                for name, key in (("ple_gate", "gate"), ("ple_gated_value", "gated"), ("ple_conv_out", "conv_out")):
                    got = self._dump(gr, f"{name}-{il}", rows=False)
                    e, p = rel(got, outs[True][key]), rel(got, outs[False][key])
                    self.TABLE.append((self.QTYPE, self.RUN, f"{name} (ple_key, ple_value inside)", ty, il,
                                       gr["g"],
                                  np.nan, e, np.nan, p))
                    self.assertLess(e, TOL_OP, f"{name} L{il} g{gr['g']}: {e:.2e}")


def _cls(qtype, run):
    name = f"{qtype}_{run.replace('-', '_')}"
    return type(name, (_QuantOps, unittest.TestCase), dict(QTYPE=qtype, RUN=run))


Q8_0_pf64 = _cls("Q8_0", "pf64")
Q8_0_ub10 = _cls("Q8_0", "ub10")
Q8_0_decode = _cls("Q8_0", "decode")
Q4_K_M_pf64 = _cls("Q4_K_M", "pf64")
Q4_K_M_ub10 = _cls("Q4_K_M", "ub10")
Q4_K_M_decode = _cls("Q4_K_M", "decode")
Q4_K_M_pf64_nr = _cls("Q4_K_M", "pf64-nr")
Q4_K_M_decode_nr = _cls("Q4_K_M", "decode-nr")


def print_table(table, model_name):
    if not table:
        return
    worst = {}
    for qt, run, site, ty, il, g, es, e, ps, p in table:
        w = worst.setdefault((qt, run, site, ty), dict(es=np.nan, e=0.0, il=-1, g=-1, ps=[], p=[], k=-1.0))
        key = es if not np.isnan(es) else e
        if key >= w["k"]:
            w.update(es=es, il=il, g=g, k=key)
        w["e"] = max(w["e"], e)
        w["ps"].append(ps)
        w["p"].append(p)
    print(f"\n{model_name}: quantized products from llama.cpp's own inputs ({COMMIT}), worst over layers and graphs. "
          f"scaled = max|got-want| / max(|W| |x_q|) (asserted < TOL_OP = {TOL_OP:g}; for a bounded product: the "
          f"largest share of its bound, asserted <= 1); rel = max|got-want| / max|want|; no-act = the same without "
          f"the activation quantization (min .. max over graphs):", file=sys.stderr)
    print(f"{'type':7s} {'run':9s} {'product':44s} {'weight':9s} {'scaled':>9s} {'rel':>9s} {'layer':>5s} {'graph':>5s}"
          f" {'no-act scaled':>21s} {'no-act rel':>21s}", file=sys.stderr)

    def rng(v):
        v = [x for x in v if not np.isnan(x)]
        return f"{min(v):9.2e} .. {max(v):8.2e}" if v else f"{'-':>21s}"
    for (qt, run, site, ty), w in worst.items():
        es = f"{w['es']:9.2e}" if not np.isnan(w["es"]) else f"{'-':>9s}"
        print(f"{qt:7s} {run:9s} {site:44s} {ty:9s} {es} {w['e']:9.2e} {w['il']:5d} {w['g']:5d} {rng(w['ps'])} "
              f"{rng(w['p'])}", file=sys.stderr)


def tearDownModule():
    print_table(TABLE, "qwen4exp")


if __name__ == "__main__":
    unittest.main()
