"""N16 (gap 12 of docs/reassessment/CORRECTNESS.md, the MXFP4 part): MXFP4 routed experts on the CPU. llama.cpp's real mimo2
graph on the N16 tiny model quantized by the pinned llama-quantize to MXFP4_MOE, against ref/mimo2.py's products on the
dequantized weights with the CPU activation quantization emulated (ref/quant.py), matrix product by matrix product, as
tests/test_qwen4exp_quant_parity.py does for Q8_0 / Q4_K_M (N10; same metric, tolerances, checks and table; the
reasons are written there).

tools/tinygen/run_parity_mimo2_decode.sh quantizes n16.gguf (tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa
--value-scale 0.75 --mtp) with `llama-quantize ... MXFP4_MOE`: every 3-D tensor with ne[2] > 1 (the routed experts
ffn_gate_exps / ffn_up_exps / ffn_down_exps) to MXFP4, the other matrices to Q8_0, the output and token embedding to
Q8_0 (src/llama-quant.cpp:490-523 @ec7630a), 1-D tensors, *_norm.weight and ffn_gate_inp kept F32
(tensor_allows_quantization, :287-306; exp_probs_b.bias does not end in "weight"). Runs (64 tokens of the N16
sequence, -ctk f32 -ctv f32 -fa off): pf64 (one 64-token ubatch), decode (48 + 16 x 1: one activation row, the gemv
path), and both with --no-repack. MXFP4 weights with ne[1] % 8 == 0 are repacked to mxfp4_8x8 on AVX2 by default
(ggml/src/ggml-cpu/repack.cpp:5094-5098; here every expert tensor: ne[1] = 32 or 64), multiplied by
ggml_gemv / gemm_mxfp4_8x8_q8_0; with --no-repack by ggml_vec_dot_mxfp4_q8_0 (two blocks per AVX2 step and a scalar
tail block: the down projection has one block per row, gate / up two). Q8_0 weights are not repacked on x86 (N10).
Both use the Q8_0 activation quantization (ref/quant.py MXFP4 section).

Products, per graph from llama.cpp's dumped tensors (cb() names of src/models/mimo2.cpp and src/llama-graph.cpp):
  attention   wqkv (fused attn_qkv, Q8_0) of attn_norm; attn_out (attn_output, Q8_0) of kqv_out
  FFN         layer 0 (dense, Q8_0): ffn_up, ffn_gate of ffn_norm, ffn_out (the down projection, named by mimo2.cpp:210)
              of ffn_swiglu; MoE layers: ffn_moe_logits (F32 router) of ffn_norm, ffn_moe_gate / ffn_moe_up (MXFP4,
              mul_mat_id with the ids of ffn_moe_topk) of ffn_norm, ffn_moe_down (MXFP4) of ffn_moe_swiglu
  head        result_output (output, Q8_0) of result_norm
  gathers     embd = the Q8_0 token_embd rows, exact
plus test_mxfp4_weights_are_the_f32_weights_quantized (each MXFP4 expert weight within 4 d of the f32 weight of
n16.gguf, d = 2^(e - 128) its block's scale: the largest kvalue gap is 4, so rounding moves a value by at most 2 d,
clipping above 12 d by less than 4 d; an E8M0 scale read off by one exponent would miss) and test_repack_path (the
default runs hold a CPU_REPACK model buffer, the --no-repack runs none; the logs of the runs at -lv 4).
Env: MIMO2_DECODE_DUMP (root: mxfp4/model.gguf, mxfp4/<run>/, n16.gguf, mxfp4_<run>.log), LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT.
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from ref import mimo2 as M  # noqa: E402
from ref import quant as QT  # noqa: E402
from tests import test_qwen4exp_quant_parity as base  # noqa: E402

DUMP = os.environ.get("MIMO2_DECODE_DUMP")
SKIP = "set MIMO2_DECODE_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_mimo2_decode.sh"
TABLE = []


def _same(a):
    return a


class _MimoQuantOps(base._QuantOpsBase):
    ROOT, SKIP, TABLE = DUMP, SKIP, TABLE
    HPARAMS = staticmethod(M.hparams)
    NOT_MATMUL = ("token_embd.weight",)
    QTYPE = "mxfp4"

    @classmethod
    def _products(cls, gr):
        hp, n, D = cls.hp, gr["n"], cls._dump
        g = gr["g"]
        out = []

        def add(site, wnames, il, got, fn):
            out.append((site, (wnames,) if isinstance(wnames, str) else wnames, il, g, got, fn, None))

        def rows(x, wname, il):
            def f(W, A=_same):
                return A(x) @ (M.layer_weights(W, il)[wname] if il >= 0 else W[wname]).T
            return f

        def experts(x, wname, il, ids, per_slot):
            def f(W, A=_same):
                w = M.layer_weights(W, il)[wname]
                return np.stack([[w[e] @ A(x[t, j] if per_slot else x[t]) for j, e in enumerate(ids[t])]
                                 for t in range(n)])
            return f

        for il in range(hp["n_layer"]):
            b = f"blk.{il}."
            add("wqkv (fused attn_qkv)", f"{b}attn_qkv.weight", il, D(gr, f"wqkv-{il}"),
                rows(D(gr, f"attn_norm-{il}"), "attn_qkv.weight", il))
            add("attn_out (attn_output of kqv_out)", f"{b}attn_output.weight", il, D(gr, f"attn_out-{il}"),
                rows(D(gr, f"kqv_out-{il}"), "attn_output.weight", il))
            x2 = D(gr, f"ffn_norm-{il}")
            if f"{b}ffn_gate_inp.weight" not in cls.W:
                for name, wn in (("ffn_up (dense)", "ffn_up"), ("ffn_gate (dense)", "ffn_gate")):
                    add(name, f"{b}{wn}.weight", il, D(gr, f"{wn}-{il}"), rows(x2, f"{wn}.weight", il))
                add("ffn_out (dense down)", f"{b}ffn_down.weight", il, D(gr, f"ffn_out-{il}"),
                    rows(D(gr, f"ffn_swiglu-{il}"), "ffn_down.weight", il))
                continue
            ids = D(gr, f"ffn_moe_topk-{il}").astype(int)
            ke = ids.shape[1]
            add("ffn_moe_logits (router)", f"{b}ffn_gate_inp.weight", il, D(gr, f"ffn_moe_logits-{il}"),
                rows(x2, "ffn_gate_inp.weight", il))
            for name, wn in (("ffn_moe_gate (experts)", "ffn_gate_exps"), ("ffn_moe_up (experts)", "ffn_up_exps")):
                add(name, f"{b}{wn}.weight", il, D(gr, f"{name.split()[0]}-{il}", rows=False).reshape(n, ke, -1),
                    experts(x2, f"{wn}.weight", il, ids, False))
            sw = D(gr, f"ffn_moe_swiglu-{il}", rows=False).reshape(n, ke, -1)
            add("ffn_moe_down (experts)", f"{b}ffn_down_exps.weight", il,
                D(gr, f"ffn_moe_down-{il}", rows=False).reshape(n, ke, -1),
                experts(sw, "ffn_down_exps.weight", il, ids, True))
        add("result_output", "output.weight", -1, D(gr, "result_output"),
            rows(D(gr, "result_norm"), "output.weight", -1))
        return out

    def test_weight_types(self):
        """MXFP4_MOE (src/llama-quant.cpp:490-523, 287-306 @ec7630a): routed experts MXFP4, the other matrices Q8_0,
        1-D tensors, *_norm.weight and ffn_gate_inp F32."""
        for name, ty in self.types.items():
            ne = self.shapes[name]
            if len(ne) < 2 or name.endswith("_norm.weight") or "ffn_gate_inp" in name:
                self.assertEqual(ty, "F32", name)
            elif len(ne) == 3 and ne[2] > 1:
                self.assertEqual(ty, "MXFP4", name)
            else:
                self.assertEqual(ty, "Q8_0", name)
        self.assertEqual(sum(ty == "MXFP4" for ty in self.types.values()), 3 * (self.hp["n_layer"] - 1))

    def test_every_type_checked(self):
        self.assertEqual(self.checked_types(), {"MXFP4", "Q8_0", "F32"})

    def test_gathers_exact(self):
        for gr in self.R.graphs:
            want = self.W["token_embd.weight"][gr["toks"]]
            self.assertTrue(np.array_equal(self._dump(gr, "embd"), want), f"embd g{gr['g']}")
            self.TABLE.append((self.QTYPE, self.RUN, "embd (gather, exact)", self.types["token_embd.weight"], -1,
                               gr["g"], 0.0, 0.0, 0.0, 0.0))

    def test_mxfp4_weights_are_the_f32_weights_quantized(self):
        sys.path.append(base.GGUF_PY)
        try:
            from gguf import GGUFReader
        finally:
            sys.path.remove(base.GGUF_PY)
        f32 = {t.name: np.array(t.data, dtype=np.float64) for t in GGUFReader(str(pathlib.Path(DUMP, "n16.gguf"))).tensors}
        r = GGUFReader(str(self.dir / "model.gguf"))
        n = 0
        for t in r.tensors:
            if t.tensor_type.name != "MXFP4":
                continue
            raw = np.asarray(t.data, dtype=np.uint8)
            deq = QT.mxfp4_dequant(raw).astype(np.float64)
            self.assertTrue(np.array_equal(deq.reshape(-1), np.asarray(self.W[t.name].w).reshape(-1)),
                            f"{t.name}: ref/quant.py and gguf-py dequantize differ")
            e, _ = QT.mxfp4_split(raw)
            d = np.repeat(QT.e8m0_to_fp32_half(e).astype(np.float64), QT.QK_MXFP4, axis=-1).reshape(deq.shape)
            w = f32[t.name].reshape(deq.shape)
            self.assertTrue((np.abs(deq - w) <= 4 * d).all(), t.name)
            # and the scale is the one the reference quantizer picks: floor(log2 amax) - 2 + 127 (ggml-quants.c:368)
            amax = np.abs(w.reshape(*w.shape[:-1], -1, QT.QK_MXFP4)).max(-1)
            self.assertTrue(np.array_equal(e, (np.floor(np.log2(amax)) - 2 + 127).astype(np.uint8)), t.name)
            n += 1
        self.assertEqual(n, 3 * (self.hp["n_layer"] - 1))

    def test_repack_path(self):
        log = pathlib.Path(DUMP, f"mxfp4_{self.RUN}.log").read_text(errors="replace")
        self.assertIn("CPU_Mapped model buffer size", log, "the run log has no buffer list (-lv 4)")
        self.assertEqual("CPU_REPACK model buffer size" in log, not self.RUN.endswith("-nr"))


def _cls(run):
    return unittest.skipUnless(DUMP and base.GGUF_PY, SKIP)(
        type(f"Mimo_MXFP4_MOE_{run.replace('-', '_')}", (_MimoQuantOps, unittest.TestCase), dict(RUN=run)))


Mimo_MXFP4_MOE_pf64 = _cls("pf64")
Mimo_MXFP4_MOE_decode = _cls("decode")
Mimo_MXFP4_MOE_pf64_nr = _cls("pf64-nr")
Mimo_MXFP4_MOE_decode_nr = _cls("decode-nr")


def tearDownModule():
    base.print_table(TABLE, "mimo2")


if __name__ == "__main__":
    unittest.main()
