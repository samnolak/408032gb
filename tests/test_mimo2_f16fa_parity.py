"""N17: the mimo2 (MiMo-V2.x-Flash) trunk at the R1 default precision of the KV cache and the attention, -ctk f16
-ctv f16 -fa on (scripts/r1/configs/llamacpp-server.json), against the precision-aware oracle (ref/mimo2.py with
prec = ref/fattn.py precision("f16", True): the K/V rows stored rounded to f16, the CPU flash attention emulated with
the layer's sinks). N16 (tests/test_mimo2_decode_parity.py) checks the same graph at f32 / flash attention off.

tools/tinygen/run_parity_mimo2_n17.sh runs n17.gguf (the N16 trunk: tools/tinygen/mimo2_tiny.py, seed 0, 6 layers,
global 0 and 5, SWA 1-4 with window 8, fused attn_qkv, sinks on the SWA layers only, value scale 0.75; K head 24,
V head 16: DK != DV) on the N16 token sequence with -c 512 -t 1 -ctk f16 -ctv f16 -fa on:
  fa/pf64      one 64-token ubatch (-ub 512): the tiled path (>= 64 query rows)
  fa/ub64      300 tokens, -ub 64: 4 tiled ubatches and one of 44 rows (one_chunk); the SWA ring (256 cells) wraps
  fa/decode    prefill 240 (-ub 64), 60 x 1: one_chunk for every token; at position >= 256 the global layers read 512
               cells, one query row: the split-KV path (ggml/src/ggml-cpu/ops.cpp:9490 @ec7630a), equal to one_chunk
               with one thread (ref/fattn.py docstring)
  fa/rollback  the N16 rollback plan (seq_rm by 2 and by 10, re-decoded rows)
  fa/hd192     n17hd.gguf (the same generator with --head-dims 192,128: MiMo-V2.6-Flash's K 192 / V 128 heads, RoPE on
               64 dims), 64 + 16 x 1 (-ub 512): the tiled path and one_chunk on the AVX2 SIMD loops of the f16 dot,
               scale and mad (DK, DV >= 32)
CPU paths and roundings: ref/fattn.py docstring (one_chunk: q to f16, f16 V accumulator, the sink after the cell loop on
the f32 accumulator, O:9034-9056; tiled: f32 over f16 K / V, the sink per row after the tiles, O:9332-9349). The mask:
llama_kv_cache::set_input_kq_mask fills the f16 flash-attention mask with the same template as the f32 one
(src/llama-kv-cache.cpp:1566-1716 and 1789-1793 @ec7630a, 1569-1719 / 1792-1796 @79e2e74, the same code).

Every graph, row by row, against (A) the full-sequence oracle at the run's precision (each position in the CPU mode of
the ubatch that computed its kept row) and (B) the incremental cell-model oracle at that precision:
  test_graphs_and_path        one graph per ubatch, embd exact; no kq_soft_max node (flash attention ran); the cache
                              tensors are f16; the mode per graph (tiled at 64 rows, one_chunk below) and, in decode, the
                              split-KV graphs (one row, 512 cells) present
  test_cache_store_and_rows   the store, exact: the cells a graph wrote (B's cell model) hold f16 of its own dumped Kcur /
                              Vcur; every cell the graph reads (the get_k / get_v views, n_kv cells) against the rows the
                              oracle stored there, stale and empty cells included (one binary16 spacing plus the
                              pre-store tolerance; at least half of the elements of K and of V exactly equal)
  test_flash_attention_op     ref/fattn.py attend() with the layer's sinks on llama.cpp's own inputs (dumped Qcur, the
                              K / V views it read, the visible cells of B's cell model = the f32 path's mask) reproduces
                              kqv_out: one_chunk within one binary16 spacing per element, tiled 1e-5 normwise; and the
                              comparison discriminates: a window of n_swa - 1 or n_swa + 1, or the sinks dropped, breaks
                              the bound on some graph of the run (so the FA mask is the cell-accurate SWA mask)
  test_attention / test_ffn / test_head_and_logits   the components of N16's tests (tests/test_mimo2_decode_parity.py)
                              against the precision-aware (A) and (B)
  test_attention_output_vs_plain_oracle   the precision-aware oracle at least twice as close to kqv_out as the plain
                              f32 oracle (2-norm), as N10 (tests/f16fa.py)
  test_run_exercises_its_mechanisms
Tolerances (N10, tests/test_qwen4exp_f16fa_parity.py): 1e-4 normwise for what layer 0 computes before its attention
(attn_norm, the projections, RoPE); TOL16 = 16 U16 = 2^-7 normwise from layer 0's attention on (the one_chunk f16
accumulator leaves a noise of about one spacing that no oracle short of llama.cpp's own f32 arithmetic removes); MoE
expert sets and greedy tokens equal (no tie replay: none was needed); the flash-attention op on llama.cpp's own inputs as
above. Env: MIMO2_N17_DUMP, LLAMA_CPP_GGUF_PY, PARITY_LLAMA_COMMIT.
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import fattn as F  # noqa: E402
from ref import mimo2 as M  # noqa: E402
from tests import test_mimo2_decode_parity as D  # noqa: E402
from tests.f16fa import cache_check, cache_share, store_check, ulp_diff  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402

DUMP = os.environ.get("MIMO2_N17_DUMP")
GGUF_PY = D.GGUF_PY
LLAMA_COMMIT = D.LLAMA_COMMIT
SKIP = "set MIMO2_N17_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_mimo2_n17.sh"
N_CTX = D.N_CTX
U16 = F.U16
TOL16 = 16 * U16
PREC = F.precision("f16", True)
PRE_ATTENTION = ("attn_norm", "wqkv", "Qcur", "Kcur", "Vcur")
TABLE = []        # (run, component, layer, graph, max abs, rel)
PLAIN = []        # the same rows against the plain f32 oracle, recorded only


class _FaRun:
    RUN = None
    MODEL = "n17.gguf"
    DIMS = (24, 16)              # K and V head sizes of the model
    RATIO = 2.0                  # test_attention_output_vs_plain_oracle: N10's factor (tests/f16fa.py)
    TABLE = TABLE
    # share of the cached elements equal to what the oracle stored (tests/f16fa.py cache_share): >= 0.5
    CACHE_AGREE = 0.5

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")
        cls.W, cls.hp = D.model(pathlib.Path(DUMP, cls.MODEL))
        W, hp = cls.W, cls.hp
        cls.R = Run(cls.RUN, DUMP)
        cls.ub = cls.R.man["n_ubatch"]
        modes = {}
        for gr in cls.R.graphs:
            for p, k in zip(gr["pos"], gr["kept"]):
                if k:
                    modes[p] = F.mode_for(True, True, gr["n"])
        cls.A = M.forward(cls.R.final, W, hp, prec=dict(PREC, modes=modes))
        cls.A32 = M.forward(cls.R.final, W, hp)
        cls.B, cls.B32, cls.cells, cls.stored = {}, {}, {}, {}
        by_call = {}
        for gr in cls.R.graphs:
            by_call.setdefault(gr["call"], []).append(gr)
        cache, cache32 = M.new_cache(hp, N_CTX, cls.ub), M.new_cache(hp, N_CTX, cls.ub)
        for c, call in enumerate(cls.R.calls):
            if call["kind"] == "rollback":
                M.cache_truncate(cache, call["pos0"])
                M.cache_truncate(cache32, call["pos0"])
                continue
            for gr in by_call[c]:
                cls.B[gr["g"]] = M.forward_chunk(gr["toks"], W, hp, cache, prec=PREC)
                cls.B32[gr["g"]] = M.forward_chunk(gr["toks"], W, hp, cache32)
                cls.cells[gr["g"]] = {k: cache[k].pos.copy() for k in ("base", "swa")}
                # the cache contents after this graph, as the oracle stored them (K, V per layer, all cells)
                cls.stored[gr["g"]] = {il: (cache["K"][il].copy(), cache["V"][il].copy()) for il in range(hp["n_layer"])}
        cls.final_cache = cache

    # ---- helpers ----
    def dump(self, g, name, k=0):
        return self.R.dump(g, name, k)[0]

    def meta(self, g, name, k=0):
        return json.loads((self.R.dir / f"g{g:03d}" / f"{name}__{k}.json").read_text())

    def tol_for(self, what, il):
        return D.TOL if il == 0 and what.startswith(PRE_ATTENTION) else TOL16

    def close(self, what, il, g, got, want, tol=None):
        tol = self.tol_for(what, il) if tol is None else tol
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} L{il} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        self.TABLE.append((self.RUN, what, il, g, mx, rel))
        self.assertLess(rel, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, il, gr, got, a_full, b_rows, a32=None, b32=None):
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        kept = np.array(gr["kept"])
        if kept.any():
            rows = np.array(gr["pos"])[kept]
            self.close(f"{what} [full]", il, gr["g"], got[kept], np.asarray(a_full)[rows])
            if a32 is not None:
                PLAIN.append((self.RUN, f"{what} [full]", il, gr["g"], *_err(got[kept], np.asarray(a32)[rows])))
        self.close(f"{what} [chunk]", il, gr["g"], got, b_rows)
        if b32 is not None:
            PLAIN.append((self.RUN, f"{what} [chunk]", il, gr["g"], *_err(got, b32)))

    def mode(self, gr):
        return F.mode_for(True, True, gr["n"])

    # ---- tests ----
    def test_graphs_and_path(self):
        """One graph per ubatch and the token rows (embd, exact); flash attention ran (no kq_soft_max node,
        src/llama-graph.cpp:2651-2694 @ec7630a); the caches are f16 (the get_k / get_v views); the CPU mode of every
        graph is the one the oracle takes; the split-KV graphs of decode (one row reading 512 cells)."""
        R, hp = self.R, self.hp
        self.assertEqual(len(R.man["graphs"]), len(R.graphs))
        for gr in R.graphs:
            g = gr["g"]
            got = self.dump(g, "embd").reshape(gr["n"], -1)
            self.close("embd (token rows, exact)", -1, g, got, self.W["token_embd.weight"][gr["toks"]], tol=1e-300)
            for il in range(hp["n_layer"]):
                self.assertFalse(R.has(g, f"kq_soft_max-{il}"), f"kq_soft_max-{il} in g{g}: flash attention off?")
                for kv in "kv":
                    m = self.meta(g, f"cache_{kv}_l{il} (view)", 1)
                    self.assertEqual(m["ggml_type"], "f16", f"cache_{kv}_l{il} g{g}")
                    self.assertEqual(m["ne"][2], self.B[g]["layers"][il]["attn"]["n_kv"], f"n_kv L{il} g{g}")
            self.assertEqual(set(self.B[g]["layers"][0]["attn"]["modes"]), {self.mode(gr)})

    def _layer_rows(self, g, il):
        """The cache rows of layer il after graph g (the set_rows result, [cells, Hkv*D]) and the views the attention
        read ([n_kv, Hkv, D]), for K and V."""
        hp = self.hp
        Hkv = hp["n_head_kv"][il]
        out = {}
        for kv, Dd in (("k", hp["k_dim"]), ("v", hp["v_dim"])):
            whole = self.dump(g, f"cache_{kv}_l{il} (view)", 0)
            whole = whole.reshape(whole.shape[-2], -1) if whole.ndim > 1 else whole
            view = self.dump(g, f"cache_{kv}_l{il} (view)", 1).reshape(-1, Hkv, Dd)
            out[kv] = (whole.reshape(-1, Hkv * Dd), view)
        return out

    def test_cache_store_and_rows(self):
        """(a) the store: the cells graph g wrote (B's cell model) hold f16 of g's own dumped Kcur / Vcur, every element
        (src/llama-kv-cache.cpp:1328-1361 ggml_set_rows, ref/fattn.py f16); (b) every cell the attention read (the
        get_k / get_v views over n_kv cells: the cells this graph and earlier ones wrote, stale and never-written ones
        included) against the rows the oracle stored there."""
        R, hp = self.R, self.hp
        stats, n_store = {}, 0
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            for il in range(hp["n_layer"]):
                a = self.B[g]["layers"][il]["attn"]
                rows = self._layer_rows(g, il)
                idxs, n_kv = a["idxs"], a["n_kv"]
                Kst, Vst = self.stored[g][il]
                n_store += store_check(self, "K", il, g, rows["k"][0][idxs], self.dump(g, f"Kcur-{il}").reshape(n, -1))
                n_store += store_check(self, "V", il, g, rows["v"][0][idxs], self.dump(g, f"Vcur-{il}").reshape(n, -1))
                cache_check(self, "K", il, g, rows["k"][1], Kst[:n_kv], stats, self.tol_for("Kcur", il))
                cache_check(self, "V", il, g, rows["v"][1], Vst[:n_kv], stats, self.tol_for("Vcur", il))
                # the view is the cache: the first n_kv rows of the whole tensor
                self.assertTrue(np.array_equal(rows["k"][1].reshape(n_kv, -1), rows["k"][0][:n_kv]), f"K view g{g}")
        self.TABLE.append((self.RUN, f"cache store: {n_store} elements equal to the rounded pre-store value", -1, -1,
                           0.0, 0.0))
        cache_share(self, stats)

    def _op(self, gr, il, n_swa=None, sinks="layer"):
        """ref/fattn.py on llama.cpp's own inputs for layer il of graph gr: (got kqv_out, want, mode)."""
        hp = self.hp
        g, n = gr["g"], gr["n"]
        a = self.B[g]["layers"][il]["attn"]
        rows = self._layer_rows(g, il)
        q = self.dump(g, f"Qcur-{il}").reshape(n, hp["n_head"][il], hp["k_dim"])
        swa = hp["swa"][il]
        w = n_swa if n_swa is not None else (hp["n_swa"] if swa else 0)
        vis = M.visible(gr["pos"], a["kpos"], w if swa else 0)
        cells = [np.nonzero(vis[t])[0] for t in range(n)]
        sk = M.layer_weights(self.W, il).get("attn_sinks.weight") if sinks == "layer" else None
        mode = self.mode(gr)
        want = F.attend(q, rows["k"][1], rows["v"][1], cells, M.kq_scale(hp, PREC), mode, sk)
        got = self.dump(g, f"kqv_out-{il}").reshape(want.shape)
        return got, want, mode

    def _op_ok(self, got, want, mode):
        if mode == "fa_chunk":
            return float(ulp_diff(got, want).max()) <= 1.0, float(ulp_diff(got, want).max())
        e = _err(got, want)[1]
        return e < 1e-5, e

    def test_flash_attention_op(self):
        """The flash-attention op with sinks on llama.cpp's own inputs (module docstring), and its power to tell the
        mask and the sinks apart."""
        hp = self.hp
        n_one = n_all = n_eq = 0
        for gr in self.R.graphs:
            for il in range(hp["n_layer"]):
                got, want, mode = self._op(gr, il)
                ok, e = self._op_ok(got, want, mode)
                if mode == "fa_chunk":
                    u = ulp_diff(got, want)
                    n_one += int((u > 0.5).sum())
                    n_eq += int((got == want).sum())
                    n_all += u.size
                    self.TABLE.append((self.RUN, f"flash attention op ({mode}, own inputs; spacings)", il, gr["g"],
                                       0.0, e))
                    self.assertTrue(ok, f"flash attention L{il} g{gr['g']}: {e:.2f} spacings")
                else:
                    self.close(f"flash attention op ({mode}, own inputs)", il, gr["g"], got, want, 1e-5)
        if n_all:
            self.TABLE.append((self.RUN, f"flash attention op (one_chunk): elements > 0.5 spacing {n_one}/{n_all}, "
                                         f"bit-equal {n_eq}/{n_all}", -1, -1, 0.0, n_one / n_all))
            # the emulation does the f16 arithmetic of one_chunk (ref/fattn.py: f32 scale, f32 exp argument, f16
            # accumulator): measured no element more than 0.5 spacing off, the rest of the difference at f32 level
            # (the sum S; docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md). At most 0.1 % allowed past half a
            # spacing, so that another libm's expf (not always correctly rounded) cannot fail the step
            self.assertLessEqual(n_one, 1e-3 * n_all, f"{self.RUN}: one_chunk {n_one}/{n_all} elements > 0.5 spacing")
        # discrimination: another window or no sinks breaks the bound on some graph of a sliding-window layer
        swa_layers = [il for il in range(hp["n_layer"]) if hp["swa"][il]]
        for label, kw in (("window n_swa - 1", dict(n_swa=hp["n_swa"] - 1)),
                          ("window n_swa + 1", dict(n_swa=hp["n_swa"] + 1)), ("sinks dropped", dict(sinks=None))):
            broken = 0
            for gr in self.R.graphs:
                for il in swa_layers:
                    got, want, mode = self._op(gr, il, **kw)
                    broken += not self._op_ok(got, want, mode)[0]
            self.TABLE.append((self.RUN, f"flash attention op with {label}: graphs x SWA layers out of bound", -1, -1,
                               0.0, float(broken)))
            self.assertGreater(broken, 0, f"{self.RUN}: the op comparison does not see '{label}'")

    def test_attention(self):
        """attn_norm, the fused projection, q / k after RoPE, v, kqv_out, attn_out and the value-scaled attn_out."""
        R, hp = self.R, self.hp
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            for il in range(hp["n_layer"]):
                La, Lb = self.A["layers"][il], self.B[g]["layers"][il]
                L32a, L32b = self.A32["layers"][il], self.B32[g]["layers"][il]
                a, b = La["attn"], Lb["attn"]
                self.both("attn_norm", il, gr, self.dump(g, f"attn_norm-{il}"), La["attn_norm"], Lb["attn_norm"])
                fa = np.concatenate([a[x].reshape(len(a[x]), -1) for x in ("q_pre", "k_pre", "v")], 1)
                fb = np.concatenate([b[x].reshape(n, -1) for x in ("q_pre", "k_pre", "v")], 1)
                self.both("wqkv (fused projection)", il, gr, self.dump(g, f"wqkv-{il}"), fa, fb)
                self.both("Qcur (RoPE)", il, gr, self.dump(g, f"Qcur-{il}"), a["q"], b["q"])
                self.both("Kcur (RoPE)", il, gr, self.dump(g, f"Kcur-{il}"), a["k"], b["k"])
                self.both("Vcur", il, gr, self.dump(g, f"Vcur-{il}"), a["v"], b["v"])
                self.both("kqv_out", il, gr, self.dump(g, f"kqv_out-{il}"), a["kqv"], b["kqv"],
                          L32a["attn"]["kqv"], L32b["attn"]["kqv"])
                self.both("attn_out (wo)", il, gr, self.dump(g, f"attn_out-{il}"), a["attn_out"], b["attn_out"])
                self.both("attn_out_scaled (value scale)", il, gr, self.dump(g, f"attn_out_scaled-{il}"), a["out"],
                          b["out"])

    def test_ffn(self):
        """ffn_inp, ffn_norm, the dense layer or the MoE (router logits, expert sets, weights, output), l_out."""
        R, hp = self.R, self.hp
        k = hp["n_expert_used"]
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            kept = np.array(gr["kept"])
            for il in range(hp["n_layer"]):
                La, Lb = self.A["layers"][il], self.B[g]["layers"][il]
                Fa, Fb = La["ffn"], Lb["ffn"]
                self.both("ffn_inp", il, gr, self.dump(g, f"ffn_inp-{il}"), La["ffn_inp"], Lb["ffn_inp"])
                self.both("ffn_norm", il, gr, self.dump(g, f"ffn_norm-{il}"), La["ffn_norm"], Lb["ffn_norm"])
                if "ids" not in Fb:
                    for nm, key in (("ffn_up", "up"), ("ffn_gate", "gate"), ("ffn_swiglu", "act"), ("ffn_out", "out")):
                        self.both(f"{nm} (dense)", il, gr, self.dump(g, f"{nm}-{il}"), Fa[key], Fb[key])
                else:
                    self.both("ffn_moe_logits", il, gr, self.dump(g, f"ffn_moe_logits-{il}"), Fa["logits"],
                              Fb["logits"])
                    ids = self.dump(g, f"ffn_moe_topk-{il}").reshape(n, k).astype(int)
                    self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(Fb["ids"], 1)), f"top-k L{il} g{g}")
                    if kept.any():
                        want = np.sort(Fa["ids"][np.array(gr["pos"])[kept]], 1)
                        self.assertTrue(np.array_equal(np.sort(ids, 1)[kept], want), f"top-k (full) L{il} g{g}")
                    wd = self.dump(g, f"ffn_moe_weights_norm-{il}").reshape(n, k)
                    wb = np.array([[dict(zip(Fb["ids"][r], Fb["weights"][r]))[e] for e in ids[r]] for r in range(n)])
                    self.close("ffn_moe_weights_norm [chunk]", il, g, wd, wb)
                    self.both("ffn_moe_out", il, gr, self.dump(g, f"ffn_moe_out-{il}"), Fa["moe_out"], Fb["moe_out"])
                nm = f"l_out-{il}"
                if il == hp["n_layer"] - 1 and not R.has(g, nm):
                    nm = "h_nextn"
                self.both("l_out", il, gr, self.dump(g, nm), La["l_out"], Lb["l_out"])

    def test_head_and_logits(self):
        R = self.R
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab).astype(np.float64)
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            self.both("result_norm", -1, gr, self.dump(g, "result_norm"), self.A["result_norm"], b["result_norm"])
            self.both("result_output", -1, gr, self.dump(g, "result_output"), self.A["logits"], b["logits"],
                      self.A32["logits"], self.B32[g]["logits"])
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            self.both("logits.bin", -1, gr, rows, self.A["logits"], b["logits"])
            self.assertTrue(np.array_equal(rows.argmax(1), b["logits"].argmax(1)), f"greedy tokens g{g}")

    def test_attention_output_vs_plain_oracle(self):
        """The precision model matters: the precision-aware (B) is at least RATIO times as close to llama.cpp's kqv_out
        as the plain f32 oracle (2-norm over every graph and layer); RATIO 2 as N10 requires (tests/f16fa.py), 1.5 for
        the 192 / 128 heads (measured 1.79 there: both oracles keep the noise of the f16 rounding positions, which the
        f32-level differences of q / K / V move, and over 192-wide dot products the deterministic roundings the plain
        oracle misses weigh less; the op test shows the f16 arithmetic itself is the CPU's, 0 elements past half a
        spacing)."""
        res = {}
        for which, B in (("aware", self.B), ("plain", self.B32)):
            d2 = n2 = 0.0
            for gr in self.R.graphs:
                for il in range(self.hp["n_layer"]):
                    got = self.dump(gr["g"], f"kqv_out-{il}")
                    want = B[gr["g"]]["layers"][il]["attn"]["kqv"]
                    d2 += float(((got.reshape(want.shape) - want) ** 2).sum())
                    n2 += float((want ** 2).sum())
            res[which] = np.sqrt(d2 / n2)
        self.TABLE.append((self.RUN, f"kqv_out 2-norm rel: aware {res['aware']:.3e}, plain f32 {res['plain']:.3e}",
                           -1, -1, 0.0, res["plain"] / res["aware"]))
        self.assertLess(self.RATIO * res["aware"], res["plain"], f"{self.RUN}: the plain f32 oracle is about as close")

    def test_run_exercises_its_mechanisms(self):
        R, hp = self.R, self.hp
        sizes = [gr["n"] for gr in R.graphs]
        expect = {"fa/pf64": [64], "fa/ub64": [64] * 4 + [44], "fa/decode": [64] * 3 + [48] + [1] * 60,
                  "fa/hd192": [64] + [1] * 16}
        if self.RUN in expect:
            self.assertEqual(sizes, expect[self.RUN])
        modes = {self.mode(gr) for gr in R.graphs}
        self.assertEqual(modes, {"fa_tiled"} if self.RUN == "fa/pf64" else {"fa_tiled", "fa_chunk"})
        if self.RUN not in ("fa/pf64", "fa/hd192"):
            self.assertTrue(any(gr["pos"] != self.B[gr["g"]]["slots"]["swa"] for gr in R.graphs), "no wrap")
        if self.RUN in ("fa/decode", "fa/rollback"):
            split = [gr["g"] for gr in R.graphs if gr["n"] == 1 and self.B[gr["g"]]["layers"][0]["attn"]["n_kv"] >= 512]
            self.assertTrue(split, "no split-KV graph (one row over >= 512 cells)")
        if self.RUN == "fa/rollback":
            self.assertEqual([(c["pos0"], c["n"]) for c in R.calls if c["kind"] == "rollback"], [(258, 2), (266, 10)])
        self.assertFalse(hp["swa"][0] or hp["swa"][5])
        self.assertEqual((hp["k_dim"], hp["v_dim"], hp["n_rot"]), (*self.DIMS, self.DIMS[0] // 3))


def _err(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    d = np.abs(got.reshape(want.shape) - want)
    mx = float(d.max()) if d.size else 0.0
    return mx, mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0


def _cls(name, run, **kw):
    return unittest.skipUnless(DUMP and GGUF_PY, SKIP)(type(name, (_FaRun, unittest.TestCase), dict(RUN=run, **kw)))


FaPrefill64 = _cls("FaPrefill64", "fa/pf64")
FaUbatch64 = _cls("FaUbatch64", "fa/ub64")
FaDecode = _cls("FaDecode", "fa/decode")
FaRollback = _cls("FaRollback", "fa/rollback")
# MiMo-V2.6-Flash's head sizes (K 192, V 128, 64 rotated dims): the CPU kernels take their AVX2 SIMD loops (f32 lanes,
# FMA) for DK, DV >= GGML_F16_STEP = 32 instead of the scalar ones ref/fattn.py transcribes (module docstring there)
FaHeadDims192 = _cls("FaHeadDims192", "fa/hd192", MODEL="n17hd.gguf", DIMS=(192, 128), RATIO=1.5)


def tearDownModule():
    if not TABLE:
        return
    worst, plain = {}, {}
    for run, what, il, g, mx, rel in (r[:6] for r in TABLE):
        if (run, what) not in worst or rel > worst[(run, what)][3]:
            worst[(run, what)] = (il, g, mx, rel)
    for run, what, il, g, mx, rel in PLAIN:
        if (run, what) not in plain or rel > plain[(run, what)]:
            plain[(run, what)] = rel
    print(f"\nmimo2 -ctk f16 -ctv f16 -fa on, llama.cpp {LLAMA_COMMIT[:7]}: worst error per run and component (rel = "
          f"max|got-want| / max|want|) against the precision-aware oracle (asserted) and the plain f32 oracle "
          f"(recorded)", file=sys.stderr)
    print(f"{'run':12s} {'component':66s} {'layer':>5s} {'graph':>5s} {'rel aware':>10s} {'rel f32':>10s}",
          file=sys.stderr)
    for (run, what), (il, g, mx, rel) in worst.items():
        p = plain.get((run, what))
        print(f"{run:12s} {what:66s} {il:5d} {g:5d} {rel:10.3e} {(p if p is not None else float('nan')):10.3e}",
              file=sys.stderr)
    per_layer = {}
    for run, what, il, g, mx, rel in (r[:6] for r in TABLE):
        if g < 0 or "spacings" in what:
            continue
        if il not in per_layer or rel > per_layer[il][4]:
            per_layer[il] = (run, what, g, mx, rel)
    print(f"\nmimo2 -ctk f16 -ctv f16 -fa on: worst error per layer (layer -1 = embedding and head), llama.cpp "
          f"{LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'layer':>5s} {'run':12s} {'component':60s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for il in sorted(per_layer):
        run, what, g, mx, rel = per_layer[il]
        print(f"{il:5d} {run:12s} {what:60s} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
