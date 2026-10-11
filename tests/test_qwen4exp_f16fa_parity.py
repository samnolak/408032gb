"""N10 (gap 3 of docs/reassessment/CORRECTNESS.md): the qwen4exp graph at the R1 default precision of the KV cache and
the attention, -ctk f16 -ctv f16 -fa on (scripts/r1/configs/llamacpp-server.json), against the precision-aware oracle.

tools/tinygen/run_parity_f16fa.sh runs the N6 tiny model (tools/tinygen/qwen4exp_tiny.py, seed 0) and the N6 token
sequence through llama.cpp on the CPU with those flags, dumped graph by graph like N6: pf64 (one 64-token ubatch, the
N2 prefill shape), ub16, ub10, decode (48 + 16 x 1) and rollback (n_rs_seq 3). Every N6 check
(tests/test_qwen4exp_decode_parity.py: graphs, PLE rows, carried state, GDN, PLE, attention, QSA k-pool cache and
selection, MoE, head and logits; (A) the full-sequence oracle and (B) the incremental replay) runs again with the
oracle at the run's precision (ref/fattn.py precision(PARITY_KV, PARITY_FA)): the K/V cache, the raw indexer keys and
the pooled keys rounded to f16 where llama.cpp stores them (ref/qwen4exp.py docstring), the attention in the CPU
flash-attention mode the ubatch takes (ref/fattn.py: one_chunk with an f16 accumulator below 64 query rows, tiled
f32 at 64 and more). Added here:
  test_cache_contents     (a) the store: the cache rows a graph wrote (dumped from the cache tensors, ggml_type f16)
                          equal f16 of its own dumped pre-store values (Kcur, Vcur, indexer_k_raw, indexer_pool_k_new),
                          every element; (b) every cached row against the rows the oracle stored
  test_flash_attention_op ref/fattn.py attend() on llama.cpp's own q, cached K/V and selection mask reproduces the
                          dumped attention output (attn_pregate = kqv_out of build_attn_qsa / build_attn)
  test_attention_path     flash attention (no kq_soft_max node) iff PARITY_FA=on; the cache tensors' type
  test_attention_output_vs_plain_oracle (tests/f16fa.py) the precision-aware oracle at least twice as close to the
                          attention output as the plain f32 oracle (2-norm)
  test_plain_f32_oracle   (tests/f16fa.py) the N6 checks against the plain f32 oracle, recorded and printed, never
                          asserted; the discrete choices of both oracles counted (the aware one: 0 after its replays)

Tolerances (normwise relative error max|got - want| / max|want|, as N6; U16 = 2^-11, the unit roundoff of binary16):
  * TOL = 16 U16 = 2^-7 = 7.8e-3 for the N6 checks downstream of the first attention. Why: the oracle rounds at the
    sites llama.cpp rounds, but its float64 values and llama.cpp's f32 values differ (by the f32 errors N6 bounds by
    1e-4), and at every rounding site an element whose two values fall on different sides of a binary16 rounding
    boundary ends up one binary16 spacing apart, at most 2 U16 relative to it. The one_chunk accumulator is rounded at
    every cell (<= 11 QSA cells, <= 64 dense ones), and which way each of those roundings goes changes with the
    smallest change of its inputs: the attention output carries a noise of a few spacings that no oracle short of
    bit-equal f32 arithmetic can remove. The bound allows two spacings (4 U16) of that noise at the attention output,
    grown by up to 4 through the rest of the model. Measured (both pins, every precision configuration of
    run_parity_f16fa.sh, docs/evidence/2026-10-10-quant-fa-parity.md): attention output <= 1.4e-3 = 2.8 U16 (the dense
    layer here), worst value anywhere 2.95e-3 = 6.0 U16 (glm5-next, kda_scan_out of layer 9), so the bound holds with
    a margin of 2.6.
    The f32 paths keep the N6 tolerance 1e-4 (tol_for): the layers before the first attention layer (their carried
    state reads only those layers) and, on that layer, what is computed before the attention reads the cache.
  * MoE expert sets and QSA selections: equal, or a valid top-k within TOL of the oracle's own scores (the scores
    carry that error), replayed and counted (TIE_RTOL = MOE_TIE_RTOL = TOL; N6 keeps 1e-6 and no MoE replay).
  * cache store (a): exact. Cache rows against the rows the oracle stored (b): within one spacing plus the tolerance
    of the pre-store value (tol_for of Kcur, Vcur, indexer_k_raw, indexer_pool_k_new), and at least CACHE_AGREE = 50 %
    of the elements of each cached quantity exactly equal (they differ only where the two pre-store values straddle a
    rounding boundary; measured >= 85 % overall; an oracle that stored unrounded values would match almost none).
  * flash-attention op on llama.cpp's own inputs: one_chunk <= 1 binary16 spacing of each output element (the f16
    accumulator, its rounding at f32 vs float64), tiled and mm16 1e-5 normwise (f32 arithmetic after the f16 roundings).
  * exact checks of N6 (embedding and PLE gathers) stay exact.
The f32 tests (N2, N6) are unchanged and still run on f32 dumps.

Env: QWEN4EXP_F16FA_DUMP (root with one directory per run and tokens.txt), QWEN4EXP_F16FA_GGUF, LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT, PARITY_KV / PARITY_FA (the flags the dumps were made with, default f16 / on).
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from ref import fattn as F  # noqa: E402
from tests import test_qwen4exp_decode_parity as base  # noqa: E402
from tests.f16fa import (FA, KV, U16, F16Mixin, Recorder, cache_check, cache_share,  # noqa: E402
                         print_tables, store_check, ulp_diff)

DUMP = os.environ.get("QWEN4EXP_F16FA_DUMP")
GGUF = os.environ.get("QWEN4EXP_F16FA_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
SKIP = ("set QWEN4EXP_F16FA_DUMP, QWEN4EXP_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh")
TOL16 = 16 * U16
TABLE, F32, MISMATCH = [], Recorder(), {}
PRE_ATTENTION = ("Qcur", "Kcur", "Vcur", "indexer_k_raw", "indexer_q")


def dumps(R, g, name):
    """Every occurrence k of a dumped name in graph g: (k, meta, float64 array in ggml order reversed)."""
    d = R.dir / f"g{g:03d}"
    k = 0
    while (d / f"{name}__{k}.json").exists():
        m = json.loads((d / f"{name}__{k}.json").read_text())
        dt = np.float32 if m["type"] == "f32" else np.int32
        yield k, m, np.fromfile(d / f"{name}__{k}.bin", dtype=dt).reshape(m["ne"][::-1]).astype(np.float64)
        k += 1


def find(R, g, name, ne_pred):
    for k, m, a in dumps(R, g, name):
        if ne_pred(m["ne"]):
            return m, a
    raise AssertionError(f"no dump {name} with the expected shape in graph {g}")


class _F16(F16Mixin, base._RunParity):
    ROOT, GGUF_PATH, SKIP, TOL, TABLE, F32, MISMATCH = DUMP, GGUF, SKIP, TOL16, TABLE, F32, MISMATCH
    TIE_RTOL = MOE_TIE_RTOL = TOL16          # a near-tie under the f16 tolerance is replayed and counted (module doc)
    BASE_TESTS = tuple(sorted(n for n in dir(base._RunParity) if n.startswith("test_")))

    def tol_for(self, what, il):
        """f32 paths keep the N6 tolerance: every layer before the first attention layer (its carried state reads
        only those layers of earlier tokens) and, on the first attention layer, what is computed before the
        attention reads the cache (q, k, v, the raw indexer key, the indexer query)."""
        first_att = min(self.ATT)
        if 0 <= il < first_att or (il == first_att and what.startswith(PRE_ATTENTION)):
            return base.TOL
        return self.TOL

    def test_run_exercises_its_mechanisms(self):
        super().test_run_exercises_its_mechanisms()
        if self.RUN == "pf64":
            self.assertEqual([gr["n"] for gr in self.R.graphs], [64])
        modes = {self.graph_mode(gr) for gr in self.R.graphs}
        if FA:            # pf64 runs the tiled flash attention, every other run only one_chunk
            self.assertEqual(modes, {"fa_tiled"} if self.RUN == "pf64" else {"fa_chunk"})

    def test_attention_path(self):
        """The CPU path the run took: flash attention leaves no kq_soft_max node (src/llama-graph.cpp:2651-2697 vs
        2698-2735), and the cache tensors have the type of -ctk / -ctv."""
        R = self.R
        for gr in R.graphs:
            for il in self.ATT:
                self.assertEqual(R.has(gr["g"], f"kq_soft_max-{il}"), not FA, f"kq_soft_max-{il} g{gr['g']}")
                m, _ = find(R, gr["g"], f"cache_k_l{il} (view)", lambda ne: ne[1] == 512)
                self.assertEqual(m["ggml_type"], KV, f"cache_k_l{il} type")
                if il in self.QSA:
                    m, _ = find(R, gr["g"], f"cache_idx_k_l{il} (view)", lambda ne: ne[1] == 512)
                    self.assertEqual(m["ggml_type"], KV, f"cache_idx_k_l{il} type")

    def _cache_rows(self, gr, il):
        """The oracle's cache contents after graph gr, as it stored them: {position: (k, v, k_raw)} from the latest
        graph that wrote the cell."""
        rows = {}
        for g2 in self.R.graphs:
            if g2["g"] > gr["g"]:
                break
            st = self.B[g2["g"]]["layers"][il]["attn"]["stored"]
            for r, p in enumerate(g2["pos"]):
                rows[p] = (st["k"][r], st["v"][r], st["k_raw"][r] if "k_raw" in st else None)
        return rows

    def test_cache_contents(self):
        """The caches after every graph (the set_rows outputs, ggml order [row, cell]): (a) the store itself, the rows
        this graph wrote equal f16 of its own dumped pre-store values (Kcur, Vcur, indexer_k_raw, indexer_pool_k_new
        into the slot of each pool it completed, a pool's last member's cell); (b) every cached row so far against the
        oracle as it stored them: K, V, the raw indexer key and the pooled key of every complete pool."""
        R, hp = self.R, self.hp
        kp, d = hp["kpool"], hp["idx_dim"]
        stats, n_store = {}, 0
        for gr in R.graphs:
            g, n, end = gr["g"], gr["n"], gr["p0"] + gr["n"]
            for il in self.ATT:
                _, kc = find(R, g, f"cache_k_l{il} (view)", lambda ne: ne[1] == 512)
                kc = kc.reshape(512, -1)
                n_store += store_check(self, "K", il, g, kc[gr["pos"]], R.dump(g, f"Kcur-{il}", 1)[0].reshape(n, -1))
                rows = self._cache_rows(gr, il)
                want_k = np.stack([rows[p][0].reshape(-1) for p in range(end)])
                cache_check(self, "K", il, g, kc[:end], want_k, stats, self.tol_for("Kcur", il))
                if FA:       # with flash attention off V is stored transposed (v_trans), not checked here
                    _, vc = find(R, g, f"cache_v_l{il} (view)", lambda ne: ne[1] == 512)
                    vc = vc.reshape(512, -1)
                    n_store += store_check(self, "V", il, g, vc[gr["pos"]],
                                           R.dump(g, f"Vcur-{il}", 0)[0].reshape(n, -1))
                    want_v = np.stack([rows[p][1].reshape(-1) for p in range(end)])
                    cache_check(self, "V", il, g, vc[:end], want_v, stats, self.tol_for("Vcur", il))
                if il not in self.QSA:
                    continue
                _, ic = find(R, g, f"cache_idx_k_l{il} (view)", lambda ne: ne[0] == 2 * d and ne[1] == 512)
                ic = ic.reshape(512, -1)
                n_store += store_check(self, "indexer k_raw", il, g, ic[gr["pos"], :d],
                                       R.dump(g, f"indexer_k_raw-{il}")[0].reshape(n, d))
                want_kr = np.stack([rows[p][2] for p in range(end)])
                cache_check(self, "indexer k_raw", il, g, ic[:end, :d], want_kr, stats,
                            self.tol_for("indexer_k_raw", il))
                b0, b1 = gr["p0"] // kp, end // kp
                if b1:
                    _, pc = find(R, g, f"cache_idx_k_l{il} (view) (view)", lambda ne: ne[0] == d and ne[1] == 512)
                    pc = pc.reshape(512, -1)
                    reps = [kp * b + kp - 1 for b in range(b1)]     # a pool's slot is its last member's cell
                    if b1 > b0:
                        new = R.dump(g, f"indexer_pool_k_new-{il}")[0].reshape(-1, d)[:b1 - b0]
                        n_store += store_check(self, "indexer pooled key", il, g, pc[reps[b0:]], new)
                    cache_check(self, "indexer pooled key", il, g, pc[reps], self.B[g]["layers"][il]["attn"]["pooled"],
                                stats, self.tol_for("indexer_pool_k_new", il))
        self.TABLE.append((self.RUN, f"cache store: {n_store} elements equal to the rounded pre-store value", -1, -1,
                           0.0, 0.0, ()))
        cache_share(self, stats)

    def test_flash_attention_op(self):
        """ref/fattn.py on llama.cpp's own inputs: the dumped rotated q (Qcur), the K/V the attention read from the
        cache (the get_k / get_v views) and the cells the dumped selection mask keeps (the causal cells 0..t on the
        dense layer) -> the dumped attention output before the gate (attn_pregate)."""
        R, hp = self.R, self.hp
        H, D = hp["n_head"], hp["head_dim"]
        n_one = n_all = 0
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            mode = self.graph_mode(gr)
            for il in self.ATT:
                q = R.dump(g, f"Qcur-{il}")[0].reshape(n, H, D)
                _, k = find(R, g, f"cache_k_l{il} (view)", lambda ne: ne[0] == D and ne[2] != 1)
                _, v = find(R, g, f"cache_v_l{il} (view)", lambda ne: ne[0] == D and ne[2] != 1) if FA else \
                    (None, None)
                k = k.reshape(-1, k.shape[-2], D)
                if v is not None:
                    v = v.reshape(-1, v.shape[-2], D)
                else:                # FA off: V is stored transposed (v_trans); use the oracle's cached V of the cells
                    v = np.zeros_like(k)
                    for p in range(gr["p0"] + n):
                        g2, r = self._writer(gr, p)
                        v[p] = self.B[g2["g"]]["layers"][il]["attn"]["v"][r]
                    v = F.f16(v) if KV == "f16" else v
                if il in self.QSA:
                    m = R.dump(g, f"kq_mask_qsa-{il}")[0].reshape(n, -1)
                    cells = [np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0] for r in range(n)]
                else:
                    cells = [np.arange(p + 1) for p in gr["pos"]]
                want = F.attend(q, k, v, cells, 1.0 / np.sqrt(D), mode)
                got = R.dump(g, f"attn_pregate-{il}")[0].reshape(want.shape)
                if mode == "fa_chunk":
                    u = ulp_diff(got, want)
                    n_one += int((u > 0.5).sum())
                    n_all += u.size
                    self.assertLessEqual(float(u.max()), 1.0, f"flash attention L{il} g{g}: {u.max():.2f} spacings")
                    self.close(f"flash attention op ({mode}, own inputs)", il, g, got, want)
                elif mode == "mm16":      # f16 probabilities of llama.cpp's f32 softmax: each within one spacing
                    bound = F.mm16_bound(q, k, v, cells, 1.0 / np.sqrt(D)) + 1e-6 * np.abs(want).max()
                    r = float((np.abs(got - want) / bound).max())
                    self.assertLessEqual(r, 1.0, f"mm16 L{il} g{g}: {r:.2f} x the one-spacing bound")
                    self.close(f"flash attention op ({mode}, own inputs)", il, g, got, want)
                else:
                    self.close(f"flash attention op ({mode}, own inputs)", il, g, got, want, 1e-5)
        if n_all:
            self.TABLE.append((self.RUN, f"flash attention op: elements > 0.5 spacing {n_one}/{n_all}", -1, -1, 0.0,
                               n_one / n_all, ()))

    def _writer(self, gr, p):
        """(graph, row) of the latest graph up to gr that computed position p (the cell's content)."""
        hit = None
        for g2 in self.R.graphs:
            if g2["g"] > gr["g"]:
                break
            if p in g2["pos"]:
                hit = (g2, g2["pos"].index(p))
        return hit

    def discrete(self, B):
        """Rows where B's QSA selection, expert set or greedy token differs from llama.cpp's."""
        R, k = self.R, self.hp["n_expert_used"]
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab)
        out = {"QSA selection": 0, "MoE experts": 0, "greedy token": 0}
        for gr in R.graphs:
            L = B[gr["g"]]["layers"]
            for il in self.QSA:
                out["QSA selection"] += sum(self.seen[il][(gr["g"], r)] != L[il]["attn"]["cells"][r]
                                            for r in range(gr["n"]))
            for il in range(self.hp["n_layer"]):
                ids = np.sort(R.dump(gr["g"], f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k).astype(int), 1)
                out["MoE experts"] += int((ids != np.sort(L[il]["moe"]["ids"], 1)).any(1).sum())
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            out["greedy token"] += int((rows.argmax(1) != B[gr["g"]]["logits"].argmax(1)).sum())
        return out

    def attention_pairs(self, which):
        B = self.B if which == "aware" else self.B32
        for gr in self.R.graphs:
            for il in self.ATT:
                got = self.R.dump(gr["g"], f"attn_pregate-{il}")[0]
                want = B[gr["g"]]["layers"][il]["attn"]["pregate"]
                yield il, gr["g"], got.reshape(want.shape), want


def _cls(name, run):
    return unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)(type(name, (_F16, unittest.TestCase), {"RUN": run}))


Prefill64 = _cls("Prefill64", "pf64")
Ubatch16 = _cls("Ubatch16", "ub16")
Ubatch10 = _cls("Ubatch10", "ub10")
Decode = _cls("Decode", "decode")
Rollback = _cls("Rollback", "rollback")


def tearDownModule():
    if TABLE:
        print_tables("qwen4exp", TABLE, F32, MISMATCH, os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a"))


if __name__ == "__main__":
    unittest.main()
