"""N10 (gap 3 of docs/reassessment/CORRECTNESS.md): the glm5-next graph at the R1 default precision of the KV cache and
the attention, -ctk f16 -ctv f16 -fa on (scripts/r1/configs/llamacpp-server.json), against the precision-aware oracle.

tools/tinygen/run_parity_f16fa.sh runs the G1 / N9 tiny model (tools/tinygen/glm5next_tiny.py --layers 12, seed 0)
and the N9 token sequence through llama.cpp on the CPU with those flags, dumped graph by graph like N9: pf64 (one
64-token ubatch, the G1 prefill shape), ub16, ub10, ub18, decode (48 + 16 x 1) and rollback (n_rs_seq 3). Every N9
check (tests/test_glm5next_decode_parity.py: mHC, KDA and its carried state, DSA projections, k-pool cache, selection
and sparse MLA, MoE, head and logits; (A) and (B)) runs again with the oracle at the run's precision (ref/fattn.py
precision(PARITY_KV, PARITY_FA)): the MLA latent, the indexer key | gate rows and the pooled keys rounded to f16 where
llama.cpp stores them (ref/glm5next.py docstring), the DSA attention in the mode the graph takes: the mask path runs
build_attn_mha, i.e. CPU flash attention (one_chunk with an f16 accumulator below 64 query rows, tiled f32 at 64 and
more); the ec7630a gather path (ubatches of <= 16 tokens) never runs flash attention and only sees the f16 latents.
Added here, as for qwen4exp (tests/test_qwen4exp_f16fa_parity.py):
  test_cache_contents     (a) the store: the MLA latent, indexer key | gate and pooled-key rows a graph wrote equal f16
                          of its own dumped pre-store values (kv_cmpr, indexer_k, indexer_gate, indexer_pool_k_new);
                          (b) every cached row against f16(oracle)
  test_flash_attention_op ref/fattn.py on llama.cpp's own q_absorbed, cached latents and selection mask, times W_v_b,
                          reproduces the dumped kqv_out (mask path only)
  test_attention_path     flash attention (fattn_mla, no kq_soft_max) iff PARITY_FA=on on the mask path; the gather
                          path where the pin has it; the cache tensors' type
  test_attention_output_vs_plain_oracle, test_plain_f32_oracle (tests/f16fa.py)

Tolerances: as tests/test_qwen4exp_f16fa_parity.py (TOL = 16 U16 = 7.8e-3 downstream of the first DSA attention, the
N9 tolerance 1e-4 before it; DSA selections and MoE expert sets replayed within TOL of the oracle's scores and
counted; the cache store exact). The flash-attention op here is checked after W_v_b: every element within the bound
sum_l |W_v_b[h, v, l]| * spacing(O[l]) (each element O[l] of the flash-attention output within one binary16 spacing)
plus 1e-6 max|kqv_out| for the f32 product.

Env: GLM5NEXT_F16FA_DUMP (root with one directory per run and tokens.txt), GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT, PARITY_KV / PARITY_FA (default f16 / on).
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from ref import fattn as F  # noqa: E402
from tests import test_glm5next_decode_parity as base  # noqa: E402
from tests.f16fa import FA, KV, U16, F16Mixin, Recorder, cache_check, print_tables, store_check  # noqa: E402
from tests.test_qwen4exp_f16fa_parity import find  # noqa: E402

DUMP = os.environ.get("GLM5NEXT_F16FA_DUMP")
GGUF = os.environ.get("GLM5NEXT_F16FA_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
SKIP = ("set GLM5NEXT_F16FA_DUMP, GLM5NEXT_F16FA_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_f16fa.sh")
TOL16 = 16 * U16
TABLE, F32, MISMATCH = [], Recorder(), {}
# computed on the first DSA layer before its attention reads the caches (pure f32 paths)
PRE_ATTENTION = ("hc_post (attn)", "hc_comb (attn", "hc_attn_pre", "attn_norm", "q_resid", "kv_cmpr", "q_absorbed",
                 "indexer_k (", "indexer_gate", "indexer_q", "indexer_weights")


class _F16(F16Mixin, base._RunParity):
    ROOT, GGUF_PATH, SKIP, TOL, TABLE, F32, MISMATCH = DUMP, GGUF, SKIP, TOL16, TABLE, F32, MISMATCH
    TIE_RTOL = MOE_TIE_RTOL = TOL16          # a near-tie under the f16 tolerance is replayed and counted (module doc)
    BASE_TESTS = tuple(sorted(n for n in dir(base._RunParity) if n.startswith("test_")))

    def tol_for(self, what, il):
        """f32 paths keep the N9 tolerance: every layer before the first DSA layer and, on that layer, what is
        computed before its attention reads the caches."""
        first = min(self.DSA)
        if 0 <= il < first or (il == first and what.startswith(PRE_ATTENTION)):
            return base.TOL
        return self.TOL

    def test_run_exercises_its_mechanisms(self):
        super().test_run_exercises_its_mechanisms()
        if self.RUN == "pf64":
            self.assertEqual([gr["n"] for gr in self.R.graphs], [64])
        modes = [self.graph_mode(gr) for gr in self.R.graphs]
        if FA:            # pf64 runs the tiled flash attention; the gather path (ec7630a, <= 16 tokens) none
            want = ["fa_tiled" if gr["n"] >= F.FA_TILE_Q else "exact" if self.gather[gr["g"]] else "fa_chunk"
                    for gr in self.R.graphs]
            self.assertEqual(modes, want)
            self.assertIn("fa_tiled" if self.RUN == "pf64" else "fa_chunk" if not base.EC7630A else
                          "exact", modes)

    def test_attention_path(self):
        """Mask path: flash attention leaves fattn_mla (src/llama-graph.cpp:2681-2688) and no kq_soft_max; gather path
        (ec7630a): kq_soft_max_gathered only. The latent and indexer caches have the type of -ctk."""
        R = self.R
        for gr in R.graphs:
            g = gr["g"]
            for il in self.DSA:
                if self.gather[g]:
                    self.assertTrue(R.has(g, f"kq_soft_max_gathered-{il}"))
                    self.assertFalse(R.has(g, f"fattn_mla-{il}") or R.has(g, f"kq_soft_max-{il}"))
                else:
                    self.assertEqual(R.has(g, f"fattn_mla-{il}"), FA, f"fattn_mla-{il} g{g}")
                    self.assertEqual(R.has(g, f"kq_soft_max-{il}"), not FA, f"kq_soft_max-{il} g{g}")
                m, _ = find(R, g, f"cache_k_l{il} (view)", lambda ne: ne[1] == 512)
                self.assertEqual(m["ggml_type"], KV, f"cache_k_l{il} type")
                m, _ = find(R, g, f"cache_idx_k_l{il} (view)", lambda ne: ne[1] == 512)
                self.assertEqual(m["ggml_type"], KV, f"cache_idx_k_l{il} type")

    def test_cache_contents(self):
        """The caches after every graph (the set_rows outputs, ggml order [row, cell]): (a) the store itself, the rows
        this graph wrote equal f16 of its own dumped pre-store values (kv_cmpr, indexer_k, indexer_gate,
        indexer_pool_k_new into the slot of each pool it completed, a pool's last member's cell); (b) every cached row
        so far against the oracle: the MLA latent, the indexer key | gate rows and the pooled key of every pool."""
        R, hp = self.R, self.hp
        kp, d, Lr = hp["kpool"], hp["idx_dim"], hp["kv_lora"]
        stats, n_store = [0, 0], 0
        for gr in R.graphs:
            g, n, end = gr["g"], gr["n"], gr["p0"] + gr["n"]
            for il in self.DSA:
                rows = {}
                for g2 in R.graphs:
                    if g2["g"] > g:
                        break
                    D2 = self.B[g2["g"]]["layers"][il]["dsa"]
                    for r, p in enumerate(g2["pos"]):
                        rows[p] = (D2["latent"][r], D2["ik"][r], D2["ig"][r])
                want = [np.stack([rows[p][j] for p in range(end)]) for j in range(3)]
                _, lat = find(R, g, f"cache_k_l{il} (view)", lambda ne: ne[1] == 512)
                lat = lat.reshape(512, -1)
                _, ic = find(R, g, f"cache_idx_k_l{il} (view)", lambda ne: ne[0] == 3 * d and ne[1] == 512)
                ic = ic.reshape(512, -1)
                for what, cached, name, j, cols in (("MLA latent", lat, "kv_cmpr", 0, slice(0, Lr)),
                                                    ("indexer key", ic, "indexer_k", 1, slice(0, d)),
                                                    ("indexer gate", ic, "indexer_gate", 2, slice(d, 2 * d))):
                    pre = R.dump(g, f"{name}-{il}")[0].reshape(n, -1)
                    n_store += store_check(self, what, il, g, cached[gr["pos"], cols], pre)
                    cache_check(self, what, il, g, cached[:end, cols], want[j], stats,
                                self.tol_for({"indexer_k": "indexer_k ("}.get(name, name), il))
                b0, b1 = gr["p0"] // kp, end // kp
                if b1:
                    _, pc = find(R, g, f"cache_idx_k_l{il} (view) (view)", lambda ne: ne[0] == d and ne[1] == 512)
                    pc = pc.reshape(512, -1)
                    reps = [kp * b + kp - 1 for b in range(b1)]
                    if b1 > b0:
                        new = R.dump(g, f"indexer_pool_k_new-{il}")[0].reshape(-1, d)[:b1 - b0]
                        n_store += store_check(self, "indexer pooled key", il, g, pc[reps[b0:]], new)
                    cache_check(self, "indexer pooled key", il, g, pc[reps], self.B[g]["layers"][il]["dsa"]["pooled"],
                                stats, self.tol_for("indexer_pool_k_new", il))
        self.TABLE.append((self.RUN, f"cache store: {n_store} elements equal to the rounded pre-store value", -1, -1,
                           0.0, 0.0, ()))
        if KV == "f16":
            self.TABLE.append((self.RUN, f"cache rows equal to f16(oracle): {stats[0]}/{stats[1]}", -1, -1, 0.0,
                               1.0 - stats[0] / stats[1], ()))

    def test_flash_attention_op(self):
        """Mask-path graphs: ref/fattn.py attend() on the dumped q_absorbed, the latents the attention read (get_k
        view, also the value) and the cells of the dumped kq_mask_dsa, then W_v_b per head -> the dumped kqv_out."""
        R, hp = self.R, self.hp
        H, Lr = hp["n_head"], hp["kv_lora"]
        wv_all = {il: np.asarray(self.W[f"blk.{il}.attn_v_b.weight"], dtype=np.float64) for il in self.DSA}
        n_graphs, worst = 0, 0.0
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            if self.gather[g]:
                continue
            n_graphs += 1
            mode = self.graph_mode(gr)
            for il in self.DSA:
                q = R.dump(g, f"q_absorbed-{il}")[0].reshape(n, H, Lr)
                _, k = find(R, g, f"cache_k_l{il} (view)", lambda ne: ne[0] == Lr and ne[2] != 1)
                k = k.reshape(-1, 1, Lr)
                m = R.dump(g, f"kq_mask_dsa-{il}")[0].reshape(n, -1)
                cells = [np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0] for r in range(n)]
                o = F.attend(q, k, k, cells, 1.0 / np.sqrt(hp["k_mla"]), mode)            # (T, H, L)
                wv = wv_all[il]                                                            # (H, v, L)
                want = np.einsum("hvl,thl->thv", wv, o).reshape(n, -1)
                got = R.dump(g, f"kqv_out-{il}")[0].reshape(want.shape)
                bound = np.einsum("hvl,thl->thv", np.abs(wv), F.ulp16(o)).reshape(n, -1) + 1e-6 * np.abs(want).max()
                if mode == "fa_chunk":
                    r = float((np.abs(got - want) / bound).max())
                    worst = max(worst, r)
                    self.assertLessEqual(r, 1.0, f"flash attention L{il} g{g}: {r:.2f} x the one-spacing bound")
                    self.close(f"flash attention op ({mode}, own inputs)", il, g, got, want)
                else:
                    self.close(f"flash attention op ({mode}, own inputs)", il, g, got, want, 1e-5)
        if worst:
            self.TABLE.append((self.RUN, f"flash attention op: max |err| / (W_v_b x one spacing of O) = {worst:.3f}",
                               -1, -1, 0.0, worst, ()))
        if base.EC7630A and self.RUN in ("ub16", "ub10"):
            self.assertEqual(n_graphs, 0)       # every graph took the gather path
        else:
            self.assertGreater(n_graphs, 0)

    def discrete(self, B):
        """Rows where B's DSA selection, expert set or greedy token differs from llama.cpp's."""
        R, k = self.R, self.hp["n_expert_used"]
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab)
        out = {"DSA selection": 0, "MoE experts": 0, "greedy token": 0}
        for gr in R.graphs:
            L = B[gr["g"]]["layers"]
            for il in self.DSA:
                out["DSA selection"] += sum(self.seen[il][(gr["g"], r)] != L[il]["dsa"]["cells"][r]
                                            for r in range(gr["n"]))
            for il in self.MOE:
                ids = np.sort(R.dump(gr["g"], f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k).astype(int), 1)
                out["MoE experts"] += int((ids != np.sort(L[il]["ffn"]["ids"], 1)).any(1).sum())
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            out["greedy token"] += int((rows.argmax(1) != B[gr["g"]]["logits"].argmax(1)).sum())
        return out

    def attention_pairs(self, which):
        B = self.B if which == "aware" else self.B32
        for gr in self.R.graphs:
            for il in self.DSA:
                want = B[gr["g"]]["layers"][il]["dsa"]["kqv"]
                yield il, gr["g"], self.R.dump(gr["g"], f"kqv_out-{il}")[0].reshape(want.shape), want


def _cls(name, run):
    return unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)(type(name, (_F16, unittest.TestCase), {"RUN": run}))


Prefill64 = _cls("Prefill64", "pf64")
Ubatch16 = _cls("Ubatch16", "ub16")
Ubatch10 = _cls("Ubatch10", "ub10")
Ubatch18 = _cls("Ubatch18", "ub18")
Decode = _cls("Decode", "decode")
Rollback = _cls("Rollback", "rollback")


def tearDownModule():
    if TABLE:
        print_tables("glm5-next", TABLE, F32, MISMATCH, os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a"))


if __name__ == "__main__":
    unittest.main()
