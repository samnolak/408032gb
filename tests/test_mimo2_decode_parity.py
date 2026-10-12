"""MiMo-V2.x-Flash (llama.cpp "mimo2") beyond the single G1 prefill: several ubatches, token-by-token decode past the
sliding-window cache's size, rollbacks inside and across the window, a rollback deeper than the SWA cache retains,
against ref/mimo2.py (N16, gap 12 of docs/reassessment/CORRECTNESS.md).

tools/tinygen/run_parity_mimo2_decode.sh writes two tiny models with the same trunk weights (tools/tinygen/mimo2_tiny.py,
seed 0: 6 layers, global layers 0 and 5, sliding-window layers 1-4 with window 8, a dense layer 0, MoE 8 experts top-2
elsewhere): n16.gguf (--fused-qkv --sinks swa --value-scale 0.75 --mtp) and split.gguf (the G1 model), runs them through
llama.cpp's real mimo2 graph on the CPU (-c 512, f32 caches, flash attention off) and dumps every graph (one per
ubatch, tools/dump/dump_tensors.cpp DUMP_PER_GRAPH / DUMP_CALLS). The SWA cache has 256 cells
(pad(min(n_ctx, n_swa + n_ubatch), 256), src/llama-kv-cache-iswa.cpp:73), so the runs over 300 positions wrap it:
  ub64      one call of 300 tokens, -ub 64 (64 x 4 + 44)
  ub13      one call, -ub 13 (23 x 13 + 1): ubatches longer than the window; one straddles the wrap
  ub5       one call, -ub 5 (60 x 5): the window spans two or three ubatches
  decode    prefill 240 (-ub 64), 60 one-token calls: the SWA cache wraps during the decode
  rollback  prefill 240, 16 x 1, a 4-token draft rolled back by 2 (inside the window), 6 x 1, a 12-token draft rolled
            back by 10 (more than the window), 34 x 1 (llama_memory_seq_rm(mem, 0, p, -1))
  evict     prefill 280, rollback to position 20, a call of 24 and 4 x 1: the SWA cache had wrapped past positions
            13..19 that position 20 needs; llama.cpp accepts the rollback (seq_rm returns true) and attends without
            them (finding in docs/evidence/2026-10-11-n16-mimo-decode-parity.md)
  split     the G1 model (split attn_q/k/v, sinks on every layer, no value scale): prefill 240, 20 x 1
  mtp       the target side of the MTP run (h_nextn exported: the last layer's l_out is named h_nextn)
Every graph is compared, row by row, with
  (A) [full]: ref/mimo2.py forward(final tokens) at the graph's positions (all keys of the sequence available,
      prefill semantics): causality; rows of rejected draft tokens have no (A); the evict run has none (its point is
      that llama.cpp differs from (A) there, test_full_oracle_differs_after_the_deep_rollback);
  (B) [chunk]: ref/mimo2.py forward_chunk replaying the same calls (rollback = cache_truncate) with the cell model of
      llama.cpp's KV caches (find_slot / apply_ubatch / seq_rm, ref/mimo2.py "KV caches"): the cells each ubatch is
      written to, the n_kv cells each graph reads, and the K / V rows each cell holds, stale ones included.
The cache itself is checked through kq (the raw K . q products over all n_kv cells the graph reads: equal to (B) on
every cell, the empty, stale and masked ones too, so the content of every cell is checked) and kq_soft_max (exactly 0
on every masked cell, the probabilities on the visible ones; (A) placed at the cells (B) says hold those positions).
Tolerances (as N2 / N6 / N9): normwise relative error max|got - want| / max|want| < 1e-4 for every f32 path; exact for
the embedding gather and for the masked (zero) cells of kq_soft_max; MoE expert ids equal as sets; greedy argmax equal.
Layouts: kq / kq_soft_max are [n_kv, n_tokens, n_head] (ggml ne), read as (H, T, n_kv). The table printed at the end
has the worst error per run and component, then per layer (gap 14 of CORRECTNESS.md).
Env: MIMO2_DECODE_DUMP (the driver's <work>/n16: models, tokens.txt, one directory per run), LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT (the commit the dumps come from; default ec7630a).
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import mimo2 as M  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402

DUMP = os.environ.get("MIMO2_DECODE_DUMP")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
LLAMA_COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a640789c393694fb194f1bbbf0369fc62d")
TOL = 1e-4
N_CTX = 512            # the driver's -c
SKIP = "set MIMO2_DECODE_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_mimo2_decode.sh"

_MODEL = {}
TABLE = []        # (run, component, layer, graph, max abs, rel)


def model(path):
    """Weights (float64) and hyper-parameters of the GGUF at path, read once per path."""
    path = str(path)
    if path not in _MODEL:
        sys.path.append(GGUF_PY)
        try:
            from gguf import GGUFReader
        finally:
            sys.path.remove(GGUF_PY)
        r = GGUFReader(path)
        _MODEL[path] = ({t.name: np.array(t.data, dtype=np.float64) for t in r.tensors},
                        M.hparams({f.name: f for f in r.fields.values()}))
    return _MODEL[path]


def place_full(p_full, kpos, rows):
    """(A) probabilities (H, T_seq, T_seq) placed at the cells: out[h, r, c] = p_full[h, rows[r], kpos[c]] where cell c
    holds a position (0 elsewhere; masked positions already have p 0 in (A))."""
    H = p_full.shape[0]
    out = np.zeros((H, len(rows), len(kpos)))
    live = np.nonzero(np.asarray(kpos) >= 0)[0]
    for r, t in enumerate(rows):
        ok = live[np.asarray(kpos)[live] < p_full.shape[2]]
        out[:, r, ok] = p_full[:, t, np.asarray(kpos)[ok]]
    return out


class _RunParity:
    RUN = None
    MODEL = "n16.gguf"
    FULL = True                # compare with (A); the evict run cannot (module docstring)

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")
        cls.W, cls.hp = model(pathlib.Path(DUMP, cls.MODEL))
        cls.R = Run(cls.RUN, DUMP)
        cls.ub = cls.R.man["n_ubatch"]
        cls.A = M.forward(cls.R.final, cls.W, cls.hp)
        cls.B, cache, by_call = {}, M.new_cache(cls.hp, N_CTX, cls.ub), {}
        for gr in cls.R.graphs:
            by_call.setdefault(gr["call"], []).append(gr)
        cls.cells = {}           # graph -> {"base"/"swa": cell positions after the graph's placement}
        for c, call in enumerate(cls.R.calls):
            if call["kind"] == "rollback":
                M.cache_truncate(cache, call["pos0"])
                continue
            for gr in by_call[c]:
                assert len(cache["tokens"]) == gr["p0"], (cls.RUN, gr["g"])
                cls.B[gr["g"]] = M.forward_chunk(gr["toks"], cls.W, cls.hp, cache)
                cls.cells[gr["g"]] = {k: cache[k].pos.copy() for k in ("base", "swa")}
        cls.final_cache = cache

    # ---- helpers ----
    def dump(self, g, name, k=0):
        return self.R.dump(g, name, k)[0]

    def close(self, what, il, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} L{il} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        TABLE.append((self.RUN, what, il, g, mx, rel))
        self.assertLess(rel, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, il, gr, got, a_full, b_rows):
        """A per-token tensor of graph gr against (A) (kept rows; FULL runs only) and (B)."""
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        kept = np.array(gr["kept"])
        if self.FULL and kept.any():
            self.close(f"{what} [full]", il, gr["g"], got[kept], np.asarray(a_full)[np.array(gr["pos"])[kept]])
        self.close(f"{what} [chunk]", il, gr["g"], got, b_rows)

    def l_out_name(self, g, il):
        """The last layer's l_out is renamed h_nextn when the graph names the final hidden state (mimo2.cpp:244-246
        @ec7630a only with embeddings_nextn; every graph @79e2e74, mimo2.cpp:236-238 there)."""
        nm = f"l_out-{il}"
        if il == self.hp["n_layer"] - 1 and not self.R.has(g, nm):
            nm = "h_nextn"
        return nm

    # ---- tests ----
    def test_graphs_and_token_mapping(self):
        """One graph per ubatch, in order; each graph's embedding rows are exactly its tokens' (proves the mapping of
        graphs to positions every other test relies on)."""
        R = self.R
        self.assertEqual(len(R.man["graphs"]), len(R.graphs))
        for c, call in enumerate(R.calls):
            if call["kind"] == "decode":
                g0, g1 = call["graphs"]
                self.assertEqual(g1 - g0, -(-call["n"] // self.ub), f"call {c}")
        self.assertEqual([gr["g"] for gr in R.graphs], list(range(len(R.graphs))))
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32)
        self.assertEqual(lo.size, R.n_rows * self.W["output.weight"].shape[0])
        for gr in R.graphs:
            got = self.dump(gr["g"], "embd").reshape(gr["n"], -1)       # build_inp_embd names it "embd" (LG:2461)
            self.close("embd (token rows, exact)", -1, gr["g"], got, self.W["token_embd.weight"][gr["toks"]], tol=1e-300)

    def test_attention(self):
        """attn_norm, the q / k / v projections (fused attn_qkv rows or split), q and k after RoPE, kqv_out, attn_out and
        the value-scaled attn_out, per layer and graph."""
        R, hp = self.R, self.hp
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            for il in range(hp["n_layer"]):
                La, Lb = self.A["layers"][il], self.B[g]["layers"][il]
                a, b = La["attn"], Lb["attn"]
                self.both("attn_norm", il, gr, self.dump(g, f"attn_norm-{il}"), La["attn_norm"], Lb["attn_norm"])
                if R.has(g, f"wqkv-{il}"):              # fused (mimo2.cpp:129-142): q, k, v are views of wqkv
                    fa = np.concatenate([a[x].reshape(len(a[x]), -1) for x in ("q_pre", "k_pre", "v")], 1)
                    fb = np.concatenate([b[x].reshape(n, -1) for x in ("q_pre", "k_pre", "v")], 1)
                    self.both("wqkv (fused projection)", il, gr, self.dump(g, f"wqkv-{il}"), fa, fb)
                    kq_, kk_ = 0, 0
                else:                                   # split (mimo2.cpp:145-156): occurrence 0 before RoPE
                    for nm, key in (("Qcur", "q_pre"), ("Kcur", "k_pre")):
                        self.both(f"{nm} (projection)", il, gr, self.dump(g, f"{nm}-{il}", 0),
                                  a[key].reshape(len(a[key]), -1), b[key].reshape(n, -1))
                    kq_, kk_ = 1, 1
                self.both("Qcur (RoPE)", il, gr, self.dump(g, f"Qcur-{il}", kq_), a["q"], b["q"])
                self.both("Kcur (RoPE)", il, gr, self.dump(g, f"Kcur-{il}", kk_), a["k"], b["k"])
                self.both("Vcur", il, gr, self.dump(g, f"Vcur-{il}"), a["v"], b["v"])
                self.both("kqv_out", il, gr, self.dump(g, f"kqv_out-{il}"), a["kqv"], b["kqv"])
                self.both("attn_out (wo)", il, gr, self.dump(g, f"attn_out-{il}"), a["attn_out"], b["attn_out"])
                if hp["v_scale"]:
                    self.both("attn_out_scaled (value scale)", il, gr, self.dump(g, f"attn_out_scaled-{il}"), a["out"],
                              b["out"])
                else:
                    self.assertFalse(R.has(g, f"attn_out_scaled-{il}"), "no value scale in this model")

    def test_kv_cells(self):
        """The KV caches across ubatches and calls: n_kv cells per graph and cache, kq (K . q, unscaled) on every cell
        the graph reads, the stale and empty ones included, kq_soft_max exactly 0 on every masked cell and equal to (B)
        and (A) on the visible ones."""
        R, hp = self.R, self.hp
        for gr in R.graphs:
            g, n = gr["g"], gr["n"]
            kept = np.array(gr["kept"])
            for il in range(hp["n_layer"]):
                b = self.B[g]["layers"][il]["attn"]
                H = hp["n_head"][il]
                kq, ne = R.dump(g, f"kq-{il}")
                self.assertEqual(ne[0], b["n_kv"], f"n_kv L{il} g{g}")
                kq = kq.reshape(H, n, -1)
                self.close("kq (all n_kv cells) [chunk]", il, g, kq, b["kq"])
                sm = self.dump(g, f"kq_soft_max-{il}").reshape(H, n, -1)
                vis = M.visible(gr["pos"], b["kpos"], hp["n_swa"] if hp["swa"][il] else 0)
                self.assertTrue((sm[:, ~vis] == 0).all(), f"a masked cell has probability > 0, L{il} g{g}")
                self.assertTrue((sm[:, vis] > 0).all(), f"a visible cell has probability 0, L{il} g{g}")
                self.close("kq_soft_max (cells) [chunk]", il, g, sm, b["probs"])
                if self.FULL and kept.any():
                    rows = np.array(gr["pos"])[kept]
                    pa = place_full(self.A["layers"][il]["attn"]["probs"], b["kpos"], rows)
                    self.close("kq_soft_max (cells) [full]", il, g, sm[:, kept], pa)
                    ka = place_full(self.A["layers"][il]["attn"]["kq"], b["kpos"], rows)
                    vk = vis[kept]
                    self.close("kq (visible cells) [full]", il, g, kq[:, kept][:, vk], ka[:, vk])

    def test_ffn(self):
        """ffn_inp, ffn_norm, the dense layer (ffn_up, ffn_gate, ffn_swiglu, ffn_out) or the MoE (router logits, the
        selected experts as sets, their normalised weights, ffn_moe_out), l_out."""
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
                    if self.FULL and kept.any():
                        want = np.sort(Fa["ids"][np.array(gr["pos"])[kept]], 1)
                        self.assertTrue(np.array_equal(np.sort(ids, 1)[kept], want), f"top-k (full) L{il} g{g}")
                    wd = self.dump(g, f"ffn_moe_weights_norm-{il}").reshape(n, k)
                    wb = np.array([[dict(zip(Fb["ids"][r], Fb["weights"][r]))[e] for e in ids[r]] for r in range(n)])
                    self.close("ffn_moe_weights_norm [chunk]", il, g, wd, wb)
                    self.both("ffn_moe_out", il, gr, self.dump(g, f"ffn_moe_out-{il}"), Fa["moe_out"], Fb["moe_out"])
                self.both("l_out", il, gr, self.dump(g, self.l_out_name(g, il)), La["l_out"], Lb["l_out"])

    def test_head_and_logits(self):
        R = self.R
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab).astype(np.float64)
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            self.both("result_norm", -1, gr, self.dump(g, "result_norm"), self.A["result_norm"], b["result_norm"])
            self.both("result_output", -1, gr, self.dump(g, "result_output"), self.A["logits"], b["logits"])
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            self.both("logits.bin", -1, gr, rows, self.A["logits"], b["logits"])
            self.assertTrue(np.array_equal(rows.argmax(1), b["logits"].argmax(1)), f"greedy tokens g{g}")
            kept = np.array(gr["kept"])
            if self.FULL and kept.any():
                self.assertTrue(np.array_equal(rows[kept].argmax(1),
                                               self.A["logits"][np.array(gr["pos"])[kept]].argmax(1)),
                                f"greedy tokens (full) g{g}")

    def test_run_exercises_its_mechanisms(self):
        """What each run is for actually happened (sizes, the SWA cache wrapping, the rollbacks)."""
        R, hp = self.R, self.hp
        self.assertEqual(R.final, self.final_tokens())
        sizes = [gr["n"] for gr in R.graphs]
        swa_size = M.pad(min(N_CTX, hp["n_swa"] + self.ub), M.KV_PAD)
        self.assertEqual(self.final_cache["swa"].size, swa_size)
        wrapped = [gr["g"] for gr in R.graphs if any(c != p for c, p in zip(self.B[gr["g"]]["slots"]["swa"], gr["pos"]))]
        expect = {"ub64": [64] * 4 + [44], "ub13": [13] * 23 + [1], "ub5": [5] * 60,
                  "decode": [64] * 3 + [48] + [1] * 60, "split": [64] * 3 + [48] + [1] * 20,
                  "mtp": [64] * 3 + [48] + [1] * 24}
        if self.RUN in expect:
            self.assertEqual(sizes, expect[self.RUN])
        if self.RUN != "evict":
            # every run but evict reaches position 256 with a 256-cell SWA cache: some ubatch is written to cells
            # other than its positions (the ring wrapped), the base cache never wraps (cell = position)
            self.assertEqual(swa_size, 256)
            self.assertTrue(wrapped, "the SWA cache never wrapped")
            self.assertTrue(len(R.final) > swa_size)
        for gr in R.graphs:
            self.assertEqual(self.B[gr["g"]]["slots"]["base"], gr["pos"], f"base cache cell != position, g{gr['g']}")
        if self.RUN == "ub13":     # one ubatch straddles the end of the SWA cache: cells 247..255 then 0..3
            self.assertIn([247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3],
                          [self.B[gr["g"]]["slots"]["swa"] for gr in R.graphs])
        if self.RUN == "ub5":      # the window (8) spans earlier ubatches: rows that see keys of an earlier graph
            self.assertGreater(hp["n_swa"], self.ub)
        if self.RUN == "rollback":
            rbs = [c for c in R.calls if c["kind"] == "rollback"]
            self.assertEqual([(c["pos0"], c["n"]) for c in rbs], [(258, 2), (266, 10)])
            self.assertLess(rbs[0]["n"], hp["n_swa"])
            self.assertGreater(rbs[1]["n"], hp["n_swa"])
            for c in rbs:          # the rolled-back tokens were wrong (and are replaced)
                draft = R.calls[R.calls.index(c) - 1]
                rej = R.fed[draft["tok0"] + draft["n"] - c["n"]:draft["tok0"] + draft["n"]]
                self.assertNotEqual(rej, R.final[c["pos0"]:c["pos0"] + c["n"]])
            # the first draft (256..259) was placed after the wrap (cells 0..3); the re-decoded 258, 259 reuse cells 2, 3
            after = [gr for gr in R.graphs if gr["p0"] in (258, 259) and all(gr["kept"])]
            self.assertEqual([self.B[gr["g"]]["slots"]["swa"] for gr in after], [[2], [3]])

    def final_tokens(self):
        t = [int(x) for x in pathlib.Path(DUMP, "tokens.txt").read_text().split(",")]
        return {"evict": t[:48], "split": t[:260], "mtp": t[:264]}.get(self.RUN, t)


def _cls(run, **kw):
    return unittest.skipUnless(DUMP and GGUF_PY, SKIP)(
        type(f"Run_{run}", (_RunParity, unittest.TestCase), dict(RUN=run, **kw)))


Run_ub64 = _cls("ub64")
Run_ub13 = _cls("ub13")
Run_ub5 = _cls("ub5")
Run_decode = _cls("decode")
Run_rollback = _cls("rollback")
Run_split = _cls("split", MODEL="split.gguf")
Run_mtp_target = _cls("mtp")


@unittest.skipUnless(DUMP and GGUF_PY, SKIP)
class Run_evict(_RunParity, unittest.TestCase):
    RUN = "evict"
    FULL = False

    def test_full_oracle_differs_after_the_deep_rollback(self):
        """After prefill 280 the SWA cache (256 cells) holds positions 24..279; llama_memory_seq_rm(0, 20, -1) frees
        all of them and succeeds (src/llama-kv-cache-iswa.cpp:113-120, src/llama-kv-cache.cpp:382-420), so position 20
        attends in the SWA layers to itself alone instead of 13..20: the dumped softmax has one live cell where (A) has
        8 positions, kqv_out of the first SWA layer differs from (A) by more than 1e-2 (relative) while it equals (B)
        (test_attention), and the difference travels on through the global layers to the logits."""
        R, hp = self.R, self.hp
        rb = [c for c in R.calls if c["kind"] == "rollback"]
        self.assertEqual([(c["pos0"], c["n"]) for c in rb], [(20, 260)])
        gr = next(gr for gr in R.graphs if gr["p0"] == 20)
        il = next(i for i in range(hp["n_layer"]) if hp["swa"][i])
        sm = self.dump(gr["g"], f"kq_soft_max-{il}").reshape(hp["n_head"][il], gr["n"], -1)
        self.assertEqual(int((sm[0, 0] > 0).sum()), 1, "position 20 should see one SWA cell")
        self.assertEqual(int((self.A["layers"][il]["attn"]["probs"][0, 20] > 0).sum()), hp["n_swa"])
        got = self.dump(gr["g"], f"kqv_out-{il}").reshape(gr["n"], -1)[0]
        want = self.A["layers"][il]["attn"]["kqv"][20]
        self.assertGreater(np.abs(got - want).max() / np.abs(want).max(), 1e-2)
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, -1)
        last = R.graphs[-1]
        self.assertGreater(np.abs(lo[last["row0"]] - self.A["logits"][last["pos"][0]]).max() /
                           np.abs(self.A["logits"][last["pos"][0]]).max(), 1e-3)
        TABLE.append((self.RUN, "kqv_out vs (A) at position 20 (expected to differ)", il, gr["g"],
                      float(np.abs(got - want).max()), float(np.abs(got - want).max() / np.abs(want).max())))


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, mx, rel in TABLE:
        key = (run, what)
        if key not in worst or rel > worst[key][3]:
            worst[key] = (il, g, mx, rel)
    print(f"\nworst error per run and component over layers and graphs (rel = max|got-want| / max|want|), "
          f"llama.cpp {LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'run':9s} {'component':52s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, what), (il, g, mx, rel) in worst.items():
        print(f"{run:9s} {what:52s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)
    per_layer = {}
    for run, what, il, g, mx, rel in TABLE:
        if "expected to differ" in what:
            continue
        if il not in per_layer or rel > per_layer[il][4]:
            per_layer[il] = (run, what, g, mx, rel)
    print(f"\nworst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp "
          f"{LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'layer':>5s} {'run':9s} {'component':52s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for il in sorted(per_layer):
        run, what, g, mx, rel = per_layer[il]
        print(f"{il:5d} {run:9s} {what:52s} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
