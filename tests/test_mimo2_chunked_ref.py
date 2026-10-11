"""Property tests of ref/mimo2.py's own continuity (N16): numpy only, a random 4-layer mimo2 model built here (no GGUF,
no llama.cpp), so they run in the CI `unit tests` step as well.

  * attend() over cells holding positions 0..T-1 is ref/mimo.py attention (G1's op oracle), with and without sinks;
  * the incremental path (forward_calls: one ubatch at a time over the cell model of llama.cpp's caches) equals the
    full-sequence forward for many ubatch sizes and call plans, also when the SWA cache wraps (a 16-cell cache here);
  * rollbacks inside and across the window (shorter and longer than n_swa) leave the state of the kept prefix: the
    re-decoded positions equal the full oracle of the final sequence;
  * a rollback deeper than the SWA cache retains does NOT: the needed keys are gone (llama.cpp's behaviour, which
    tests/test_mimo2_decode_parity.py Run_evict shows on the real graph), while a rollback inside the retained span does;
  * the cell model: one token per call fills cell p % size; the purge of KV:1160-1179 on a hand-made state; the base
    cache never reuses an occupied cell; seq_rm moves the head back; n_kv padding;
  * the MTP block over chunks equals the MTP block over the whole sequence;
  * the fused attn_qkv and the split projections give the same model; the value scale is linear; the sinks matter.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import mimo2 as M  # noqa: E402
from ref.mimo import attention as mimo_attention  # noqa: E402

TOL = 1e-10


def random_model(seed=0, fused=False, sinks="all", v_scale=0.0, mtp=False, n_swa=4):
    """A random mimo2 model: 4 trunk layers (global 0 and 3, SWA 1-2), dense layer 0, MoE 4 experts top-2, GQA with
    1 (global) / 2 (SWA) KV heads, K dim 8, V dim 6, RoPE on 4 dims; optionally an MTP layer 4 (SWA, fused qkv, dense)."""
    rng = np.random.default_rng(seed)
    E, H, Dk, Dv, F, X, Fx, V, L = 32, 4, 8, 6, 16, 4, 8, 20, 4
    swa = [False, True, True, False] + ([True] if mtp else [])
    hkv = [2 if s else 1 for s in swa]
    hp = dict(arch="mimo2", n_layer=L, n_nextn=1 if mtp else 0, n_all=L + (1 if mtp else 0), n_embd=E,
              n_head=[H] * len(swa), n_head_kv=hkv, k_dim=Dk, v_dim=Dv, rms_eps=1e-5, n_rot=4, rope_base=10000.0,
              rope_base_swa=5000.0, n_swa=n_swa, swa=swa, n_expert=X, n_expert_used=2, v_scale=v_scale)

    def r(*shape, s=None):
        return (s if s is not None else 1.0 / np.sqrt(shape[-1])) * rng.standard_normal(shape)

    W = {"token_embd.weight": r(V, E, s=1.0), "output_norm.weight": 1 + 0.05 * r(E, s=1.0), "output.weight": r(V, E)}
    for il in range(len(swa)):
        b = f"blk.{il}."
        q, k, v = r(H * Dk, E), r(hkv[il] * Dk, E), r(hkv[il] * Dv, E)
        if fused or il == L:
            W[b + "attn_qkv.weight"] = np.concatenate([q, k, v])
        else:
            W[b + "attn_q.weight"], W[b + "attn_k.weight"], W[b + "attn_v.weight"] = q, k, v
        W[b + "attn_output.weight"] = r(E, H * Dv)
        W[b + "attn_norm.weight"] = 1 + 0.05 * r(E, s=1.0)
        W[b + "ffn_norm.weight"] = 1 + 0.05 * r(E, s=1.0)
        sk = r(H, s=1.0)
        if sinks == "all" or swa[il]:
            W[b + "attn_sinks.weight"] = sk
        if il in (0, L):
            W[b + "ffn_gate.weight"], W[b + "ffn_up.weight"], W[b + "ffn_down.weight"] = r(F, E), r(F, E), r(E, F)
        else:
            W[b + "ffn_gate_inp.weight"], W[b + "exp_probs_b.bias"] = r(X, E), r(X, s=0.1)
            W[b + "ffn_gate_exps.weight"], W[b + "ffn_up_exps.weight"] = r(X, Fx, E), r(X, Fx, E)
            W[b + "ffn_down_exps.weight"] = r(X, E, Fx)
    if mtp:
        b = f"blk.{L}."
        W[b + "nextn.eh_proj.weight"] = r(E, 2 * E)
        W[b + "nextn.enorm.weight"] = 1 + 0.05 * r(E, s=1.0)
        W[b + "nextn.hnorm.weight"] = 1 + 0.05 * r(E, s=1.0)
        W[b + "layer_output_norm.weight"] = 1 + 0.05 * r(E, s=1.0)
    return W, hp


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


class Mimo2ChunkedRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = random_model(1, fused=True, sinks="swa", v_scale=0.75, mtp=True)
        cls.toks = list(np.random.default_rng(7).integers(3, 20, 40))
        cls.A = M.forward(cls.toks, cls.W, cls.hp)

    def rows(self, outs):
        return np.vstack([o["logits"] for o in outs]), np.concatenate([o["pos"] for o in outs])

    def test_attend_is_ref_mimo_attention(self):
        rng = np.random.default_rng(3)
        T, H, Hkv, Dk, Dv = 11, 4, 2, 8, 6
        q, k, v = rng.standard_normal((T, H, Dk)), rng.standard_normal((T, Hkv, Dk)), rng.standard_normal((T, Hkv, Dv))
        sinks = rng.standard_normal(H)
        for n_swa in (0, 3):
            want = mimo_attention(q, k, v, sinks, n_swa)
            got = M.attend(q, np.arange(T), k, v, np.arange(T), sinks, n_swa, 1 / np.sqrt(Dk))[2]
            self.assertLess(rel(got, want), 1e-12)
            no_sink = M.attend(q, np.arange(T), k, v, np.arange(T), None, n_swa, 1 / np.sqrt(Dk))[2]
            self.assertLess(rel(no_sink, mimo_attention(q, k, v, np.full(H, -1e300), n_swa)), 1e-12)
            self.assertGreater(rel(no_sink, want), 1e-3)
        # cell order does not matter, empty cells (-1) and stale data in them do not count
        perm = rng.permutation(T)
        kpos = np.concatenate([np.arange(T)[perm], [-1, -1]])
        K = np.concatenate([k[perm], rng.standard_normal((2, Hkv, Dk))])
        Vv = np.concatenate([v[perm], rng.standard_normal((2, Hkv, Dv))])
        got = M.attend(q, np.arange(T), K, Vv, kpos, sinks, 3, 1 / np.sqrt(Dk))[2]
        self.assertLess(rel(got, mimo_attention(q, k, v, sinks, 3)), 1e-12)

    def test_ubatches_and_decode_equal_the_full_sequence(self):
        """One call of 40 split into ubatches of 1..40, and a prefill + one-token decode, with llama.cpp's SWA cache
        size and with a 16-cell SWA cache that wraps after 16 positions."""
        T = len(self.toks)
        # llama.cpp's SWA cache has n_swa + n_ubatch cells at least (KVI:73); a 16-cell cache takes ubatches <= 12
        for swa_size, ubs in ((None, (1, 3, 4, 5, 7, 13, 40)), (16, (1, 3, 4, 5, 7, 12))):
            for ub in ubs:
                outs, cache = M.forward_calls(self.toks, self.W, self.hp, [T], n_ctx=64, n_ubatch=ub, swa_size=swa_size)
                lo, pos = self.rows(outs)
                self.assertEqual(list(pos), list(range(T)))
                self.assertLess(rel(lo, self.A["logits"]), TOL, f"ub {ub} swa_size {swa_size}")
            outs, cache = M.forward_calls(self.toks, self.W, self.hp, [10] + [1] * 30, n_ctx=64, n_ubatch=4,
                                          swa_size=swa_size)
            self.assertLess(rel(self.rows(outs)[0], self.A["logits"]), TOL)
            for il in range(self.hp["n_layer"]):
                lb = np.vstack([o["layers"][il]["l_out"] for o in outs])
                self.assertLess(rel(lb, self.A["layers"][il]["l_out"]), TOL)
            if swa_size:   # the ring wrapped: the last position is not in cell 39
                self.assertEqual(cache["swa"].size, 16)
                self.assertNotEqual(int(np.nonzero(cache["swa"].pos == T - 1)[0][0]), T - 1)

    def test_rollback_inside_and_across_the_window(self):
        """prefill 20, a 4-token draft whose last 2 are wrong, rollback 2 (< n_swa = 4), 6 x 1, a 9-token draft whose
        last 7 are wrong, rollback 7 (> n_swa), the rest one by one; 16-cell SWA cache (it wraps)."""
        t = self.toks
        wrong = [(x + 5) % 17 + 3 for x in t]
        fed = t[:20] + t[20:22] + wrong[22:24] + t[22:28] + t[28:30] + wrong[30:37] + t[30:]
        plan = [20, 4, "r2"] + [1] * 6 + [9, "r7"] + [1] * 10
        outs, cache = M.forward_calls(fed, self.W, self.hp, plan, n_ctx=64, n_ubatch=8, swa_size=16)
        self.assertEqual(cache["tokens"], t)
        kept = [(o, r) for o in outs for r in range(len(o["pos"]))]
        # rows written after the second rollback and before it on kept positions equal the full oracle
        final = {}
        for o, r in kept:
            final[int(o["pos"][r])] = o["logits"][r]
        self.assertEqual(sorted(final), list(range(len(t))))
        self.assertLess(rel(np.vstack([final[p] for p in range(len(t))]), self.A["logits"]), TOL)
        # the ubatches: 8, 8, 4 | draft 20..23 | 6 x 1 | draft 28..35, 36 | 10 x 1; the rows of the rejected tokens
        # (22, 23 of the first draft, 30..36 of the second) really differed from the final sequence's
        self.assertEqual([len(o["pos"]) for o in outs], [8, 8, 4, 4] + [1] * 6 + [8, 1] + [1] * 10)
        rej = [(outs[3], r) for r in (2, 3)] + [(outs[10], r) for r in range(2, 8)] + [(outs[11], 0)]
        self.assertEqual([int(o["pos"][r]) for o, r in rej], [22, 23] + list(range(30, 37)))
        for o, r in rej:
            self.assertGreater(rel(o["logits"][r], self.A["logits"][int(o["pos"][r])]), 1e-3)

    def test_deep_rollback_loses_the_window(self):
        """16-cell SWA cache, n_swa 4: after 30 positions it holds 14..29. A rollback to 22 (needs 19..21, retained)
        equals the full oracle; a rollback to 10 (needs 7..9, gone) does not: position 10 attends in the SWA layers to
        itself alone, and the difference reaches the logits."""
        t = self.toks
        ok, _ = M.forward_calls(t[:30] + t[22:30], self.W, self.hp, [30, "r8", 8], n_ctx=64, n_ubatch=8, swa_size=16)
        self.assertLess(rel(ok[-1]["logits"], self.A["logits"][22:30]), TOL)
        bad, cache = M.forward_calls(t[:30] + t[10:18], self.W, self.hp, [30, "r20", 8], n_ctx=64, n_ubatch=8,
                                     swa_size=16)
        il = 1
        p = bad[-1]["layers"][il]["attn"]["probs"]
        self.assertEqual(int((p[0, 0] > 0).sum()), 1)                 # position 10 sees one cell
        self.assertEqual(int((self.A["layers"][il]["attn"]["probs"][0, 10] > 0).sum()), 4)
        self.assertGreater(rel(bad[-1]["logits"], self.A["logits"][10:18]), 1e-3)

    def test_cell_model(self):
        c = M.Cells(8, 2)
        for p in range(21):                                            # one token per call: a ring
            idx = c.find_slot(1)
            c.apply(idx, [p])
            self.assertEqual(idx, [p % 8])
        self.assertEqual(sorted(c.pos.tolist()), list(range(13, 21)))
        # purge (KV:1160-1179): reusing the cell of position 5 while 3 is still cached removes 3 (and 4, absent)
        c = M.Cells(4, 2)
        c.pos[:] = [5, 3, 6, 7]
        c.head = 0
        idx = c.find_slot(1)
        self.assertEqual(idx, [0])
        c.apply(idx, [8])
        self.assertEqual(c.pos.tolist(), [8, -1, 6, 7])
        self.assertEqual(c.head, 1)
        # the base cache (no window) never reuses an occupied cell
        b = M.Cells(4, 0)
        b.apply(b.find_slot(4), [0, 1, 2, 3])
        with self.assertRaises(RuntimeError):
            b.find_slot(1)
        # seq_rm frees the positions >= p and moves the head back to the first freed cell
        b.seq_rm(2, -1)
        self.assertEqual((b.pos.tolist(), b.head), ([0, 1, -1, -1], 2))
        self.assertEqual(b.find_slot(2), [2, 3])
        # n_kv: padded to 256, at most the size
        self.assertEqual(M.Cells(512).n_kv(), 256)
        big = M.Cells(512)
        big.apply(big.find_slot(300), list(range(300)))
        self.assertEqual(big.n_kv(), 512)
        self.assertEqual(M.Cells(16, 4).n_kv(), 16)

    def test_mtp_chunks_equal_the_full_block(self):
        hp, W = self.hp, self.W
        h = self.A["h_nextn"]
        h_prev = np.vstack([np.zeros((1, hp["n_embd"])), h[:-1]])
        full = M.mtp_forward_chunk(self.toks, h_prev, W, hp)
        for ub, swa_size in ((5, None), (3, 16), (40, None)):
            cache = M.new_cache(hp, 64, ub, mtp=True, swa_size=swa_size)
            outs = [M.mtp_forward_chunk(self.toks[i:i + ub], h_prev[i:i + ub], W, hp, cache)
                    for i in range(0, len(self.toks), ub)]
            self.assertLess(rel(np.vstack([o["logits"] for o in outs]), full["logits"]), TOL)
            self.assertLess(rel(np.vstack([o["h_nextn"] for o in outs]), full["h_nextn"]), TOL)
        # the head norm is layer_output_norm when present
        self.assertLess(rel(full["head_norm"], M.rms_norm(full["post_ffn"], hp["rms_eps"],
                                                          W["blk.4.layer_output_norm.weight"])), 1e-14)

    def test_fused_split_value_scale_and_sinks(self):
        Wf, hpf = random_model(2, fused=True)
        Ws, hps = random_model(2, fused=False)
        self.assertLess(rel(M.forward(self.toks, Wf, hpf)["logits"], M.forward(self.toks, Ws, hps)["logits"]), 1e-12)
        W1, hp1 = random_model(2, v_scale=0.5)
        W0, hp0 = random_model(2)
        for il in range(hp0["n_layer"]):
            W0[f"blk.{il}.attn_output.weight"] = 0.5 * W0[f"blk.{il}.attn_output.weight"]
        self.assertLess(rel(M.forward(self.toks, W1, hp1)["logits"], M.forward(self.toks, W0, hp0)["logits"]), 1e-12)
        Wn, hpn = random_model(2, sinks="swa")
        self.assertGreater(rel(M.forward(self.toks, Wn, hpn)["logits"], M.forward(self.toks, Ws, hps)["logits"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
