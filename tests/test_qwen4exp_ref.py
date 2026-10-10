"""Property tests of ref/qwen4exp.py (numpy only, no llama.cpp needed).

Reductions to known cases and algebraic identities; the dump parity against the real graph is
tests/test_qwen4exp_tiny_parity.py.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import qwen4exp as Q  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402
from ref.mimo import rope_neox  # noqa: E402


class HyperConnections(unittest.TestCase):
    def test_zero_inject_is_plain_residual_add(self):
        # G:378: 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
        rng = np.random.default_rng(0)
        res, blk = rng.standard_normal((5, 4, 8)), rng.standard_normal((5, 8))
        out = Q.hc_combine(res, blk, np.zeros((5, 4)))
        self.assertTrue(np.allclose(out, res + blk[:, None, :]))

    def test_mixer_of_identical_streams_with_zero_gate(self):
        # identical streams, equal gammas, zero low-rank weights: gate = 0, sigmoid = 1/2, and the stream mean
        # of identical streams is the stream itself: mixed = rms_norm(x) * gamma / 2
        rng = np.random.default_rng(1)
        x = rng.standard_normal((3, 8))
        gamma = np.tile(rng.standard_normal(8), (4, 1))
        m = Q.hc_mix(Q.hc_init(x, 4), gamma, np.zeros((2, 32)), np.zeros((32, 2)), None, 1e-6)
        self.assertTrue(np.allclose(m["mixed"], 0.5 * Q.rms_norm(x, 1e-6) * gamma[0]))


class Rope(unittest.TestCase):
    def test_imrope_equals_neox_when_sections_cover_all_pairs(self):
        # text positions (p, p, p, 0) and every pair in t/h/w: IMRoPE reduces to NEOX RoPE at p
        rng = np.random.default_rng(2)
        x = rng.standard_normal((9, 3, 16))
        sec = [3, 3, 2, 0]                                  # 8 pairs, like [11, 11, 10, 0] for 32 pairs
        self.assertNotIn(3, Q.imrope_sector_map(16, sec).tolist())
        got = Q.imrope(x, Q.text_pos4(9), 16, sec, 1000.0)
        self.assertTrue(np.allclose(got, rope_neox(x, np.arange(9), 16, 1000.0)))

    def test_fourth_section_uses_the_fourth_position(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((6, 1, 8))
        sec = [1, 1, 1, 1]
        self.assertEqual(Q.imrope_sector_map(8, sec).tolist(), [0, 1, 2, 3])
        a = Q.imrope(x, Q.text_pos4(6), 8, sec, 100.0)                    # pair 3 rotates by 0
        self.assertTrue(np.allclose(a[..., [3, 7]], x[..., [3, 7]]))
        p = np.arange(6)
        b = Q.imrope(x, np.stack([p, p, p, p], 1), 8, sec, 100.0)        # pooled-key positions rotate it
        self.assertFalse(np.allclose(b[1:, :, [3, 7]], x[1:, :, [3, 7]]))
        self.assertTrue(np.allclose(a[..., [0, 1, 2, 4, 5, 6]], b[..., [0, 1, 2, 4, 5, 6]]))

    def test_rotation_preserves_pair_norms(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((7, 2, 12))
        y = Q.imrope(x, Q.text_pos4(7), 8, [2, 1, 1, 0], 10000.0)
        nx = x[..., :4] ** 2 + x[..., 4:8] ** 2
        ny = y[..., :4] ** 2 + y[..., 4:8] ** 2
        self.assertTrue(np.allclose(nx, ny))
        self.assertTrue(np.array_equal(x[..., 8:], y[..., 8:]))          # partial rotary: the rest passes


class Qsa(unittest.TestCase):
    def test_budget_covering_every_pool_is_causal_attention(self):
        rng = np.random.default_rng(5)
        T, kp = 23, 4
        scores = Q.qsa_scores(rng.standard_normal((T, 2, 8)), rng.standard_normal((T // kp, 8)), kp)
        cells = Q.qsa_select(scores, top_k=kp * 64, kpool=kp)
        self.assertEqual(cells, [list(range(t + 1)) for t in range(T)])
        q, k, v = rng.standard_normal((T, 4, 8)), rng.standard_normal((T, 2, 8)), rng.standard_normal((T, 2, 8))
        self.assertTrue(np.allclose(Q.attention(q, k, v, cells, 0.3), Q.attention(q, k, v, None, 0.3)))

    def test_selection_is_top_pools_plus_tail(self):
        T, kp = 16, 4
        sc = np.full((T, 4), -np.inf)
        for t in range(T):
            nv = (t + 1) // kp
            sc[t, :nv] = np.arange(nv)[::-1]                # earlier pools score higher
        cells = Q.qsa_select(sc, top_k=4, kpool=kp)         # one pool per token
        self.assertEqual(cells[2], [0, 1, 2])               # no complete pool: tail only
        self.assertEqual(cells[9], [0, 1, 2, 3, 8, 9])      # pool 0 + tail 8..9
        self.assertEqual(cells[15], [0, 1, 2, 3])           # 4 complete pools, no tail

    def test_tie_variant_accepts_only_genuine_ties(self):
        t, kp = 13, 4                                       # 3 complete pools, tail 12..13, budget 1 pool
        row = np.array([0.5, 0.5, 0.2, -np.inf])
        ok = lambda cells: Q.qsa_tie_variant(row, cells, t, 4, kp)[0]  # noqa: E731
        self.assertTrue(ok([0, 1, 2, 3, 12, 13]))           # the oracle's own choice
        self.assertTrue(ok([4, 5, 6, 7, 12, 13]))           # the other member of the exact tie
        self.assertFalse(ok([8, 9, 10, 11, 12, 13]))        # a lower-scoring pool
        self.assertFalse(ok([0, 1, 2, 3, 13]))              # tail cell 12 missing
        self.assertFalse(ok([0, 1, 2, 3, 4, 12, 13]))       # a partial pool
        self.assertFalse(ok([0, 1, 2, 3, 4, 5, 6, 7, 12, 13]))   # over budget

    def test_pool_is_mean_of_raw_keys(self):
        rng = np.random.default_rng(6)
        k_raw = rng.standard_normal((9, 8))
        pooled = Q.qsa_pooled_keys(k_raw, np.ones(8), 4, 1e-6, n_rot=4, sections=[1, 1, 0, 0], base=10.0)
        self.assertEqual(pooled.shape, (2, 8))
        # pool 0 starts at position 0: no rotation, so it is the normalised mean
        self.assertTrue(np.allclose(pooled[0], Q.rms_norm(k_raw[:4].mean(0), 1e-6)))


class Gdn(unittest.TestCase):
    def test_value_head_reads_key_head_mod_hk(self):
        # Hk < Hv: the result must equal the same scan with q/k tiled to Hv heads (ggml_repeat_4d, G:1116-1120)
        rng = np.random.default_rng(7)
        T, Hk, Hv, S = 6, 2, 4, 8
        q, k = rng.standard_normal((T, Hk, S)), rng.standard_normal((T, Hk, S))
        v = rng.standard_normal((T, Hv, S))
        g, beta = -rng.random((T, Hv)), rng.random((T, Hv))
        a, _ = gated_delta_rule(q, k, v, g, beta)
        b, _ = gated_delta_rule(np.tile(q, (1, 2, 1)), np.tile(k, (1, 2, 1)), v, g, beta)
        self.assertTrue(np.allclose(a, b))

    def test_short_conv_last_tap_is_identity(self):
        x = np.random.default_rng(8).standard_normal((5, 3))
        w = np.zeros((3, 4))
        w[:, 3] = 1.0
        self.assertTrue(np.allclose(Q.short_conv(x, w), x))


class Moe(unittest.TestCase):
    def test_softmax_route(self):
        rng = np.random.default_rng(9)
        logits = rng.standard_normal((10, 8))
        probs, ids, w = Q.route_softmax(logits, 3)
        self.assertTrue(np.allclose(w.sum(1), 1.0))
        self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(np.argsort(-logits, 1)[:, :3], 1)))
        # renormalised softmax over the chosen experts = softmax of their logits
        self.assertTrue(np.allclose(w, Q.softmax(np.take_along_axis(logits, ids, 1))))


class Ple(unittest.TestCase):
    M = [0x1A2B3C4D5E1, 0x0F1E2D3C4B5, 0x13579BDF023]

    def test_hash_matches_uint64_arithmetic(self):
        toks = [5, 9, 17, 3]
        rows = Q.ple_rows(toks, self.M, [0, 7], [7, 11], ngram=3, heads_per_ngram=1, eos=2)
        m = np.array(self.M, dtype=np.uint64)
        t = np.array(toks, dtype=np.uint64)
        bi = (t[3] * m[0]) ^ (t[2] * m[1])
        tri = bi ^ (t[1] * m[2])
        self.assertEqual(rows[3].tolist(), [int(bi % np.uint64(7)), int(tri % np.uint64(11)) + 7])

    def test_hash_wraps_like_uint64(self):
        # products above 2^64 wrap (llama.cpp multiplies uint64_t, G:1269-1272); the tiny dump never wraps
        big = [0xFFFFFFFFFFFFFFF1, 0xF0F0F0F0F0F0F0F1, 0x9E3779B97F4A7C15]
        toks = [95, 77, 63]
        rows = Q.ple_rows(toks, big, [0, 1000003], [1000003, 999983], ngram=3, heads_per_ngram=1, eos=2)
        m = np.array(big, dtype=np.uint64)
        t = np.array(toks[::-1], dtype=np.uint64)          # ctx[0] = newest token
        bi = np.bitwise_xor.reduce(t[:2] * m[:2])            # array products wrap silently mod 2^64
        tri = np.bitwise_xor.reduce(t * m)
        self.assertGreater(int(toks[2]) * big[0], 2 ** 64)
        self.assertEqual(rows[2].tolist(), [int(bi % np.uint64(1000003)), int(tri % np.uint64(999983)) + 1000003])

    def test_eos_and_sequence_start_reset_the_window(self):
        kw = dict(multipliers=self.M, offsets=[0, 50], vocab_sizes=[50, 53], ngram=3, heads_per_ngram=1, eos=2)
        # after an EOS every older predecessor reads as EOS: same rows whatever came before it
        a = Q.ple_rows([40, 41, 2, 7], **kw)
        b = Q.ple_rows([11, 12, 2, 7], **kw)
        self.assertTrue(np.array_equal(a[3], b[3]))
        # the first token sees EOS for its missing predecessors, like a token right after an EOS
        c = Q.ple_rows([7], **kw)
        d = Q.ple_rows([2, 7], **kw)
        self.assertTrue(np.array_equal(c[0], d[1]))
        # the token's own EOS does not cut its own context
        e = Q.ple_rows([30, 31, 2], **kw)
        f = Q.ple_rows([33, 31, 2], **kw)
        self.assertFalse(np.array_equal(e[2], f[2]))

    def test_dilated_conv_taps(self):
        x = np.arange(12, dtype=float).reshape(12, 1)
        w = np.array([[1.0, 0.0, 0.0]])                    # oldest tap only: x[t - 2*dil]
        y = Q.dilated_causal_dwconv(x, w, dilation=3)
        self.assertTrue(np.array_equal(y[:6, 0], np.zeros(6)))
        self.assertTrue(np.array_equal(y[6:, 0], x[:6, 0]))


if __name__ == "__main__":
    unittest.main()
