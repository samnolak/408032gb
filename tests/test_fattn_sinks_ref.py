"""Property tests of the N17 additions to ref/fattn.py: attention sinks in every CPU attention mode, K and V heads of
different sizes, the f32 attention scale and the f32 argument of expf in one_chunk. numpy only, no dumps
(tests/test_mimo2_f16fa_parity.py and tests/test_mimo2_mtp_heads_parity.py compare the same code with llama.cpp).
"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref import fattn as F  # noqa: E402


def case(seed, T=5, H=4, Hkv=2, DK=24, DV=16, S=40, n_vis=11, scale_v=1.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(T, H, DK))
    k = rng.normal(size=(S, Hkv, DK))
    v = rng.normal(size=(S, Hkv, DV)) * scale_v
    cells = [np.sort(rng.choice(S, size=n_vis, replace=False)) for _ in range(T)]
    sinks = rng.normal(size=H) * 2.0
    return q, k, v, cells, sinks


def dot_f16_transcribed(x, y):
    """ggml_vec_dot_f16 (ggml/src/ggml-cpu/vec.cpp:345-377 @ec7630a) on x86 AVX2, element by element: below 32
    elements the scalar loop in double; from 32 on 4 accumulators of 8 f32 lanes, lane l of accumulator j adding
    x[i + 8j + l] * y[i + 8j + l] with one rounding (FMA), reduced as GGML_F32x8_REDUCE (simd-mappings.h:602-620),
    the leftovers in double; returns the f32 *s."""
    f32 = np.float32
    n = len(x)
    n_simd = n & ~31
    sumf = 0.0
    if n_simd:
        acc = [[f32(0.0)] * 8 for _ in range(4)]
        for i in range(0, n_simd, 32):
            for j in range(4):
                for lane in range(8):
                    e = i + 8 * j + lane
                    acc[j][lane] = f32(float(acc[j][lane]) + float(x[e]) * float(y[e]))
        a0 = [f32(acc[0][l] + acc[2][l]) for l in range(8)]
        a1 = [f32(acc[1][l] + acc[3][l]) for l in range(8)]
        a0 = [f32(a0[l] + a1[l]) for l in range(8)]
        t0 = [f32(a0[l] + a0[l + 4]) for l in range(4)]                  # low half + high half
        sumf = float(f32(f32(t0[0] + t0[1]) + f32(t0[2] + t0[3])))       # hadd, hadd: lane 0
    for e in range(n_simd, n):
        sumf += float(x[e]) * float(y[e])
    return f32(sumf)


def one_chunk_transcribed(q, k, v, scale, sink):
    """ggml_compute_forward_flash_attn_ext_f16_one_chunk for one query row of one head, statement by statement in f32
    (ggml/src/ggml-cpu/ops.cpp:8961-9068 @ec7630a): q (DK,), k (n, DK) and v (n, DV) f16 values in cell order, scale
    and sink f32 (sink None: no sinks tensor). exp: correctly rounded expf of the f32 argument."""
    f32, f16 = np.float32, F.f16
    expf = lambda x: f32(np.exp(np.float64(f32(x))))                    # noqa: E731
    qq = f16(q)                                                           # q_to_vec_dot (O:8961)
    M, S = f32(-np.inf), f32(0.0)
    acc = np.zeros(v.shape[1])                                            # VKQ16
    for i in range(k.shape[0]):
        s = f32(dot_f16_transcribed(qq, k[i]) * f32(scale))              # O:8976-8978
        if s > M:                                                         # O:8994-9000
            ms = expf(M - s) if M != -np.inf else f32(0.0)
            vs = f32(1.0)
            M = s
            acc = f16(acc * ms)                                           # ggml_vec_scale_f16
        else:
            ms, vs = f32(1.0), expf(s - M)                                # O:9003
        acc = f16(acc + v[i] * vs)                                        # ggml_vec_mad_f16 (O:9007)
        S = f32(f32(S * ms) + vs)                                         # O:9031
    acc = F.f32(acc)                                                      # O:9034-9038
    if sink is not None:                                                  # O:9041-9056
        s = f32(sink)
        if s > M:
            ms = expf(M - s) if M != -np.inf else f32(0.0)
            acc = F.f32(acc * ms)
            S = f32(f32(S * ms) + f32(1.0))
        else:
            S = f32(S + expf(s - M))
    inv = f32(0.0) if S == 0 else f32(f32(1.0) / S)                       # O:9067
    return F.f32(acc * inv)                                               # O:9068


class SinkModes(unittest.TestCase):
    def test_no_sink_and_minus_infinity_sink_are_the_same(self):
        """sinks=None keeps every mode as before; a sink of -inf adds exp(-inf) = 0 to the sum (bit for bit)."""
        q, k, v, cells, _ = case(1)
        for mode in F.MODES:
            a = F.attend(q, k, v, cells, 0.25, mode)
            b = F.attend(q, k, v, cells, 0.25, mode, np.full(q.shape[1], -np.inf))
            self.assertTrue(np.array_equal(a, b), mode)

    def test_sink_is_one_more_logit_with_a_zero_value_row(self):
        """exact mode: the sink joins the denominator only, i.e. a phantom cell whose score is the sink and whose V
        row is 0 (ggml_soft_max_add_sinks: max over the row and the sink, sum += exp(sink - max), O:5896-5912)."""
        q, k, v, cells, sinks = case(2, Hkv=4)
        got = F.attend(q, k, v, cells, 0.25, "exact", sinks)
        T, H, DK = q.shape
        for t in range(T):
            for h in range(H):
                sc = np.append(q[t, h] @ k[cells[t], h].T * 0.25, sinks[h])
                p = np.exp(sc - sc.max())
                p /= p.sum()
                want = p[:-1] @ v[cells[t], h]
                self.assertLess(np.abs(got[t, h] - want).max(), 1e-12)

    def test_one_chunk_is_the_transcribed_kernel(self):
        """fa_chunk with sinks equals the statement-by-statement f32 transcription of one_chunk bit for bit: long rows
        (200 cells, so the running maximum moves and the f32 rounding of the exp argument matters), sinks below and
        above every score, K 24 / V 16 (mimo2's tiny heads: the scalar dot) and K 192 / V 128 (MiMo-V2.6-Flash: the
        AVX2 lanes of the dot), and K 40 (32 in lanes, 8 left over)."""
        for seed, (DK, DV), sink_scale in ((3, (24, 16), 1.0), (4, (24, 16), 40.0), (5, (192, 128), 1.0),
                                           (12, (40, 16), 1.0)):
            q, k, v, cells, sinks = case(seed, T=2, H=2, Hkv=1, DK=DK, DV=DV, S=220, n_vis=200)
            sinks = sinks * sink_scale
            k, v = F.f16(k), F.f16(v)
            scale = F.kq_scale_f32(DK)
            got = F.attend(q, k, v, cells, scale, "fa_chunk", sinks)
            self.assertEqual(got.shape, (2, 2, DV))
            for t in range(2):
                for h in range(2):
                    want = one_chunk_transcribed(q[t, h], k[cells[t], 0], v[cells[t], 0], scale, sinks[h])
                    self.assertTrue(np.array_equal(got[t, h], want), (seed, t, h, np.abs(got[t, h] - want).max()))

    def test_one_chunk_with_sink_against_the_float64_softmax(self):
        """fa_chunk with sinks against the float64 attention with sinks over the same f16 K / V: within the
        first-order bound of tests/test_fattn_ref.py (Q rounded once, the f16 accumulator twice per cell), and the
        rounding is really there; a sink above every score scales the f32 accumulator (not f16)."""
        for seed, sink_scale in ((6, 1.0), (7, 50.0)):
            q, k, v, cells, sinks = case(seed, scale_v=3.0)
            sinks = sinks * sink_scale
            k, v = F.f16(k), F.f16(v)
            a = F.attend(q, k, v, cells, 0.25, "exact", sinks)
            b = F.attend(q, k, v, cells, 0.25, "fa_chunk", sinks)
            n = max(len(c) for c in cells)
            qk = max(float(np.abs(q[t, h] * k[c, h // 2]).sum(-1).max()) for t in range(q.shape[0])
                     for h in range(q.shape[1]) for c in [cells[t]]) * 0.25
            bound = (2 * n * F.U16 + 4 * F.U16 * qk) * np.abs(v).max()
            self.assertLess(np.abs(a - b).max(), bound)
            self.assertGreater(np.abs(a - b).max(), 1e-9)

    def test_sink_changes_every_mode(self):
        """With finite sinks every mode differs from the same mode without them."""
        q, k, v, cells, sinks = case(8)
        for mode in F.MODES:
            self.assertGreater(np.abs(F.attend(q, k, v, cells, 0.25, mode, sinks) -
                                      F.attend(q, k, v, cells, 0.25, mode)).max(), 1e-3, mode)

    def test_tiled_with_sinks_is_exact_on_f16_values(self):
        q, k, v, cells, sinks = case(9)
        k, v = F.f16(k), F.f16(v)
        a = F.attend(q, k, v, cells, 0.25, "exact", sinks)
        b = F.attend(q, k, v, cells, 0.25, "fa_tiled", sinks)
        self.assertLess(np.abs(a - b).max(), 1e-12)

    def test_mm16_with_sinks_rounds_the_probabilities(self):
        q, k, v, cells, sinks = case(10)
        k, v = F.f16(k), F.f16(v)
        got = F.attend(q, k, v, cells, 0.25, "mm16", sinks)
        t, h = 3, 1
        sc = F.f16(q[t, h]) @ k[cells[t], h // 2].T * 0.25
        m = max(sc.max(), sinks[h])
        p = np.exp(sc - m) / (np.exp(sc - m).sum() + np.exp(sinks[h] - m))
        self.assertLess(np.abs(got[t, h] - F.f16(p) @ v[cells[t], h // 2]).max(), 1e-12)
        bound = F.mm16_bound(q, k, v, cells, 0.25, sinks)
        ex = F.attend(F.f16(q), k, v, cells, 0.25, "exact", sinks)
        self.assertTrue((np.abs(got - ex) <= bound).all())

    def test_gqa_and_head_sizes(self):
        """Query head h reads kv head h // (H / Hkv) and uses sink h; DV sets the output width."""
        q, k, v, cells, sinks = case(11, H=4, Hkv=2, DK=24, DV=16)
        a = F.attend(q, k, v, cells, 0.25, "fa_chunk", sinks)
        self.assertEqual(a.shape, (5, 4, 16))
        s2 = sinks.copy()
        s2[3] += 1.0
        b = F.attend(q, k, v, cells, 0.25, "fa_chunk", s2)
        self.assertTrue(np.array_equal(a[:, :3], b[:, :3]))
        self.assertFalse(np.array_equal(a[:, 3], b[:, 3]))
        rows = F.attend_rows(q, k, v, cells, 0.25, ["fa_chunk", "exact", "fa_tiled", "fa_chunk", "mm16"], sinks)
        for r, m in enumerate(["fa_chunk", "exact", "fa_tiled", "fa_chunk", "mm16"]):
            self.assertTrue(np.array_equal(rows[r], F.attend(q[r:r + 1], k, v, cells[r:r + 1], 0.25, m, sinks)[0]))

    def test_kq_scale_f32(self):
        """1.0f/sqrtf(d) in f32 (the graphs' kq_scale): for d = 24 one f32 step below float32(1/sqrt(24))."""
        self.assertEqual(F.kq_scale_f32(24), float(np.float32(0.20412414)))
        self.assertNotEqual(F.kq_scale_f32(24), float(np.float32(1 / np.sqrt(24))))
        for d in (16, 128, 192):
            self.assertEqual(F.kq_scale_f32(d), float(np.float32(1 / np.sqrt(d))))


if __name__ == "__main__":
    unittest.main()
