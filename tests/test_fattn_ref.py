"""Property tests of ref/fattn.py (N10): the f16 store, the CPU attention modes of llama.cpp and their precision
helpers. numpy only, no dumps (tests/test_qwen4exp_f16fa_parity.py and tests/test_glm5next_f16fa_parity.py compare
the same code with llama.cpp)."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref import fattn as F  # noqa: E402


def case(seed, T=5, H=4, Hkv=2, D=16, S=40, scale_v=1.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(T, H, D))
    k = rng.normal(size=(S, Hkv, D))
    v = rng.normal(size=(S, Hkv, D)) * scale_v
    cells = [np.sort(rng.choice(S, size=11, replace=False)) for _ in range(T)]
    return q, k, v, cells


class F16Rounding(unittest.TestCase):
    def test_round_to_nearest_even(self):
        one = 1.0
        self.assertEqual(F.f16(one + 2.0 ** -11), 1.0)                       # tie -> even (1.0)
        self.assertEqual(F.f16(one + 3 * 2.0 ** -11), 1.0 + 2.0 ** -9)       # tie -> even (1 + 2^-9)
        self.assertEqual(F.f16(one + 2.0 ** -11 + 2.0 ** -20), 1.0 + 2.0 ** -10)
        with np.errstate(over="ignore"):
            self.assertEqual(F.f16(65520.0), np.inf)                         # overflow to inf, like the hardware

    def test_idempotent_and_within_half_a_spacing(self):
        """For f32 values (what the graph stores) f16() is one correctly rounded step: within half a spacing. (From a
        float64 value it rounds twice, f64 -> f32 -> f16, like an f32 graph value of the oracle's float64 one.)"""
        x = np.random.default_rng(1).normal(size=10000) * 10.0 ** np.random.default_rng(2).uniform(-4, 3, 10000)
        x = F.f32(x)
        r = F.f16(x)
        self.assertTrue(np.array_equal(F.f16(r), r))
        self.assertTrue((np.abs(r - x) <= 0.5 * F.ulp16(r) * (1 + 1e-12)).all())
        self.assertTrue((np.abs(r - x) <= F.U16 * np.abs(x) * (1 + 1e-9) + 2.0 ** -25).all())

    def test_store_and_modes(self):
        self.assertIs(F.store(np.ones(3), None).dtype, np.dtype(np.float64))
        self.assertEqual(F.store(1.0 + 2.0 ** -12, F.precision("f16"))[()], 1.0)
        self.assertEqual(F.store(1.0 + 2.0 ** -12, F.precision("f32"))[()], 1.0 + 2.0 ** -12)
        self.assertEqual(F.mode_for(True, True, 63), "fa_chunk")
        self.assertEqual(F.mode_for(True, True, 64), "fa_tiled")
        self.assertEqual(F.mode_for(False, True, 1), "fa_chunk")
        self.assertEqual(F.mode_for(True, False, 64), "mm16")
        self.assertEqual(F.mode_for(False, False, 64), "exact")
        self.assertEqual(F.row_modes(None, [3, 4]), ["exact", "exact"])
        self.assertEqual(F.row_modes(F.precision("f16", True, {4: "exact"}), [3, 4]), ["fa_chunk", "exact"])


class AttentionModes(unittest.TestCase):
    def test_tiled_equals_exact_on_f16_values(self):
        """The tiled path rounds nothing but K and V (already f16 here): it equals the float64 attention."""
        q, k, v, cells = case(3)
        k, v = F.f16(k), F.f16(v)
        a = F.attend(q, k, v, cells, 0.25, "exact")
        b = F.attend(q, k, v, cells, 0.25, "fa_tiled")
        self.assertLess(np.abs(a - b).max(), 1e-12)

    def test_one_cell_is_the_rounded_value(self):
        """One visible cell: VKQ16 = f16(0 * 0 + v * 1) and S = 1, so one_chunk returns f16(v) exactly."""
        q, k, v, _ = case(4)
        cells = [np.array([7])] * q.shape[0]
        out = F.attend(q, k, v, cells, 0.25, "fa_chunk")
        want = np.repeat(F.f16(v[7])[None], q.shape[0], 0).repeat(2, axis=1)
        self.assertTrue(np.array_equal(out, want))

    def test_one_chunk_error_bound(self):
        """one_chunk against the float64 attention over the same f16 K/V: Q rounded once (relative error <= U16 per
        element of q, at most U16 * |q||k| * scale per score), the f16 accumulator rounded twice per cell (scale and
        mad), each step <= half a spacing of the running sum <= U16 * max|v|; bound the output by
        (2 n U16 + U16 sum_d |q_d k_d| scale * 2) * max|v| per element (first order)."""
        for seed in range(5):
            q, k, v, cells = case(seed, scale_v=3.0)
            k, v = F.f16(k), F.f16(v)
            a = F.attend(q, k, v, cells, 0.25, "exact")
            b = F.attend(q, k, v, cells, 0.25, "fa_chunk")
            n = max(len(c) for c in cells)
            qk = max(float(np.abs(q[t, h] * k[c, h // 2]).sum(-1).max()) for t in range(q.shape[0])
                     for h in range(q.shape[1]) for c in [cells[t]]) * 0.25
            bound = (2 * n * F.U16 + 4 * F.U16 * qk) * np.abs(v).max()
            self.assertLess(np.abs(a - b).max(), bound)
            self.assertGreater(np.abs(a - b).max(), 1e-6)          # the rounding is really there

    def test_one_chunk_depends_on_order_only_through_rounding(self):
        """The online softmax is order-free in exact arithmetic; one_chunk visits cells in cell order. Reversing the
        cell indices (and the rows of K, V with them) changes the result by at most the rounding bound."""
        q, k, v, cells = case(6)
        S = k.shape[0]
        rev = [np.sort(S - 1 - c) for c in cells]
        a = F.attend(q, k, v, cells, 0.25, "fa_chunk")
        b = F.attend(q, k[::-1], v[::-1], rev, 0.25, "fa_chunk")
        ex = F.attend(q, F.f16(k), F.f16(v), cells, 0.25, "exact")
        self.assertLess(np.abs(a - b).max(), 50 * F.U16 * np.abs(v).max())
        self.assertLess(np.abs(a - ex).max(), 50 * F.U16 * np.abs(v).max())

    def test_mm16_rounds_q_and_probabilities(self):
        q, k, v, cells = case(7)
        k, v = F.f16(k), F.f16(v)
        got = F.attend(q, k, v, cells, 0.25, "mm16")
        t, h = 2, 3
        kc, vc = k[cells[t], h // 2], v[cells[t], h // 2]
        sc = F.f16(q[t, h]) @ kc.T * 0.25
        p = np.exp(sc - sc.max())
        p = F.f16(p / p.sum())
        self.assertLess(abs(got[t, h] - p @ vc).max(), 1e-12)

    def test_gqa_head_mapping(self):
        """Query head h reads kv head h // (H / Hkv): moving kv head 1 changes only query heads 2 and 3."""
        q, k, v, cells = case(8)
        a = F.attend(q, k, v, cells, 0.25, "fa_chunk")
        k2 = k.copy()
        k2[:, 1] += 1.0
        b = F.attend(q, k2, v, cells, 0.25, "fa_chunk")
        self.assertTrue(np.array_equal(a[:, :2], b[:, :2]))
        self.assertFalse(np.array_equal(a[:, 2:], b[:, 2:]))

    def test_attend_rows_groups_by_mode(self):
        q, k, v, cells = case(9)
        modes = ["exact", "fa_chunk", "exact", "fa_tiled", "fa_chunk"]
        got = F.attend_rows(q, k, v, cells, 0.25, modes)
        for r, m in enumerate(modes):
            self.assertTrue(np.array_equal(got[r], F.attend(q[r:r + 1], k, v, cells[r:r + 1], 0.25, m)[0]))


if __name__ == "__main__":
    unittest.main()
