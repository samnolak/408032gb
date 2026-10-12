"""Property tests of ref/quant.py (N10, gap 2): the activation quantization of ggml's CPU matrix products and the
QWeight wrapper. numpy only, no dumps (tests/test_qwen4exp_quant_parity.py compares the same code with llama.cpp)."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref import quant as QT  # noqa: E402


def rows(seed, n=6, k=256, spread=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    if spread:
        x *= 10.0 ** rng.uniform(-3, 2, size=(n, 1))
    return x.astype(np.float32)


class ActivationQuantization(unittest.TestCase):
    def test_q8_0_block_rule(self):
        """AVX2 quantize_row_q8_0: per 32-block d = f16(amax / 127), q = rint(x * 127 / amax); the element of largest
        |x| maps to +-127, every value is within half a step (plus the f16 rounding of d) of x."""
        x = rows(1)
        q, d = QT.q8_0(x)
        b = x.reshape(x.shape[0], -1, 32).astype(np.float64)
        amax = np.abs(b).max(-1, keepdims=True)
        self.assertTrue((np.abs(q).max(-1) == 127).all())
        self.assertTrue((np.abs(q) <= 127).all())
        self.assertTrue(np.array_equal(d.astype(np.float64), (amax / 127).astype(np.float32).astype(np.float16)))
        v = q * d.astype(np.float64)
        step = amax / 127
        slack = 127 * step * 2.0 ** -22                   # id = 127 / amax and the product x * id are f32
        self.assertTrue((np.abs(v - b) <= 0.5 * step + 127 * np.abs(d.astype(np.float64) - step) + slack).all())

    def test_q8_0_rounds_half_to_even(self):
        """x * id exactly on a half step: the x86 path rounds to even (_MM_ROUND_NEAREST), the scalar reference
        quantize_row_q8_0_ref would round half away from zero."""
        x = np.zeros((1, 32), dtype=np.float32)
        x[0, 0] = 127.0                    # amax 127: id = 1, d = 1
        x[0, 1:5] = [0.5, 1.5, 2.5, -2.5]
        q, d = QT.q8_0(x)
        self.assertEqual(float(d[0, 0, 0]), 1.0)
        self.assertEqual(q[0, 0, 1:5].tolist(), [0.0, 2.0, 2.0, -2.0])

    def test_q8_K_block_rule(self):
        """quantize_row_q8_K_ref: per 256-block iscale = -127 / max, q = nearest_int(iscale * x), d = 1 / iscale; in the
        sign convention of ref/quant.py the largest |x| maps to +-127 and d * q is within half a step of x."""
        x = rows(2, k=512)
        q, d = QT.q8_K(x)
        b = x.reshape(x.shape[0], -1, 256).astype(np.float64)
        amax = np.abs(b).max(-1, keepdims=True)
        self.assertTrue((np.abs(q).max(-1) == 127).all())
        v = q * d.astype(np.float64)
        self.assertTrue((np.abs(v - b) <= (0.5 + 127 * 2.0 ** -22) * amax / 127).all())   # d = 1 / s in f32
        # the reference's own arithmetic (negative iscale for a positive max) gives the same values
        for r in range(x.shape[0]):
            for blk in range(2):
                xb = x[r, blk * 256:(blk + 1) * 256]
                mx = xb[int(np.argmax(np.abs(xb)))]
                iscale = np.float32(-127.0) / np.float32(mx)
                qr = np.minimum(np.rint((iscale * xb).astype(np.float32)), 127)
                dr = np.float32(1.0) / iscale
                self.assertTrue(np.array_equal(qr.astype(np.float64) * np.float64(dr), v[r, blk]))

    def test_zero_block(self):
        x = np.zeros((2, 256), dtype=np.float32)
        x[1, :32] = 1.0
        for vdt in ("Q8_0", "Q8_K"):
            v = QT.act_quant(x, vdt)
            self.assertTrue(np.isfinite(v).all())
            self.assertTrue((v[0] == 0).all())

    def test_f16_and_f32(self):
        x = np.array([[1.0 + 2.0 ** -12, 3.0]])
        self.assertEqual(QT.act_quant(x, "F16")[0, 0], 1.0)
        self.assertEqual(QT.act_quant(x, None)[0, 0], 1.0 + 2.0 ** -12)
        with self.assertRaises(NotImplementedError):
            QT.act_quant(np.ones((1, 32)), "Q8_1")

    def test_block_size_rule(self):
        with self.assertRaises(ValueError):
            QT.q8_K(np.ones((1, 128), dtype=np.float32))

    def test_integer_dot_identity(self):
        """llama.cpp's Q8_0 x Q8_0 dot is sum_b d_w d_a sum(q_w q_a); the oracle's dequantized-row product is the same
        number up to float64 rounding."""
        x = rows(3, n=1, k=256)[0]
        w = rows(4, n=1, k=256)[0]
        qw, dw = QT.q8_0(w)
        qa, da = QT.q8_0(x)
        exact = sum(float(dw[b, 0]) * float(da[b, 0]) * int(np.dot(qw[b].astype(np.int64), qa[b].astype(np.int64)))
                    for b in range(8))
        deq = (qw * dw.astype(np.float64)).reshape(-1) @ QT.act_quant(x, "Q8_0")
        self.assertLess(abs(deq - exact), 1e-12 * abs(exact) + 1e-300)


class Wrapper(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.deq = rng.normal(size=(16, 256))
        self.x = rows(6, n=3)

    def test_rmatmul_quantizes_the_activations(self):
        w = QT.QWeight(self.deq, "Q4_K")
        got = self.x @ w.T
        self.assertTrue(np.array_equal(got, QT.act_quant(self.x, "Q8_K") @ self.deq.T))
        self.assertFalse(np.allclose(got, self.x.astype(np.float64) @ self.deq.T, rtol=1e-6, atol=0))
        w0 = QT.QWeight(self.deq, "Q4_K", act=False)
        self.assertTrue(np.array_equal(self.x @ w0.T, self.x.astype(np.float64) @ self.deq.T))
        w8 = QT.QWeight(self.deq, "Q8_0")
        self.assertTrue(np.array_equal(self.x @ w8.T, QT.act_quant(self.x, "Q8_0") @ self.deq.T))
        x3 = self.x.reshape(3, 1, 256)
        self.assertTrue(np.allclose(x3 @ w8.T, (self.x @ w8.T)[:, None, :], rtol=1e-12, atol=0))   # leading dims

    def test_matmul_one_row_and_rows_and_experts(self):
        w = QT.QWeight(self.deq, "Q6_K")
        self.assertTrue(np.array_equal(w @ self.x[0], self.deq @ QT.act_quant(self.x[0], "Q8_K")))
        with self.assertRaises(ValueError):
            w @ self.x
        self.assertTrue(np.array_equal(w[[3, 1]], self.deq[[3, 1]]))           # get_rows: dequantized values
        self.assertTrue(np.array_equal(np.asarray(w), self.deq))
        e = QT.QWeight(self.deq.reshape(4, 4, 256), "Q8_0")[2]
        self.assertIsInstance(e, QT.QWeight)
        self.assertTrue(np.array_equal(e @ self.x[1], self.deq[8:12] @ QT.act_quant(self.x[1], "Q8_0")))

    def test_ndarray_never_silently_multiplies_a_qweight(self):
        w = QT.QWeight(self.deq, "Q8_0")
        with self.assertRaises(TypeError):
            np.multiply(self.x[:, :16], w)
        with self.assertRaises(TypeError):
            self.x @ w                     # needs w.T: the products of the oracle are x @ W.T

    def test_f16_weight(self):
        w = QT.QWeight(self.deq.astype(np.float16).astype(np.float64), "F16")
        self.assertTrue(np.array_equal(self.x @ w.T, QT.act_quant(self.x, "F16") @ np.asarray(w).T))
        with self.assertRaises(NotImplementedError):
            QT.QWeight(self.deq, "Q4_1") @ self.x[0]


if __name__ == "__main__":
    unittest.main()
