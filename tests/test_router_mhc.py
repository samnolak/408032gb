"""Property tests for ref/router.py and ref/mhc.py."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref.mhc import hc_init, hc_mean, hc_post, hc_pre, sinkhorn  # noqa: E402
from ref.router import route, sigmoid, swiglu_clamp  # noqa: E402

R = np.random.default_rng(5)


class Router(unittest.TestCase):
    def test_bias_selects_but_does_not_weight(self):
        logits = R.standard_normal((6, 16))
        bias = np.zeros(16)
        bias[3] = 100.0                                  # forces expert 3 into every selection
        ids, w = route(logits, bias, k=4)
        self.assertTrue(np.all((ids == 3).any(axis=1)))
        p = sigmoid(logits)
        pos = np.argmax(ids == 3, axis=1)
        expect = p[:, 3] / np.take_along_axis(p, ids, 1).sum(1) * 2.5
        np.testing.assert_allclose(w[np.arange(6), pos], expect, rtol=1e-12)

    def test_weights_sum_to_scale(self):
        ids, w = route(R.standard_normal((5, 288)), R.standard_normal(288) * 0.1, k=8)
        np.testing.assert_allclose(w.sum(1), 2.5, rtol=1e-12)
        self.assertEqual(ids.shape, (5, 8))

    def test_swiglu_clamp_is_pre_activation(self):
        g = np.array([-20.0, 0.0, 5.0, 10.0, 30.0])
        u = np.array([1.0, 1.0, 50.0, -50.0, 1.0])
        out = swiglu_clamp(g, u, 10.0)
        silu10 = 10.0 * sigmoid(10.0)
        np.testing.assert_allclose(out[4], silu10)            # silu(min(30,10)) * 1, not 10
        np.testing.assert_allclose(out[2], 5.0 * sigmoid(5.0) * 10.0)
        np.testing.assert_allclose(out[3], -silu10 * 10.0)
        self.assertLess(out[4], 10.0)                         # the other variant would give 10.0


class MHC(unittest.TestCase):
    def test_sinkhorn_doubly_stochastic(self):
        c = sinkhorn(R.standard_normal((7, 4, 4)) * 3, iters=20, eps=1e-6)
        np.testing.assert_allclose(c.sum(axis=2), 1.0, atol=1e-5)   # last step: over src
        # over dst only approximately: 20 iterations do not fully converge on skewed inputs
        # (measured up to 1.8% at logit scale 3); kernels must reproduce this, not fix it
        np.testing.assert_allclose(c.sum(axis=1), 1.0, atol=0.05)
        self.assertTrue(np.all(c > 0))

    def test_identity_like_comb_and_zero_post_keeps_streams(self):
        T, hc, n = 3, 4, 16
        s = R.standard_normal((T, hc, n))
        comb = np.broadcast_to(np.eye(hc), (T, hc, hc))
        out = hc_post(np.zeros((T, n)), s, np.zeros((T, hc)), comb)
        np.testing.assert_allclose(out, s)

    def test_pre_shapes_and_ranges(self):
        T, hc, n = 2, 4, 32
        s = hc_init(R.standard_normal((T, n)), hc)
        mix, post, comb = hc_pre(s, R.standard_normal((24, hc * n)) * 0.1, np.array([1.0, 1.0, 1.0]),
                                 R.standard_normal(24) * 0.1)
        self.assertEqual(mix.shape, (T, n))
        self.assertTrue(np.all((post > 0) & (post < 2)))
        np.testing.assert_allclose(comb.sum(2), 1.0, atol=1e-5)
        np.testing.assert_allclose(hc_mean(s), s[:, 0])      # identical streams after init


if __name__ == "__main__":
    unittest.main()
