"""Property tests for ref/kda.py (the KDA / GDN oracle). numpy + stdlib only."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref.kda import gated_delta_rule, kda_gate, l2_norm  # noqa: E402

RNG = np.random.default_rng(20261001)


def rand_inputs(T=7, H_k=2, H_v=4, S=8, kda=True):
    q = RNG.standard_normal((T, H_k, S))
    k = RNG.standard_normal((T, H_k, S))
    v = RNG.standard_normal((T, H_v, S))
    g = -RNG.uniform(0.01, 2.0, (T, H_v, S) if kda else (T, H_v))
    beta = 1.0 / (1.0 + np.exp(-RNG.standard_normal((T, H_v))))
    return q, k, v, g, beta


class KDAOracle(unittest.TestCase):
    def test_constant_channel_gate_equals_scalar_gdn(self):
        q, k, v, g, beta = rand_inputs(kda=False)
        g_kda = np.repeat(g[:, :, None], q.shape[-1], axis=2)
        o1, s1 = gated_delta_rule(q, k, v, g, beta)
        o2, s2 = gated_delta_rule(q, k, v, g_kda, beta)
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s1, s2, rtol=1e-12, atol=1e-12)

    def test_state_continuity(self):
        # decode after prefill must equal one long pass: what chunked prefill and MTP rely on
        q, k, v, g, beta = rand_inputs(T=10)
        o_all, s_all = gated_delta_rule(q, k, v, g, beta)
        o_a, s_a = gated_delta_rule(q[:6], k[:6], v[:6], g[:6], beta[:6])
        o_b, s_b = gated_delta_rule(q[6:], k[6:], v[6:], g[6:], beta[6:], state=s_a)
        np.testing.assert_allclose(np.concatenate([o_a, o_b]), o_all, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s_b, s_all, rtol=1e-12, atol=1e-12)

    def test_matrix_form(self):
        # independent derivation: S_t = (I - b k k^T) diag(a) S_{t-1} + b k v^T, out = scale * S_t^T q
        q, k, v, g, beta = rand_inputs(T=5, H_k=1, H_v=1, S=6)
        out, _ = gated_delta_rule(q, k, v, g, beta)
        qn, kn = l2_norm(q), l2_norm(k)
        S = np.zeros((6, 6))
        for t in range(5):
            a = np.diag(np.exp(g[t, 0]))
            kk = kn[t, 0][:, None]
            S = (np.eye(6) - beta[t, 0] * kk @ kk.T) @ a @ S + beta[t, 0] * kk @ v[t, 0][None, :]
            np.testing.assert_allclose(out[t, 0], S.T @ qn[t, 0] / np.sqrt(6), rtol=1e-11, atol=1e-12)

    def test_associative_memory(self):
        # no decay, beta = 1, orthonormal keys: reading key i returns value i (times scale)
        S_dim, T = 8, 8
        k = np.eye(S_dim)[:, None, :]
        q = np.eye(S_dim)[:, None, :]
        v = RNG.standard_normal((T, 1, S_dim))
        out, _ = gated_delta_rule(q, k, v, np.zeros((T, 1)), np.ones((T, 1)))
        np.testing.assert_allclose(out[:, 0], v[:, 0] / np.sqrt(S_dim), atol=1e-6)

    def test_gqa_head_mapping_is_tiled(self):
        # v head h reads q/k head h % H_k (ggml tiled broadcast; Strata gdn.hpp idx[h] = h % h_k)
        q, k, v, g, beta = rand_inputs(H_k=2, H_v=4)
        o1, _ = gated_delta_rule(q, k, v, g, beta)
        o2, _ = gated_delta_rule(np.tile(q, (1, 2, 1)), np.tile(k, (1, 2, 1)), v, g, beta)
        np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)

    def test_l2_norm_equals_llama_cpp_rms_form(self):
        x = RNG.standard_normal((3, 4, 128))
        n, eps = x.shape[-1], 1e-6
        rms = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps / n)
        np.testing.assert_allclose(l2_norm(x, eps), rms / np.sqrt(n), rtol=1e-12)

    def test_kda_gate_bounds_and_formula(self):
        raw = RNG.standard_normal((5, 3, 4)) * 4
        dt = RNG.standard_normal((3, 4))
        A_log = RNG.standard_normal(3)
        g = kda_gate(raw, dt, A_log, lower_bound=-5.0)
        self.assertTrue(np.all(g < 0) and np.all(g > -5.0))
        A = -np.exp(A_log)[None, :, None]
        explicit = -5.0 * (1.0 / (1.0 + np.exp(-(-(raw + dt[None]) * A))))
        np.testing.assert_allclose(g, explicit, rtol=1e-12)
        g_sp = kda_gate(raw, dt, A_log, lower_bound=-np.inf)
        np.testing.assert_allclose(g_sp, np.log1p(np.exp(raw + dt[None])) * A, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
