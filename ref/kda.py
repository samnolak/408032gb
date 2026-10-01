"""Oracle: gated delta rule with a per-channel (KDA) or scalar (GDN) decay gate.

Transcribed from the pinned sources, llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d:

* op semantics  ggml/src/ggml-cpu/ops.cpp  ggml_compute_forward_gated_delta_net_one_chunk
  (line 11124 at the pin):
      scale = 1/sqrt(S_v)
      KDA:    S[i][:] *= exp(g[i])          scalar: S *= exp(g)
      delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta
      S[i][j] += k[i] * delta[j]
      out[j]   = scale * sum_i S[i][j] q[i]
* op contract   ggml/include/ggml.h  ggml_gated_delta_net: g is [1, H_v] (scalar gate) or
  [S_v, H_v] (KDA); q, k are [S_k, H_k] with H_v % H_k == 0 (k/q head = h % H_k is Strata's
  `idx[h] = h % h_k`, include/strata/kernels/gdn.hpp @c499bd1).
* q/k normalisation  src/models/models.h build_gdn_l2_norm: rms_norm(x, eps/n) / sqrt(n),
  which equals x * rsqrt(sum(x^2) + eps), eps = 1e-6 (glm5-next.cpp build_kda_layer).
* KDA gate      src/models/glm5-next.cpp build_kda_layer:
      g = (f_b(f_a(x)) + dt_bias) * A,   A = ssm_a = -exp(A_log)
      lower bound set:  g = lower_bound * sigmoid(-g)     (GLM-5.3-Flash: lower_bound = -5)
      otherwise:        g = softplus(f_b(f_a(x)) + dt_bias) * A

Shapes here are numpy-natural: q, k (T, H_k, S); v (T, H_v, S); g (T, H_v, S) or (T, H_v);
beta (T, H_v); state (H_v, S_k, S_v) indexed [h, i, j] like S[i][j] above.
Everything runs in float64; cast at the boundary when comparing with a kernel.
"""
import numpy as np


def l2_norm(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def kda_gate(raw, dt_bias, A_log, lower_bound=-5.0):
    """Log-decay g for KDA. raw: (T, H, S) = f_b(f_a(x)); dt_bias: (H, S); A_log: (H,)."""
    raw = np.asarray(raw, dtype=np.float64)
    A = -np.exp(np.asarray(A_log, dtype=np.float64))[None, :, None]
    pre = raw + np.asarray(dt_bias, dtype=np.float64)[None]
    if np.isfinite(lower_bound):
        return lower_bound / (1.0 + np.exp(pre * A))      # lower_bound * sigmoid(-(pre * A))
    return np.logaddexp(0.0, pre) * A                       # softplus(pre) * A


def gated_delta_rule(q, k, v, g, beta, state=None, normalize_qk=True, eps=1e-6):
    """Return (out (T, H_v, S_v), final state (H_v, S_k, S_v)).

    g of shape (T, H_v) is the scalar GDN gate; (T, H_v, S_k) is the KDA per-channel gate.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    T, H_k, S_k = q.shape
    _, H_v, S_v = v.shape
    assert H_v % H_k == 0, "H_v must be a multiple of H_k"
    assert S_k == S_v, "ggml_gated_delta_net requires S_k == S_v"
    if normalize_qk:
        q, k = l2_norm(q, eps), l2_norm(k, eps)
    kda = g.ndim == 3
    S = np.zeros((H_v, S_k, S_v)) if state is None else np.array(state, dtype=np.float64)
    out = np.empty((T, H_v, S_v))
    scale = 1.0 / np.sqrt(S_v)
    for t in range(T):
        for h in range(H_v):
            hk = h % H_k
            if kda:
                S[h] *= np.exp(g[t, h])[:, None]            # row i scaled by exp(g[i])
            else:
                S[h] *= np.exp(g[t, h])
            delta = (v[t, h] - k[t, hk] @ S[h]) * beta[t, h]
            S[h] += np.outer(k[t, hk], delta)
            out[t, h] = scale * (q[t, hk] @ S[h])
    return out, S
