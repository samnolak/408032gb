"""Oracle: MiMo-V2.x-Flash attention (llama.cpp mimo2 @ec7630a).

    q, k   = NEOX RoPE on the first n_rot dims of each head (pairs i, i + n_rot/2), theta = base^(-2i/n_rot);
             SWA layers use rope.freq_base_swa                             (rope type: src/llama-model.cpp:3186)
    mask   = causal; on SWA layers key p0 is hidden from query p1 when p1 - p0 >= n_swa
                                                                           (src/llama-hparams.h is_masked_swa)
    GQA    = query head h reads KV head h // (n_head / n_kv)               (ggml mul_mat broadcast)
    sinks  = one logit per head that joins the softmax denominator only:   p_j = e^{s_j} / (sum_k e^{s_k} + e^{sink})
    scale  = 1 / sqrt(k_dim); output = concat heads -> wo
MoE: sigmoid + bias selection, weights normalised, no shared expert, plain SwiGLU (ref/router.py, w_scale=1).
"""
import numpy as np


def rope_neox(x, pos, n_rot, base):
    """x (T, H, D); rotates x[..., :n_rot] in NEOX pairing."""
    x = np.array(x, dtype=np.float64)
    half = n_rot // 2
    inv = base ** (-np.arange(half) * 2.0 / n_rot)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]          # (T, half)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half].copy(), x[..., half:n_rot].copy()
    x[..., :half] = a * c - b * s
    x[..., half:n_rot] = a * s + b * c
    return x


def attention(q, k, v, sinks, n_swa=0):
    """q (T, H, Dk); k (T, Hkv, Dk); v (T, Hkv, Dv); sinks (H,). Returns (T, H * Dv)."""
    T, H, Dk = q.shape
    Hkv = k.shape[1]
    r = H // Hkv
    out = np.empty((T, H, v.shape[2]))
    p1 = np.arange(T)[:, None]
    p0 = np.arange(T)[None, :]
    keep = p0 <= p1
    if n_swa:
        keep &= (p1 - p0) < n_swa
    for h in range(H):
        kh, vh = k[:, h // r], v[:, h // r]
        sc = np.where(keep, q[:, h] @ kh.T / np.sqrt(Dk), -np.inf)
        m = np.maximum(sc.max(1, keepdims=True), sinks[h])
        e = np.exp(sc - m)
        p = e / (e.sum(1, keepdims=True) + np.exp(sinks[h] - m))
        out[:, h] = p @ vh
    return out.reshape(T, -1)
