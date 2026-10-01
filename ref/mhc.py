"""Oracle: manifold-constrained hyper-connections (mHC) as used by GLM-5.3-Flash.

Transcribed from llama.cpp @ec7630a, src/models/glm5-next.cpp:
build_hc_pre (line 454), build_hc_sinkhorn (line 417), build_hc_post (line 512),
build_hc_mean (line 370), stream init with ggml_repeat_4d (line 580).

Residual state per token: hc = 4 streams of n_embd.
    mixes = hc_fn @ rms_norm(concat(streams), eps=rms_norm_eps)          (24 = (2 + hc) * hc values)
    pre   = sigmoid(mixes[0:4]  * scale[0] + base[0:4])  + hc_eps
    post  = sigmoid(mixes[4:8]  * scale[1] + base[4:8])  * 2
    comb  = sinkhorn(mixes[8:24] * scale[2] + base[8:24], as C[dst][src] with dst fastest)
    mixer input  = sum_h pre[h] * stream[h]
    new stream d = post[d] * mixer_output + sum_s comb[d][s] * stream[s]
Sinkhorn: softmax over dst for each src; + eps; normalise over src for each dst ("cols");
then (iters - 1) x [normalise over dst for each src ("rows"), then over src for each dst].
Every normalisation adds eps to the sum.
"""
import numpy as np


def rms_norm(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def sinkhorn(comb, iters=20, eps=1e-6):
    """comb (T, dst, src) logits -> doubly (near-)stochastic matrices, same shape."""
    c = np.asarray(comb, dtype=np.float64)
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True)                 # softmax over dst
    c = c + eps
    c = c / (c.sum(axis=2, keepdims=True) + eps)         # cols: over src, per dst
    for _ in range(1, iters):
        c = c / (c.sum(axis=1, keepdims=True) + eps)     # rows: over dst, per src
        c = c / (c.sum(axis=2, keepdims=True) + eps)
    return c


def hc_pre(streams, hc_fn, hc_scale, hc_base, rms_eps=1e-5, hc_eps=1e-6, iters=20):
    """streams (T, hc, n); hc_fn (24, hc*n); hc_scale (3,); hc_base (24,).
    Returns mixer_input (T, n), post (T, hc), comb (T, hc, hc) as C[dst][src]."""
    T, hc, n = streams.shape
    mixes = rms_norm(streams.reshape(T, hc * n), rms_eps) @ np.asarray(hc_fn, dtype=np.float64).T
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    pre = sig(mixes[:, 0:hc] * hc_scale[0] + hc_base[0:hc]) + hc_eps
    post = sig(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]) * 2.0
    # comb values are stored dst-fastest: index = src * hc + dst
    raw = (mixes[:, 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(T, hc, hc).transpose(0, 2, 1)
    comb = sinkhorn(raw, iters, hc_eps)
    mix_in = np.einsum("th,thn->tn", pre, streams)
    return mix_in, post, comb


def hc_post(mixer_out, streams, post, comb):
    """new streams (T, hc, n)."""
    return post[:, :, None] * mixer_out[:, None, :] + np.einsum("tds,tsn->tdn", comb, streams)


def hc_init(x, hc=4):
    return np.repeat(np.asarray(x, dtype=np.float64)[:, None, :], hc, axis=1)


def hc_mean(streams):
    return streams.mean(axis=1)
