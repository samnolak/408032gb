"""Oracle: GLM-5.3-Flash DeepSeek-sparse-attention layer (k-pool indexer + nope MLA), prefill.

Transcribed from llama.cpp @ec7630a src/models/glm5-next.cpp build_dsa_layer and
build_kpool_select, plus the selection rule observed in the dumped kq_mask_dsa (tests/
test_glm_tiny_parity.py checks it against the real graph):

Indexer (one key head, n_idx query heads, kpool = 4):
    ik[t]   = layer_norm(Wk x[t]) * g + b          ig[t] = Wgate x[t]
    pool b (cells 4b..4b+3, complete when 4(b+1) <= t+1):
        pooled[b] = sum_j softmax_j(ig[4b+j] + ape[j]) * ik[4b+j]     (softmax per channel over j)
    iq[t,h] = Wq_b q_resid[t]          w[t,h] = Wproj x[t] / sqrt(dim * n_idx)
    score[t,b] = sum_h w[t,h] * relu(iq[t,h] . pooled[b])     (visible pools only)
    selected cells = cells of the top (top_k / kpool) visible pools by score
                     + the incomplete tail 4*floor((t+1)/4) .. t
Attention (nope MLA, latent cache of width kv_lora, keys = values = latent):
    q_resid = rms_norm(Wq_a x) * n1        q = Wq_b q_resid (heads x k_mla)
    latent[s] = rms_norm(Wkv_a x[s]) * n2
    q_abs[h] = Wk_b[h] q[h]                p = softmax over selected s of q_abs[h] . latent[s] / sqrt(k_mla)
    out_h = Wv_b[h] (sum_s p_s latent[s])  attn_out = Wo concat_h(out_h)
"""
import numpy as np


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def indexer_pools(ik, ig, ape, kpool=4):
    """ik, ig (T, dim); ape (kpool, dim). Returns pooled keys (n_complete, dim)."""
    n = ik.shape[0] // kpool
    out = np.empty((n, ik.shape[1]))
    for b in range(n):
        lg = ig[b * kpool:(b + 1) * kpool] + ape
        p = np.exp(lg - lg.max(0, keepdims=True))
        p /= p.sum(0, keepdims=True)
        out[b] = (p * ik[b * kpool:(b + 1) * kpool]).sum(0)
    return out


def indexer_scores(iq, w, pooled, kpool=4):
    """iq (T, n_idx, dim); w (T, n_idx); pooled (n_pool, dim). -inf for pools not yet complete."""
    T = iq.shape[0]
    s = np.einsum("th,thb->tb", w, np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0))
    visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(T)[:, None] + 1)
    return np.where(visible, s, -np.inf)


def select_cells(scores, top_k, kpool=4):
    """List of selected cell indices per token."""
    n_top = top_k // kpool
    out = []
    for t in range(scores.shape[0]):
        vis = np.where(np.isfinite(scores[t]))[0]
        top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top]]
        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]
        tail0 = ((t + 1) // kpool) * kpool
        out.append(sorted(cells + list(range(tail0, t + 1))))
    return out


def mla_attention(q_abs, latent, cells, wv_b, k_mla):
    """q_abs (T, H, L); latent (S, L); wv_b (H, v, L). Returns (T, H * v)."""
    T, H, _ = q_abs.shape
    out = np.empty((T, H * wv_b.shape[1]))
    for t in range(T):
        kv = latent[cells[t]]
        sc = np.einsum("hl,sl->hs", q_abs[t], kv) / np.sqrt(k_mla)
        p = np.exp(sc - sc.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ctx = p @ kv                                            # (H, L)
        out[t] = np.einsum("hvl,hl->hv", wv_b, ctx).reshape(-1)
    return out
