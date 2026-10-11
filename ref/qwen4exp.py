"""Oracle: llama.cpp architecture "qwen4exp" (the reassessment's candidate #1, docs/reassessment/ROADMAP.md N2):
hyper-connections + gated delta net + QSA block-sparse attention + softmax MoE with a gated shared expert +
PLE n-gram hash embeddings.

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d. Paths below are relative to that tree;
"G" = src/models/qwen4exp.cpp. Every formula cites the lines it comes from.

Residual (G:442-499): the residual is hc parallel streams res (T, hc, n). hc_init repeats the token embedding
into every stream (G:443-446). Per layer: [PLE on its layer] -> attention mixer -> combine -> MoE mixer ->
combine. The final mixer replaces the output norm (G:514-524); logits = output @ result_norm.

HC mixer, build_hc_mix (G:314-368), with gamma (hc, n), w_down (r, hc*n), w_up (hc*n, r), w_inject (hc, hc*n):
    xn      = rms_norm(res[s]) * gamma[s]        per stream, eps = rms_norm_epsilon       (G:326-330)
    gate    = w_up silu(w_down xn / hc)                                                  (G:332-335)
    mixed   = (1/hc) sum_s xn[s] * sigmoid(gate[s])        unfused G:345-358; fused ggml_dsv4_hc_pre_gated
                                                           (ggml/include/ggml.h:2706-2713, same formula)
    inject  = w_inject xn                                                                (G:362-365)
HC combine, build_hc_combine (G:370-398):
    res'[s] = res[s] + block_out * 2*sigmoid(inject[s] / hc)       unfused G:388-393; fused ggml_dsv4_hc_post
                                                                   with comb = NULL (ggml.h:2715-2720)

Gated delta net layer, build_layer_attn_linear (G:1017-1142):
    qkv = W_qkv x (channels [q | k | v], q,k: Hk heads of S, v: Hv heads of S)          (G:626, 1087-1103)
    z = W_gate x; beta = sigmoid(W_beta x); g = softplus(W_alpha x + dt_bias) * ssm_a  (G:630, 1040-1055)
        (ssm_a stores A = -exp(A_log), conversion/qwen.py:395-396)
    short conv: ggml_ssm_conv over [zero state | qkv], kernel d_conv, then silu        (G:1069-1080,
        ggml/src/ggml-cpu/ops.cpp:9932-9980: out[c,t] = sum_k w[c,k] x[c, t+k] on the padded sequence)
    q, k = l2_norm(.) with eps = rms_norm_epsilon (src/models/models.h:14-18: rms_norm(x, eps/n)/sqrt(n)
        == x / sqrt(sum x^2 + eps))                                                     (G:1112-1113)
    out = gated delta rule, scalar gate per value head, value head h reads key head h % Hk
        (ggml/src/ggml-cpu/ops.cpp:11124-11200; oracle: ref/kda.py gated_delta_rule)      (G:1126)
    final = rms_norm(out per head) * ssm_norm * sigmoid(z)   (sigmoid, not silu: G:636-646)
    y = W_out final                                                                      (G:1136)

Full-attention layer, build_layer_attn (G:930-1015):
    W_q x holds [q | gate] interleaved per head (D each)                                 (G:947-972)
    q = rms_norm(q) * q_norm; k = rms_norm(W_k x) * k_norm per head; v = W_v x           (G:955-975)
    IMRoPE on q and k (G:978-988); scale 1/sqrt(D) (G:994, attention.scale absent)
    attention over the selected cells (QSA) or causal (compress ratio 0 -> dense, G:942, 996-1002), GQA:
    query head h reads kv head h // (H/Hkv) (ggml mul_mat broadcast); softmax(kq*scale + mask)
    (src/llama-graph.cpp:2696-2752, flash attention off)
    y = W_o (attn * sigmoid(gate))                                                       (G:1005-1012)
QSA selection, build_qsa_sel (G:744-869), pools = blocks of kpool consecutive cells in sequence order
(src/llama-memory-hybrid-idx.cpp kpool layout; G:80-83 by_order, select_tail):
    k_raw = W_ik x                                                                       (G:763)
    pooled[b] = imrope(rms_norm(mean_j k_raw[b*kpool + j]) * ik_norm, pos of first member)  (G:777-791)
    iq = imrope(rms_norm(W_iq x per head) * iq_norm, pos)                                (G:807-813)
    score[t,b] = sum_h relu(iq[t,h] . pooled[b]) / sqrt(idx_dim), -inf unless pool b ends at or before t
        (G:815-829; mask: llama-memory-hybrid-idx.cpp:977-993, nv = (rank+1)/kpool)
    cells = members of the top min(n_pool, top_k/kpool) pools (ggml_top_k = std::partial_sort, order among
        equal scores unspecified, ops.cpp:8786-8823) + the tail kpool*floor((t+1)/kpool) .. t
        (G:831-866; tail: llama-memory-hybrid-idx.cpp:1017-1018); causal mask added (G:865)
IMRoPE (ggml/src/ggml-cpu/ops.cpp:6220-6290 ggml_mrope_cache_init, 6377 is_imrope, 6440 NEOX pairing
(i, i + n_rot/2), theta_scale = base^(-2/n_rot)): pair i uses section sector = i % sum(sections):
    sector%3==1 and sector < 3*s[1] -> p_h; sector%3==2 and sector < 3*s[2] -> p_w;
    sector%3==0 and sector < 3*s[0] -> p_t; otherwise -> p_e.
    Positions: text q/k (and the indexer query) use (p, p, p, 0) (src/llama-graph.cpp:131-141); a pooled
    indexer key uses (p0, y, x, p0) of its first member (llama-memory-hybrid-idx.cpp:931-937), and for a
    token batch the cell's x = y = p0 (src/llama-batch.cpp:817-823, src/llama-kv-cache.cpp:1133-1139), so
    (p0, p0, p0, p0). The two agree unless a rotary pair falls in the 4th section.

MoE, build_layer_ffn (G:1144-1192) -> build_moe_ffn (src/llama-graph.cpp:2015-2340):
    probs = softmax(W_r x) (2063-2065); ids = top-k of probs (2131); w = probs[ids] / max(sum, 6.1035e-5)
    (2156-2167, norm_w = true at G:1155); no scaling: qwen4exp never reads expert_weights_scale, default
    0 (src/llama-hparams.h:124) and 0 skips the scale (2171); expert = W_down (silu(W_gate x) * W_up x)
    (2268, swiglu_clamp_exp = 0); routed = sum_j w_j expert_j.
    shared = (W_sd (silu(W_sg x) * W_su x)) * sigmoid(w_sgate . x)                       (G:1165-1186)
    y = routed + shared

PLE (one linear layer, before its attention mixer, G:453-455), build_ple (G:1370-1458):
    rows: hash on the host (G:1222-1280): ctx[0] = token, ctx[s] = token s positions back, replaced by EOS
        from the first missing or EOS predecessor on (the token's own EOS does not cut, G:1254-1266);
        for n = 2..ngram: mixed = XOR_{j<n} (ctx[j] * mult[j]) in uint64 (wraps mod 2^64);
        row[(n-2)*hpn + g] = mixed % vocab[h] + offset[h]
    emb = concat_h table[row_h]   (n_heads * head_dim = n_embd)                          (G:1363-1364)
    key = gnorm(W_k emb, nk); query = gnorm(res, nq)   gnorm = per-stream rms_norm * gamma (G:1378-1388)
    s = sum(key*query)/sqrt(n); gate = sigmoid(sign(s) sqrt(max(|s|, 1e-6)))            (G:1391-1395)
    gated = (W_v emb) * gate (broadcast over the streams)                                (G:1399-1402)
    conv  = silu(sum_k w[c,k] gnorm(gated, nc)[c, t - (K-1-k)*ngram]), zero history     (G:1405-1455)
    res'  = res + gated + conv                                                           (G:1457)

Precision (N10, gap 3 of docs/reassessment/CORRECTNESS.md): every function that touches the attention takes
prec=None (the reference contract: f32 caches, flash attention off) or ref/fattn.py precision(kv, fa). With an f16
cache the oracle rounds what llama.cpp stores in f16, where it stores it: the rotated K and V of every cell
(cpy_k/cpy_v into the attention cache, build_attn_qsa G:906-907, and build_attn), the raw indexer key (cpy_k into the
indexer cache, G:765-768, which llama_memory_hybrid_idx creates with the attention cache types,
src/llama-memory-hybrid-idx.cpp:69-72 and src/llama-model.cpp:2745-2748) and the pooled key (scattered into the same
cache and gathered back for the scores, G:793-797); the pooling reads the cached raw keys (G:772-774). The attention
itself runs in the mode of ref/fattn.py the run takes (flash attention one_chunk / tiled, or mul_mat with f16
operands). Recurrent states stay f32 (GGML_TYPE_F32, src/llama-model.cpp:2754-2755).

Shapes are numpy-natural (token-major); weights are the GGUF arrays as numpy reads them (ggml ne reversed).
Everything runs in float64; cast at the boundary when comparing with a kernel.
"""
import numpy as np

from ref.fattn import attend_rows, row_modes, store
from ref.kda import gated_delta_rule

U64 = (1 << 64) - 1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * sigmoid(x)


def softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def rms_norm(x, eps, w=None):
    """ggml_rms_norm over the last axis (x / sqrt(mean(x^2) + eps)), times w if given."""
    x = np.asarray(x, dtype=np.float64)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def grouped_rms_norm(x, w, eps):
    """x (T, hc, n), w (hc, n): RMS over each stream's n values, then the per-stream gamma (G:328, 1382-1385)."""
    return rms_norm(x, eps) * np.asarray(w, dtype=np.float64)[None]


# ---------------------------------------------------------------- hyper-connections

def hc_init(x, hc):
    return np.repeat(np.asarray(x, dtype=np.float64)[:, None, :], hc, axis=1)


def hc_mix(res, w_norm, w_down, w_up, w_inject, eps):
    """res (T, hc, n). Returns dict norm (T, hc*n), gate (T, hc*n), mixed (T, n), inject (T, hc) or None."""
    T, hc, n = res.shape
    xn = grouped_rms_norm(res, w_norm, eps).reshape(T, hc * n)
    gate = silu((xn @ w_down.T) / hc) @ w_up.T
    mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1) / hc
    inject = None if w_inject is None else xn @ w_inject.T
    return dict(norm=xn, gate=gate, mixed=mixed, inject=inject)


def hc_combine(res, block_out, inject):
    """res (T, hc, n); block_out (T, n); inject (T, hc) -> new residual (T, hc, n)."""
    hc = res.shape[1]
    w = 2.0 * sigmoid(np.asarray(inject) / hc)
    return res + np.asarray(block_out)[:, None, :] * w[:, :, None]


# ---------------------------------------------------------------- PLE n-gram hash embedding

def ple_rows(tokens, multipliers, offsets, vocab_sizes, ngram, heads_per_ngram, eos):
    """Gather row per (token, head) for one sequence starting at position 0 (G:1254-1280). (T, n_heads) int."""
    tokens = [int(t) for t in tokens]
    n_heads = (ngram - 1) * heads_per_ngram
    out = np.zeros((len(tokens), n_heads), dtype=np.int64)
    for i, tok in enumerate(tokens):
        ctx, cut = [tok], False
        for s in range(1, ngram):
            prev = tokens[i - s] if i - s >= 0 else None          # before the sequence start: LLAMA_TOKEN_NULL
            cut = cut or prev is None or prev == eos
            ctx.append(eos if cut else prev)
        for n in range(2, ngram + 1):
            mixed = (ctx[0] * int(multipliers[0])) & U64
            for j in range(1, n):
                mixed ^= (ctx[j] * int(multipliers[j])) & U64
            for g in range(heads_per_ngram):
                h = (n - 2) * heads_per_ngram + g
                out[i, h] = mixed % int(vocab_sizes[h]) + int(offsets[h])
    return out


def ple_embed(table, rows):
    """table (n_rows, head_dim), rows (T, n_heads) -> (T, n_heads * head_dim), heads slowest (G:1362-1364)."""
    return np.asarray(table, dtype=np.float64)[rows].reshape(rows.shape[0], -1)


def dilated_causal_dwconv(x, w, dilation):
    """x (T, C), w (C, K): out[t, c] = sum_k w[c, k] x[t - (K-1-k)*dilation, c], zero before t = 0 (G:1410-1451)."""
    T, C = x.shape
    K = w.shape[1]
    out = np.zeros((T, C))
    for k in range(K):
        shift = (K - 1 - k) * dilation
        if shift < T:
            out[shift:] += x[:T - shift] * w[:, k][None]
    return out


def ple_forward(emb, res, w_key, w_value, w_norm_key, w_norm_query, w_norm_conv, w_conv, eps, ngram):
    """emb (T, n); res (T, hc, n). Returns dict gate (T, hc), gated (T, hc, n), conv_out (T, hc, n), out."""
    T, hc, n = res.shape
    key = grouped_rms_norm((emb @ w_key.T).reshape(T, hc, n), w_norm_key, eps)
    value = emb @ w_value.T
    query = grouped_rms_norm(res, w_norm_query, eps)
    s = (key * query).sum(-1) / np.sqrt(n)
    gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
    gated = value[:, None, :] * gate[:, :, None]
    normed = grouped_rms_norm(gated, w_norm_conv, eps).reshape(T, hc * n)
    conv_out = silu(dilated_causal_dwconv(normed, np.asarray(w_conv, dtype=np.float64), ngram)).reshape(T, hc, n)
    return dict(key=key, query=query, score=s, gate=gate, gated=gated, conv_out=conv_out,
                out=res + gated + conv_out)


# ---------------------------------------------------------------- gated delta net

def short_conv(x, w):
    """ggml_ssm_conv with a zero initial state: x (T, C), w (C, K) -> (T, C)."""
    T, C = x.shape
    K = w.shape[1]
    pad = np.vstack([np.zeros((K - 1, C)), x])
    return np.stack([np.sum(pad[t:t + K].T * w, axis=1) for t in range(T)])


def gdn_output_norm(scan, w_norm, z, eps):
    """Gated RMS norm of the scan output per value head: rms_norm(scan) * w * sigmoid(z) (G:636-646, 1131)."""
    return rms_norm(scan, eps) * np.asarray(w_norm) * sigmoid(z)


def gdn_layer(x, w, hp):
    """x (T, n) mixer input; w: dict of this layer's weights without the 'blk.N.' prefix."""
    T = x.shape[0]
    S, Hk, Hv, eps = hp["ssm_state"], hp["ssm_groups"], hp["ssm_v_heads"], hp["eps"]
    kd = S * Hk
    qkv = x @ w["attn_qkv.weight"].T
    z = x @ w["attn_gate.weight"].T
    beta = sigmoid(x @ w["ssm_beta.weight"].T)
    alpha = x @ w["ssm_alpha.weight"].T
    a_sp = softplus(alpha + w["ssm_dt.bias"])
    g = a_sp * w["ssm_a"]
    conv_raw = short_conv(qkv, w["ssm_conv1d.weight"])
    conv = silu(conv_raw)
    q = conv[:, :kd].reshape(T, Hk, S)
    k = conv[:, kd:2 * kd].reshape(T, Hk, S)
    v = conv[:, 2 * kd:].reshape(T, Hv, S)
    out, state = gated_delta_rule(q, k, v, g, beta, normalize_qk=True, eps=eps)
    final = gdn_output_norm(out, w["ssm_norm.weight"], z.reshape(T, Hv, S), eps).reshape(T, Hv * S)
    return dict(qkv=qkv, z=z, beta=beta, alpha=alpha, a_softplus=a_sp, g=g, conv_raw=conv_raw, conv=conv,
                q=q, k=k, v=v, scan=out, state=state, final=final, out=final @ w["ssm_out.weight"].T)


# ---------------------------------------------------------------- IMRoPE

def imrope_sector_map(n_rot, sections):
    """Section (0 t, 1 h, 2 w, 3 e) used by each rotary pair (ops.cpp ggml_mrope_cache_init, is_imrope)."""
    s = list(sections)
    sect_dims = sum(s)
    out = []
    for i in range(n_rot // 2):
        sector = i % sect_dims
        if sector % 3 == 1 and sector < 3 * s[1]:
            out.append(1)
        elif sector % 3 == 2 and sector < 3 * s[2]:
            out.append(2)
        elif sector % 3 == 0 and sector < 3 * s[0]:
            out.append(0)
        else:
            out.append(3)
    return np.array(out)


def imrope(x, pos4, n_rot, sections, base):
    """x (T, H, D); pos4 (T, 4) positions (t, h, w, e). Rotates pairs (i, i + n_rot/2), i < n_rot/2."""
    x = np.array(x, dtype=np.float64)
    half = n_rot // 2
    sec = imrope_sector_map(n_rot, sections)
    inv = base ** (-np.arange(half) * 2.0 / n_rot)
    ang = np.asarray(pos4, dtype=np.float64)[:, sec] * inv[None, :]            # (T, half)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half].copy(), x[..., half:n_rot].copy()
    x[..., :half] = a * c - b * s
    x[..., half:n_rot] = a * s + b * c
    return x


def text_pos4(T):
    """M-RoPE positions of a text token batch: (p, p, p, 0) (src/llama-graph.cpp:131-141)."""
    p = np.arange(T)
    return np.stack([p, p, p, np.zeros_like(p)], axis=1)


def pool_pos4(n_pool, kpool):
    """Position of each pooled key: its first member's (p0, y, x, p0) = (p0, p0, p0, p0) for text."""
    p0 = np.arange(n_pool) * kpool
    return np.stack([p0, p0, p0, p0], axis=1)


# ---------------------------------------------------------------- QSA

def qsa_pooled_keys(k_raw, k_norm, kpool, eps, n_rot, sections, base):
    """k_raw (T, d) -> pooled keys of the complete pools (n_pool, d) (G:777-791)."""
    n_pool = k_raw.shape[0] // kpool
    pooled = k_raw[:n_pool * kpool].reshape(n_pool, kpool, -1).mean(axis=1)
    pooled = rms_norm(pooled, eps, k_norm)
    return imrope(pooled[:, None, :], pool_pos4(n_pool, kpool), n_rot, sections, base)[:, 0]


def qsa_scores(iq, pooled, kpool):
    """iq (T, n_idx, d); pooled (n_pool, d) -> (T, n_pool), -inf for pools not yet complete (G:815-829)."""
    T, _, d = iq.shape
    s = np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0).sum(axis=1) / np.sqrt(d)
    visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(T)[:, None] + 1)
    return np.where(visible, s, -np.inf)


def qsa_tail(t, kpool):
    """Cells of the incomplete block token t always keeps: kpool*floor((t+1)/kpool) .. t
    (llama-memory-hybrid-idx.cpp:1017-1018, n_tail = (rank + 1) % kpool, newest first)."""
    tail0 = ((t + 1) // kpool) * kpool
    return list(range(tail0, t + 1))


def qsa_select(scores, top_k, kpool):
    """Selected cells per token: top visible pools (stable order on ties) + the incomplete tail."""
    n_top = top_k // kpool
    out = []
    for t in range(scores.shape[0]):
        vis = np.where(np.isfinite(scores[t]))[0]
        top = vis[np.argsort(-scores[t, vis], kind="stable")[:n_top]]
        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]
        out.append(sorted(cells + qsa_tail(t, kpool)))
    return out


def qsa_tie_variant(scores_row, cells, t, top_k, kpool, rtol=1e-6):
    """Whether `cells` is another valid selection for token t: the oracle's tail, whole visible pools, the
    budgeted pool count, and no left-out visible pool scoring above a chosen one beyond f32 rounding (the
    order of ggml_top_k among equal scores is unspecified, ops.cpp:8786-8823). Returns (ok, reason)."""
    cells = sorted(int(c) for c in cells)
    tail = qsa_tail(t, kpool)
    if not set(tail) <= set(cells):
        return False, "tail missing"
    rest = sorted(set(cells) - set(tail))
    pools = sorted({c // kpool for c in rest})
    if rest != [c for b in pools for c in range(b * kpool, (b + 1) * kpool)]:
        return False, "not whole pools"
    vis = set(np.where(np.isfinite(scores_row))[0].tolist())
    if not set(pools) <= vis:
        return False, "invisible pool chosen"
    if len(pools) != min(top_k // kpool, len(vis)):
        return False, "pool count"
    left = sorted(vis - set(pools))
    if left and pools:
        edge = max(scores_row[left])
        if min(scores_row[pools]) < edge - rtol * max(1.0, abs(edge)):
            return False, "not a top-k"
    return True, ""


def attention(q, k, v, cells, scale):
    """q (T, H, D); k, v (T, Hkv, D); cells: list of key cells per query, None = causal. Returns (T, H*D)."""
    T, H, D = q.shape
    r = H // k.shape[1]
    out = np.empty((T, H, v.shape[2]))
    for t in range(T):
        idx = np.arange(t + 1) if cells is None else np.asarray(cells[t])
        for h in range(H):
            sc = k[idx, h // r] @ q[t, h] * scale
            p = np.exp(sc - sc.max())
            out[t, h] = (p / p.sum()) @ v[idx, h // r]
    return out.reshape(T, -1)


def gate_attention(pre, gate):
    """Output gate of the full-attention layer: attn * sigmoid(gate) (G:1005-1009)."""
    return np.asarray(pre) * sigmoid(gate)


def attn_layer(x, w, hp, qsa, cells_override=None, prec=None):
    """Full-attention layer on the mixer input x (T, n). qsa: compress ratio > 0 for this layer. prec: see the module
    docstring (the rows are one ubatch from position 0 unless prec["modes"] says otherwise)."""
    T = x.shape[0]
    H, Hkv, D, eps = hp["n_head"], hp["n_head_kv"], hp["head_dim"], hp["eps"]
    rope = dict(n_rot=hp["n_rot"], sections=hp["rope_sections"], base=hp["rope_base"])
    pos = text_pos4(T)
    qf = (x @ w["attn_q.weight"].T).reshape(T, H, 2 * D)
    q_raw, gate = qf[..., :D], qf[..., D:].reshape(T, H * D)
    q = imrope(rms_norm(q_raw, eps, w["attn_q_norm.weight"]), pos, **rope)
    k_lin = x @ w["attn_k.weight"].T
    k = imrope(rms_norm(k_lin.reshape(T, Hkv, D), eps, w["attn_k_norm.weight"]), pos, **rope)
    v = (x @ w["attn_v.weight"].T).reshape(T, Hkv, D)
    r = dict(q_full=qf.reshape(T, -1), q_normed=rms_norm(q_raw, eps, w["attn_q_norm.weight"]), q=q,
             k_lin=k_lin, k=k, v=v, gate=gate)
    cells = None
    if qsa:
        kp, nh, d = hp["kpool"], hp["idx_heads"], hp["idx_dim"]
        k_raw = x @ w["indexer.k_proj.weight"].T
        pooled_raw = qsa_pooled_keys(store(k_raw, prec), w["indexer.k_norm.weight"], kp, eps, **rope)
        pooled = store(pooled_raw, prec)
        iq = imrope(rms_norm((x @ w["indexer.q_proj.weight"].T).reshape(T, nh, d), eps, w["indexer.q_norm.weight"]),
                    pos, **rope)
        scores = qsa_scores(iq, pooled, kp)
        cells = qsa_select(scores, hp["idx_top_k"], kp)
        if isinstance(cells_override, dict):      # {position: cells} for single tokens
            cells = [cells_override.get(t, c) for t, c in enumerate(cells)]
        elif cells_override is not None:
            cells = cells_override
        r.update(k_raw=k_raw, pooled_raw=pooled_raw, pooled=pooled, iq=iq, scores=scores, cells=cells)
        r.setdefault("stored", {})["k_raw"] = store(k_raw, prec)
    if prec:
        modes = row_modes(prec, np.arange(T))
        r.update(modes=modes, k_cache=store(k, prec), v_cache=store(v, prec))
        r.setdefault("stored", {}).update(k=r["k_cache"], v=r["v_cache"])
        pre = attend_rows(q, r["k_cache"], r["v_cache"], cells, 1.0 / np.sqrt(D), modes).reshape(T, -1)
    else:
        pre = attention(q, k, v, cells, 1.0 / np.sqrt(D))
    gated = gate_attention(pre, gate)
    r.update(pregate=pre, gated=gated, out=gated @ w["attn_output.weight"].T)
    return r


# ---------------------------------------------------------------- MoE

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def swiglu(gate, up):
    """Expert activation: silu(gate) * up (llama-graph.cpp:2268, ggml_swiglu_split)."""
    return silu(gate) * np.asarray(up)


def route_softmax(logits, k, ids=None):
    """softmax gating, top-k, renormalised weights, no scale. Returns probs, ids (T, k), weights (T, k).
    ids: optional expert choice to weight instead of the oracle's own top-k (replay of a tie)."""
    probs = softmax(logits)
    if ids is None:
        ids = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    ids = np.asarray(ids)
    wts = np.take_along_axis(probs, ids, axis=-1)
    wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-5)
    return probs, ids, wts


def moe_layer(x, w, k, ids_override=None):
    """x (T, n) mixer input; routed experts + sigmoid-gated shared expert. ids_override: (T, k) expert ids, or
    {row: ids} for single rows, replays the reference's choice among (near-)tied experts."""
    T = x.shape[0]
    logits = x @ w["ffn_gate_inp.weight"].T
    if isinstance(ids_override, dict):
        own = route_softmax(logits, k)[1].copy()
        for r, e in ids_override.items():
            own[r] = np.asarray(e)
        ids_override = own
    probs, ids, wts = route_softmax(logits, k, ids_override)
    ge, ue, de = w["ffn_gate_exps.weight"], w["ffn_up_exps.weight"], w["ffn_down_exps.weight"]
    gate = np.stack([[ge[e] @ x[t] for e in ids[t]] for t in range(T)])            # (T, k, F)
    up = np.stack([[ue[e] @ x[t] for e in ids[t]] for t in range(T)])
    act = swiglu(gate, up)
    down = np.stack([[de[e] @ act[t, j] for j, e in enumerate(ids[t])] for t in range(T)])   # (T, k, n)
    routed = (down * wts[:, :, None]).sum(axis=1)
    sh = (silu(x @ w["ffn_gate_shexp.weight"].T) * (x @ w["ffn_up_shexp.weight"].T)) @ w["ffn_down_shexp.weight"].T
    sh_gate = sigmoid(x @ w["ffn_gate_inp_shexp.weight"])
    return dict(logits=logits, probs=probs, ids=ids, weights=wts, gate=gate, up=up, act=act, down=down,
                routed=routed, shexp=sh, sh_gate=sh_gate, out=routed + sh * sh_gate[:, None])


# ---------------------------------------------------------------- whole model

def layer_weights(W, il):
    pre = f"blk.{il}."
    return {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}


def forward(tokens, W, hp, overrides=None, prec=None):
    """Prefill of one sequence from position 0. W: GGUF weights as float64 numpy arrays; hp: hyper-parameters
    (see tests/test_qwen4exp_tiny_parity.py read_hparams). overrides: optional {("qsa", il): cells,
    ("moe", il): ids (T, k) or {position: ids}} to replay a discrete choice the reference made among (near-)tied
    candidates.
    prec: KV cache type and flash attention (module docstring; None = f32, flash attention off).
    Returns dict with per-layer intermediates and logits (T, n_vocab)."""
    overrides = overrides or {}
    tokens = np.asarray(tokens)
    T, hc, eps = len(tokens), hp["hc"], hp["eps"]
    res = hc_init(W["token_embd.weight"][tokens], hc)
    out = dict(hc_init=res, layers=[])
    emb = None
    if hp["ple_layers"]:
        rows = ple_rows(tokens, hp["ple_multipliers"], hp["ple_offsets"], hp["ple_vocab_sizes"],
                        hp["ple_ngram"], hp["ple_heads_per_ngram"], hp["ple_eos"])
        emb = ple_embed(W["per_layer_token_embd.weight"], rows)
        out.update(ple_rows=rows, ple_embd=emb)
    for il in range(hp["n_layer"]):
        w = layer_weights(W, il)
        L = dict(res_in=res)
        if il in hp["ple_layers"]:
            L["ple"] = ple_forward(emb, res, w["ple_key.weight"], w["ple_value.weight"], w["ple_norm_key.weight"],
                                   w["ple_norm_query.weight"], w["ple_norm_conv.weight"], w["ple_conv1d.weight"],
                                   eps, hp["ple_ngram"])
            res = L["ple"]["out"]
        L["mix_attn"] = m = hc_mix(res, w["hc_attn_norm.weight"], w["hc_attn_down.weight"], w["hc_attn_up.weight"],
                                   w["hc_attn_inject.weight"], eps)
        if hp["recurrent"][il]:
            L["gdn"] = blk = gdn_layer(m["mixed"], w, hp)
        else:
            L["attn"] = blk = attn_layer(m["mixed"], w, hp, hp["compress"][il] > 0, overrides.get(("qsa", il)), prec)
        res = L["combine_attn"] = hc_combine(res, blk["out"], m["inject"])
        L["mix_ffn"] = m = hc_mix(res, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"], w["hc_ffn_up.weight"],
                                  w["hc_ffn_inject.weight"], eps)
        L["moe"] = moe = moe_layer(m["mixed"], w, hp["n_expert_used"], overrides.get(("moe", il)))
        res = L["combine_ffn"] = hc_combine(res, moe["out"], m["inject"])
        out["layers"].append(L)
    head = hc_mix(res, W["output_hc_norm.weight"], W["output_hc_down.weight"], W["output_hc_up.weight"], None, eps)
    out.update(head=head, result_norm=head["mixed"], logits=head["mixed"] @ W["output.weight"].T)
    return out


# ---------------------------------------------------------------- incremental path: several ubatches, decode, rollback
#
# llama.cpp runs a long prompt as several ubatches and a generation as one llama_decode call per token; every ubatch
# is its own graph that reads what the earlier ones left in the memory (src/llama-memory-hybrid-idx.cpp; one
# sequence, cell = position here). What this oracle carries between chunks, and where llama.cpp keeps it:
#   tokens        token ids of the cached cells: PLE predecessors are read from the attention KV cells (ext.tok),
#                 this ubatch included (G:1250-1252; src/llama-kv-cache.cpp:1848-1900 get_prev_tokens)
#   gdn[il].conv  the last d_conv-1 rows of the conv input qkv (recurrent row r_l): build_conv_state_at prepends it
#                 to the ubatch and writes the new tail back (G:1296-1348; dumped as "conv_state_at", G:1320)
#   gdn[il].state delta-rule state (H_v, S_k, S_v) (row s_l; dumped as "state_predelta" G:1073-1074, written back
#                 as "new_state" src/models/delta-net-base.cpp:546-560; ggml keeps S[i][j] at [j*S_v + i] per head,
#                 ggml/src/ggml-cpu/ops.cpp:11228-11229, so a dump reads as state[h].T)
#   ple[il].hist  the last (K-1)*ngram rows of the grouped-normed gated PLE value (row p_l, G:1405-1424, through
#                 the same build_conv_state_at)
#   kv[il]        rotated K and V of every cell (attention KV cache)
#   idx[il]       raw indexer key of every cell (G:763-768) and the pooled key of every complete pool, computed once,
#                 by the ubatch that completes the pool, and read back from the pooled slot afterwards
#                 (src/llama-memory-hybrid-idx.cpp:672-680, 791-1043; G:772-797)
# forward_chunk over the chunks of a sequence must equal forward() of the whole sequence row by row (causality):
# tests/test_qwen4exp_ref.py checks that; tests/test_qwen4exp_decode_parity.py checks both against llama.cpp per
# ubatch and per decode step. A rollback (llama_memory_seq_rm of the last n positions with n_rs_seq snapshots,
# src/llama-memory-recurrent.cpp:199-215) must leave the state of the kept prefix: cache_truncate.

def new_cache():
    """Empty sequence memory for forward_chunk. modes: the attention mode each cached position was computed with
    (a rollback rebuilds the kept prefix with the modes of its original ubatches)."""
    return dict(tokens=[], gdn={}, ple={}, kv={}, idx={}, modes=[])


def text_pos4_at(pos):
    """M-RoPE positions (p, p, p, 0) of text tokens at absolute positions pos (src/llama-graph.cpp:131-141)."""
    p = np.asarray(pos)
    return np.stack([p, p, p, np.zeros_like(p)], axis=1)


def ple_rows_chunk(history, chunk, multipliers, offsets, vocab_sizes, ngram, heads_per_ngram, eos):
    """PLE rows of the chunk tokens; the predecessors before the chunk come from the cached cells `history`
    (G:1250-1266: prev[] of the KV cells; missing ones before the sequence start read as EOS)."""
    n_prev = ngram - 1
    prefix = [int(t) for t in history[max(0, len(history) - n_prev):]]
    rows = ple_rows(prefix + [int(t) for t in chunk], multipliers, offsets, vocab_sizes, ngram, heads_per_ngram, eos)
    return rows[len(prefix):]


def short_conv_hist(x, w, hist):
    """ggml_ssm_conv over [hist | x] (G:1069-1077, 1322): hist (K-1, C) = the last K-1 conv input rows of the
    earlier ubatches (zeros at the sequence start). Returns (out (T, C), new hist (K-1, C))."""
    x = np.asarray(x, dtype=np.float64)
    K = w.shape[1]
    pad = np.vstack([hist, x])
    out = np.stack([np.sum(pad[t:t + K].T * w, axis=1) for t in range(x.shape[0])])
    return out, pad[pad.shape[0] - (K - 1):]


def gdn_layer_chunk(x, w, hp, st):
    """Gated delta net on one ubatch; st = dict(conv=(K-1, C), state=(H_v, S, S)) carried in, replaced on return."""
    T = x.shape[0]
    S, Hk, Hv, eps = hp["ssm_state"], hp["ssm_groups"], hp["ssm_v_heads"], hp["eps"]
    kd = S * Hk
    qkv = x @ w["attn_qkv.weight"].T
    z = x @ w["attn_gate.weight"].T
    beta = sigmoid(x @ w["ssm_beta.weight"].T)
    alpha = x @ w["ssm_alpha.weight"].T
    a_sp = softplus(alpha + w["ssm_dt.bias"])
    g = a_sp * w["ssm_a"]
    conv_in, state_in = st["conv"], st["state"]
    conv_raw, conv_hist = short_conv_hist(qkv, w["ssm_conv1d.weight"], conv_in)
    conv = silu(conv_raw)
    q = conv[:, :kd].reshape(T, Hk, S)
    k = conv[:, kd:2 * kd].reshape(T, Hk, S)
    v = conv[:, 2 * kd:].reshape(T, Hv, S)
    out, state = gated_delta_rule(q, k, v, g, beta, state=state_in, normalize_qk=True, eps=eps)
    st["conv"], st["state"] = conv_hist, state
    final = gdn_output_norm(out, w["ssm_norm.weight"], z.reshape(T, Hv, S), eps).reshape(T, Hv * S)
    return dict(qkv=qkv, z=z, beta=beta, alpha=alpha, a_softplus=a_sp, g=g, conv_in=conv_in, state_in=state_in,
                conv_raw=conv_raw, conv=conv, q=q, k=k, v=v, scan=out, state=state, final=final,
                out=final @ w["ssm_out.weight"].T)


def ple_forward_chunk(emb, res, w_key, w_value, w_norm_key, w_norm_query, w_norm_conv, w_conv, eps, ngram, st):
    """ple_forward on one ubatch; st = dict(hist=((K-1)*ngram, hc*n)) carried in, replaced on return (G:1405-1455:
    the history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one)."""
    T, hc, n = res.shape
    key = grouped_rms_norm((emb @ w_key.T).reshape(T, hc, n), w_norm_key, eps)
    value = emb @ w_value.T
    query = grouped_rms_norm(res, w_norm_query, eps)
    s = (key * query).sum(-1) / np.sqrt(n)
    gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
    gated = value[:, None, :] * gate[:, :, None]
    normed = grouped_rms_norm(gated, w_norm_conv, eps).reshape(T, hc * n)
    hist_in = st["hist"]
    full = np.vstack([hist_in, normed])
    conv = dilated_causal_dwconv(full, np.asarray(w_conv, dtype=np.float64), ngram)[hist_in.shape[0]:]
    st["hist"] = full[full.shape[0] - hist_in.shape[0]:]
    conv_out = silu(conv).reshape(T, hc, n)
    return dict(gate=gate, gated=gated, normed=normed, hist_in=hist_in, conv_out=conv_out,
                out=res + gated + conv_out)


def qsa_scores_at(iq, pooled, kpool, pos):
    """qsa_scores for tokens at absolute positions pos: pool b is visible once (b+1)*kpool <= pos+1."""
    d = iq.shape[-1]
    s = np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0).sum(axis=1) / np.sqrt(d)
    visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)
    return np.where(visible, s, -np.inf)


def qsa_select_at(scores, top_k, kpool, pos):
    """qsa_select for tokens at absolute positions pos (top visible pools + the incomplete tail of the token)."""
    n_top = top_k // kpool
    out = []
    for r, t in enumerate(np.asarray(pos).tolist()):
        vis = np.where(np.isfinite(scores[r]))[0]
        top = vis[np.argsort(-scores[r, vis], kind="stable")[:n_top]]
        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]
        out.append(sorted(cells + qsa_tail(t, kpool)))
    return out


def attn_layer_chunk(x, w, hp, qsa, pos, kvc, idxc, cells_override=None, prec=None):
    """Full-attention layer on one ubatch at absolute positions pos (cell = position). kvc = dict(k, v) of the
    cached cells, idxc = dict(k_raw, pooled) of the indexer cache; both are extended in place with the values as the
    cache stores them (prec: module docstring). cells_override: {position: cells} replays a reference choice among
    tied pools."""
    T = x.shape[0]
    H, Hkv, D, eps = hp["n_head"], hp["n_head_kv"], hp["head_dim"], hp["eps"]
    rope = dict(n_rot=hp["n_rot"], sections=hp["rope_sections"], base=hp["rope_base"])
    p4 = text_pos4_at(pos)
    qf = (x @ w["attn_q.weight"].T).reshape(T, H, 2 * D)
    q_raw, gate = qf[..., :D], qf[..., D:].reshape(T, H * D)
    q_normed = rms_norm(q_raw, eps, w["attn_q_norm.weight"])
    q = imrope(q_normed, p4, **rope)
    k_lin = x @ w["attn_k.weight"].T
    k = imrope(rms_norm(k_lin.reshape(T, Hkv, D), eps, w["attn_k_norm.weight"]), p4, **rope)
    v = (x @ w["attn_v.weight"].T).reshape(T, Hkv, D)
    kvc["k"] = np.concatenate([kvc["k"], store(k, prec)])
    kvc["v"] = np.concatenate([kvc["v"], store(v, prec)])
    r = dict(q_full=qf.reshape(T, -1), q_normed=q_normed, q=q, k_lin=k_lin, k=k, v=v, gate=gate,
             stored=dict(k=kvc["k"][kvc["k"].shape[0] - T:], v=kvc["v"][kvc["v"].shape[0] - T:]))   # as cached
    cells = [list(range(0, t + 1)) for t in np.asarray(pos).tolist()]
    if qsa:
        kp, nh, d = hp["kpool"], hp["idx_heads"], hp["idx_dim"]
        k_raw = x @ w["indexer.k_proj.weight"].T
        idxc["k_raw"] = np.concatenate([idxc["k_raw"], store(k_raw, prec)])
        b0, b1 = idxc["pooled"].shape[0], idxc["k_raw"].shape[0] // kp       # pools this ubatch completes
        new = idxc["k_raw"][b0 * kp:b1 * kp].reshape(b1 - b0, kp, d).mean(axis=1)
        new = rms_norm(new, eps, w["indexer.k_norm.weight"])
        new = imrope(new[:, None, :], pool_pos4(b1, kp)[b0:], **rope)[:, 0]
        idxc["pooled"] = np.concatenate([idxc["pooled"], store(new, prec)])
        iq = imrope(rms_norm((x @ w["indexer.q_proj.weight"].T).reshape(T, nh, d), eps, w["indexer.q_norm.weight"]),
                    p4, **rope)
        scores = qsa_scores_at(iq, idxc["pooled"], kp, pos)
        cells = qsa_select_at(scores, hp["idx_top_k"], kp, pos)
        if cells_override is not None:
            cells = [cells_override.get(int(t), c) for t, c in zip(np.asarray(pos).tolist(), cells)]
        r.update(k_raw=k_raw, pooled_new=new, pool_range=(b0, b1), pooled=idxc["pooled"].copy(), iq=iq,
                 scores=scores)
        r["stored"]["k_raw"] = idxc["k_raw"][idxc["k_raw"].shape[0] - T:]
    if prec:
        r["modes"] = modes = row_modes(prec, pos)
        pre = attend_rows(q, kvc["k"], kvc["v"], cells, 1.0 / np.sqrt(D), modes).reshape(T, -1)
    else:
        pre = attention(q, kvc["k"], kvc["v"], cells, 1.0 / np.sqrt(D))
    gated = gate_attention(pre, gate)
    r.update(pregate=pre, gated=gated, cells=cells, out=gated @ w["attn_output.weight"].T)
    return r


def _cache_init_layer(cache, W, hp, il):
    w = layer_weights(W, il)
    if hp["recurrent"][il] and il not in cache["gdn"]:
        C, K = w["ssm_conv1d.weight"].shape
        S, Hv = hp["ssm_state"], hp["ssm_v_heads"]
        cache["gdn"][il] = dict(conv=np.zeros((K - 1, C)), state=np.zeros((Hv, S, S)))
    if il in hp["ple_layers"] and il not in cache["ple"]:
        K = w["ple_conv1d.weight"].shape[1]
        cache["ple"][il] = dict(hist=np.zeros(((K - 1) * hp["ple_ngram"], hp["hc"] * hp["n_embd"])))
    if not hp["recurrent"][il] and il not in cache["kv"]:
        Hkv, D = hp["n_head_kv"], hp["head_dim"]
        cache["kv"][il] = dict(k=np.zeros((0, Hkv, D)), v=np.zeros((0, Hkv, D)))
        cache["idx"][il] = dict(k_raw=np.zeros((0, hp["idx_dim"])), pooled=np.zeros((0, hp["idx_dim"])))


def forward_chunk(tokens, W, hp, cache, overrides=None, prec=None):
    """One ubatch of a single sequence that continues `cache` (new_cache() at the sequence start); updates cache in
    place. overrides: {("qsa", il): {position: cells}} replays a QSA choice the reference made among exactly tied
    pools, {("moe", il): {position: expert ids}} an expert choice among (near-)tied experts. prec: module
    docstring. Returns a dict like forward() for the chunk rows, plus the carried-in states (gdn conv_in/state_in,
    ple hist_in) and the absolute positions `pos`."""
    overrides = overrides or {}
    tokens = [int(t) for t in tokens]
    T, hc, eps = len(tokens), hp["hc"], hp["eps"]
    p0 = len(cache["tokens"])
    pos = np.arange(p0, p0 + T)
    res = hc_init(W["token_embd.weight"][tokens], hc)
    out = dict(pos=pos, hc_init=res, layers=[])
    emb = None
    if hp["ple_layers"]:
        rows = ple_rows_chunk(cache["tokens"], tokens, hp["ple_multipliers"], hp["ple_offsets"],
                              hp["ple_vocab_sizes"], hp["ple_ngram"], hp["ple_heads_per_ngram"], hp["ple_eos"])
        emb = ple_embed(W["per_layer_token_embd.weight"], rows)
        out.update(ple_rows=rows, ple_embd=emb)
    cache["tokens"] = cache["tokens"] + tokens
    cache.setdefault("modes", []).extend(row_modes(prec, pos))
    for il in range(hp["n_layer"]):
        _cache_init_layer(cache, W, hp, il)
        w = layer_weights(W, il)
        L = dict(res_in=res)
        if il in hp["ple_layers"]:
            L["ple"] = ple_forward_chunk(emb, res, w["ple_key.weight"], w["ple_value.weight"],
                                         w["ple_norm_key.weight"], w["ple_norm_query.weight"],
                                         w["ple_norm_conv.weight"], w["ple_conv1d.weight"], eps, hp["ple_ngram"],
                                         cache["ple"][il])
            res = L["ple"]["out"]
        L["mix_attn"] = m = hc_mix(res, w["hc_attn_norm.weight"], w["hc_attn_down.weight"], w["hc_attn_up.weight"],
                                   w["hc_attn_inject.weight"], eps)
        if hp["recurrent"][il]:
            L["gdn"] = blk = gdn_layer_chunk(m["mixed"], w, hp, cache["gdn"][il])
        else:
            L["attn"] = blk = attn_layer_chunk(m["mixed"], w, hp, hp["compress"][il] > 0, pos, cache["kv"][il],
                                               cache["idx"][il], overrides.get(("qsa", il)), prec)
        res = L["combine_attn"] = hc_combine(res, blk["out"], m["inject"])
        L["mix_ffn"] = m = hc_mix(res, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"], w["hc_ffn_up.weight"],
                                  w["hc_ffn_inject.weight"], eps)
        mo = overrides.get(("moe", il))
        L["moe"] = moe = moe_layer(m["mixed"], w, hp["n_expert_used"],
                                   {int(t) - p0: e for t, e in mo.items()} if mo else None)
        res = L["combine_ffn"] = hc_combine(res, moe["out"], m["inject"])
        out["layers"].append(L)
    head = hc_mix(res, W["output_hc_norm.weight"], W["output_hc_down.weight"], W["output_hc_up.weight"], None, eps)
    out.update(head=head, result_norm=head["mixed"], logits=head["mixed"] @ W["output.weight"].T)
    return out


def cache_truncate(cache, n_keep, W, hp, overrides=None, prec=None):
    """Roll the sequence back to its first n_keep positions. The state of a prefix depends on that prefix alone
    (causality), so the oracle rebuilds it from the kept tokens; llama.cpp restores it from its n_rs_seq snapshots
    (src/llama-memory-recurrent.cpp:199-215, G:1324-1345, src/models/delta-net-base.cpp:563-603). With prec, every
    kept row is recomputed in the attention mode it had (cache["modes"])."""
    kept = cache["tokens"][:n_keep]
    if prec:
        prec = dict(prec, modes={**{p: m for p, m in enumerate(cache.get("modes", [])[:n_keep])},
                                 **(prec.get("modes") or {})})
    fresh = new_cache()
    if kept:
        forward_chunk(kept, W, hp, fresh, overrides, prec)
    cache.clear()
    cache.update(fresh)
    return cache


def mtp_forward_chunk(tokens, h_prev, W, hp, cache, overrides=None, prec=None):
    """The MTP block (nextn_predict_layers = 1), graph_mtp G:529-618, on one ubatch of the MTP draft context.

    tokens: the ubatch's ids; h_prev (T, hc*n): the hc-wide residual paired with each token. common/speculative.cpp
    pairs token p with the target's h_nextn of position p-1 (lines 1511-1529; the first token of a call with the
    last row of the previous call, zeros at the start) and, while drafting, the drafted token with the MTP's own
    h_nextn of the previous step (lines 1700-1704). The target's h_nextn is the residual before its final mixer
    (G:501-506), i.e. forward()'s last combine_ffn. cache: new_cache() of the MTP context, whose memory holds the
    MTP layer alone (src/llama-model.cpp:2713-2720); positions continue from it. With il = n_layer:
        e      = rms_norm(tok_embd[token]) * enorm, repeated over the hc streams                    (G:584-586)
        h_norm = per-stream rms_norm(h_prev) * hnorm                                                (G:581-582)
        res    = eh_proj [e ; h_norm_s] per stream (concat on the feature axis)                     (G:588-589)
        attention mixer -> full-attention QSA layer il -> combine -> MoE mixer -> MoE -> combine    (G:591-598)
        h_nextn = res (the next draft step's h)                                                     (G:600-605)
        logits = output @ hc_mix(res, nextn.hc_head_*) (no inject)                                  (G:607-617)
    Returns a dict like forward_chunk's layer entry plus e_norm, h_norm, eh_proj, h_nextn, result_norm, logits."""
    overrides = overrides or {}
    il = hp["n_layer"]
    w = layer_weights(W, il)
    tokens = [int(t) for t in tokens]
    T, hc, n, eps = len(tokens), hp["hc"], hp["n_embd"], hp["eps"]
    p0 = len(cache["tokens"])
    pos = np.arange(p0, p0 + T)
    cache["tokens"] = cache["tokens"] + tokens
    cache.setdefault("modes", []).extend(row_modes(prec, pos))
    _cache_init_layer(cache, W, hp, il)
    e = rms_norm(W["token_embd.weight"][tokens], eps, w["nextn.enorm.weight"])
    h_norm = grouped_rms_norm(np.asarray(h_prev, dtype=np.float64).reshape(T, hc, n), w["nextn.hnorm.weight"], eps)
    e_rep = np.repeat(e[:, None, :], hc, axis=1)
    res = np.concatenate([e_rep, h_norm], axis=-1) @ w["nextn.eh_proj.weight"].T                     # (T, hc, n)
    out = dict(pos=pos, e_norm=e_rep, h_norm=h_norm, eh_proj=res)
    out["mix_attn"] = m = hc_mix(res, w["hc_attn_norm.weight"], w["hc_attn_down.weight"], w["hc_attn_up.weight"],
                                 w["hc_attn_inject.weight"], eps)
    out["attn"] = blk = attn_layer_chunk(m["mixed"], w, hp, hp["compress"][il] > 0, pos, cache["kv"][il],
                                         cache["idx"][il], overrides.get(("qsa", il)), prec)
    res = out["combine_attn"] = hc_combine(res, blk["out"], m["inject"])
    out["mix_ffn"] = m = hc_mix(res, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"], w["hc_ffn_up.weight"],
                                w["hc_ffn_inject.weight"], eps)
    out["moe"] = moe = moe_layer(m["mixed"], w, hp["n_expert_used"])
    res = out["combine_ffn"] = hc_combine(res, moe["out"], m["inject"])
    out["h_nextn"] = res.reshape(T, hc * n)
    head = hc_mix(res, w["nextn.hc_head_norm.weight"], w["nextn.hc_head_down.weight"], w["nextn.hc_head_up.weight"],
                  None, eps)
    out.update(head=head, result_norm=head["mixed"], logits=head["mixed"] @ W["output.weight"].T)
    return out


def forward_chunks(tokens, W, hp, sizes, overrides=None, prec=None):
    """Run a sequence as consecutive chunks of the given sizes; returns the list of forward_chunk results."""
    cache, outs, i = new_cache(), [], 0
    for n in sizes:
        outs.append(forward_chunk(tokens[i:i + n], W, hp, cache, overrides, prec))
        i += n
    assert i == len(tokens), (i, len(tokens))
    return outs
