"""Oracle: llama.cpp architecture "glm5-next" (GLM-5.3-Flash), the whole model, built from the op oracles
ref/mhc.py (mHC + Sinkhorn), ref/kda.py (KDA scan), ref/dsa.py (k-pool indexer + nope MLA) and ref/router.py
(sigmoid noaux_tc router, clamped SwiGLU), which tests/test_glm_tiny_parity.py checks op by op (G1).

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d. "G" = src/models/glm5-next.cpp, "M" =
src/llama-memory-hybrid-idx.cpp, "LG" = src/llama-graph.cpp, all at that commit; every cited line was printed from
the pinned checkout. The trunk graph (G:549-682) is the same at the R1 pin 79e2e74 (only the DSA decode path and the
k-pool padding differ there, see "pin differences" below).

Residual: hc = 4 parallel streams per token, res (T, hc, n). hc_init repeats the token embedding (G:579-581).
Per layer (G:585-651):
    x, post, comb = hc_pre(res, hc_attn_*)              (G:592, build_hc_pre G:454-510; ref/mhc.py hc_pre)
    x = rms_norm(x) * attn_norm                          (G:595)
    y = KDA layer (head_count_kv = 0) or DSA layer       (G:599-604)
    res = hc_post(y, res, post, comb)                    (G:606, build_hc_post G:512-547) -> "hc_attn_post"
    x, post, comb = hc_pre(res, hc_ffn_*); x = rms_norm(x) * ffn_norm                       (G:610-614)
    y = dense SwiGLU (il < leading_dense_block_count) or MoE + shared expert              (G:616-646)
    res = hc_post(y, res, post, comb)                    (G:648-650) -> "l_out"
Head (G:653-679): hc_head = mean of the streams (build_hc_mean G:370-378), result_norm = rms_norm(hc_head) *
    output_norm (G:665, also "h_nextn" G:668), logits = output @ result_norm (G:677).

KDA layer, build_kda_layer (G:686-762), H heads of D, d_inner = H*D:
    q, k, v = silu(ssm_conv(conv_hist | W_{q,k,v} x))   per channel, kernel d_conv (glm5_conv1d G:196-235)
    g = lower_bound * sigmoid(-((f_b(f_a x) + dt_bias) * ssm_a))  per channel (G:708-718; ssm_a stores -exp(A_log))
        (no lower bound: softplus(f_b(f_a x) + dt_bias) * ssm_a, G:719-722)
    beta = sigmoid(W_beta x)                             (G:728-730)
    o = gated delta rule (KDA gate), q, k l2-normalised with eps 1e-6 (G:737-742; ref/kda.py gated_delta_rule)
    y = W_o (rms_norm(o per head) * ssm_norm * sigmoid(g_b(g_a x)))                        (G:747-758)

DSA layer, build_dsa_layer (G:923-1013) + build_kpool_select (G:766-919); ref/dsa.py:
    qr = rms_norm(W_q_a x) * q_a_norm; q = W_q_b qr (H x k_mla, nope only)                 (G:937-942)
    latent = rms_norm(W_kv_a x) * kv_a_norm, the MLA cache row of the token               (G:944-947, 968)
    q_abs[h] = W_k_b[h] q[h]                                                               (G:950-953)
    indexer: ik = layer_norm(W_ik x) * g + b (eps = attention.layer_norm_epsilon, absent -> 0),
             ig = W_gate x, cached per cell (G:783-794); pooled[b] = sum_j softmax_j(ig + ape)[j] ik[4b+j]
             computed by the ubatch that completes pool b (G:799-819); iq = W_iq_b qr (G:778);
             w = W_proj x / sqrt(dim * n_idx) (G:836-837); score[t,b] = sum_h w[t,h] relu(iq[t,h] . pooled[b]),
             -inf unless pool b ends at or before t (G:845-854; M:978-1011)
    cells = members of the top min(n_pool, top_k/kpool) pools + the incomplete tail kpool*floor((t+1)/kpool)..t
            (G:858-878; tail M:1013-1037); attention softmax(q_abs . latent / sqrt(k_mla)) over the cells,
            y = W_o concat_h W_v_b[h] (sum p latent)                                       (G:970-1010)

FFN: dense layer = W_down swiglu_clamp(W_gate x, W_up x, swiglu_clamp_shexp[il]) (LG:1850-1858, GLM5_NEXT clamps
before silu: ref/router.py swiglu_clamp); MoE = build_moe_ffn (sigmoid gating, exp_probs_b selection bias, norm_w,
expert_weights_scale; clamp swiglu_clamp_exp[il], LG:2245-2251; ref/router.py route) + the shared expert as the
dense layer (G:624-645).

Precision (N10, gap 3 of docs/reassessment/CORRECTNESS.md): the DSA functions take prec=None (the reference
contract: f32 caches, flash attention off) or ref/fattn.py precision(kv, fa). With an f16 cache the oracle rounds what
llama.cpp stores in f16: the MLA latent (cpy_k into the K-only MLA cache, G:966-968), the indexer key and gate rows
(cpy_k of key | gate | pooled into the indexer cache, G:790-794, which llama_memory_hybrid_idx creates with the
attention cache types, M:69-72, src/llama-model.cpp:2474-2477) and the pooled key (scattered and gathered back when
cache_safe, G:817-824); the pooling reads the cached key | gate rows (G:799-800). Attention: the mask path runs
build_attn_mha (G:998-1005), i.e. flash attention when -fa on (ref/fattn.py modes); the ec7630a gather path (n_tokens
<= 16) gathers the cached latents with ggml_get_rows (f16 -> f32) and runs f32 mul_mat / soft_max (G:971-996), so only
the latent rounding reaches it ("exact" mode over the f16 latents; the tests force it per row with prec["modes"]).
KDA states stay f32 (GGML_TYPE_F32, src/llama-model.cpp:2483-2484).

Shapes are numpy-natural (token-major); weights are the GGUF arrays as numpy reads them (ggml ne reversed).
Everything runs in float64; cast at the boundary when comparing with a kernel.
"""
import numpy as np

from ref.dsa import indexer_pools, indexer_scores, layer_norm, mla_attention, select_cells
from ref.fattn import row_modes, store
from ref.kda import gated_delta_rule
from ref.mhc import hc_init, hc_mean, hc_post, hc_pre
from ref.router import route, swiglu_clamp

L2_EPS = 1e-6          # G:738, build_gdn_l2_norm
CLAMP_EPS = 1e-6       # LG:1854 / 2248: a limit <= 1e-6 means no clamp


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * sigmoid(x)


def rms_norm(x, eps, w=None):
    """ggml_rms_norm over the last axis, times w if given."""
    x = np.asarray(x, dtype=np.float64)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------- hyper-parameters from a GGUF

def hparams(fields):
    """Hyper-parameters from GGUFReader fields ({name: field}); keys as G:7-59 reads them."""
    def raw(name):
        f = fields[name]
        return [f.parts[i] for i in f.data]

    arch = bytes(raw("general.architecture")[0]).decode()

    def get(key, default=None, array=False):
        name = f"{arch}.{key}"
        if name not in fields:
            if default is None:
                raise KeyError(name)
            return default
        vals = [p.tolist()[0] if hasattr(p, "tolist") else p for p in raw(name)]
        return vals if array or len(vals) > 1 else vals[0]

    n_all = int(get("block_count"))                                        # counts the MTP block, if any
    n_nextn = int(get("nextn_predict_layers", 0))
    n_layer = n_all - n_nextn                                               # n_layer() = n_layer_all - n_layer_nextn
    hkv = get("attention.head_count_kv", array=True)
    hkv = hkv * n_all if len(hkv) == 1 else hkv
    clamp_exp = get("swiglu_clamp_exp", [0.0] * n_all, array=True)
    clamp_exp = clamp_exp * n_all if len(clamp_exp) == 1 else clamp_exp
    clamp_sh = get("swiglu_clamp_shexp", clamp_exp, array=True)            # G:35-37: defaults to the expert limit
    clamp_sh = clamp_sh * n_all if len(clamp_sh) == 1 else clamp_sh
    return dict(
        arch=arch, n_layer=n_layer, n_nextn=n_nextn, n_embd=int(get("embedding_length")),
        n_head=int(get("attention.head_count")),
        kda=[int(h) == 0 for h in hkv[:n_all]],                             # G:21-23 is_recr = head_count_kv == 0
        rms_eps=float(get("attention.layer_norm_rms_epsilon")),
        norm_eps=float(get("attention.layer_norm_epsilon", 0.0)),           # G:9, optional; the tiny model has none
        kda_head_dim=int(get("kda.head_dim")), d_conv=int(get("ssm.conv_kernel")),
        kda_lower_bound=float(get("kda.gate_lower_bound", -np.inf)),
        k_mla=int(get("attention.key_length_mla")), v_mla=int(get("attention.value_length_mla")),
        kv_lora=int(get("attention.kv_lora_rank")), q_lora=int(get("attention.q_lora_rank")),
        dense_lead=int(get("leading_dense_block_count", 0)), n_expert_used=int(get("expert_used_count")),
        w_scale=float(get("expert_weights_scale", 0.0)), norm_w=bool(get("expert_weights_norm", False)),
        clamp_exp=[float(v) for v in clamp_exp[:n_all]], clamp_shexp=[float(v) for v in clamp_sh[:n_all]],
        idx_heads=int(get("attention.indexer.head_count")), idx_dim=int(get("attention.indexer.key_length")),
        idx_top_k=int(get("attention.indexer.top_k")), kpool=int(get("attention.indexer.kpool")),
        select_tail=bool(get("attention.indexer.kpool_select_tail", True)),
        hc=int(get("hyper_connection.count")), sinkhorn_iters=int(get("hyper_connection.sinkhorn_iterations")),
        hc_eps=float(get("hyper_connection.epsilon")),
    )


def layer_weights(W, il):
    pre = f"blk.{il}."
    return {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}


# ---------------------------------------------------------------- FFN

def swiglu(gate, up, limit):
    """GLM5_NEXT activation: swiglu_clamp when the layer's limit > 1e-6 (LG:1853-1858, 2247-2251), else
    silu(gate) * up (ggml_swiglu_split)."""
    if limit > CLAMP_EPS:
        return swiglu_clamp(gate, up, limit)
    return silu(gate) * np.asarray(up, dtype=np.float64)


def ffn_layer(x, w, hp, il, ids_override=None):
    """x (T, n) after ffn_norm. Dense SwiGLU before leading_dense_block_count, else routed experts + shared
    expert (G:616-646). Returns dict with out and the MoE intermediates (sel: the selection scores, sigmoid(logits)
    + exp_probs_b). ids_override {row: expert ids} replays the reference's choice among (near-)tied experts; those
    rows are weighted like route() weights its own choice."""
    lim_sh = hp["clamp_shexp"][il]
    if il < hp["dense_lead"]:
        act = swiglu(x @ w["ffn_gate.weight"].T, x @ w["ffn_up.weight"].T, lim_sh)
        return dict(out=act @ w["ffn_down.weight"].T)
    T, k = x.shape[0], hp["n_expert_used"]
    logits = x @ w["ffn_gate_inp.weight"].T
    ids, wts = route(logits, w["exp_probs_b.bias"], k, w_scale=hp["w_scale"] or 1.0, norm=hp["norm_w"])
    sel = sigmoid(logits) + np.asarray(w["exp_probs_b.bias"], dtype=np.float64)[None]
    if ids_override:
        ids, wts = ids.copy(), wts.copy()
        probs = sigmoid(logits)
        for r, e in ids_override.items():
            ids[r] = np.asarray(e)
            wr = probs[r, ids[r]]
            if hp["norm_w"]:
                wr = wr / max(wr.sum(), 6.103515625e-5)
            wts[r] = wr * (hp["w_scale"] or 1.0)
    ge, ue, de = w["ffn_gate_exps.weight"], w["ffn_up_exps.weight"], w["ffn_down_exps.weight"]
    gate = np.stack([[ge[e] @ x[t] for e in ids[t]] for t in range(T)])          # (T, k, F)
    up = np.stack([[ue[e] @ x[t] for e in ids[t]] for t in range(T)])
    act = swiglu(gate, up, hp["clamp_exp"][il])
    down = np.stack([[de[e] @ act[t, j] for j, e in enumerate(ids[t])] for t in range(T)])   # (T, k, n)
    moe = (down * wts[:, :, None]).sum(axis=1)
    sh = swiglu(x @ w["ffn_gate_shexp.weight"].T, x @ w["ffn_up_shexp.weight"].T, lim_sh) @ w["ffn_down_shexp.weight"].T
    return dict(logits=logits, sel=sel, ids=ids, weights=wts, gate=gate, up=up, act=act, moe_out=moe, shexp=sh,
                out=moe + sh)


# ---------------------------------------------------------------- KDA

def conv_hist(x, w, hist):
    """ggml_ssm_conv over [hist | x] (G:206-213, 229-232): hist (K-1, C) = the last K-1 projected rows of the
    earlier ubatches (zeros at the sequence start). Returns (out (T, C) before silu, new hist (K-1, C))."""
    x = np.asarray(x, dtype=np.float64)
    K = w.shape[1]
    pad = np.vstack([hist, x])
    out = np.stack([np.sum(pad[t:t + K].T * w, axis=1) for t in range(x.shape[0])])
    return out, pad[pad.shape[0] - (K - 1):]


def kda_gate(x, w, hp):
    """Per-channel log decay (T, H, D) (G:708-723)."""
    T, H, D = x.shape[0], hp["n_head"], hp["kda_head_dim"]
    raw = (x @ w["ssm_f_a.weight"].T) @ w["ssm_f_b.weight"].T + w["ssm_dt.bias"]
    A = np.asarray(w["ssm_a"], dtype=np.float64)[None, :, None]
    lb = hp["kda_lower_bound"]
    if np.isfinite(lb):
        return lb * sigmoid(-(raw.reshape(T, H, D) * A))
    return np.logaddexp(0.0, raw).reshape(T, H, D) * A


def kda_layer(x, w, hp, st):
    """KDA layer on one ubatch; st = dict(conv={q,k,v: (K-1, d_inner)}, state=(H, D, D)) carried in and replaced on
    return (st None = sequence start, nothing carried out)."""
    T, H, D = x.shape[0], hp["n_head"], hp["kda_head_dim"]
    d_inner = H * D
    K = hp["d_conv"]
    if st is None:
        st = dict(conv={c: np.zeros((K - 1, d_inner)) for c in "qkv"}, state=np.zeros((H, D, D)))
    r = dict(conv_in={c: st["conv"][c] for c in "qkv"}, state_in=st["state"])
    conv, hist = {}, {}
    for c in "qkv":
        proj = x @ w[f"attn_{c}.weight"].T
        cw = np.asarray(w[f"ssm_conv1d_{c}.weight"], dtype=np.float64).reshape(d_inner, -1)
        raw, hist[c] = conv_hist(proj, cw, st["conv"][c])
        conv[c] = silu(raw).reshape(T, H, D)
        r[f"proj_{c}"] = proj
    g = kda_gate(x, w, hp)
    beta = sigmoid(x @ w["ssm_beta.weight"].T)
    scan, state = gated_delta_rule(conv["q"], conv["k"], conv["v"], g, beta, state=st["state"], normalize_qk=True,
                                   eps=L2_EPS)
    st["conv"], st["state"] = hist, state
    g2 = ((x @ w["ssm_g_a.weight"].T) @ w["ssm_g_b.weight"].T).reshape(T, H, D)
    normed = rms_norm(scan, hp["rms_eps"], w["ssm_norm.weight"])
    gated = (normed * sigmoid(g2)).reshape(T, d_inner)
    r.update(q=conv["q"], k=conv["k"], v=conv["v"], g=g, beta=beta, scan=scan, state=state, g2=g2, normed=normed,
             out=gated @ w["attn_output.weight"].T)
    return r


# ---------------------------------------------------------------- DSA

def tail_cells(t, kpool, select_tail=True):
    """Incomplete-pool cells token t always keeps: kpool*floor((t+1)/kpool) .. t (M:1013-1037 n_tail)."""
    return list(range(((t + 1) // kpool) * kpool, t + 1)) if select_tail else []


def scores_at(iq, wts, pooled, kpool, pos):
    """ref/dsa.py indexer_scores for tokens at absolute positions pos: pool b visible once (b+1)*kpool <= pos+1."""
    s = np.einsum("th,thb->tb", wts, np.maximum(np.einsum("thd,bd->thb", iq, pooled), 0.0))
    visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)
    return np.where(visible, s, -np.inf)


def select_at(scores, top_k, kpool, pos, select_tail=True):
    """ref/dsa.py select_cells for tokens at absolute positions pos (top visible pools, stable order on ties)."""
    n_top = top_k // kpool
    out = []
    for r, t in enumerate(np.asarray(pos).tolist()):
        vis = np.where(np.isfinite(scores[r]))[0]
        top = vis[np.argsort(-scores[r, vis], kind="stable")[:n_top]]
        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]
        out.append(sorted(cells + tail_cells(t, kpool, select_tail)))
    return out


def tie_variant(scores_row, cells, t, top_k, kpool, select_tail=True, rtol=1e-6):
    """Whether `cells` is another valid selection for token t: the tail, whole visible pools, the budgeted pool
    count, and no left-out visible pool scoring above a chosen one beyond f32 rounding (ggml_top_k's order among
    equal scores is unspecified, ggml/src/ggml-cpu/ops.cpp std::partial_sort). Returns (ok, reason)."""
    cells = sorted(int(c) for c in cells)
    tail = tail_cells(t, kpool, select_tail)
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


def dsa_project(x, w, hp):
    """Per-token parts of the DSA layer that need no cache."""
    T, H = x.shape[0], hp["n_head"]
    nh, d = hp["idx_heads"], hp["idx_dim"]
    qr = rms_norm(x @ w["attn_q_a.weight"].T, hp["rms_eps"], w["attn_q_a_norm.weight"])
    q = (qr @ w["attn_q_b.weight"].T).reshape(T, H, -1)
    latent = rms_norm(x @ w["attn_kv_a_mqa.weight"].T, hp["rms_eps"], w["attn_kv_a_norm.weight"])
    q_abs = np.einsum("hld,thd->thl", np.asarray(w["attn_k_b.weight"], dtype=np.float64), q)
    ik = layer_norm(x @ w["indexer.attn_k.weight"].T, w["indexer.k_norm.weight"], w["indexer.k_norm.bias"],
                    hp["norm_eps"])
    ig = x @ w["indexer_compressor_gate.weight"].T
    iq = (qr @ w["indexer.attn_q_b.weight"].T).reshape(T, nh, d)
    wts = (x @ w["indexer.proj.weight"].T) / np.sqrt(d * nh)
    return dict(q_resid=qr, q=q, latent=latent, q_abs=q_abs, ik=ik, ig=ig, iq=iq, wts=wts)


def dsa_layer_full(x, w, hp, cells_override=None, prec=None):
    """DSA layer over a whole sequence from position 0 with the ref/dsa.py prefill oracles. prec: module docstring
    (the rows are one ubatch unless prec["modes"] says otherwise)."""
    r = dsa_project(x, w, hp)
    kp = hp["kpool"]
    pooled_raw = indexer_pools(store(r["ik"], prec), store(r["ig"], prec), w["indexer_compressor_ape.weight"], kp)
    pooled = store(pooled_raw, prec)
    scores = indexer_scores(r["iq"], r["wts"], pooled, kp)
    cells = select_cells(scores, hp["idx_top_k"], kp) if hp["select_tail"] else \
        select_at(scores, hp["idx_top_k"], kp, np.arange(x.shape[0]), False)
    if isinstance(cells_override, dict):          # {position: cells} for single tokens
        cells = [cells_override.get(t, c) for t, c in enumerate(cells)]
    elif cells_override is not None:
        cells = cells_override
    modes = row_modes(prec, np.arange(x.shape[0])) if prec else None
    kqv = mla_attention(r["q_abs"], store(r["latent"], prec), cells,
                        np.asarray(w["attn_v_b.weight"], dtype=np.float64), hp["k_mla"], modes)
    r.update(pooled_raw=pooled_raw, pooled=pooled, scores=scores, cells=cells, kqv=kqv, modes=modes,
             stored={key: store(r[key], prec) for key in ("latent", "ik", "ig")},
             out=kqv @ w["attn_output.weight"].T)
    return r


def dsa_layer_chunk(x, w, hp, pos, st, cells_override=None, prec=None):
    """DSA layer on one ubatch at absolute positions pos (cell = position for one sequence). st = dict(latent,
    ik, ig, pooled) of the cached cells, extended in place: the MLA latent cache (G:968), the indexer key | gate rows
    (G:790-794) and the pooled key of every complete pool, pooled once by the ubatch that completes it and read
    back afterwards (G:799-832; M:673-679); with prec, the values as the cache stores them (module docstring).
    cells_override: {position: cells} replays a reference tie choice."""
    r = dsa_project(x, w, hp)
    kp = hp["kpool"]
    for key in ("latent", "ik", "ig"):
        st[key] = np.concatenate([st[key], store(r[key], prec)])
    b0, b1 = st["pooled"].shape[0], st["ik"].shape[0] // kp            # pools this ubatch completes
    new = indexer_pools(st["ik"][b0 * kp:b1 * kp], st["ig"][b0 * kp:b1 * kp], w["indexer_compressor_ape.weight"], kp)
    st["pooled"] = np.concatenate([st["pooled"], store(new.reshape(-1, hp["idx_dim"]), prec)])
    scores = scores_at(r["iq"], r["wts"], st["pooled"], kp, pos)
    cells = select_at(scores, hp["idx_top_k"], kp, pos, hp["select_tail"])
    if cells_override:
        cells = [cells_override.get(int(t), c) for t, c in zip(np.asarray(pos).tolist(), cells)]
    modes = row_modes(prec, pos) if prec else None
    kqv = mla_attention(r["q_abs"], st["latent"], cells, np.asarray(w["attn_v_b.weight"], dtype=np.float64),
                        hp["k_mla"], modes)
    T = x.shape[0]
    r.update(pooled_new=new.reshape(-1, hp["idx_dim"]), pool_range=(b0, b1), pooled=st["pooled"].copy(),
             scores=scores, cells=cells, kqv=kqv, modes=modes, out=kqv @ w["attn_output.weight"].T,
             stored={key: st[key][st[key].shape[0] - T:] for key in ("latent", "ik", "ig")})     # as cached
    return r


# ---------------------------------------------------------------- whole model

def _mhc(res, w, hp, kind):
    return hc_pre(res, w[f"hc_{kind}_fn.weight"], w[f"hc_{kind}_scale.weight"], w[f"hc_{kind}_base.weight"],
                  rms_eps=hp["rms_eps"], hc_eps=hp["hc_eps"], iters=hp["sinkhorn_iters"])


def _head(res, W, hp, out):
    head = hc_mean(res)
    norm = rms_norm(head, hp["rms_eps"], W["output_norm.weight"])
    out.update(hc_head=head, result_norm=norm, logits=norm @ W["output.weight"].T)
    return out


def _layer(res, w, hp, il, mixer, moe_ids=None):
    L = dict(res_in=res)
    x, post, comb = _mhc(res, w, hp, "attn")
    L.update(attn_pre=x, attn_post_w=post, attn_comb=comb)
    L["attn_norm"] = xn = rms_norm(x, hp["rms_eps"], w["attn_norm.weight"])
    L["kda" if hp["kda"][il] else "dsa"] = blk = mixer(xn)
    res = L["hc_attn_post"] = hc_post(blk["out"], res, post, comb)
    x, post, comb = _mhc(res, w, hp, "ffn")
    L.update(ffn_pre=x, ffn_post_w=post, ffn_comb=comb)
    L["ffn_norm"] = xn = rms_norm(x, hp["rms_eps"], w["ffn_norm.weight"])
    L["ffn"] = f = ffn_layer(xn, w, hp, il, moe_ids)
    res = L["l_out"] = hc_post(f["out"], res, post, comb)
    return res, L


def forward(tokens, W, hp, overrides=None, prec=None):
    """Prefill of one sequence from position 0 with zero recurrent state. W: GGUF weights as float64 arrays.
    overrides: {("dsa", il): cells per token, or {position: cells}} replays a selection the reference made among
    tied pools; {("moe", il): {position: expert ids}} an expert choice among (near-)tied experts.
    prec: KV cache type and flash attention (module docstring; None = f32, flash attention off).
    Returns dict with per-layer intermediates (layers[il]) and logits (T, n_vocab)."""
    overrides = overrides or {}
    tokens = np.asarray(tokens)
    res = hc_init(W["token_embd.weight"][tokens], hp["hc"])
    out = dict(hc_init=res, layers=[])
    for il in range(hp["n_layer"]):
        w = layer_weights(W, il)
        if hp["kda"][il]:
            mixer = lambda xn: kda_layer(xn, w, hp, None)                    # noqa: E731
        else:
            mixer = lambda xn: dsa_layer_full(xn, w, hp, overrides.get(("dsa", il)), prec)  # noqa: E731
        res, L = _layer(res, w, hp, il, mixer, overrides.get(("moe", il)))
        out["layers"].append(L)
    return _head(res, W, hp, out)


# ---------------------------------------------------------------- incremental path: several ubatches, decode, rollback
#
# llama.cpp runs a long prompt as several ubatches and generation as one llama_process call per token; every ubatch
# is its own graph that reads what earlier ones left in the hybrid memory (src/llama-memory-hybrid-idx.cpp). What
# this oracle carries between chunks (one sequence, cell = position), and where llama.cpp keeps it:
#   tokens         ids of the cached cells (positions continue from len(tokens))
#   kda[il].conv   last d_conv-1 rows of each of the q, k, v projections (recurrent r_l row: glm5_conv1d concatenates
#                  it before the ubatch and writes the new tail back, G:206-227; n_rs_seq + 1 snapshot groups)
#   kda[il].state  KDA state (H, D, D) (s_l row; written back as "new_state" when n_rs_seq = 0,
#                  src/models/delta-net-base.cpp:546-560, else as n_rs_seq + 1 snapshots, :563-603)
#   dsa[il].latent the MLA latent of every cell (K-only MLA cache, G:968)
#   dsa[il].ik/ig  the indexer key | gate row of every cell (indexer cache, G:790-794)
#   dsa[il].pooled the pooled key of every complete pool, pooled by the ubatch that completes it, scattered into the
#                  rep (last member) row and gathered from there afterwards (G:799-832; M:673-679, 912-916)
#   mHC streams are per token: nothing is carried.
# A rollback (llama_memory_seq_rm of the last n positions with n_rs_seq snapshots) must leave the state of the kept
# prefix: cache_truncate.

def new_cache():
    """Empty sequence memory for forward_chunk. modes: the attention mode each cached position was computed with
    (a rollback rebuilds the kept prefix with the modes of its original ubatches)."""
    return dict(tokens=[], kda={}, dsa={}, modes=[])


def _cache_init_layer(cache, hp, il):
    if hp["kda"][il]:
        if il not in cache["kda"]:
            H, D, K = hp["n_head"], hp["kda_head_dim"], hp["d_conv"]
            cache["kda"][il] = dict(conv={c: np.zeros((K - 1, H * D)) for c in "qkv"}, state=np.zeros((H, D, D)))
    elif il not in cache["dsa"]:
        d = hp["idx_dim"]
        cache["dsa"][il] = dict(latent=np.zeros((0, hp["kv_lora"])), ik=np.zeros((0, d)), ig=np.zeros((0, d)),
                                pooled=np.zeros((0, d)))


def forward_chunk(tokens, W, hp, cache, overrides=None, prec=None):
    """One ubatch of a single sequence that continues `cache` (new_cache() at the sequence start); updates cache in
    place. overrides: {("dsa", il): {position: cells}}, {("moe", il): {position: expert ids}} (replays of the
    reference's choice among tied candidates). Returns a dict like forward() for the chunk rows, with the
    carried-in KDA state (layers[il]["kda"]["conv_in"/"state_in"]) and the absolute positions `pos`."""
    overrides = overrides or {}
    tokens = [int(t) for t in tokens]
    p0 = len(cache["tokens"])
    pos = np.arange(p0, p0 + len(tokens))
    cache["tokens"] = cache["tokens"] + tokens
    cache.setdefault("modes", []).extend(row_modes(prec, pos))
    res = hc_init(W["token_embd.weight"][tokens], hp["hc"])
    out = dict(pos=pos, hc_init=res, layers=[])
    for il in range(hp["n_layer"]):
        _cache_init_layer(cache, hp, il)
        w = layer_weights(W, il)
        if hp["kda"][il]:
            mixer = lambda xn: kda_layer(xn, w, hp, cache["kda"][il])      # noqa: E731
        else:
            mixer = lambda xn: dsa_layer_chunk(xn, w, hp, pos, cache["dsa"][il],  # noqa: E731
                                               overrides.get(("dsa", il)), prec)
        moe = overrides.get(("moe", il))
        res, L = _layer(res, w, hp, il, mixer, {int(t) - p0: e for t, e in moe.items()} if moe else None)
        out["layers"].append(L)
    return _head(res, W, hp, out)


def cache_truncate(cache, n_keep, W, hp, overrides=None, prec=None):
    """Roll the sequence back to its first n_keep positions. The state of a prefix depends on that prefix alone
    (causality), so the oracle rebuilds it from the kept tokens; llama.cpp restores the KDA state from its n_rs_seq
    snapshots (G:217-227, src/models/delta-net-base.cpp:563-603) and drops the cells (and the pools ending) at or
    after n_keep from the MLA / indexer caches (M:681-731: an edit stales the pools from the edited position on).
    With prec, every kept row is recomputed in the attention mode it had (cache["modes"])."""
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


def forward_chunks(tokens, W, hp, sizes, overrides=None, prec=None):
    """Run a sequence as consecutive chunks of the given sizes; returns the list of forward_chunk results."""
    cache, outs, i = new_cache(), [], 0
    for n in sizes:
        outs.append(forward_chunk(tokens[i:i + n], W, hp, cache, overrides, prec))
        i += n
    assert i == len(tokens), (i, len(tokens))
    return outs


# ---------------------------------------------------------------- MTP block (nextn_predict_layers = 1)

def mtp_forward_chunk(tokens, h_prev, W, hp, cache, overrides=None, prec=None):
    """The MTP block, graph_mtp of src/models/glm5-next.cpp @79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 ("M79" below;
    ec7630a has none: G:189-191 throws), on one ubatch of the MTP draft context. No mHC: a plain residual.

    tokens: the ubatch's ids; h_prev (T, n_embd): the row paired with each token. common/speculative.cpp pairs token p
    with the target's h_nextn of position p-1 (the first token of a call with the last row of the previous call,
    zeros at the start) and, while drafting, the drafted token with the MTP's own h_nextn of the previous step
    (tools/dump/dump_tensors.cpp DUMP_MTP). The target's h_nextn is its post-norm hidden state
    rms_norm(hc_mean(res)) * output_norm (G:662-669; M79:780-787), i.e. forward()'s result_norm.
    cache: new_cache() of the MTP context, whose memory holds the MTP layer alone (src/llama-model.cpp:2503-2512
    @79e2e74: attention and indexer filters il >= n_layer, no recurrent layer). With il = n_layer:
        e       = rms_norm(embed_tokens or tok_embd [token]) * enorm                           (M79:583-592)
        h       = rms_norm(h_prev) * hnorm                                                        (M79:594-595)
        x       = eh_proj [e ; h]                                                                 (M79:597-598)
        y       = DSA layer il on rms_norm(x) * attn_norm, over its own MLA + indexer cache       (M79:600-607)
        f       = y + x; out = MoE(rms_norm(f) * ffn_norm) + shared expert + f                    (M79:615-645)
        h_nextn = rms_norm(out) * (shared_head_norm or output_norm), the next draft step's h     (M79:647-651)
        logits  = (shared_head_head or output) @ h_nextn                                         (M79:658-662)
    Returns a dict with those intermediates and logits."""
    overrides = overrides or {}
    il = hp["n_layer"]
    w = layer_weights(W, il)
    tokens = [int(t) for t in tokens]
    eps = hp["rms_eps"]
    p0 = len(cache["tokens"])
    pos = np.arange(p0, p0 + len(tokens))
    cache["tokens"] = cache["tokens"] + tokens
    cache.setdefault("modes", []).extend(row_modes(prec, pos))
    _cache_init_layer(cache, hp, il)
    emb = w.get("nextn.embed_tokens.weight", W["token_embd.weight"])
    e = rms_norm(emb[tokens], eps, w["nextn.enorm.weight"])
    h = rms_norm(h_prev, eps, w["nextn.hnorm.weight"])
    x = np.concatenate([e, h], axis=-1) @ w["nextn.eh_proj.weight"].T
    out = dict(pos=pos, e_norm=e, h_norm=h, eh_proj=x)
    out["attn_norm"] = an = rms_norm(x, eps, w["attn_norm.weight"])
    out["dsa"] = blk = dsa_layer_chunk(an, w, hp, pos, cache["dsa"][il], overrides.get(("dsa", il)), prec)
    out["ffn_inp"] = f = blk["out"] + x
    out["ffn_norm"] = fn = rms_norm(f, eps, w["ffn_norm.weight"])
    out["ffn"] = ffn = ffn_layer(fn, w, hp, il)
    out["post_ffn"] = post = ffn["out"] + f
    out["h_nextn"] = hn = rms_norm(post, eps, w.get("nextn.shared_head_norm.weight", W["output_norm.weight"]))
    head = w.get("nextn.shared_head_head.weight", W["output.weight"])
    out["logits"] = hn @ head.T
    return out
