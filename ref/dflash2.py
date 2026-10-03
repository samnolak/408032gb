"""numpy oracle for the DFlash2 drafter primitives (ADR-011, design docs/design/G8-dflash2.md).

Reference: vLLM @bc21cba, vendored byte-exact in third_party/vllm-dflash2 (Apache-2.0):
    grouped convolution     vllm/model_executor/models/qwen3_dflash2.py  _grouped_conv (:144), DFlashGroupedConv (:170)
    candidate edge scores   same file, _score_edges (:290), CandidateSelector (:314)
    candidate walk          vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py  _selector_walk_kernel (:16)
    context K/V             vllm/model_executor/models/qwen3_dflash.py  combine_hidden_states (:845),
                            _project_context_kv (:580), _normalize_context_k (:620)

What is checked against executed reference code (tests/test_dflash2_ref.py, torch on CPU): grouped_conv and score_edges.
The walk is a Triton GPU kernel and the context K/V path uses vLLM's fused ops; both are transcribed by reading and are
PROVISIONAL until a GPU run. Everything is float64 here.
"""
import numpy as np


def rms_norm(x, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight


def grouped_conv(hidden, delta, base, block_size, group_size):
    """hidden [n, H]: rows are consecutive block positions, row r is position r % block_size.
    delta [n, taps, G] (one coefficient per group, input-dependent), base [taps, H] (learned, per element), G = H / group_size.
    out[r] = sum over tap <= position(r) of (base[tap] + delta[r, tap] spread over its group) * hidden[r - tap]:
    a causal filter inside the block that never reaches into the previous block."""
    hidden = np.asarray(hidden, dtype=np.float64)
    n, h = hidden.shape
    taps = base.shape[0]
    groups = h // group_size
    coef = np.asarray(base, dtype=np.float64).reshape(1, taps, groups, group_size) + np.asarray(delta, dtype=np.float64)[..., None]
    blocks = hidden.reshape(n, groups, group_size)
    pos = np.arange(n) % block_size
    out = coef[:, 0] * blocks
    for tap in range(1, taps):
        shifted = np.zeros_like(blocks)
        shifted[tap:] = blocks[:-tap]
        out += coef[:, tap] * shifted * (pos >= tap)[:, None, None]
    return out.reshape(n, h)


def conv_prepare(hidden, kernel_projection, base_kernel, block_size, group_size):
    """DFlashGroupedConv.prepare: one projection of the layer input yields the coefficients of both convolutions;
    the first is applied now, the second is returned for conv_finish. kernel_projection [2*taps*G, H], base_kernel [2, taps, H]."""
    hidden = np.asarray(hidden, dtype=np.float64)
    taps = base_kernel.shape[1]
    groups = hidden.shape[1] // group_size
    coef = (hidden @ np.asarray(kernel_projection, dtype=np.float64).T).reshape(hidden.shape[0], 2, taps, groups)
    return grouped_conv(hidden, coef[:, 0], base_kernel[0], block_size, group_size), coef[:, 1]


def conv_finish(hidden, coef, base_kernel, block_size, group_size):
    return grouped_conv(hidden, coef, base_kernel[1], block_size, group_size)


def top_k_candidates(logits, k):
    """[.., V] -> ids [.., k] and their logits, highest first (ties: lowest id first)."""
    logits = np.asarray(logits, dtype=np.float64)
    ids = np.argsort(-logits, axis=-1, kind="stable")[..., :k]
    return ids, np.take_along_axis(logits, ids, axis=-1)


def score_edges(predecessor_codebook, successor_codebook, candidate_ids, unary_logits, hidden, anchor_token_ids):
    """scores [b, L, k, k]: scores[b, l, p, c] = unary[b, l, c] + (P[pred(l, p)] * hidden[b, l]) . S[cand(l, c)],
    where pred(0, p) is the anchor token for every p and pred(l, p) = candidate_ids[b, l-1, p] otherwise.
    hidden [b, L, rank] is the selector's projection of the drafter output; codebooks are [V, rank]."""
    cand = np.asarray(candidate_ids)
    b, steps, k = cand.shape
    P = np.asarray(predecessor_codebook, dtype=np.float64)
    S = np.asarray(successor_codebook, dtype=np.float64)
    pred_ids = np.concatenate([np.broadcast_to(np.asarray(anchor_token_ids)[:, None, None], (b, 1, k)), cand[:, :-1]], axis=1)
    pred = P[pred_ids] * np.asarray(hidden, dtype=np.float64)[:, :, None, :]
    return np.asarray(unary_logits, dtype=np.float64)[:, :, None, :] + np.einsum("blpr,blcr->blpc", pred, S[cand])


def selector_walk(scores, candidate_ids):
    """Greedy walk (temperature 0): the predecessor index starts at 0 (all predecessors of step 0 are the anchor), each
    step takes the best candidate given the previous choice. Returns tokens [b, L] and the chosen indices [b, L]."""
    scores = np.asarray(scores)
    cand = np.asarray(candidate_ids)
    b, steps, k, _ = scores.shape
    tokens = np.zeros((b, steps), dtype=np.int64)
    picks = np.zeros((b, steps), dtype=np.int64)
    for r in range(b):
        prev = 0
        for l in range(steps):
            prev = int(np.argmax(scores[r, l, prev]))
            picks[r, l] = prev
            tokens[r, l] = cand[r, l, prev]
    return tokens, picks


def context_kv(aux_hidden, fc_weight, hidden_norm_weight, k_proj, v_proj, k_norm_weight, n_kv_heads, head_dim, eps,
               fc_bias=None, k_bias=None, v_bias=None):
    """K and V that the target's hidden states contribute to every drafter layer, before RoPE (PROVISIONAL: by reading).
    aux_hidden [n_ctx, n_taps * target_hidden] is the concatenation of the tapped target states of each context token;
    k_proj, v_proj [L, n_kv_heads * head_dim, hidden]; k_norm_weight [L, head_dim]. Returns K, V [L, n_ctx, n_kv_heads, head_dim].
    The drafter's layers are never run over the context: this projection is all the context costs."""
    ctx = np.asarray(aux_hidden, dtype=np.float64) @ np.asarray(fc_weight, dtype=np.float64).T
    if fc_bias is not None:
        ctx = ctx + fc_bias
    ctx = rms_norm(ctx, hidden_norm_weight, eps)
    n_layers = k_proj.shape[0]
    n_ctx = ctx.shape[0]
    K = np.empty((n_layers, n_ctx, n_kv_heads, head_dim))
    V = np.empty_like(K)
    for l in range(n_layers):
        k = ctx @ np.asarray(k_proj[l], dtype=np.float64).T
        v = ctx @ np.asarray(v_proj[l], dtype=np.float64).T
        if k_bias is not None:
            k = k + k_bias[l]
        if v_bias is not None:
            v = v + v_bias[l]
        K[l] = rms_norm(k.reshape(n_ctx, n_kv_heads, head_dim), k_norm_weight[l], eps)
        V[l] = v.reshape(n_ctx, n_kv_heads, head_dim)
    return K, V
