"""Oracle: GLM-5.3-Flash MoE routing and the expert SwiGLU with clamp.

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d,
src/llama-graph.cpp llm_graph_context::build_moe_ffn (second overload, from line 2015):
    probs     = sigmoid(logits)                       gating SIGMOID (GLM: scoring_func sigmoid)
    selection = probs + exp_probs_b                   e_score_correction_bias, used ONLY to select
    ids       = top_k(selection, k)                   n_group = 1: no group step
    w         = probs[ids]                            unbiased weights
    w         = w / max(sum(w), 6.103515625e-5)       norm_w (norm_topk_prob)
    w         = w * w_scale                           routed_scaling_factor = 2.5
Expert activation for GLM5_NEXT (src/llama-graph.cpp lines 2246-2250 and 1856; the CPU op is
ggml/src/ggml-cpu/ops.cpp ggml_compute_forward_swiglu_clamp_f32):
    out = silu(min(gate, L)) * clamp(up, -L, L)       L = swiglu_limit = 10, clamp BEFORE silu
Other architectures use min(silu(gate), L) instead; GLM must not.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def route(logits, bias, k, w_scale=2.5, norm=True):
    """logits (T, E) fp32 router outputs; bias (E,). Returns ids (T, k), weights (T, k)."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = sigmoid(logits)
    sel = probs + np.asarray(bias, dtype=np.float64)[None]
    ids = np.argsort(-sel, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(probs, ids, axis=-1)
    if norm:
        w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    return ids, w * w_scale


def swiglu_clamp(gate, up, limit=10.0):
    g = np.minimum(np.asarray(gate, dtype=np.float64), limit)
    u = np.clip(np.asarray(up, dtype=np.float64), -limit, limit)
    return g * sigmoid(g) * u


def moe_forward(x, logits, bias, w_gate, w_up, w_down, k, limit=10.0, w_scale=2.5):
    """x (T, H); w_gate/w_up (E, F, H); w_down (E, H, F). Routed part only (no shared expert)."""
    ids, w = route(logits, bias, k, w_scale)
    out = np.zeros((x.shape[0], w_down.shape[1]))
    for t in range(x.shape[0]):
        for j in range(k):
            e = ids[t, j]
            h = swiglu_clamp(w_gate[e] @ x[t], w_up[e] @ x[t], limit)
            out[t] += w[t, j] * (w_down[e] @ h)
    return out, ids, w
