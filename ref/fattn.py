"""Oracle: the f16 KV cache store and the CPU attention paths llama.cpp takes with -ctk/-ctv and -fa (N10, gap 3 of
docs/reassessment/CORRECTNESS.md).

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d ("O" = ggml/src/ggml-cpu/ops.cpp, "LG" =
src/llama-graph.cpp, "C" = ggml/src/ggml-cpu/ggml-cpu.c, "V" = ggml/src/ggml-cpu/vec.h, "VC" =
ggml/src/ggml-cpu/vec.cpp); the flash-attention code of O (one_chunk, tiled, the dispatcher) and build_attn_mha of LG
are identical at the R1 pin 79e2e74 (diffed; there the O lines are one lower and the LG lines 153 higher). Every cited
line was printed from the pinned checkout.

f16 rounding: every f32 -> f16 store goes through ggml_cpu_fp32_to_fp16, the from_float of F16 (C:222-226), which on
x86 with F16C is _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT) (C:3480-3493): IEEE binary16, round to nearest even.
The KV cache store is ggml_set_rows into the cache tensor (src/llama-kv-cache.cpp:1328-1361 cpy_k), which calls that
from_float per row (O:5477, 5491). f16() below rounds float64 -> float32 -> float16, the way an f32 graph value of the
oracle's float64 one would be.

The attention modes (attend()):
  "exact"    float64 softmax attention: f32 cache, flash attention off (the reference precision of N2/N6/G1/N9).
  "mm16"     flash attention off with an f16 cache: kq = ggml_mul_mat(k, q) and kqv = ggml_mul_mat(v, kq) (LG:2696,
             2740) with src0 f16 convert src1 to the vec_dot_type of F16, which is F16 (C:225): q and the softmax
             probabilities are rounded to f16, the products summed in f32 / double (VC ggml_vec_dot_f16).
  "fa_chunk" flash attention, ggml_compute_forward_flash_attn_ext_f16_one_chunk (O:8843-9079), taken when the ubatch
             has fewer than GGML_FA_TILE_Q = 64 query rows (O:9547-9551, ggml/src/ggml-cpu/common.h:9); the split-KV
             decode path (one query row, n_kv >= 512, O:9490) runs the same loop per KV chunk (not reached here):
               Q_q = f16(q)                (O:8917-8918 q_to_vec_dot = from_float of the vec_dot_type F16, O:8961)
               s   = f32(sum_d Q_q[d] k[d]) * scale                  (O:8976 ggml_vec_dot_f16, O:8978)
               online softmax over the unmasked cells in cell order, -inf cells skipped (O:8967-8970):
                 new maximum: M = s, ms = expf(Mold - M), VKQ16 = f16(VKQ16 * ms)  (O:8994-9000, ggml_vec_scale_f16)
                 else vs = expf(s - M)                                              (O:9003)
                 VKQ16 = f16(VKQ16 + v * vs)          (O:9007, ggml_vec_mad_f16: the ACCUMULATOR IS f16)
                 S = S*ms + vs                        (O:9031, f32)
               out = f32(VKQ16) * (1/S)               (O:9036, 9067-9068)
             ggml_prec_set_acc(cur, GGML_PREC_F32) (LG:2675) does not change this path: the CPU dispatcher runs the
             same function for GGML_PREC_DEFAULT and GGML_PREC_F32 (O:9581-9585).
  "fa_tiled" flash attention, ggml_compute_forward_flash_attn_ext_tiled (O:9081-), taken for >= 64 query rows of f32 q
             with an f32 or f16 cache (O:9547-9551): q stays f32 (O:9214), K and V are widened to f32 (O:9256, 9319),
             scores, softmax and the V accumulation are f32 (simd_gemm, O:9266, 9329). So only the f16 cache rounding
             remains: exact attention over f16(K), f16(V).
With flash attention on, an f32 K or V is cast to f16 before ggml_flash_attn_ext (LG:2660-2666), so both fa modes
round K and V even with -ctk f32 -ctv f32.

Shapes: q (T, H, DK); k (S, Hkv, DK); v (S, Hkv, DV); query head h reads kv head h // (H / Hkv) (GQA broadcast
O:8891, 8954, 8958); cells: per query row the sorted cell indices it may attend to (None = causal 0..t).
The emulation of "fa_chunk" follows the scalar code the AVX2 build runs for DK, DV < GGML_F16_STEP = 32
(ggml/src/ggml-cpu/simd-mappings.h:635-636; leftover loops V:580 mad, V:854 scale, VC:373 and 377 dot): products of
f16 values are exact in f32, the dot is summed in double and rounded to f32 once.
"""
import numpy as np

U16 = 2.0 ** -11          # unit roundoff of IEEE binary16 (11-bit significand): |f16(x) - x| <= U16 |x|
FA_TILE_Q = 64            # ggml/src/ggml-cpu/common.h:9 GGML_FA_TILE_Q
MODES = ("exact", "mm16", "fa_chunk", "fa_tiled")


def f16(x):
    """Round like an f32 graph value stored as f16: float64 -> float32 -> float16 (round to nearest even), back to
    float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float16).astype(np.float64)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def ulp16(x):
    """Spacing of binary16 at |x| (normal range; subnormal spacing 2^-24 below 2^-14)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -14)))
    return 2.0 ** (e - 10)


def mode_for(kv_f16, fa, n_query_rows):
    """The attention mode llama.cpp takes for a ubatch of n_query_rows tokens (see the module docstring)."""
    if fa:
        return "fa_tiled" if n_query_rows >= FA_TILE_Q else "fa_chunk"
    return "mm16" if kv_f16 else "exact"


def _cells(cells, T):
    return [np.arange(t + 1) for t in range(T)] if cells is None else [np.asarray(c, dtype=int) for c in cells]


def _online_f16(qh, kc, vc, scale):
    """flash_attn_ext_f16_one_chunk for one query row and its kv heads: qh (G, DK) query heads sharing one kv head,
    kc (n, DK), vc (n, DV) the unmasked cells in cell order. Returns (G, DV) and the f32 sums S (G,)."""
    G, DV = qh.shape[0], vc.shape[1]
    qq = f16(qh)                                                          # Q_q = from_float(q) to F16
    sc32 = np.float32(scale)
    s_all = (f32(qq @ kc.T).astype(np.float32) * sc32).astype(np.float64)  # (G, n): f32(dot) * scale in f32
    M = np.full(G, -np.inf)
    S = np.zeros(G, dtype=np.float32)
    acc = np.zeros((G, DV))                                               # VKQ16, values of binary16
    for i in range(kc.shape[0]):
        s = s_all[:, i]
        new = s > M
        ms = np.where(new, f32(np.exp(np.where(new, M - s, 0.0))), 1.0)   # expf(Mold - M); exp(-inf) = 0
        vs = np.where(new, 1.0, f32(np.exp(np.where(new, 0.0, s - M))))
        M = np.where(new, s, M)
        acc = np.where(new[:, None], f16(acc * ms[:, None]), acc)          # ggml_vec_scale_f16
        acc = f16(acc + vc[i][None, :] * vs[:, None])                     # ggml_vec_mad_f16
        S = (S * ms.astype(np.float32) + vs.astype(np.float32)).astype(np.float32)
    S = S.astype(np.float64)
    inv = np.where(S == 0.0, 0.0, f32(1.0 / np.where(S == 0.0, 1.0, S)))
    return f32(acc * inv[:, None]), S


# ---------------------------------------------------------------- precision of a run, for the model oracles
#
# prec = None: the reference contract of N2/N6/G1/N9 (-ctk f32 -ctv f32 -fa off), every value float64.
# prec = precision(kv, fa): the cache type of -ctk/-ctv and the -fa flag of the run. Optional prec["modes"]
# {absolute position: mode} forces the attention mode of single rows (a path the oracle cannot infer from the ubatch
# size: the ec7630a glm5-next gather path, or rows of earlier ubatches when a cache is rebuilt after a rollback).

def precision(kv="f32", fa=False, modes=None):
    if kv not in ("f16", "f32"):
        raise ValueError(kv)
    return dict(kv=kv, fa=bool(fa), modes=dict(modes or {}))


def kv_f16(prec):
    return bool(prec) and prec.get("kv", "f32") == "f16"


def store(x, prec):
    """A value as the cache holds it: f16(x) with an f16 cache (ggml_set_rows, O:5478-5495), else unchanged."""
    return f16(x) if kv_f16(prec) else np.asarray(x, dtype=np.float64)


def row_modes(prec, pos):
    """Attention mode of each query row of one ubatch at absolute positions pos."""
    pos = [int(p) for p in np.asarray(pos).tolist()]
    if not prec:
        return ["exact"] * len(pos)
    default = mode_for(kv_f16(prec), prec.get("fa", False), len(pos))
    forced = prec.get("modes") or {}
    return [forced.get(p, default) for p in pos]


def attend_rows(q, k, v, cells, scale, modes):
    """attend() with a mode per query row (rows grouped by mode)."""
    q = np.asarray(q, dtype=np.float64)
    cl = _cells(cells, q.shape[0])
    out = np.empty((q.shape[0], q.shape[1], np.shape(v)[2]))
    for m in sorted(set(modes)):
        rows = [i for i, mi in enumerate(modes) if mi == m]
        out[rows] = attend(q[rows], k, v, [cl[i] for i in rows], scale, m)
    return out


def attend(q, k, v, cells, scale, mode="exact"):
    """Attention of every query row over its cells; returns (T, H, DV) float64. k and v are the cache contents
    (already f16 values when the cache is f16; the fa modes round them anyway, as the f32->f16 cast of LG:2660-2666)."""
    if mode not in MODES:
        raise ValueError(mode)
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    T, H, _ = q.shape
    Hkv = k.shape[1]
    r = H // Hkv
    if mode in ("fa_chunk", "fa_tiled"):
        k, v = f16(k), f16(v)
    out = np.empty((T, H, v.shape[2]))
    for t, idx in enumerate(_cells(cells, T)):
        idx = np.sort(idx)
        for g in range(Hkv):
            hs = slice(g * r, (g + 1) * r)
            kc, vc = k[idx, g], v[idx, g]
            if mode == "fa_chunk":
                out[t, hs] = _online_f16(q[t, hs], kc, vc, scale)[0]
                continue
            qh = f16(q[t, hs]) if mode == "mm16" else q[t, hs]
            sc = qh @ kc.T * scale
            p = np.exp(sc - sc.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if mode == "mm16":
                p = f16(p)
            out[t, hs] = p @ vc
    return out
