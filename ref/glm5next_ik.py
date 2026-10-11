"""Oracle: what ik_llama.cpp computes for a GLM-5.3-Flash GGUF (architecture "glm5next", or the alias "glm5-next"),
as a set of deviations from ref/glm5next.py (the transcription of llama.cpp's glm5-next graph). N12, gap 5 of
docs/reassessment/CORRECTNESS.md.

ref/glm5next.py stays the reference and is not modified: this module runs it inside semantics(cfg), which swaps four
of its module-level functions for the ik versions below while the context is open, and restores them on exit.
"IK" = ikawrakow/ik_llama.cpp @5194a9e1451b885e24dd5a065f650864e8807c59 (tools/engines/pins.json); "B" =
src/graphs/build_glm5next.cpp, "H" = src/llama-hparams.cpp, "L" = src/llama.cpp, "C" = common/common.h, all at that
commit; every cited line was printed from the pinned checkout. "ML" = llama.cpp @79e2e74 (the R1 pin).

What differs from llama.cpp's glm5-next (each one measured on the tiny model, tests/test_ik_glm5next_parity.py):
1. SwiGLU clamp after the SiLU. ik: min(silu(gate), L) * clamp(up, -L, L) (ggml/src/ggml.c:17374-17377 fused
   mul-unary of the dense and shared FFN; ggml/src/iqk/iqk_mul_mat.cpp:155-157, 169-170 fused MoE up/gate; the limit
   is hparams.swiglu_limits[_shared], H:2506-2509, model.swiglu_limit src/llama-model.h:686-691). llama.cpp:
   silu(min(gate, L)) * clamp(up, -L, L) (ggml_swiglu_clamp, ML ggml/src/ggml-cpu/ops.cpp:3614-3615), as the
   DeepSeek-V4.1 reference does (third_party/deepseek-v41-flash/inference/model.py:845-848). The two agree while
   gate <= L and differ by at most L * sigmoid(-L) per element otherwise (silu(L) vs a value in (silu(L), L]).
2. Indexer key LayerNorm eps: attention.layer_norm_epsilon if present, else the RMS eps (H:2461-2468); llama.cpp
   uses the key and 0 when it is absent (ref/glm5next.py hparams "norm_eps"). ik_hparams() applies the rule.
3. KDA q/k L2 norm: x / max(|x|, rms_eps) (ggml/src/ggml.c:19218, eps = f_norm_rms_eps passed at
   src/llama-kda.cpp:225); llama.cpp x / sqrt(|x|^2 + 1e-6).
4. DSA k-pool selection is opt-in (--dsa, C:454; B:268); without it every MLA layer attends densely (causal).
   With it, the tail of the incomplete pool is passed as kpool-1 slots per query whose unused slots hold cell 0
   (L:5934 std::fill(dst_tail, ..., 0)), and the sparse mask unmasks every listed cell (ggml_indexer_mask ->
   iqk_mask_topk, ggml/src/iqk/iqk_cpu_ops.cpp:1106-1108): cell 0 is attended whenever the tail has fewer than
   kpool-1 cells, i.e. (t+1) % kpool != kpool-1. llama.cpp fills unused tail slots with the out-of-range sentinel n_kv,
   which selects no cell (ML src/llama-memory-hybrid-idx.cpp:814, 949-975). For one sequence from an empty cache cell = position.
   ik pools each complete pool afresh in every graph from the cached key | gate rows (B:109-137) instead of caching
   the pooled key (same values in f32).
5. Cache precision. The MLA latent cache takes the -ctv type when flash attention is off and -mla is not 1 (L:1496-1497;
   -mla defaults to 3, C:446), and -ctv f32 is rejected without flash attention (L:9257-9259): the latent is always
   rounded to f16 (or bf16) with -fa off. The log line still prints the -ctk type as "c^KV" (L:9808-9811). The
   indexer key | gate cache has its own type, -ictk, default f16 (C:495), not the -ctk type. With -fa off the
   attention itself is f32: kq = latent(f16) x q(f32) (B:345-348) and the probabilities stay f32 (measured, N12).
6. Flash attention (-fa on, ik's default, C:445) on the CPU runs the iqk kernels (ggml/src/iqk/iqk_flash_attn.cpp);
   an f32 K/V cache aborts there (iqk_flash_attn.cpp:280-298), so the latent is f16. Their result over an f16 latent
   matches ref/fattn.py's "fa_chunk" (q rounded to f16, online softmax, f16 V accumulator) for every ubatch size,
   where llama.cpp would switch to its f32 tiled kernel at 64 query rows: measured 1.6e-7 on a 64-row ubatch and
   1.0e-5 on single-token graphs (tests/test_ik_glm5next_parity.py, N12); the kernel itself is not transcribed here
   (PROVISIONAL mechanism, CONFIRMED agreement on these runs). config(fa=True) uses that mode.
Not modelled (not exercised by the tests): MTP (B:381-499: the token embedding at position 0 is zeroed,
B:396-404), multiple sequences.

cfg = config(dsa=..., latent="f16"|"f32", index="f16"|"f32", fa=...); latent "f32" is not reachable in ik on the CPU
(item 5 with -fa off, item 6 with -fa on).
"""
import contextlib

import numpy as np

from ref import glm5next as G
from ref.dsa import indexer_pools, mla_attention

_ORIG = dict(swiglu=G.swiglu, gated_delta_rule=G.gated_delta_rule, dsa_layer_full=G.dsa_layer_full,
             dsa_layer_chunk=G.dsa_layer_chunk)
_CFG = {}


def config(dsa=True, latent="f16", index="f32", fa=False):
    for v in (latent, index):
        if v not in ("f16", "f32"):
            raise ValueError(v)
    return dict(dsa=bool(dsa), latent=latent, index=index, fa=bool(fa))


def ik_hparams(hp):
    """ref/glm5next.hparams() as ik reads the same GGUF (item 2 of the module docstring)."""
    hp = dict(hp)
    if hp["norm_eps"] <= 0.0:
        hp["norm_eps"] = hp["rms_eps"]
    return hp


def _cast(x, t):
    x = np.asarray(x, dtype=np.float64)
    return x.astype(np.float32).astype(np.float16).astype(np.float64) if t == "f16" else x


def swiglu_post_silu(gate, up, limit):
    """Item 1: ik's clamped SwiGLU."""
    if limit > G.CLAMP_EPS:
        return np.minimum(G.silu(gate), limit) * np.clip(np.asarray(up, dtype=np.float64), -limit, limit)
    return G.silu(gate) * np.asarray(up, dtype=np.float64)


def _gdr(q, k, v, g, beta, state=None, normalize_qk=True, eps=1e-6):
    """Item 3: ik's L2 norm of q and k, then the reference scan without its own normalisation."""
    if normalize_qk:
        e = _CFG["rms_eps"]
        q, k = (np.asarray(a, dtype=np.float64) for a in (q, k))
        q = q / np.maximum(np.sqrt(np.sum(q * q, axis=-1, keepdims=True)), e)
        k = k / np.maximum(np.sqrt(np.sum(k * k, axis=-1, keepdims=True)), e)
    return _ORIG["gated_delta_rule"](q, k, v, g, beta, state=state, normalize_qk=False)


def ik_cells(cells, pos, kpool):
    """Item 4: the reference selection plus cell 0 wherever ik pads the tail with it."""
    return [sorted(set(c) | {0}) if (int(t) + 1) % kpool != kpool - 1 else list(c) for c, t in zip(cells, pos)]


def dsa_layer_chunk(x, w, hp, pos, st, cells_override=None, prec=None):
    """ik's DSA / MLA layer on one ubatch at absolute positions pos (prec is ignored: the cache types come from cfg).
    st = dict(latent, ik, ig, pooled) of the cached cells as ik stores them (pooled: recomputed, kept for the caller)."""
    cfg = _CFG
    r = G.dsa_project(x, w, hp)
    kp = hp["kpool"]
    st["latent"] = np.concatenate([st["latent"], _cast(r["latent"], cfg["latent"])])
    st["ik"] = np.concatenate([st["ik"], _cast(r["ik"], cfg["index"])])
    st["ig"] = np.concatenate([st["ig"], _cast(r["ig"], cfg["index"])])
    nb = st["ik"].shape[0] // kp                                     # complete pools in the cache (B:113-137)
    pooled = indexer_pools(st["ik"][:nb * kp], st["ig"][:nb * kp], w["indexer_compressor_ape.weight"], kp)
    st["pooled"] = pooled.reshape(-1, hp["idx_dim"])
    if cfg["dsa"]:
        scores = G.scores_at(r["iq"], r["wts"], st["pooled"], kp, pos)
        ref_cells = G.select_at(scores, hp["idx_top_k"], kp, pos, True)
        cells = ik_cells(ref_cells, pos, kp)
    else:
        scores, ref_cells = None, None
        cells = [list(range(int(t) + 1)) for t in pos]
    if cells_override:
        cells = [cells_override.get(int(t), c) for t, c in zip(np.asarray(pos).tolist(), cells)]
    T = x.shape[0]
    kqv = mla_attention(r["q_abs"], st["latent"], cells, np.asarray(w["attn_v_b.weight"], dtype=np.float64),
                        hp["k_mla"], ["fa_chunk"] * T if cfg.get("fa") else None)
    r.update(pooled=st["pooled"].copy(), scores=scores, ref_cells=ref_cells, cells=cells, kqv=kqv,
             out=kqv @ w["attn_output.weight"].T,
             stored={key: st[key][st[key].shape[0] - T:] for key in ("latent", "ik", "ig")})
    return r


def dsa_layer_full(x, w, hp, cells_override=None, prec=None):
    d = hp["idx_dim"]
    st = dict(latent=np.zeros((0, hp["kv_lora"])), ik=np.zeros((0, d)), ig=np.zeros((0, d)), pooled=np.zeros((0, d)))
    return dsa_layer_chunk(x, w, hp, np.arange(x.shape[0]), st, cells_override)


@contextlib.contextmanager
def semantics(cfg, hp):
    """Run ref/glm5next.py with ik's semantics (module docstring) while the context is open."""
    global _CFG
    saved = dict(_CFG)
    _CFG = dict(cfg, rms_eps=hp["rms_eps"])
    G.swiglu, G.gated_delta_rule = swiglu_post_silu, _gdr
    G.dsa_layer_full, G.dsa_layer_chunk = dsa_layer_full, dsa_layer_chunk
    try:
        yield
    finally:
        for k, v in _ORIG.items():
            setattr(G, k, v)
        _CFG = saved


def forward(tokens, W, hp, cfg, overrides=None):
    """G.forward under ik semantics; hp from G.hparams (ik_hparams is applied here)."""
    hp = ik_hparams(hp)
    with semantics(cfg, hp):
        return G.forward(tokens, W, hp, overrides)


def forward_chunks(tokens, W, hp, sizes, cfg, overrides=None):
    """G.forward_chunks under ik semantics (one ubatch per chunk)."""
    hp = ik_hparams(hp)
    with semantics(cfg, hp):
        return G.forward_chunks(tokens, W, hp, sizes, overrides)
