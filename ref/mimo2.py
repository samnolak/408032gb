"""Oracle: llama.cpp architecture "mimo2" (MiMo-V2.x-Flash), the whole model, its incremental path (the KV caches of the
global and the sliding-window layers, one ubatch at a time, rollback) and its MTP (NextN) block (N16, gap 12 of
docs/reassessment/CORRECTNESS.md). Built from ref/mimo.py (NEOX partial RoPE; attention with sinks, window and GQA,
checked op by op against the real graph in G1, tests/test_mimo_tiny_parity.py) and ref/router.py (route: sigmoid
gating, exp_probs_b selection bias, normalised weights); ref/mimo.py and ref/router.py are used unchanged.

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d. "M" = src/models/mimo2.cpp, "LG" =
src/llama-graph.cpp, "KV" = src/llama-kv-cache.cpp, "KVI" = src/llama-kv-cache-iswa.cpp, "KC" = src/llama-kv-cells.h,
"HP" = src/llama-hparams.h, "LM" = src/llama-model.cpp, "OPS" = ggml/src/ggml-cpu/ops.cpp, all at that commit; every
cited line was printed from the pinned checkout. At the R1 pin 79e2e74 KV, KVI and the parts of LG used here are the
same; mimo2.cpp differs only in the loader's NextN flags (nextn_flags, same flags) and in naming the final hidden state
h_nextn in every trunk graph (M:236-238 @79e2e74), which renames the last layer's l_out in the dumps.

Trunk, per layer il (M:108-234):
    x       = rms_norm(inpL) * attn_norm                                                       (M:122)
    q, k, v = W_q x, W_k x, W_v x, or the rows [q | k | v] of the fused W_qkv x                  (M:129-157)
    q, k    = NEOX RoPE on the first n_rot dims of each head; base rope.freq_base_swa on sliding-window layers
              (absent: 10000, HP:151), rope.freq_base on the others                  (M:115, 159-169; LM:2329-2331)
    k, v    are stored in the layer's cache cells (LG:3181, 3187); then over the n_kv cells of that cache:
    kq      = K_cell . q per head (LG:2696-2697); query head h reads KV head h // (n_head / n_head_kv) (ggml_mul_mat
              broadcast)
    p       = softmax(kq / sqrt(k_dim) + mask), the layer's sink logit (attn_sinks, optional, M:57) joining the
              denominator only: max over the row and the sink, sum += exp(sink - max) (LG:2730-2732; OPS:5896-5912)
    kqv_out = sum over cells of p V (head-major concat)                                         (LG:2740-2752)
    attn    = W_o kqv_out, times attention.value_scale when the GGUF has one != 1                  (M:177-185, 14-17)
    ffn_inp = attn + inpL                                                                       (M:193)
    y       = rms_norm(ffn_inp) * ffn_norm, then a dense SwiGLU W_down (silu(W_gate y) * W_up y) when the layer has no
              ffn_gate_inp (M:202-210, LG build_ffn LLM_FFN_SILU / LLM_FFN_PAR), else build_moe_ffn (M:213-224):
              probs = sigmoid(W_router y), ids = top n_expert_used of probs + exp_probs_b, w = probs[ids] / max(sum,
              6.103515625e-5, LG:2156-2170), no scale (mimo2 never reads expert_weights_scale: M:3-23; hparams default
              0, LG:2171 scales only if != 0 and != 1), plain silu(gate) * up (swiglu_clamp_exp = 0), sum of
              w * W_down_e act
    l_out   = y + ffn_inp                                                                       (M:227-230)
Head: result_norm = rms_norm(l_out of the last layer) * output_norm, logits = W_output result_norm   (M:236-264).
The target's h_nextn is that last l_out, before the norm (M:244-246).

KV caches (one sequence, one stream; KVI:34-106 creates the pair, LM:2815-2862 picks the iSWA memory for a SWA model):
  * base cache, the non-SWA layers: kv_size = n_ctx cells, swa_type NONE: a cell is reused only once empty;
  * SWA cache, the sliding-window layers: pad(min(n_ctx, n_swa + n_ubatch), 256) cells (KVI:73; swa_full off), so it
    holds at least 256 cells however small the window, and wraps only after that many positions;
  * the target context holds the trunk layers, an MTP context the MTP layer(s) alone (filters, LM:2807-2812);
  * the buffers start at zero (KV:293); a freed cell keeps its K/V data until a new token is written there.
Placement of a ubatch (prepare() -> find_slot(ubatch, cont = false), KV:751, 769, 898-1095; apply_ubatch KV:1097-1186):
walk the cells from the head (from 0 when head > used + 2 n_tokens), take every empty cell and every cell whose
position p0 is masked by the window for the cache's seq_pos_max + 1 (is_masked_swa: p1 - p0 >= n_swa, HP:479-491),
wrapping at the end; overwrite those cells, then remove every position of the sequence from seq_pos_min up to the
largest position overwritten (KV:1160-1177); head = last cell + 1 (KV:1180-1185). n_kv = min(size, max(256,
pad(used_max_p1, 256))) cells enter the graph (KV:1260-1274; KC:97-99), with the mask (KV:1656-1700): a cell is visible
to the query at p1 when it holds the sequence, p0 <= p1, and on SWA layers p1 - p0 < n_swa.
Rollback llama_memory_seq_rm(0, p, -1) (KVI:113-120 -> KV:382-420): free every cell holding a position >= p in both
caches; head moves back to the first freed cell. seq_rm reports success whatever the SWA cache still holds: a
rollback to p needs positions p - n_swa + 1 .. p - 1 in the SWA cache, and once the cache has wrapped past them they
are gone (not restored); the cell model below reproduces that (cache_truncate), the full-sequence oracle cannot.

MTP block, graph_mtp (M:272-407), layer il = n_layer, on the MTP context with its own iSWA cache:
    e = rms_norm(embed_tokens or tok_embd [token]) * nextn.enorm; h = rms_norm(h_prev) * nextn.hnorm    (M:303-318)
    x = nextn.eh_proj [e ; h]                                                                   (M:320-324)
    attention of layer il as above on rms_norm(x) * attn_norm (fused attn_qkv only, M:285, 331-356), value scale
    f = attn + x; post = dense SwiGLU(rms_norm(f) * ffn_norm) + f  (dense FFN only, M:375-385); h_nextn = post
    logits = (shared_head_head or output) (rms_norm(post) * (shared_head_norm or layer_output_norm or output_norm))
    (M:387-405). common/speculative.cpp pairs token p with the target's h_nextn of p-1; a draft step feeds the MTP's
    own h_nextn (tools/dump/dump_tensors.cpp DUMP_MTP).

Quantized weights (N10): a weight may be a ref/quant.py QWeight; every product with a weight is written x @ w.T or
w @ x, which for a QWeight applies the CPU activation conversion. Everything else runs in float64; shapes are
numpy-natural (token-major), weights are the GGUF arrays as numpy reads them (ggml ne reversed).
"""
import numpy as np

from ref.mimo import rope_neox
from ref.router import route

NORM_W_MIN = 6.103515625e-5      # LG build_moe_ffn: the weight sum is clamped to the smallest f16 normal
KV_PAD = 256                     # KVI:73 (SWA cache size) and KV:1265 (n_kv padding)


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


def pad(n, m):
    return (n + m - 1) // m * m


# ---------------------------------------------------------------- hyper-parameters from a GGUF

def hparams(fields):
    """Hyper-parameters from GGUFReader fields ({name: field}); keys as M:3-23 and LM:1327-1404 read them."""
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

    n_all = int(get("block_count"))                                    # counts the MTP blocks (LM:1327-1330)
    n_nextn = int(get("nextn_predict_layers", 0))

    def per_layer(key, default=None):
        v = get(key, default, array=True)
        v = v * n_all if len(v) == 1 else v
        return [int(x) for x in v[:n_all]]

    vs = float(get("attention.value_scale", 1.0))                      # M:14-17: kept only if != 1
    return dict(
        arch=arch, n_layer=n_all - n_nextn, n_nextn=n_nextn, n_all=n_all, n_embd=int(get("embedding_length")),
        n_head=per_layer("attention.head_count"), n_head_kv=per_layer("attention.head_count_kv"),
        k_dim=int(get("attention.key_length")), v_dim=int(get("attention.value_length")),
        rms_eps=float(get("attention.layer_norm_rms_epsilon")), n_rot=int(get("rope.dimension_count")),
        rope_base=float(get("rope.freq_base")), rope_base_swa=float(get("rope.freq_base_swa", 10000.0)),
        n_swa=int(get("attention.sliding_window")),
        swa=[bool(x) for x in get("attention.sliding_window_pattern", array=True)][:n_all],
        n_expert=int(get("expert_count")), n_expert_used=int(get("expert_used_count")),
        v_scale=0.0 if vs == 1.0 else vs,
    )


def layer_weights(W, il):
    pre = f"blk.{il}."
    return {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}


# ---------------------------------------------------------------- KV cells (KV / KC, one stream, sequence 0)

class Cells:
    """The cells of one llama_kv_cache stream that holds sequence 0 only. pos[i] = position held by cell i, -1 empty;
    head = v_heads[0]. n_swa = 0: a non-SWA cache (the base cache of the iSWA pair is created with swa_type NONE and
    n_swa 0, KVI:95-98)."""

    def __init__(self, size, n_swa=0):
        self.size, self.n_swa = int(size), int(n_swa)
        self.pos = np.full(self.size, -1, dtype=np.int64)
        self.head = 0

    def copy(self):
        c = Cells(self.size, self.n_swa)
        c.pos, c.head = self.pos.copy(), self.head
        return c

    def used(self):
        return int((self.pos >= 0).sum())

    def used_max_p1(self):                                             # KC:97-99
        nz = np.nonzero(self.pos >= 0)[0]
        return int(nz[-1]) + 1 if nz.size else 0

    def seq_pos_max(self):
        return int(self.pos.max()) if (self.pos >= 0).any() else -1

    def seq_pos_min(self):
        live = self.pos[self.pos >= 0]
        return int(live.min()) if live.size else -1

    def masked_swa(self, p0, p1):                                      # HP:479-491, LLAMA_SWA_TYPE_STANDARD
        return self.n_swa > 0 and p1 - p0 >= self.n_swa

    def find_slot(self, n):
        """KV:898-1095 with cont = false (prepare() passes false, KV:769): the cells for a ubatch of n tokens."""
        head_cur = self.head
        if head_cur > self.used() + 2 * n:                            # KV:1004-1006
            head_cur = 0
        if n > self.size:
            raise RuntimeError(f"n_tokens = {n} > size = {self.size}")
        pmax1 = self.seq_pos_max() + 1                                 # cells.seq_pos_max(seq_id_cell) + 1, KV:1057
        idxs, n_tested = [], 0
        while True:
            if head_cur + 1 > self.size:                               # KV:1020-1024
                n_tested += self.size - head_cur
                head_cur = 0
                continue
            idx = head_cur
            head_cur += 1
            n_tested += 1
            p0 = int(self.pos[idx])
            if p0 < 0 or self.masked_swa(p0, pmax1):                   # KV:1042-1061
                idxs.append(idx)
            if len(idxs) == n:
                return idxs
            if n_tested >= self.size:
                raise RuntimeError("no slot")                          # KV:1080-1083: find_slot fails

    def apply(self, idxs, positions):
        """KV:1097-1186: write the ubatch's positions into its cells; then purge every position from seq_pos_min up
        to the largest position overwritten (KV:1160-1177); head = last cell + 1 (KV:1180-1185)."""
        rm_max = -1
        for idx, p in zip(idxs, positions):
            if self.pos[idx] >= 0:
                rm_max = max(rm_max, int(self.pos[idx]))
            self.pos[idx] = int(p)
        if rm_max >= 0 and self.seq_pos_min() <= rm_max:
            self.seq_rm(self.seq_pos_min(), rm_max + 1)
        self.head = int(idxs[-1]) + 1

    def seq_rm(self, p0, p1=-1):
        """KV:382-420 for sequence 0: free every cell with p0 <= pos < p1 (p1 < 0: no upper end, KC:421-425); head
        moves to the first freed cell if that is lower."""
        p0 = max(int(p0), 0)
        hi = np.iinfo(np.int64).max if p1 < 0 else int(p1)
        hit = np.nonzero((self.pos >= p0) & (self.pos < hi))[0]
        self.pos[hit] = -1
        if hit.size and hit[0] < self.head:
            self.head = int(hit[0])

    def n_kv(self):
        """KV:1260-1274 (n_pad = 1 for mimo2's caches, LM:2842, 2859): the cells the graph reads."""
        return min(self.size, max(KV_PAD, pad(self.used_max_p1(), KV_PAD)))


def new_cache(hp, n_ctx, n_ubatch, mtp=False, swa_size=None):
    """The memory of one llama context of mimo2: the iSWA pair (module docstring). mtp=False: the target context (the
    trunk layers il < n_layer); mtp=True: an MTP context (the MTP layer il = n_layer alone). swa_size: a SWA cache of
    that many cells instead of llama.cpp's pad(min(n_ctx, n_swa + n_ubatch), 256) (property tests only)."""
    layers = [hp["n_layer"]] if mtp else list(range(hp["n_layer"]))
    size_swa = swa_size or pad(min(n_ctx, hp["n_swa"] + n_ubatch), KV_PAD)
    c = dict(tokens=[], n_ctx=n_ctx, n_ubatch=n_ubatch, layers=layers, base=Cells(n_ctx), swa=Cells(size_swa, hp["n_swa"]),
             K={}, V={})
    for il in layers:
        size = size_swa if hp["swa"][il] else n_ctx
        c["K"][il] = np.zeros((size, hp["n_head_kv"][il], hp["k_dim"]))
        c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["v_dim"]))
    return c


# ---------------------------------------------------------------- attention

def project_qkv(x, w, hp, il):
    """The q, k, v projections (before RoPE), (T, H, Dk), (T, Hkv, Dk), (T, Hkv, Dv): split W_q / W_k / W_v or the
    rows [q | k | v] of the fused W_qkv (M:129-157)."""
    T = x.shape[0]
    H, Hkv, Dk, Dv = hp["n_head"][il], hp["n_head_kv"][il], hp["k_dim"], hp["v_dim"]
    if "attn_qkv.weight" in w:
        y = x @ w["attn_qkv.weight"].T
        q, k, v = y[:, :H * Dk], y[:, H * Dk:(H + Hkv) * Dk], y[:, (H + Hkv) * Dk:]
    else:
        q, k, v = x @ w["attn_q.weight"].T, x @ w["attn_k.weight"].T, x @ w["attn_v.weight"].T
    return q.reshape(T, H, Dk), k.reshape(T, Hkv, Dk), v.reshape(T, Hkv, Dv)


def visible(qpos, kpos, n_swa):
    """(T, C) the mask of KV:1656-1700 for queries at qpos over key slots holding kpos (-1 = empty cell)."""
    qpos, kpos = np.asarray(qpos)[:, None], np.asarray(kpos)[None, :]
    vis = (kpos >= 0) & (kpos <= qpos)
    if n_swa:
        vis &= (qpos - kpos) < n_swa
    return vis


def attend(q, qpos, K, V, kpos, sinks, n_swa, scale):
    """q (T, H, Dk) after RoPE; K (C, Hkv, Dk), V (C, Hkv, Dv): the key slots (cells), kpos (C,) their positions (-1
    empty). Returns kq (H, T, C) = the raw products K . q (ggml's kq, LG:2696-2697), p (H, T, C) the probabilities (0
    where masked; sinks None: no sink), out (T, H * Dv)."""
    q, K, V = (np.asarray(a, dtype=np.float64) for a in (q, K, V))
    T, H, _ = q.shape
    grp = np.arange(H) // (H // K.shape[1])                            # GQA: head h reads KV head h // r
    kq = np.einsum("thd,chd->htc", q, K[:, grp])
    vis = visible(qpos, kpos, n_swa)
    s = np.where(vis[None], kq * scale, -np.inf)
    sink = np.full(H, -np.inf) if sinks is None else np.asarray(sinks, dtype=np.float64)
    m = np.maximum(s.max(-1, keepdims=True), sink[:, None, None])
    e = np.exp(s - m)
    p = e / (e.sum(-1, keepdims=True) + np.exp(sink[:, None, None] - m))
    out = np.einsum("htc,chv->thv", p, V[:, grp]).reshape(T, -1)
    return kq, p, out


def attention_layer(x, w, hp, il, pos, cache=None, slots=None):
    """One attention block on x = rms_norm(inpL) * attn_norm at absolute positions pos. cache None: the full sequence
    from position 0 (the keys are this call's rows, prefill semantics); else the context's cache (new_cache) and slots
    {"base"/"swa": cells of this ubatch}: k, v are written there and the attention runs over its n_kv cells."""
    pos = np.asarray(pos)
    swa = hp["swa"][il]
    q0, k0, v = project_qkv(x, w, hp, il)
    base = hp["rope_base_swa"] if swa else hp["rope_base"]
    q, k = rope_neox(q0, pos, hp["n_rot"], base), rope_neox(k0, pos, hp["n_rot"], base)
    out = dict(q_pre=q0, k_pre=k0, v=v, q=q, k=k)
    if cache is None:
        K, V, kpos = k, v, pos
    else:
        which = "swa" if swa else "base"
        idxs = slots[which]
        cache["K"][il][idxs] = k
        cache["V"][il][idxs] = v
        n = cache[which].n_kv()
        K, V, kpos = cache["K"][il][:n], cache["V"][il][:n], cache[which].pos[:n].copy()
        out.update(idxs=list(idxs), n_kv=n)
    kq, p, kqv = attend(q, pos, K, V, kpos, w.get("attn_sinks.weight"), hp["n_swa"] if swa else 0,
                        1.0 / np.sqrt(hp["k_dim"]))
    att = kqv @ w["attn_output.weight"].T
    out.update(kpos=kpos, kq=kq, probs=p, kqv=kqv, attn_out=att,
               out=att * hp["v_scale"] if hp["v_scale"] else att)
    return out


# ---------------------------------------------------------------- FFN

def ffn_layer(x, w, hp, ids_override=None):
    """x (T, n) after ffn_norm. Dense SwiGLU without ffn_gate_inp, else the routed experts (module docstring).
    ids_override {row: expert ids} replays a reference choice among (near-)tied experts."""
    if "ffn_gate_inp.weight" not in w:
        up, gate = x @ w["ffn_up.weight"].T, x @ w["ffn_gate.weight"].T
        act = silu(gate) * up
        return dict(up=up, gate=gate, act=act, out=act @ w["ffn_down.weight"].T)
    T, k = x.shape[0], hp["n_expert_used"]
    logits = x @ w["ffn_gate_inp.weight"].T
    bias = np.asarray(w["exp_probs_b.bias"], dtype=np.float64)
    ids, wts = route(logits, bias, k, w_scale=1.0, norm=True)
    probs = sigmoid(logits)
    if ids_override:
        ids, wts = ids.copy(), wts.copy()
        for r, e in ids_override.items():
            ids[r] = np.asarray(e)
            wr = probs[r, ids[r]]
            wts[r] = wr / max(wr.sum(), NORM_W_MIN)
    ge, ue, de = w["ffn_gate_exps.weight"], w["ffn_up_exps.weight"], w["ffn_down_exps.weight"]
    gate = np.stack([[ge[e] @ x[t] for e in ids[t]] for t in range(T)])          # (T, k, F)
    up = np.stack([[ue[e] @ x[t] for e in ids[t]] for t in range(T)])
    act = silu(gate) * up
    down = np.stack([[de[e] @ act[t, j] for j, e in enumerate(ids[t])] for t in range(T)])   # (T, k, n)
    moe = (down * wts[:, :, None]).sum(axis=1)
    return dict(logits=logits, probs=probs, sel=probs + bias[None], ids=ids, weights=wts, gate=gate, up=up, act=act,
                down=down, moe_out=moe, out=moe)


# ---------------------------------------------------------------- trunk

def _layer(h, W, hp, il, pos, cache=None, slots=None, moe_ids=None):
    w = layer_weights(W, il)
    L = dict(inp=h)
    L["attn_norm"] = x = rms_norm(h, hp["rms_eps"], w["attn_norm.weight"])
    L["attn"] = a = attention_layer(x, w, hp, il, pos, cache, slots)
    L["ffn_inp"] = f = a["out"] + h
    L["ffn_norm"] = fn = rms_norm(f, hp["rms_eps"], w["ffn_norm.weight"])
    L["ffn"] = ff = ffn_layer(fn, w, hp, moe_ids)
    L["l_out"] = ff["out"] + f
    return L["l_out"], L


def _head(h, W, hp, out):
    norm = rms_norm(h, hp["rms_eps"], W["output_norm.weight"])
    out.update(h_nextn=h, result_norm=norm, logits=norm @ W["output.weight"].T)
    return out


def forward(tokens, W, hp, overrides=None):
    """The full sequence from position 0 in one pass (prefill semantics: every key of the sequence is available).
    overrides {("moe", il): {position: expert ids}} replays reference choices among tied experts.
    Returns dict(layers=[per layer], h_nextn, result_norm, logits)."""
    overrides = overrides or {}
    tokens = np.asarray(tokens)
    pos = np.arange(len(tokens))
    h = W["token_embd.weight"][tokens]
    out = dict(pos=pos, inp_embd=h, layers=[])
    for il in range(hp["n_layer"]):
        h, L = _layer(h, W, hp, il, pos, moe_ids=overrides.get(("moe", il)))
        out["layers"].append(L)
    return _head(h, W, hp, out)


# ---------------------------------------------------------------- incremental path: ubatches, decode, rollback
#
# llama.cpp runs a call of n tokens as ceil(n / n_ubatch) ubatches, each its own graph over the context's iSWA memory.
# forward_chunk is one such ubatch; cache_truncate is llama_memory_seq_rm(0, n_keep, -1). The state carried between
# ubatches is the KV caches alone (module docstring): the cells' positions, head, and the K / V rows.

def place(cache, n):
    """Cells for the next ubatch of n tokens in both caches (prepare + apply, KV:751-815, 1097-1186); returns the
    slots {"base": [...], "swa": [...]} and the positions."""
    p0 = len(cache["tokens"])
    pos = np.arange(p0, p0 + n)
    slots = {}
    for which in ("base", "swa"):
        idxs = cache[which].find_slot(n)
        cache[which].apply(idxs, pos)
        slots[which] = idxs
    return slots, pos


def forward_chunk(tokens, W, hp, cache, overrides=None):
    """One ubatch of a single sequence that continues `cache` (new_cache() at the start); updates cache in place.
    Returns a dict like forward() for the ubatch's rows, with the absolute positions and per layer the cells written
    (attn["idxs"]), the cells read (attn["kpos"], n_kv) and kq / probs over those cells."""
    overrides = overrides or {}
    tokens = [int(t) for t in tokens]
    slots, pos = place(cache, len(tokens))
    cache["tokens"] = cache["tokens"] + tokens
    h = W["token_embd.weight"][tokens]
    out = dict(pos=pos, slots=slots, inp_embd=h, layers=[])
    for il in range(hp["n_layer"]):
        moe = overrides.get(("moe", il))
        h, L = _layer(h, W, hp, il, pos, cache, slots,
                      {int(t) - int(pos[0]): e for t, e in moe.items()} if moe else None)
        out["layers"].append(L)
    return _head(h, W, hp, out)


def cache_truncate(cache, n_keep):
    """llama_memory_seq_rm(mem, 0, n_keep, -1) (KVI:113-120 -> KV:382-420): free the cells holding positions >= n_keep
    in both caches. Nothing is restored: what the SWA cache already overwrote stays lost (module docstring)."""
    cache["base"].seq_rm(n_keep, -1)
    cache["swa"].seq_rm(n_keep, -1)
    cache["tokens"] = cache["tokens"][:n_keep]
    return cache


def forward_calls(tokens, W, hp, plan, n_ctx=512, n_ubatch=512, swa_size=None):
    """Replay a DUMP_CALLS plan ("<n>" a call of n tokens, "r<n>" a rollback of n positions) the way llama.cpp runs it
    (ceil(n / n_ubatch) ubatches per call). Returns the list of per-ubatch results (forward_chunk) and the cache."""
    cache, outs, ti = new_cache(hp, n_ctx, n_ubatch, swa_size=swa_size), [], 0
    for step in plan:
        if isinstance(step, str) and step.startswith("r"):
            cache_truncate(cache, len(cache["tokens"]) - int(step[1:]))
            continue
        n = int(step)
        for j in range(0, n, n_ubatch):
            m = min(n_ubatch, n - j)
            outs.append(forward_chunk(tokens[ti + j:ti + j + m], W, hp, cache))
        ti += n
    assert ti == len(tokens), (ti, len(tokens))
    return outs, cache


# ---------------------------------------------------------------- MTP block (nextn_predict_layers >= 1, offset 0)

def mtp_forward_chunk(tokens, h_prev, W, hp, cache=None, pos=None):
    """graph_mtp (M:272-407) on one ubatch of the MTP context (module docstring). tokens: the ubatch's ids; h_prev
    (T, n_embd): the pre-norm hidden row paired with each token. cache: new_cache(..., mtp=True) of the MTP context
    (incremental); None: all positions 0..T-1 at once (prefill semantics, pos ignored)."""
    il = hp["n_layer"]
    w = layer_weights(W, il)
    eps = hp["rms_eps"]
    tokens = [int(t) for t in tokens]
    if cache is None:
        slots, pos = None, np.arange(len(tokens))
    else:
        slots, pos = place(cache, len(tokens))
        cache["tokens"] = cache["tokens"] + tokens
    emb = w.get("nextn.embed_tokens.weight", W["token_embd.weight"])
    e = rms_norm(np.asarray(emb)[tokens], eps, w["nextn.enorm.weight"])
    h = rms_norm(h_prev, eps, w["nextn.hnorm.weight"])
    x = np.concatenate([e, h], axis=-1) @ w["nextn.eh_proj.weight"].T
    out = dict(pos=pos, slots=slots, tok_embd=np.asarray(emb)[tokens], e_norm=e, h_norm=h, eh_proj=x)
    out["attn_norm"] = an = rms_norm(x, eps, w["attn_norm.weight"])
    out["attn"] = a = attention_layer(an, w, hp, il, pos, cache, slots)
    out["ffn_inp"] = f = a["out"] + x
    out["ffn_norm"] = fn = rms_norm(f, eps, w["ffn_norm.weight"])
    out["ffn"] = ff = ffn_layer(fn, w, hp)
    out["post_ffn"] = post = ff["out"] + f
    head_norm = w.get("nextn.shared_head_norm.weight", w.get("layer_output_norm.weight", W["output_norm.weight"]))
    out["h_nextn"] = post
    out["head_norm"] = hn = rms_norm(post, eps, head_norm)
    out["logits"] = hn @ w.get("nextn.shared_head_head.weight", W["output.weight"]).T
    return out
