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

Several MTP heads (N17; MiMo-V2-Flash / V2.5 / V2.6 ship 3, conversion/mimo.py:22-24 @79e2e74): head k is layer
n_layer + k (graph_mtp takes il = n_layer + cparams.nextn_layer_offset and asserts 0 <= offset < n_layer_nextn,
M:276-279 @ec7630a, M:267-270 @79e2e74); every head is a sliding-window layer of the one MTP context, whose memory holds
all n_layer_nextn layers on shared cells (the filter il >= n_layer, LM:2807-2812). common/speculative.cpp drives them
(chain_heads = n_mtp_layers > 1, :1482 @79e2e74, :1421 @ec7630a; the same code at both pins):
  process (every target call): for head k = 0 .. H-1: llama_memory_seq_rm(mtp, seq, first position of the call, -1),
      nextn_layer_offset = k, decode the call's tokens with the TARGET's h rows (token p with the target h_nextn of p-1,
      the first with the pending row); offset back to 0 (:1583-1621 @79e2e74, :1519-1557 @ec7630a). So every head
      holds the K/V of its own layer for every processed position, each computed from the target's h.
  draft (one sequence, id_last at pos0, pending = the target row of pos0-1; at most H steps, :1485): step k = 0 .. H-1
      runs head k on the whole prefix after seq_rm(pos0): tokens [id_last, d_1, .., d_k] at pos0 .. pos0+k with the
      rows [pending, h_0, .., h_{k-1}], where h_j is the MTP h_nextn that step j output at its last row (pos0 + j);
      only the last row is an output, its argmax is d_{k+1} (:1681-1688, 1700-1708, 1767-1777 @79e2e74; :1604-1611,
      1623-1631, 1686-1696 @ec7630a). Head k thus takes, at its new token, the h of head k-1 one position back, and at
      the earlier rows those of heads 0 .. k-2 (and the target's at pos0).
  llama-server around it (tools/server/server-context.cpp @79e2e74; 3063, 3998 @ec7630a): after drafting it removes
      the draft positions from the MTP memory (seq_rm(pos0), :3423); after verification, accept() pairs the next token
      with the target row of the last accepted position (speculative.cpp:1814-1827) and common_memory::seq_rm cuts both
      contexts after it (:4505, common/common.cpp:1558-1563 @ec7630a).
mtp_forward_chunk(..., offset=k) is head k; mtp_process / mtp_draft are the two loops above (tools/dump/dump_tensors.cpp
DUMP_MTP replays them, with the server's seq_rm steps as plan steps "d<n>" and "r<n>").
The cells are shared but each head writes only its own layer, and seq_rm moves the cache head back to the first freed
cell (KV:410-419): when head 0's ubatch wraps around the end of the SWA ring, heads 1 .. H-1 place it from that cell on,
the cells' positions follow them, and head 0's layer keeps under those cells K/V of other positions. The cache records
per layer the position each cell's K/V was computed for (cache["P"]); attention_layer reports the rows that see such a
cell (attn["foreign"]): there llama.cpp follows this cell model and not the full-sequence oracle
(docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md).

Precision (N17; prec=None keeps every result above bit for bit): prec = ref/fattn.py precision(kv, fa) is the cache
type of -ctk / -ctv and the -fa flag. With an f16 cache the K and V rows are stored rounded to f16 (cpy_k / cpy_v are
ggml_set_rows, src/llama-kv-cache.cpp:1328-1361; ref/fattn.py store()); flash attention casts f32 K / V to f16 anyway
(LG:2660-2666) and runs the CPU path of ref/fattn.py attend() with the layer's sinks (one_chunk below 64 query rows,
tiled at 64 and more; one query row over n_kv >= 512 cells takes the split-KV path, which with -t 1 equals one_chunk,
ref/fattn.py docstring). Every layer's mask is the one of the f32 path: llama_kv_cache::set_input_kq_mask fills an f16
(flash attention) or f32 mask with the same template (KV:1566-1716, 1789-1793 @ec7630a), 0 or -inf, both exact in f16.
The rows the online softmax visits are the visible cells in CELL order (a wrapped SWA ring visits position 256 in cell 0
before 249..255); the full-sequence oracle (cache None) visits positions in order.

Quantized weights (N10): a weight may be a ref/quant.py QWeight; every product with a weight is written x @ w.T or
w @ x, which for a QWeight applies the CPU activation conversion. Everything else runs in float64; shapes are
numpy-natural (token-major), weights are the GGUF arrays as numpy reads them (ggml ne reversed).
"""
import numpy as np

from ref import fattn as F
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
    trunk layers il < n_layer); mtp=True: an MTP context (the MTP layers il = n_layer .. n_layer + n_nextn - 1, one per
    head, on shared cells). swa_size: a SWA cache of that many cells instead of llama.cpp's pad(min(n_ctx, n_swa +
    n_ubatch), 256) (property tests only)."""
    layers = list(range(hp["n_layer"], hp["n_layer"] + max(1, hp.get("n_nextn", 1)))) if mtp else \
        list(range(hp["n_layer"]))
    size_swa = swa_size or pad(min(n_ctx, hp["n_swa"] + n_ubatch), KV_PAD)
    c = dict(tokens=[], n_ctx=n_ctx, n_ubatch=n_ubatch, layers=layers, base=Cells(n_ctx), swa=Cells(size_swa, hp["n_swa"]),
             K={}, V={}, P={})
    for il in layers:
        size = size_swa if hp["swa"][il] else n_ctx
        c["K"][il] = np.zeros((size, hp["n_head_kv"][il], hp["k_dim"]))
        c["V"][il] = np.zeros((size, hp["n_head_kv"][il], hp["v_dim"]))
        c["P"][il] = np.full(size, -1, dtype=np.int64)    # the position each cell's K/V of this layer was computed for
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


def kq_scale(hp, prec=None):
    """The attention scale. prec None (the f32 contract): 1/sqrt(k_dim) in float64. With prec: llama.cpp's f32 value
    1.0f/sqrtf(k_dim) (M:179 @ec7630a, M:171 @79e2e74; ref/fattn.py kq_scale_f32), which one_chunk multiplies an f32
    score by (ggml/src/ggml-cpu/ops.cpp:8978 @ec7630a): for k_dim 24 it is one f32 step below float32(1/sqrt(24))."""
    return F.kq_scale_f32(hp["k_dim"]) if prec else 1.0 / np.sqrt(hp["k_dim"])


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


def attention_layer(x, w, hp, il, pos, cache=None, slots=None, prec=None):
    """One attention block on x = rms_norm(inpL) * attn_norm at absolute positions pos. cache None: the full sequence
    from position 0 (the keys are this call's rows, prefill semantics); else the context's cache (new_cache) and slots
    {"base"/"swa": cells of this ubatch}: k, v are written there and the attention runs over its n_kv cells.
    prec: module docstring ("Precision"); the rows' modes are prec["modes"][position] where given, else the mode the
    ubatch size implies (ref/fattn.py row_modes); returned as out["modes"], the stored rows as out["stored"]."""
    pos = np.asarray(pos)
    swa = hp["swa"][il]
    q0, k0, v = project_qkv(x, w, hp, il)
    base = hp["rope_base_swa"] if swa else hp["rope_base"]
    q, k = rope_neox(q0, pos, hp["n_rot"], base), rope_neox(k0, pos, hp["n_rot"], base)
    ks, vs = (F.store(k, prec), F.store(v, prec)) if prec else (k, v)
    out = dict(q_pre=q0, k_pre=k0, v=v, q=q, k=k, stored=dict(k=ks, v=vs))
    if cache is None:
        K, V, kpos = ks, vs, pos
    else:
        which = "swa" if swa else "base"
        idxs = slots[which]
        cache["K"][il][idxs] = ks
        cache["V"][il][idxs] = vs
        cache["P"][il][idxs] = pos
        n = cache[which].n_kv()
        K, V, kpos = cache["K"][il][:n], cache["V"][il][:n], cache[which].pos[:n].copy()
        # a visible cell whose K/V of THIS layer were computed for another position than the cell now holds: the
        # cells are shared by the layers of a cache, and only a layer that is written in every placement stays in
        # step (the MTP heads are not, module docstring and docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md)
        vis_ = visible(pos, kpos, hp["n_swa"] if swa else 0)
        foreign = vis_ & (cache["P"][il][:n][None, :] != kpos[None, :])
        out.update(idxs=list(idxs), n_kv=n, foreign=foreign.any(axis=1), foreign_cells=foreign)
    scale = kq_scale(hp, prec)
    n_swa, sinks = hp["n_swa"] if swa else 0, w.get("attn_sinks.weight")
    if prec and F.kv_f16(prec) and not prec.get("fa"):
        # mm16: kq = ggml_mul_mat(K f16, q) converts q to f16 (LG:2696, the vec_dot_type of F16)
        kq, p, _ = attend(F.f16(q), pos, K, V, kpos, sinks, n_swa, scale)
    else:
        kq, p, kqv = attend(q, pos, K, V, kpos, sinks, n_swa, scale)
    if prec:
        modes = F.row_modes(prec, pos)
        vis = visible(pos, kpos, n_swa)
        cells = [np.nonzero(vis[t])[0] for t in range(len(pos))]
        kqv = F.attend_rows(q, K, V, cells, scale, modes, sinks).reshape(len(pos), -1)
        out["modes"] = modes
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

def _layer(h, W, hp, il, pos, cache=None, slots=None, moe_ids=None, prec=None):
    w = layer_weights(W, il)
    L = dict(inp=h)
    L["attn_norm"] = x = rms_norm(h, hp["rms_eps"], w["attn_norm.weight"])
    L["attn"] = a = attention_layer(x, w, hp, il, pos, cache, slots, prec)
    L["ffn_inp"] = f = a["out"] + h
    L["ffn_norm"] = fn = rms_norm(f, hp["rms_eps"], w["ffn_norm.weight"])
    L["ffn"] = ff = ffn_layer(fn, w, hp, moe_ids)
    L["l_out"] = ff["out"] + f
    return L["l_out"], L


def _head(h, W, hp, out):
    norm = rms_norm(h, hp["rms_eps"], W["output_norm.weight"])
    out.update(h_nextn=h, result_norm=norm, logits=norm @ W["output.weight"].T)
    return out


def forward(tokens, W, hp, overrides=None, prec=None):
    """The full sequence from position 0 in one pass (prefill semantics: every key of the sequence is available).
    overrides {("moe", il): {position: expert ids}} replays reference choices among tied experts. prec: module
    docstring (give prec["modes"] for every position: the ubatch size of this one pass is not llama.cpp's).
    Returns dict(layers=[per layer], h_nextn, result_norm, logits)."""
    overrides = overrides or {}
    tokens = np.asarray(tokens)
    pos = np.arange(len(tokens))
    h = W["token_embd.weight"][tokens]
    out = dict(pos=pos, inp_embd=h, layers=[])
    for il in range(hp["n_layer"]):
        h, L = _layer(h, W, hp, il, pos, moe_ids=overrides.get(("moe", il)), prec=prec)
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


def forward_chunk(tokens, W, hp, cache, overrides=None, prec=None):
    """One ubatch of a single sequence that continues `cache` (new_cache() at the start); updates cache in place.
    Returns a dict like forward() for the ubatch's rows, with the absolute positions and per layer the cells written
    (attn["idxs"]), the cells read (attn["kpos"], n_kv) and kq / probs over those cells. prec: module docstring."""
    overrides = overrides or {}
    tokens = [int(t) for t in tokens]
    slots, pos = place(cache, len(tokens))
    cache["tokens"] = cache["tokens"] + tokens
    h = W["token_embd.weight"][tokens]
    out = dict(pos=pos, slots=slots, inp_embd=h, layers=[])
    for il in range(hp["n_layer"]):
        moe = overrides.get(("moe", il))
        h, L = _layer(h, W, hp, il, pos, cache, slots,
                      {int(t) - int(pos[0]): e for t, e in moe.items()} if moe else None, prec)
        out["layers"].append(L)
    return _head(h, W, hp, out)


def cache_truncate(cache, n_keep):
    """llama_memory_seq_rm(mem, 0, n_keep, -1) (KVI:113-120 -> KV:382-420): free the cells holding positions >= n_keep
    in both caches. Nothing is restored: what the SWA cache already overwrote stays lost (module docstring)."""
    cache["base"].seq_rm(n_keep, -1)
    cache["swa"].seq_rm(n_keep, -1)
    cache["tokens"] = cache["tokens"][:n_keep]
    return cache


def forward_calls(tokens, W, hp, plan, n_ctx=512, n_ubatch=512, swa_size=None, prec=None):
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
            outs.append(forward_chunk(tokens[ti + j:ti + j + m], W, hp, cache, prec=prec))
        ti += n
    assert ti == len(tokens), (ti, len(tokens))
    return outs, cache


# ---------------------------------------------------------------- MTP block (nextn_predict_layers >= 1)

def mtp_forward_chunk(tokens, h_prev, W, hp, cache=None, pos=None, offset=0, prec=None):
    """graph_mtp (M:272-407) on one ubatch of the MTP context (module docstring). tokens: the ubatch's ids; h_prev
    (T, n_embd): the pre-norm hidden row paired with each token. cache: new_cache(..., mtp=True) of the MTP context
    (incremental); None: all positions 0..T-1 at once (prefill semantics, pos ignored). offset: the head,
    cparams.nextn_layer_offset (layer n_layer + offset, 0 <= offset < n_nextn as M:277-279 asserts). prec: module
    docstring."""
    if not 0 <= offset < max(1, hp.get("n_nextn", 1)):
        raise ValueError(f"nextn_layer_offset {offset} out of range [0, {hp.get('n_nextn')})")
    il = hp["n_layer"] + offset
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
    out["attn"] = a = attention_layer(an, w, hp, il, pos, cache, slots, prec)
    out["ffn_inp"] = f = a["out"] + x
    out["ffn_norm"] = fn = rms_norm(f, eps, w["ffn_norm.weight"])
    out["ffn"] = ff = ffn_layer(fn, w, hp)
    out["post_ffn"] = post = ff["out"] + f
    head_norm = w.get("nextn.shared_head_norm.weight", w.get("layer_output_norm.weight", W["output_norm.weight"]))
    out["h_nextn"] = post
    out["head_norm"] = hn = rms_norm(post, eps, head_norm)
    out["logits"] = hn @ w.get("nextn.shared_head_head.weight", W["output.weight"]).T
    out["head"] = offset
    return out


def _snap(out, cache, hp, offset):
    """The K / V rows of the head's layer after a ubatch (the cache the graph read), as out["snap"]."""
    il = hp["n_layer"] + offset
    out["snap"] = (cache["K"][il].copy(), cache["V"][il].copy())
    return out


def mtp_ubatches(tokens, h_rows, W, hp, cache, n_ubatch, offset=0, prec=None, snap=False):
    """One MTP-context call: ceil(T / n_ubatch) ubatches of mtp_forward_chunk under head `offset`. snap: keep the
    head's cache rows after every ubatch (out["snap"])."""
    h_rows = np.asarray(h_rows, dtype=np.float64)
    outs = []
    for j in range(0, len(tokens), n_ubatch):
        o = mtp_forward_chunk(tokens[j:j + n_ubatch], h_rows[j:j + n_ubatch], W, hp, cache, offset=offset, prec=prec)
        outs.append(_snap(o, cache, hp, offset) if snap else o)
    return outs


def mtp_process(tokens, h_rows, W, hp, cache, n_ubatch, prec=None, snap=False):
    """common/speculative.cpp process() for one target call on the MTP context (module docstring): returns
    {head: [ubatch results]}. One head: a plain call. H > 1: every head after seq_rm from the call's first position."""
    n_heads = max(1, hp.get("n_nextn", 1))
    p0 = len(cache["tokens"])
    out = {}
    for head in range(n_heads):
        if n_heads > 1:
            cache_truncate(cache, p0)
        out[head] = mtp_ubatches(tokens, h_rows, W, hp, cache, n_ubatch, head, prec, snap)
    return out


def mtp_draft(id_last, pending, W, hp, cache, n_req, tokens=None, prec=None, snap=False):
    """common/speculative.cpp draft() for one sequence, greedy (module docstring), from the cache's next position.
    tokens: the draft tokens to feed (a reference's, e.g. llama.cpp's; the oracle's own argmax chain if None).
    Returns (the oracle's argmax per step, [per step dict(head, tokens, rows, pos0, out)]). The caller applies the
    server's seq_rm(pos0) afterwards."""
    n_heads = max(1, hp.get("n_nextn", 1))
    chain = n_heads > 1
    n_max = min(n_req, n_heads) if chain else n_req
    pos0 = len(cache["tokens"])
    pending = np.asarray(pending, dtype=np.float64)
    chain_h, own, fed, steps = [pending], [], [], []
    ids, rows = [int(id_last)], [pending]
    for i in range(n_max):
        if chain:
            cache_truncate(cache, pos0)
        p_call = len(cache["tokens"])
        out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=i if chain else 0, prec=prec)
        if snap:
            _snap(out, cache, hp, i if chain else 0)
        steps.append(dict(head=i if chain else 0, tokens=list(ids), rows=np.vstack(rows), pos0=p_call, out=out))
        own.append(int(np.argmax(out["logits"][-1])))
        fed.append(own[-1] if tokens is None or i >= len(tokens) else int(tokens[i]))
        if len(fed) >= n_max:
            break
        h = out["h_nextn"][-1]
        if chain:
            chain_h.append(h)
            ids, rows = [int(id_last)] + fed, list(chain_h)
        else:
            ids, rows = [fed[-1]], [h]
    return own, steps
