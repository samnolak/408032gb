"""Graph branches of llama.cpp glm5-next / qwen4exp that depend on GGUF hyperparameters or on which tensors a file has.

Each probe names the code it is about by literal anchors (a substring of one source line); tools/gguf_headers/
analyze.py finds each anchor's line in every llama.cpp checkout it is given and prints file:line per pin, so no line
number here is typed by hand. `pins` lists the checkouts that must contain the anchor (tests/test_gguf_headers.py
checks it when the sources are available); "ec7630a" is the CI pin, "79e2e74" the R1 pin (tools/engines/pins.json).

A probe's regime(m) returns the set of branch labels a model takes; a branch of the real file is "exercised by the tiny
models" when its labels are a subset of the union over the tiny variants (default, --mtp, --quant-dims). detail(m) is
the numbers behind it. m is analyze.Model (KV by arch-relative key, tensor names and shapes).
Code read at ec7630a640789c393694fb194f1bbbf0369fc62d and 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (N15).
"""
import math

BOTH = ("ec7630a", "79e2e74")
R1 = ("79e2e74",)
EC = ("ec7630a",)
GLM = "src/models/glm5-next.cpp"
QWEN = "src/models/qwen4exp.cpp"
GRAPH = "src/llama-graph.cpp"
MODEL = "src/llama-model.cpp"


class Probe:
    def __init__(self, pid, graphs, what, anchors, regime, detail=lambda m: ""):
        self.id, self.graphs, self.what, self.anchors, self.regime, self.detail = \
            pid, graphs, what, anchors, regime, detail


def _f(v):
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return v


def _layers(m):
    n = m.a("block_count") or 0
    return n - (m.a("nextn_predict_layers") or 0)


def _per_layer(v, n):
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v] * n


def _dsa_layers(m):
    kv = _per_layer(m.a("attention.head_count_kv"), m.a("block_count") or 0)
    return [i for i, x in enumerate(kv) if x]


def _glm_kda_bound(m):
    v = _f(m.a("kda.gate_lower_bound"))
    return {"softplus decay (no lower bound)"} if v is None or v == -math.inf else {"bounded sigmoid decay"}


def _glm_indexer(m):
    t = m.a("attention.indexer.types")
    if t is None:
        return {"full indexer in every DSA layer"}
    dsa = [i for i in _dsa_layers(m) if i < len(t)]
    out = {"full indexer"} if any(t[i] for i in dsa) else set()
    if any(not t[i] for i in dsa):
        out.add("shared indexer (reuses the previous layer's selection)")
    return out


def _rope_glm(m):
    v = m.a("rope.dimension_count")
    if v is None:
        return {"rope.dimension_count absent: n_rot = key_length, the nope-only assert fails"}
    return {"n_rot 0 (nope MLA)"} if v == 0 else {f"n_rot {v}: the nope-only assert fails"}


def _groups(m):
    g, u = m.a("expert_group_count"), m.a("expert_group_used_count")
    return {f"group-limited routing ({u} of {g} groups)"} if (g or 0) > 1 else {"no group step"}


def _clamp(key, fused):
    def f(m):
        v = [x for x in _per_layer(m.a(key), m.a("block_count") or 0) if x is not None]
        if not any(_f(x) > 1e-6 for x in v):
            return {"no clamp"}
        return {"fused swiglu_clamp" if fused else "generic clamp (arch not in the fused list)"}
    return f


def _clamp_detail(key):
    def f(m):
        v = m.a(key)
        return f"{key} = {sorted(set(_per_layer(v, m.a('block_count') or 0)))}" if v is not None else f"{key} absent"
    return f


def _sections(m):
    s = m.a("rope.dimension_sections") or []
    names = ("t", "h", "w", "e")
    return {f"section {names[i]} non-empty" for i, x in enumerate(s[:4]) if x}


def _qwen_full_attn(m):
    n = _layers(m)
    rec = m.a("attention.recurrent_layers")
    if rec is not None:
        full = [i for i, r in enumerate(rec[:n]) if not r]
    else:
        k = m.a("full_attention_interval") or 4
        full = [i for i in range(n) if (i + 1) % k == 0]
    cr = _per_layer(m.a("attention.compress_ratios"), m.a("block_count") or 0)
    out = set()
    for i in full:
        out.add("QSA full attention" if i < len(cr) and cr[i] else "dense full attention (compress ratio 0)")
    return out


def _n_sel(m):
    k, kp = m.a("attention.indexer.top_k") or 0, m.a("attention.indexer.kpool") or 1
    tail = kp - 1 if m.a("attention.indexer.kpool_select_tail", True) else 0
    return f"top_k {k}, kpool {kp}: at most {k // kp * kp + tail} selected cells per token (ec7630a gathers them)"


def _has(m, pattern):
    return any(pattern(n) for n in m.names)


PROBES = [
    # ------------------------------------------------------------------ glm5-next
    Probe("glm.kda_gate_bound", ("glm5-next",), "KDA decay gate: bounded sigmoid vs softplus",
          [(GLM, "ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,", BOTH),
           (GLM, "if (hparams.kda_gate_lower_bound > -INFINITY) {", BOTH)],
          _glm_kda_bound, lambda m: f"kda.gate_lower_bound = {m.a('kda.gate_lower_bound')}"),
    Probe("glm.layer_pattern", ("glm5-next",), "KDA (head_count_kv 0) vs DSA layers",
          [(GLM, "hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;", BOTH)],
          lambda m: {"KDA + DSA every 4th layer (3, 7, ...)"} if _dsa_layers(m)[:_layers(m)] and
          [i for i in _dsa_layers(m) if i < _layers(m)] == [i for i in range(_layers(m)) if i % 4 == 3]
          else {"other KDA/DSA pattern"},
          lambda m: f"DSA layers {_dsa_layers(m)} of {m.a('block_count')}"),
    Probe("glm.indexer_types", ("glm5-next",), "DSA indexer full per layer vs shared with the previous DSA layer",
          [(GLM, "ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES,", BOTH),
           (GLM, "if (il >= (int) hparams.n_layer() || hparams.is_indexer_full(il)) {", BOTH)],
          _glm_indexer, lambda m: f"attention.indexer.types = {m.a('attention.indexer.types')}"),
    Probe("glm.kpool_select_tail", ("glm5-next",), "DSA selection appends the incomplete k-pool tail",
          [(GLM, "if (hparams.indexer_kpool_select_tail) {", BOTH)],
          lambda m: {"tail appended"} if m.a("attention.indexer.kpool_select_tail", True) else {"no tail"},
          lambda m: f"kpool_select_tail = {m.a('attention.indexer.kpool_select_tail')}"),
    Probe("glm.dsa_threshold", ("glm5-next",),
          "sparse DSA selection only above top_k context tokens; at ec7630a ubatches <= 16 gather the selection",
          [(GLM, "const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);", BOTH),
           (GLM, "inp->gather = (int64_t) n_tokens <= max_ub && (int64_t) n_kv > n_sel;", EC)],
          lambda m: {"selection active above top_k"}, _n_sel),
    Probe("glm.rope", ("glm5-next",), "MLA must be nope-only (rope.dimension_count 0)",
          [(GLM, 'GGML_ASSERT(hparams.n_rot() == 0 && "GLM5-Next MLA is nope-only");', BOTH),
           (MODEL, "ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);", BOTH)],
          _rope_glm, lambda m: f"rope.dimension_count = {m.a('rope.dimension_count')}, key_length = {m.a('attention.key_length')}"),
    Probe("glm.hc", ("glm5-next",), "mHC with 4 streams only", [(GLM, "GGML_ASSERT(hparams.dsv4_hc_mult == 4", BOTH)],
          lambda m: {f"hc {m.a('hyper_connection.count')}"},
          lambda m: f"sinkhorn {m.a('hyper_connection.sinkhorn_iterations')}, eps {m.a('hyper_connection.epsilon')}"),
    Probe("glm.dense_lead", ("glm5-next",), "leading dense FFN layers, then MoE + shared expert",
          [(GLM, "if ((uint32_t) il < hparams.n_layer_dense_lead) {", BOTH)],
          lambda m: ({"dense FFN"} if (m.a("leading_dense_block_count") or 0) > 0 else set()) | {"MoE + shared expert"},
          lambda m: f"leading_dense_block_count = {m.a('leading_dense_block_count')}"),
    Probe("glm.gating", ("glm5-next",), "router gating function (absent or none = sigmoid)",
          [(GLM, "if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {", BOTH)],
          lambda m: {{None: "sigmoid", 0: "sigmoid", 1: "softmax", 2: "sigmoid"}.get(m.a("expert_gating_func"),
                                                                                    f"gating {m.a('expert_gating_func')}")},
          lambda m: f"expert_gating_func = {m.a('expert_gating_func')}"),
    Probe("glm.shexp_clamp", ("glm5-next",), "shared-expert SwiGLU clamp: own key or the routed one",
          [(GLM, "if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,", BOTH),
           (GRAPH, "const float limit = hparams.swiglu_clamp_shexp[il];", BOTH)],
          lambda m: {"own shexp clamp key"} if m.has("swiglu_clamp_shexp") else {"shexp clamp = routed clamp"},
          _clamp_detail("swiglu_clamp_shexp")),
    Probe("glm.layer_norm_eps", ("glm5-next",), "LayerNorm eps of the indexer key norm (key absent: 0)",
          [(GLM, "ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,", BOTH),
           (GRAPH, "case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);", BOTH)],
          lambda m: {f"eps {m.a('attention.layer_norm_epsilon')}"} if m.has("attention.layer_norm_epsilon")
          else {"eps 0 (key absent)"}),
    Probe("glm.ff_exp_array", ("glm5-next",), "expert FFN width: one value or per layer",
          [(GLM, "ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);",
            BOTH)],
          lambda m: {"per-layer array"} if isinstance(m.a("expert_feed_forward_length"), list) else {"scalar"},
          lambda m: f"expert_feed_forward_length = {m.a('expert_feed_forward_length')}"),
    Probe("glm.kda_qkv", ("glm5-next",), "KDA q/k/v projections: separate (the graph reads wq/wk/wv) or fused attn_qkv",
          [(GLM, "create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner,", BOTH),
           (MODEL, "void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,", BOTH)],
          lambda m: {"fused attn_qkv (not read by the glm5-next graph)"} if _has(m, lambda n: n.endswith(".attn_qkv.weight"))
          else {"separate attn_q/k/v"}),
    Probe("glm.conv_rank", ("glm5-next",), "KDA conv1d weights 4-D {d_conv,1,d_inner,1} or 3-D",
          [(GLM, 'ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);',
            BOTH)],
          lambda m: {f"{len(s)}-D conv1d" for n, s in m.shapes.items() if ".ssm_conv1d_" in n} or {"no conv1d"}),
    Probe("glm.mtp", ("glm5-next",), "MTP (NextN) block: count; the MTP graph exists at 79e2e74 only",
          [(GLM, 'GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM5-Next MTP currently supports a single NextN block");',
            R1)],
          lambda m: {f"nextn_predict_layers {m.a('nextn_predict_layers') or 0}"}),
    Probe("glm.mtp_heads", ("glm5-next",), "MTP block's own embedding / head / head norm instead of the trunk's",
          [(GLM, "layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd", R1),
           (GLM, "layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm", R1),
           (GLM, "layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;", R1)],
          lambda m: {f"own nextn.{k}" for k in ("embed_tokens", "shared_head_head", "shared_head_norm")
                     if _has(m, lambda n, k=k: f".nextn.{k}." in n)} or {"trunk embedding, head and norm"}),
    # ------------------------------------------------------------------ both (generic builders)
    Probe("moe.groups", ("glm5-next", "qwen4exp"), "group-limited routing (expert_group_count > 1)",
          [(MODEL, "ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);", BOTH),
           (GRAPH, "if (hparams.n_expert_groups > 1 && n_tokens > 0) {", BOTH)],
          _groups, lambda m: f"expert_group_count = {m.a('expert_group_count')}, used = {m.a('expert_group_used_count')}"),
    Probe("moe.norm_w", ("glm5-next",), "renormalise the selected expert weights",
          [(GRAPH, "if (norm_w) {", BOTH)],
          lambda m: {"renormalised"} if m.a("expert_weights_norm") else {"not renormalised"},
          lambda m: f"expert_weights_norm = {m.a('expert_weights_norm')}"),
    Probe("moe.scale", ("glm5-next", "qwen4exp"), "routed weights scaled (scale not 0 or 1)",
          [(GRAPH, "if (w_scale != 0.0f && w_scale != 1.0f) {", BOTH)],
          lambda m: {"scaled"} if (m.a("expert_weights_scale") or 0) not in (0, 1) else {"unscaled"},
          lambda m: f"expert_weights_scale = {m.a('expert_weights_scale')}"),
    Probe("moe.clamp_glm", ("glm5-next",), "routed SwiGLU clamp (glm5-next: fused swiglu_clamp)",
          [(GRAPH, "const float limit = hparams.swiglu_clamp_exp[il];", BOTH)],
          _clamp("swiglu_clamp_exp", True), _clamp_detail("swiglu_clamp_exp")),
    Probe("moe.clamp_qwen", ("qwen4exp",), "routed SwiGLU clamp (qwen4exp is not in the fused list: generic clamp)",
          [(GRAPH, "const float limit = hparams.swiglu_clamp_exp[il];", BOTH)],
          _clamp("swiglu_clamp_exp", False), _clamp_detail("swiglu_clamp_exp")),
    Probe("moe.used_array", ("glm5-next", "qwen4exp"), "experts per token: one value or per layer",
          [(MODEL, "ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);",
            BOTH)],
          lambda m: {"per-layer array"} if isinstance(m.a("expert_used_count"), list) else {"scalar"},
          lambda m: f"expert_used_count = {m.a('expert_used_count')}, expert_count = {m.a('expert_count')}"),
    Probe("rope.scaling", ("qwen4exp",), "RoPE scaling type (YaRN etc. change ggml_rope_multi's factors)",
          [(MODEL, "ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);", BOTH)],
          lambda m: {f"rope scaling {m.a('rope.scaling.type') or 'none'}"},
          lambda m: (f"factor {m.a('rope.scaling.factor')}, original ctx {m.a('rope.scaling.original_context_length')}, "
                     f"freq_base {m.a('rope.freq_base')}")),
    # ------------------------------------------------------------------ qwen4exp
    Probe("qwen.hc", ("qwen4exp",), "low-rank hyper-connections (count > 1)",
          [(QWEN, "if (hparams.dsv4_hc_mult <= 1) {", BOTH)],
          lambda m: {"hyper-connections"} if (m.a("hyper_connection.count") or 0) > 1 else {"rejected (count <= 1)"},
          lambda m: f"hc {m.a('hyper_connection.count')}, low rank {m.a('hyper_connection.low_rank')}"),
    Probe("qwen.recurrent_source", ("qwen4exp",), "linear/full layer pattern from the array or the interval rule",
          [(QWEN, "if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS,", BOTH)],
          lambda m: {"recurrent_layers array"} if m.has("attention.recurrent_layers") else {"full_attention_interval rule"},
          lambda m: f"full_attention_interval = {m.a('full_attention_interval')}"),
    Probe("qwen.full_attention", ("qwen4exp",), "full-attention layers: QSA (compress ratio > 0) or dense",
          [(QWEN, "const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;", BOTH)],
          _qwen_full_attn, lambda m: f"compress_ratios = {m.a('attention.compress_ratios')}"),
    Probe("qwen.ple", ("qwen4exp",), "per-layer n-gram embeddings (PLE) on a linear layer",
          [(QWEN, "if (hparams.is_ple(il)) {", BOTH)],
          lambda m: {"PLE layer"} if m.a("ple.layers") else {"no PLE"},
          lambda m: (f"ple.layers {m.a('ple.layers')}, ngram {m.a('ple.ngram_size')}, heads/ngram "
                     f"{m.a('ple.heads_per_ngram')}, conv {m.a('ple.conv_kernel')}")),
    Probe("qwen.ple_image_token", ("qwen4exp",), "PLE image token id: own key or the EOS fallback",
          [(QWEN, "ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);", BOTH)],
          lambda m: (set() if not m.a("ple.layers") else
                     {"image token id key"} if m.has("ple.image_token_id") else {"EOS fallback"})),
    Probe("qwen.output", ("qwen4exp",), "own output.weight or tied to token_embd",
          [(QWEN, "output = create_tensor(tn(LLM_TENSOR_OUTPUT, \"weight\"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);",
            BOTH)],
          lambda m: {"own output.weight"} if "output.weight" in m.names else {"tied to token_embd"}),
    Probe("qwen.attn_scale", ("qwen4exp",), "attention scale: 1/sqrt(head_dim) or attention.scale",
          [(QWEN, "const float kq_scale = hparams.f_attention_scale == 0.0f ?", BOTH)],
          lambda m: {f"attention.scale {m.a('attention.scale')}"} if m.a("attention.scale") else {"1/sqrt(head_dim)"}),
    Probe("qwen.ff_defaults", ("qwen4exp",), "expert / shared-expert widths: explicit keys or derived from n_ff",
          [(QWEN, "const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;", BOTH),
           (QWEN, "const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;", BOTH)],
          lambda m: {"explicit n_ff_exp" if m.a("expert_feed_forward_length") else "n_ff_exp = n_ff / n_expert_used",
                     "explicit n_ff_shexp" if m.a("expert_shared_feed_forward_length") else "n_ff_shexp = n_ff"}),
    Probe("qwen.rope_sections", ("qwen4exp",), "IMRoPE sections in use",
          [(QWEN, "ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);", BOTH)],
          _sections, lambda m: f"sections {m.a('rope.dimension_sections')}, n_rot {m.a('rope.dimension_count')}, "
                               f"head {m.a('attention.key_length')}"),
    Probe("qwen.partial_rope", ("qwen4exp",), "partial rotary (n_rot < head_dim)",
          [(QWEN, "Qcur = ggml_rope_multi(", BOTH)],
          lambda m: {"partial rotary"} if (m.a("rope.dimension_count") or 0) < (m.a("attention.key_length") or 0)
          else {"full rotary"}),
    Probe("qwen.gate_up_fused", ("qwen4exp",), "routed experts: separate gate/up or fused gate_up_exps",
          [(GRAPH, "if (gate_up_exps) {", BOTH)],
          lambda m: {"fused ffn_gate_up_exps"} if _has(m, lambda n: n.endswith(".ffn_gate_up_exps.weight"))
          else {"separate gate/up"}),
    Probe("qwen.mtp", ("qwen4exp",), "MTP (NextN) block count",
          [(QWEN, 'GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP has a single block");', BOTH)],
          lambda m: {f"nextn_predict_layers {m.a('nextn_predict_layers') or 0}"}),
]
