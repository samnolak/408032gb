#!/usr/bin/env python3
"""Offline analysis of the GGUF headers committed by the gguf-headers workflow (third_party/gguf-headers/<name>/).

Works from the committed bytes only (no network, no weights). Per request with PIN.json status "ok":
  integrity  every <file>.header.gz decompresses to the bytes PIN.json hashes, and gguf_header.summarize() of them
             reproduces the committed <file>.summary.json byte for byte;
  (e) identity: architecture string, tokenizer model / pre-tokenizer, vocabulary size, special token ids, and
             whether each given llama.cpp checkout knows the architecture string, the tokenizer model and the
             pre-tokenizer (file:line of the match in src/llama-arch.cpp / src/llama-vocab.cpp, or "not found");
  (d) expert_group_count / expert_group_used_count and what llama.cpp does with them;
  (c) the embedded tokenizer.chat_template against the vendored HF template named in the request ("analysis":
             {"hf_chat_template": ...}): byte-equal, or the first differing byte and a line diff;
  (b) the real hyperparameters (all arch-prefixed KV) next to the tinygen models the parity tests use
             (tools/tinygen/<tiny>_tiny.py variants default / --mtp / --quant-dims [/ --dialect ik], built here; needs
             numpy and LLAMA_CPP_GGUF_PY), the config.json cross-check, and the graph branches of tools/gguf_headers/
             gguf_branches.py: the labels the real file and the tiny models take, "exercised by tiny" or NOT, with the
             file:line of every anchor at every --src checkout;
  (a)+(f) tools/gguf_quant_audit.py on the header-only shards rebuilt from the committed bytes (it reads headers
             only), i.e. the ADR-012 D5 floor: routed experts and non-expert matrices >= 4.0 bpw.
  keys       (N18) the keys load_hparams / load_arch_hparams of each --src checkout read without a default
             (src/llama-model.cpp, src/models/<arch>.cpp; calls inside a nested block are listed as conditional), and
             which of them the file lacks;
  experts    (N18) the ggml type of every routed-expert projection, counted over the layers.
Architectures: glm5-next, qwen4exp, mimo2 (N18: config map, graph-branch probes, tiny model tools/tinygen/mimo2_tiny.py).
A request with "analysis": {"mode": "mtp_files"} (N18) holds separate MTP (NextN) files: each file is its own model;
per file the KV / tensor layout, the NextN layer against a reference (another request's NextN layer, "mtp_reference":
{"request": NAME}, or a tiny model's, {"tiny": ..., "variant": ...}), the trunk request's KV ("trunk": NAME), the
required keys, the quant audit, and per --src checkout what that pin's loader asks of an MTP-only file
(gguf_branches.MTP_LOAD) and how a separate MTP file is attached (gguf_branches.MTP_ATTACH); the loader verdict is
PROVISIONAL (by reading; nothing is loaded).
A discover request's LISTING.json is printed as a table (searches and named repos). Exit 1 if an integrity check fails.

Usage: python3 tools/gguf_headers/analyze.py [--root third_party/gguf-headers] [--name NAME ...]
         [--src ec7630a=<llama.cpp checkout>] [--src 79e2e74=<checkout>] [--src ik=<ik_llama.cpp>] [--json OUT]
"""
import argparse
import difflib
import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import gguf_branches as branches  # noqa: E402
import gguf_header as G  # noqa: E402
import gguf_quant_audit as QA  # noqa: E402

TINY = {  # request "analysis.tiny" -> (tools/tinygen module, variants: (label, build kwargs, ARCH override))
    "glm5next": ("glm5next_tiny", [("default", {}, "glm5-next"), ("--mtp", {"mtp": True}, "glm5-next"),
                                   ("--quant-dims", {"quant": True}, "glm5-next"),
                                   ("--dialect ik", {"dialect": "ik"}, "glm5next")]),
    "qwen4exp": ("qwen4exp_tiny", [("default", {}, None), ("--mtp", {"mtp": True}, None),
                                   ("--quant-dims", {"quant": True}, None)]),
    # mimo2_tiny.build(path, seed, ...) has no cfg; a variant whose options build() does not take is skipped (noted)
    "mimo2": ("mimo2_tiny", [("default", {}, None),
                             ("--fused-qkv --sinks swa --value-scale 0.75 --mtp",
                              {"fused_qkv": True, "sinks": "swa", "value_scale": 0.75, "mtp": True}, None)]),
}

# config.json (text_config) key -> GGUF key suffix [, (transform(config value, config, gguf Model), label)]; "layers" =
# block_count - nextn_predict_layers; ABSENT in the report = the GGUF has no such key


def _mimo_kv_arr(v, cfg, m):
    """per-layer KV heads as conversion/mimo.py:151-157 writes them: SWA count where the pattern is 1, else the full
    count, plus the SWA count for each MTP layer of this file"""
    n = (m.a("nextn_predict_layers") or 0) if m is not None else 0
    return [cfg.get("swa_num_key_value_heads") if x == 1 else cfg.get("num_key_value_heads") for x in v] + \
        [cfg.get("swa_num_key_value_heads")] * n


def _mimo_pattern_arr(v, cfg, m):
    """conversion/mimo.py:155, 160: the HF pattern plus 1 (sliding window) per MTP layer of this file"""
    n = (m.a("nextn_predict_layers") or 0) if m is not None else 0
    return [int(x) for x in v] + [1] * n


CONFIG_MAP = {
    "glm5-next": [("num_hidden_layers", "layers"), ("num_nextn_predict_layers", "nextn_predict_layers"),
                  ("hidden_size", "embedding_length"), ("intermediate_size", "feed_forward_length"),
                  ("moe_intermediate_size", "expert_feed_forward_length"), ("n_routed_experts", "expert_count"),
                  ("num_experts_per_tok", "expert_used_count"), ("n_shared_experts", "expert_shared_count"),
                  ("first_k_dense_replace", "leading_dense_block_count"),
                  ("routed_scaling_factor", "expert_weights_scale"), ("norm_topk_prob", "expert_weights_norm"),
                  ("n_group", "expert_group_count"), ("topk_group", "expert_group_used_count"),
                  ("swiglu_limit", "swiglu_clamp_exp"), ("num_attention_heads", "attention.head_count"),
                  ("q_lora_rank", "attention.q_lora_rank"), ("kv_lora_rank", "attention.kv_lora_rank"),
                  ("qk_head_dim", "attention.key_length_mla"), ("v_head_dim", "attention.value_length_mla"),
                  ("qk_rope_head_dim", "rope.dimension_count"), ("rms_norm_eps", "attention.layer_norm_rms_epsilon"),
                  ("index_n_heads", "attention.indexer.head_count"), ("index_head_dim", "attention.indexer.key_length"),
                  ("index_topk", "attention.indexer.top_k"), ("index_kpool", "attention.indexer.kpool"),
                  ("index_kpool_always_select_tail", "attention.indexer.kpool_select_tail"),
                  ("hc_mult", "hyper_connection.count"), ("hc_sinkhorn_iters", "hyper_connection.sinkhorn_iterations"),
                  ("hc_eps", "hyper_connection.epsilon"), ("max_position_embeddings", "context_length"),
                  ("linear_attn_config.head_dim", "kda.head_dim"),
                  ("linear_attn_config.gate_lower_bound", "kda.gate_lower_bound"),
                  ("linear_attn_config.short_conv_kernel_size", "ssm.conv_kernel"), ("vocab_size", "@vocab")],
    "qwen4exp": [("num_hidden_layers", "layers"), ("mtp_num_hidden_layers", "nextn_predict_layers"),
                 ("hidden_size", "embedding_length"), ("num_attention_heads", "attention.head_count"),
                 ("num_key_value_heads", "attention.head_count_kv"), ("head_dim", "attention.key_length"),
                 ("num_experts", "expert_count"), ("num_experts_per_tok", "expert_used_count"),
                 ("moe_intermediate_size", "expert_feed_forward_length"),
                 ("shared_expert_intermediate_size", "expert_shared_feed_forward_length"),
                 ("full_attention_interval", "full_attention_interval"), ("linear_conv_kernel_dim", "ssm.conv_kernel"),
                 ("linear_key_head_dim", "ssm.state_size"), ("linear_num_key_heads", "ssm.group_count"),
                 ("linear_num_value_heads", "ssm.time_step_rank"), ("hc_count", "hyper_connection.count"),
                 ("hc_lowrank", "hyper_connection.low_rank"), ("indexer_n_heads", "attention.indexer.head_count"),
                 ("indexer_head_dim", "attention.indexer.key_length"), ("indexer_budget", "attention.indexer.top_k"),
                 ("rms_norm_eps", "attention.layer_norm_rms_epsilon"), ("max_position_embeddings", "context_length"),
                 ("rope_parameters.rope_theta", "rope.freq_base"),
                 # 1-based in the HF config; llama.cpp's converter writes i - 1 (conversion/qwen4exp.py:91-94 @79e2e74)
                 ("ple_layer_ids", "ple.layers", (lambda v, cfg, m: [i - 1 for i in v], "1-based")),
                 ("ngram_size", "ple.ngram_size"), ("heads_per_ngram", "ple.heads_per_ngram"),
                 ("ple_conv_kernel_size", "ple.conv_kernel"), ("vocab_size", "@vocab")],
    # conversion/mimo.py (identical at ec7630a and 79e2e74): set_gguf_parameters :143-175 and the TextModel base
    "mimo2": [("num_hidden_layers", "layers"), ("num_nextn_predict_layers", "nextn_predict_layers"),
              ("hidden_size", "embedding_length"), ("intermediate_size", "feed_forward_length"),
              ("moe_intermediate_size", "expert_feed_forward_length"), ("n_routed_experts", "expert_count"),
              ("num_experts_per_tok", "expert_used_count"), ("num_attention_heads", "attention.head_count"),
              ("hybrid_layer_pattern", "attention.head_count_kv",
               (_mimo_kv_arr, "swa_num_key_value_heads where 1 else num_key_value_heads, + SWA per MTP layer")),
              ("head_dim", "attention.key_length"), ("v_head_dim", "attention.value_length"),
              ("sliding_window", "attention.sliding_window"),
              ("hybrid_layer_pattern", "attention.sliding_window_pattern", (_mimo_pattern_arr, "+ 1 per MTP layer")),
              ("partial_rotary_factor", "rope.dimension_count",
               (lambda v, cfg, m: int(cfg.get("head_dim", 0) * v), "int(head_dim x factor)")),
              ("rope_theta", "rope.freq_base"), ("swa_rope_theta", "rope.freq_base_swa"),
              ("attention_value_scale", "attention.value_scale"),
              ("layernorm_epsilon", "attention.layer_norm_rms_epsilon"),
              ("max_position_embeddings", "context_length"), ("n_group", "expert_group_count"),
              ("topk_group", "expert_group_used_count"), ("vocab_size", "@vocab")],
}


class Model:
    """KV of the first shard (split.no 0) and the tensors of all shards."""

    def __init__(self, summaries):
        first = next((s for s in summaries if _kv(s).get("split.no", {}).get("value", 0) == 0), summaries[0])
        self.kv = _kv(first)
        self.arch = self.v("general.architecture")
        self.tensors = [t for s in summaries for t in s["tensors"]]
        self.names = {t["name"] for t in self.tensors}
        self.shapes = {t["name"]: t["shape"] for t in self.tensors}

    def v(self, key, default=None):
        e = self.kv.get(key)
        if e is None:
            return default
        return e.get("value", f"<{e.get('count')} x {e.get('elem_type')}, sha256 {e.get('sha256', '')[:12]}>")

    def a(self, suffix, default=None):
        return self.v(f"{self.arch}.{suffix}", default)

    def has(self, suffix):
        return f"{self.arch}.{suffix}" in self.kv

    def arch_keys(self):
        p = self.arch + "."
        return {k[len(p):]: self.v(k) for k in self.kv if k.startswith(p)}


def _kv(summary):
    return {e["key"]: e for e in summary["kv"]}


def fmt(v, width=60):
    if isinstance(v, list):
        if len(v) <= 6:
            s = json.dumps(v)
        else:
            counts = {}
            for x in v:
                counts[json.dumps(x)] = counts.get(json.dumps(x), 0) + 1
            s = f"[{len(v)}] " + ", ".join(f"{k} x{n}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    else:
        s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= width else s[:width - 3] + "..."


# ------------------------------------------------------------------------------------------------ sources

def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def locate(root, rel, needle=None, regex=None):
    """1-based line numbers of the lines of root/rel containing needle (or matching regex)."""
    try:
        with open(os.path.join(root, rel), errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    rx = re.compile(regex) if regex else None
    return [i + 1 for i, ln in enumerate(lines) if (rx.search(ln) if rx else needle in ln)]


def cite(rel, hits):
    if hits is None:
        return f"{rel}: file absent"
    return f"{rel}:{','.join(map(str, hits))}" if hits else "not found"


def known(root, kind, value):
    if value is None:
        return "n/a"
    rx = {"arch": (r'\{\s*LLM_ARCH_\w+\s*,\s*"%s"\s*\}', "src/llama-arch.cpp"),
          "pre": (r'tokenizer_pre\s*==\s*"%s"', "src/llama-vocab.cpp"),
          "model": (r'tokenizer_model\s*==\s*"%s"', "src/llama-vocab.cpp")}[kind]
    return cite(rx[1], locate(root, rx[1], regex=rx[0] % re.escape(value)))


# ------------------------------------------------------------------------------------------------ inputs

def load_request(d):
    with open(os.path.join(d, "PIN.json")) as f:
        pin = json.load(f)
    out = {"pin": pin, "files": [], "integrity": []}
    for rec in pin.get("files", []):
        with open(os.path.join(d, rec["header_gz"]), "rb") as f:
            gz = f.read()
        hdr = gzip.decompress(gz)
        with open(os.path.join(d, rec["summary"]), "rb") as f:
            committed = f.read()
        derived = G.dumps(G.summarize(G.parse(hdr), hdr)).encode()
        ok = (hashlib.sha256(hdr).hexdigest() == rec["header_sha256"] and len(hdr) == rec["header_bytes"]
              and hashlib.sha256(gz).hexdigest() == rec["header_gz_sha256"] and derived == committed)
        out["integrity"].append((rec["path"], ok))
        out["files"].append({"rec": rec, "header": hdr, "summary": json.loads(committed)})
    return out


def build_tiny(tiny, gguf_py, workdir):
    """Summaries of the tinygen variants, or (None, reason)."""
    if tiny not in TINY:
        return None, f"no tiny model for {tiny!r}"
    if not gguf_py:
        return None, "NOT RUN: set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py (and install numpy) to build the tiny models"
    try:
        import numpy  # noqa: F401
    except ImportError:
        return None, "NOT RUN: numpy is not installed"
    for p in (gguf_py, os.path.join(ROOT, "tools", "tinygen")):
        if p not in sys.path:
            sys.path.insert(0, p)
    modname, variants = TINY[tiny]
    mod = importlib.import_module(modname)
    os.makedirs(workdir, exist_ok=True)
    out, skipped = {}, []
    takes = set(inspect.signature(mod.build).parameters)
    for label, kw, arch in variants:
        kw = dict(kw)
        if hasattr(mod, "DEFAULT"):
            cfg = dict(mod.QUANT if kw.pop("quant", False) else mod.DEFAULT)
            if "n_layer" in cfg and modname == "glm5next_tiny":
                cfg["n_layer"] = 12            # the CLI default the parity drivers use
            kw["cfg"] = cfg
        if not set(kw) <= takes:
            skipped.append(f"{label} (tools/tinygen/{modname}.py build() takes no {sorted(set(kw) - takes)})")
            continue
        path = os.path.join(workdir, f"{tiny}-{len(out)}.gguf")
        saved = mod.ARCH
        if arch:
            mod.ARCH = arch
        try:
            mod.build(path, **kw)
        finally:
            mod.ARCH = saved
        h, hb = G.read_file_header(path)
        out[label] = Model([G.summarize(h, hb)])
    return out, ("tiny variants skipped: " + "; ".join(skipped)) if skipped else None


# ------------------------------------------------------------------------------------------------ sections

def section_identity(m, srcs):
    tok = m.kv.get("tokenizer.ggml.tokens", {})
    rows = {"architecture": m.arch, "general.name": m.v("general.name"),
            "tokenizer.ggml.model": m.v("tokenizer.ggml.model"), "tokenizer.ggml.pre": m.v("tokenizer.ggml.pre"),
            "vocab (tokenizer.ggml.tokens)": tok.get("count", len(tok.get("value", [])) if tok else None),
            "merges": m.kv.get("tokenizer.ggml.merges", {}).get("count"),
            "token_embd shape": m.shapes.get("token_embd.weight"), "output shape": m.shapes.get("output.weight")}
    for k in sorted(m.kv):
        if k.startswith("tokenizer.ggml.") and (k.endswith("_token_id") or k.startswith("tokenizer.ggml.add_")):
            rows[k] = m.v(k)
    rows["chat templates"] = [k for k in sorted(m.kv) if k.startswith("tokenizer.chat_template")]
    rows["general.* keys"] = {k: m.v(k) for k in sorted(m.kv) if k.startswith("general.")
                              and k not in ("general.architecture", "general.name")}
    support = {}
    for name, root in srcs.items():
        support[name] = {"arch " + str(m.arch): known(root, "arch", m.arch),
                         "tokenizer model " + str(m.v("tokenizer.ggml.model")): known(root, "model",
                                                                                     m.v("tokenizer.ggml.model")),
                         "pre-tokenizer " + str(m.v("tokenizer.ggml.pre")): known(root, "pre", m.v("tokenizer.ggml.pre"))}
    return {"identity": rows, "support": support}


def section_groups(m, srcs):
    g, u = m.a("expert_group_count"), m.a("expert_group_used_count")
    note = ("group-limited routing in llama.cpp (n_expert_groups > 1)" if (g or 0) > 1 else
            "no group step in llama.cpp (absent or <= 1 -> n_expert_groups 0/1)")
    cites = {n: cite(branches.GRAPH, locate(r, branches.GRAPH, "if (hparams.n_expert_groups > 1 && n_tokens > 0) {"))
             for n, r in srcs.items()}
    return {"expert_group_count": g, "expert_group_used_count": u, "present": [m.has("expert_group_count"),
                                                                                 m.has("expert_group_used_count")],
            "llama.cpp": note, "branch": cites}


def section_template(m, rel):
    t = m.v("tokenizer.chat_template")
    out = {"vendored": rel, "embedded_present": t is not None}
    if rel is None or t is None:
        return out
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        out["error"] = f"{rel} not found"
        return out
    with open(path, "rb") as f:
        ref = f.read()
    emb = t.encode("utf-8", errors="surrogateescape")
    out.update(embedded_bytes=len(emb), embedded_sha256=hashlib.sha256(emb).hexdigest(), vendored_bytes=len(ref),
               vendored_sha256=hashlib.sha256(ref).hexdigest(), byte_equal=emb == ref)
    if emb != ref:
        i = next((k for k in range(min(len(emb), len(ref))) if emb[k] != ref[k]), min(len(emb), len(ref)))
        out["first_difference_at_byte"] = i
        out["equal_after_stripping_trailing_newlines"] = emb.rstrip(b"\n") == ref.rstrip(b"\n")
        diff = list(difflib.unified_diff(ref.decode("utf-8", "replace").splitlines(),
                                         emb.decode("utf-8", "replace").splitlines(),
                                         "vendored", "embedded", lineterm="", n=1))
        out["diff"] = diff[:80] + ([f"... {len(diff) - 80} more diff lines"] if len(diff) > 80 else [])
    return out


def _cfg_get(cfg, path):
    v = cfg
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    return a == b


def section_config(m, rel, graph):
    if not rel or graph not in CONFIG_MAP or not os.path.exists(os.path.join(ROOT, rel)):
        return None
    with open(os.path.join(ROOT, rel)) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    rows = []
    for ck, gk, *fn in CONFIG_MAP[graph]:
        raw = cv = _cfg_get(cfg, ck)
        mapping = None
        if fn and raw is not None:
            cv, mapping = fn[0][0](raw, cfg, m), fn[0][1]
        if gk == "layers":
            gv = (m.a("block_count") or 0) - (m.a("nextn_predict_layers") or 0) if m.has("block_count") else None
        elif gk == "@vocab":
            gv = m.kv.get("tokenizer.ggml.tokens", {}).get("count")
            sh = m.shapes.get("token_embd.weight")
            gv = {"tokens": gv, "token_embd rows": sh[1] if sh and len(sh) > 1 else None}
        else:
            gv = m.a(gk)
        if gk == "@vocab":
            ok = cv in gv.values()
        elif isinstance(gv, str) and gv in ("inf", "-inf", "nan"):
            ok = cv is not None and _same(float(gv), cv)
        elif isinstance(gv, list) and not isinstance(cv, list):
            ok = len(gv) > 0 and all(_same(x, cv) for x in gv)
        else:
            ok = gv is not None and _same(gv, cv)
        row = {"config": ck, "config_value": raw, "gguf": gk, "gguf_value": gv, "match": ok if gv is not None else None}
        if mapping:                      # compared after the converter's mapping; the report shows both
            row.update(mapping=mapping, config_value_compared=cv)
        rows.append(row)
    return rows


def section_hparams(m, tiny_models):
    keys = set(m.arch_keys())
    tiny = tiny_models or {}
    for t in tiny.values():
        keys |= set(t.arch_keys())
    rows = []
    for k in sorted(keys):
        rows.append({"key": k, "real": m.a(k) if m.has(k) else "(absent)",
                     **{f"tiny {lab}": (t.a(k) if t.has(k) else "(absent)") for lab, t in tiny.items()}})
    return rows


def section_branches(m, graph, tiny_models, srcs):
    out = []
    for p in branches.PROBES:
        if graph not in p.graphs:
            continue
        real = set(_labels(p.regime(m)))
        tiny = {lab: set(_labels(p.regime(t))) for lab, t in (tiny_models or {}).items()}
        union = set().union(*tiny.values()) if tiny else None
        out.append({"id": p.id, "what": p.what, "real": sorted(real), "real_detail": p.detail(m),
                    "tiny": {lab: sorted(v) for lab, v in tiny.items()},
                    "exercised_by_tiny": (real <= union) if union is not None else None,
                    "not_exercised": sorted(real - union) if union is not None else None,
                    "anchors": {name: [cite(f, locate(root, f, needle)) for f, needle, _ in p.anchors]
                                for name, root in srcs.items()}})
    return out


def _labels(x):
    return {x} if isinstance(x, str) else set(x)


def section_audit(files, workdir):
    d = os.path.join(workdir, "headers")
    os.makedirs(d, exist_ok=True)
    names = []
    for f in files:
        base = os.path.basename(f["rec"]["path"])
        with open(os.path.join(d, base), "wb") as g:
            g.write(f["header"])
        names.append(base)
    first = sorted(names)[0]
    r = QA.audit(os.path.join(d, first))
    low = sorted(({"tensor": t["name"], "type": t["type"], "bpw": QA.GGML_TYPES[t["type_id"]][2] * 8 /
                   QA.GGML_TYPES[t["type_id"]][1], "class": QA.classify(t["name"])}
                  for f in files for t in f["summary"]["tensors"]
                  if t["type_id"] in QA.GGML_TYPES and len(t["shape"]) >= 2), key=lambda x: x["bpw"])
    return {"report": QA.render(r), "verdict": r["verdict"], "classes": r["classes"], "findings": r["findings"],
            "unknown_types": r["unknown_types"], "total_gib": r["total_gib"], "total_bpw": r["total_bpw"],
            "lowest_bpw_matrices": low[:12],
            "file_bytes": sum(f["rec"]["size"] for f in files)}


# ------------------------------------------------------------------------------------------------ N18: keys, experts

# a statement-level call: the arguments stay on one line and the call ends the statement; a call inside an if
# condition (`if (ml.get_key(..., false)) {`) does not match (it is optional there)
_KV_CALL = re.compile(r"ml\.(get_key|get_key_or_arr|get_arr|get_arr_n)\((LLM_KV_\w+)([^;{\n]*)\);")


def _kv_names(root):
    try:
        with open(os.path.join(root, "src", "llama-arch.cpp"), errors="replace") as f:
            return dict(re.findall(r'\{\s*(LLM_KV_\w+)\s*,\s*"([^"]+)"\s*\}', f.read()))
    except OSError:
        return None


def kv_calls(root, rel, start, end):
    """[(LLM_KV name, required, depth, line)] of the ml.get_* calls in the function whose text starts at `start` and
    ends before `end` in root/rel (one call per statement; required = the last argument is not `false`; depth 1 =
    directly in the function body, > 1 = inside a nested block, i.e. conditional). None if not found."""
    try:
        with open(os.path.join(root, rel), errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    i = text.find(start)
    if i < 0:
        return None
    b = text.index("{", i)
    j = text.find(end, b)
    body = text[b:j if j > 0 else len(text)]
    out = []
    for mt in _KV_CALL.finditer(body):
        depth = body.count("{", 0, mt.start()) - body.count("}", 0, mt.start())
        out.append((mt.group(2), not mt.group(3).strip().endswith("false"), depth,
                    text.count("\n", 0, b + mt.start()) + 1))
    return out


def section_required_keys(m, srcs):
    """per checkout: the keys llama_model_base::load_hparams (directly in its body) and <arch>::load_arch_hparams read
    without a default, plus tokenizer.ggml.model (the vocabulary loader requires it); which of them the file lacks"""
    out = {}
    rel = branches.ARCH_SOURCES.get(m.arch)
    for pin, root in srcs.items():
        names = _kv_names(root)
        gen = kv_calls(root, branches.MODEL, "void llama_model_base::load_hparams(llama_model_loader & ml) {",
                       "load_arch_hparams(ml);")
        arch = kv_calls(root, rel, "::load_arch_hparams(llama_model_loader & ml) {", "::load_arch_tensors") \
            if rel else None
        if names is None or gen is None or arch is None:
            out[pin] = {"error": f"not readable: src/llama-arch.cpp, {branches.MODEL} load_hparams or "
                                 f"{rel or 'no source for ' + str(m.arch)} load_arch_hparams"}
            continue
        rows, seen = [], set()
        for src, calls, generic in ((branches.MODEL, gen, True), (rel, arch, False)):
            for kv, req, depth, line in calls:
                if not req or (generic and depth > 1) or (kv, depth > 1) in seen:
                    continue
                seen.add((kv, depth > 1))
                key = names.get(kv, "?").replace("%s", str(m.arch))
                rows.append({"key": key, "kv": kv, "at": f"{src}:{line}", "conditional": depth > 1,
                             "present": key in m.kv})
        voc = locate(root, branches.VOCAB, "ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);")
        if voc:
            rows.append({"key": "tokenizer.ggml.model", "kv": "LLM_KV_TOKENIZER_MODEL", "at": cite(branches.VOCAB, voc),
                         "conditional": False, "present": "tokenizer.ggml.model" in m.kv})
        out[pin] = {"rows": rows, "missing": [r["key"] for r in rows if not r["present"] and not r["conditional"]],
                    "conditional_missing": [r["key"] for r in rows if not r["present"] and r["conditional"]]}
    return out


_EXP = re.compile(r"^blk\.(\d+)\.(ffn_(?:gate_up|gate|up|down)_exps)\.weight$")


def section_experts(m):
    """ggml type of each routed-expert projection, counted over the layers"""
    out = {}
    for t in m.tensors:
        mt = _EXP.match(t["name"])
        if mt:
            d = out.setdefault(mt.group(2), {})
            d[t["type"]] = d.get(t["type"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


# ------------------------------------------------------------------------------------------------ N18: MTP files

def _same_val(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return _same(a, b)


def _array_id(e):
    """a summary array entry's identity: its sha256 (arrays over the inline limit) or its values"""
    if not e:
        return None
    return ("sha256", e["sha256"]) if "sha256" in e else ("value", json.dumps(e.get("value")))


def _request_model(root, name):
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, "PIN.json")):
        return None, f"request {name} has no result yet"
    r = load_request(d)
    if r["pin"]["status"] != "ok":
        return None, f"request {name}: status {r['pin']['status']}"
    if not all(ok for _, ok in r["integrity"]):
        return None, f"request {name}: integrity failure"
    return Model([f["summary"] for f in r["files"]]), None


def mtp_reference(an, root, gguf_py, workdir):
    """(tensor names of the reference NextN layer without "blk.<i>.", label) or (None, reason)"""
    ref = an.get("mtp_reference") or {}
    if "request" in ref:
        rm, why = _request_model(root, ref["request"])
        label = f"request {ref['request']}"
    elif "tiny" in ref:
        tm, why = build_tiny(ref["tiny"], gguf_py, workdir)
        rm = (tm or {}).get(ref.get("variant"))
        if rm is None and tm is not None:
            why = f"tiny variant {ref.get('variant')!r} not built" + (f" ({why})" if why else "")
        label = f"tiny {ref['tiny']} {ref.get('variant')}"
    else:
        return None, "no mtp_reference in the request"
    if rm is None:
        return None, why
    layers = [i for i in branches._mtp_layers(rm) if branches._suffixes(rm, i)]
    if not layers:
        return None, f"{label} has no NextN layer"
    return branches._suffixes(rm, layers[0]), f"{label}, blk.{layers[0]}"


def mtp_verdict(m, root, ref):
    """what the loader of the checkout at root asks of this file as a separate MTP model (by reading)"""
    rules = branches.MTP_LOAD.get(m.arch)
    if rules is None:
        return {"verdict": f"UNKNOWN: no MTP_LOAD rules for {m.arch!r}", "hard": [], "check": [], "anchors": {}}

    def hits(a):
        return locate(root, a[0], a[1])

    def found(group):
        return [cite(a[0], hits(a)) for a in rules.get(group, []) if hits(a)]

    anchors = {g: found(g) for g in ("detect", "no_detect", "no_graph", "graph")}
    hard, check = [], []
    layout = branches.file_layout(m, rules["trunk_probe"])
    if layout == "MTP-only file":
        if not anchors["detect"]:
            hard.append(f"no MTP-only handling ({'; '.join(anchors['no_detect']) or 'detection anchor not found'}): "
                        f"every trunk tensor is required, the file has none ({rules['trunk_probe']} absent)")
    elif layout != "trunk + MTP in one file":
        hard.append(f"file layout: {layout}")
    if anchors["no_graph"]:
        hard.append(f"the MTP graph is not implemented ({'; '.join(anchors['no_graph'])})")
    elif not anchors["graph"]:
        check.append("MTP graph anchor not found")
    for name, a in rules["globals"]:
        if hits(a) and name not in m.names:
            hard.append(f"{name} absent, created with flags 0 ({cite(a[0], hits(a))})")
        elif not hits(a):
            check.append(f"anchor of {name} not found")
    if "ple_table" in rules and m.has("ple.layers") and hits(rules["ple_table"]) and \
            "per_layer_token_embd.weight" not in m.names:
        hard.append(f"ple.layers is set, so per_layer_token_embd.weight is required "
                    f"({cite(rules['ple_table'][0], hits(rules['ple_table']))}) and absent")
    if "tokenizer.ggml.model" not in m.kv:
        hard.append("tokenizer.ggml.model absent: the vocabulary loader requires it")
    if ref is not None:
        for i, r in ref.items():
            if r["missing"]:
                hard.append(f"blk.{i} lacks {r['missing']} (present in the reference NextN layer, not optional)")
            if r["extra"]:
                check.append(f"blk.{i} has {r['extra']}, which the reference NextN layer lacks: a tensor the loader "
                             f"does not create fails the tensor count")
    if hard:
        verdict = "NOT loadable as a separate MTP model"
    elif check:
        verdict = "CHECK"
    else:
        verdict = "loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)"
    return {"verdict": verdict, "layout": layout, "hard": hard, "check": check, "anchors": anchors}


# loader helpers that accept either form (src/llama-model.cpp create_tensor_qkv / create_tensor_gate_up_exps: the fused
# tensor first, the split ones otherwise): a fused/split difference against the reference is neither missing nor extra
ALTERNATIVES = [({"attn_qkv.weight"}, {"attn_q.weight", "attn_k.weight", "attn_v.weight"}),
                ({"ffn_gate_up_exps.weight"}, {"ffn_gate_exps.weight", "ffn_up_exps.weight"})]


def ref_diff(ref, have, optional):
    """NextN layer tensor names of this file (have) against the reference: missing (not optional), missing optional,
    extra, and the fused/split alternatives that differ"""
    ref, have = set(ref), set(have)
    alt = []
    for a, b in ALTERNATIVES:
        for x, y in ((a, b), (b, a)):
            if x <= have and y <= ref and not (x <= ref):
                alt.append(f"{sorted(x)} here, {sorted(y)} in the reference")
                have, ref = have - x, ref - y
    return {"missing": sorted(ref - have - optional), "missing_optional": sorted((ref - have) & optional),
            "extra": sorted(have - ref), "alternatives": alt}


def section_mtp_file(m, srcs, ref_names, trunk):
    rules = branches.MTP_LOAD.get(m.arch) or {}
    tok = m.kv.get("tokenizer.ggml.tokens", {})
    res = {"arch": m.arch, "block_count": m.a("block_count"), "nextn_predict_layers": m.a("nextn_predict_layers"),
           "n_layer": branches._layers(m),
           "layout": branches.file_layout(m, rules.get("trunk_probe", "blk.0.attn_norm.weight")),
           "tokenizer": {"model": m.v("tokenizer.ggml.model"), "pre": m.v("tokenizer.ggml.pre"),
                         "tokens": tok.get("count", len(tok.get("value", [])) if tok else None)},
           "chat_template": "tokenizer.chat_template" in m.kv,
           "globals": {t["name"]: [t["type"], t["shape"]] for t in m.tensors if not t["name"].startswith("blk.")},
           "layers_present": sorted({int(n.split(".")[1]) for n in m.names if n.startswith("blk.")}),
           "mtp_layers": {i: sorted(branches._suffixes(m, i)) for i in branches._mtp_layers(m)}}
    if ref_names is not None:
        res["reference"] = {i: ref_diff(ref_names, have, rules.get("optional", set()))
                            for i, have in res["mtp_layers"].items()}
    if trunk is not None:
        tk, mk = trunk.arch_keys(), m.arch_keys()
        res["trunk"] = {"arch": trunk.arch, "keys_differ": [[k, tk.get(k, "(absent)"), mk.get(k, "(absent)")]
                                                            for k in sorted(set(tk) | set(mk))
                                                            if not _same_val(tk.get(k), mk.get(k))],
                        "tokens_equal": _array_id(trunk.kv.get("tokenizer.ggml.tokens")) == _array_id(tok)
                        if tok else None,
                        "token_embd": [trunk.shapes.get("token_embd.weight"), m.shapes.get("token_embd.weight")]}
    res["loader"] = {pin: mtp_verdict(m, root, res.get("reference")) for pin, root in srcs.items()}
    return res


def section_attach(srcs):
    return {pin: [{"at": cite(rel, locate(root, rel, needle)), "what": what, "expected": pin in pins}
                  for rel, needle, pins, what in branches.MTP_ATTACH] for pin, root in srcs.items()}


# ------------------------------------------------------------------------------------------------ main

def analyze_dir(d, srcs, gguf_py, workdir):
    os.makedirs(workdir, exist_ok=True)
    res = {"dir": os.path.relpath(d, ROOT)}
    if os.path.exists(os.path.join(d, "LISTING.json")):
        with open(os.path.join(d, "LISTING.json")) as f:
            res["listing"] = json.load(f)
        return res
    if not os.path.exists(os.path.join(d, "PIN.json")):
        res["status"] = "PENDING (no PIN.json" + (", ERROR.json present)" if os.path.exists(
            os.path.join(d, "ERROR.json")) else ")")
        return res
    req = load_request(d)
    pin = req["pin"]
    res.update(status=pin["status"], repo=pin["repo"], revision=pin.get("revision"), integrity=req["integrity"])
    if pin["status"] != "ok":
        res["pin"] = {k: pin.get(k) for k in ("http_status", "http_body", "gguf_listing")}
        return res
    an = pin["request"].get("analysis", {})
    res["files"] = [{**{k: f["rec"][k] for k in ("path", "size", "lfs_sha256", "header_bytes", "header_sha256",
                                                 "bytes_fetched", "header_gz_bytes", "size_check", "tensor_offsets")},
                     "size_check_recomputed": G.size_check(f["summary"], f["rec"]["size"]),
                     "n_tensors": f["summary"]["n_tensors"]}
                    for f in req["files"]]
    if an.get("mode") == "mtp_files":
        root = os.path.dirname(os.path.abspath(d))
        ref, res["mtp_reference"] = mtp_reference(an, root, gguf_py, os.path.join(workdir, "ref"))
        trunk, why = _request_model(root, an["trunk"]) if an.get("trunk") else (None, "no trunk request named")
        res["trunk"] = an.get("trunk") if trunk is not None else why
        res["mode"], res["mtp_files"] = "mtp_files", []
        for k, f in enumerate(req["files"]):
            m = Model([f["summary"]])
            item = {"path": f["rec"]["path"], **section_mtp_file(m, srcs, ref, trunk)}
            item["identity"] = section_identity(m, srcs)
            item["config"] = section_config(m, an.get("hf_config"), an.get("graph"))
            item["required_keys"] = section_required_keys(m, srcs)
            item["experts"] = section_experts(m)
            item["audit"] = section_audit([f], os.path.join(workdir, f"mtp{k}"))
            res["mtp_files"].append(item)
        res["attach"] = section_attach(srcs)
        return res
    m = Model([f["summary"] for f in req["files"]])
    res["identity"] = section_identity(m, srcs)
    res["groups"] = section_groups(m, srcs)
    res["chat_template"] = section_template(m, an.get("hf_chat_template"))
    tiny, why = build_tiny(an.get("tiny"), gguf_py, workdir) if an.get("tiny") else (None, "no tiny model named")
    res["tiny_note"] = why
    res["hparams"] = section_hparams(m, tiny)
    res["config"] = section_config(m, an.get("hf_config"), an.get("graph"))
    res["branches"] = section_branches(m, an.get("graph"), tiny, srcs)
    res["required_keys"] = section_required_keys(m, srcs)
    res["experts"] = section_experts(m)
    res["audit"] = section_audit(req["files"], workdir)
    return res


def render(results, srcs):
    out = []
    w = out.append
    w("sources: " + ", ".join(f"{n} = {git_head(r) or '?'} ({r})" for n, r in srcs.items()) if srcs else
      "sources: none given (--src NAME=PATH): no file:line citations")
    for name, r in results.items():
        w(f"\n=== {name} ({r['dir']})")
        if "listing" in r:
            for x in r["listing"].get("repos", []):
                files = x.get("gguf_files") or []
                w(f"named repo {x['repo']} @ {x.get('revision')} (requested {x['requested_revision']}): {x['status']}" +
                  (f", arch={x.get('gguf_architecture')}, lastModified={x.get('lastModified')}, {len(files)} .gguf, "
                   f"{sum(f['size'] or 0 for f in files) / 2**30:.1f} GiB" if x["status"] == "ok" else
                   f", HTTP {x.get('http_status')}"))
                for f in files:
                    w(f"    {f['path']}  {f['size']} B = {(f['size'] or 0) / 2**30:.2f} GiB  lfs {str(f['lfs_sha256'])[:12]}")
            for s in r["listing"]["searches"]:
                w(f"search {s['search']!r} (architectures {s['architectures']}): {s['n_results']} repos")
                for it in s["results"]:
                    files = it.get("gguf_files")
                    w(f"  {'*' if it['arch_match'] else ' '} {it['id']}  arch={it['gguf_architecture']}  "
                      f"downloads={it['downloads']}  sha={str(it['sha'])[:12]}" +
                      (f"  {len(files)} .gguf, {sum(x['size'] or 0 for x in files) / 2**30:.1f} GiB" if files else ""))
            continue
        w(f"status {r.get('status')}" + (f", {r.get('repo')} @ {r.get('revision')}" if r.get("repo") else ""))
        if r.get("pin"):
            w(json.dumps(r["pin"], indent=1)[:3000])
        if "integrity" in r:
            w("integrity: " + ", ".join(f"{os.path.basename(p)} {'OK' if ok else 'FAIL'}" for p, ok in r["integrity"]))
        if r.get("status") != "ok":
            continue
        if r.get("mode") == "mtp_files":
            render_mtp(r, w)
            continue
        for f in r["files"]:
            w(f"  {f['path']}: {f['size']} B, LFS sha256 {f['lfs_sha256']}, header {f['header_bytes']} B "
              f"(sha256 {f['header_sha256']}), fetched {f['bytes_fetched']} B, committed {f['header_gz_bytes']} B gz, "
              f"{f['n_tensors']} tensors, size check {f['size_check_recomputed']}" +
              (f" (PIN.json recorded: {f['size_check']})" if f["size_check"] != f["size_check_recomputed"] else "") +
              f", offsets {f['tensor_offsets']}")
        w("\n(e) identity")
        for k, v in r["identity"]["identity"].items():
            w(f"  {k}: {fmt(v, 200)}")
        for n, sup in r["identity"]["support"].items():
            for k, v in sup.items():
                w(f"  [{n}] {k}: {v}")
        g = r["groups"]
        w(f"\n(d) expert_group_count = {g['expert_group_count']}, expert_group_used_count = "
          f"{g['expert_group_used_count']} (keys present: {g['present']}): {g['llama.cpp']}")
        for n, c in g["branch"].items():
            w(f"  [{n}] {c}")
        t = r["chat_template"]
        w(f"\n(c) chat template vs {t['vendored']}: " + (
            "no embedded tokenizer.chat_template" if not t["embedded_present"] else
            "BYTE-EQUAL" if t.get("byte_equal") else
            f"DIFFERENT (embedded {t.get('embedded_bytes')} B sha256 {t.get('embedded_sha256')}, vendored "
            f"{t.get('vendored_bytes')} B sha256 {t.get('vendored_sha256')}, first difference at byte "
            f"{t.get('first_difference_at_byte')}, equal up to trailing newlines: "
            f"{t.get('equal_after_stripping_trailing_newlines')})"))
        if t.get("byte_equal"):
            w(f"  {t['embedded_bytes']} B, sha256 {t['embedded_sha256']}")
        for ln in t.get("diff", []):
            w("  | " + ln)
        w("\n(b) hyperparameters, real vs tiny" + (f" ({r['tiny_note']})" if r["tiny_note"] else ""))
        for row in r["hparams"]:
            w("  " + " | ".join([f"{row['key']:42s}"] + [f"{k}: {fmt(v, 48)}" for k, v in row.items() if k != "key"]))
        if r["config"]:
            w("  config.json cross-check:")
            for c in r["config"]:
                cfgv = fmt(c["config_value"], 40)
                if c.get("mapping"):
                    cfgv += f" ({c['mapping']} -> {fmt(c['config_value_compared'], 40)})"
                w(f"    {'OK    ' if c['match'] else 'ABSENT' if c['match'] is None else 'DIFF  '} {c['config']} = "
                  f"{cfgv}  ->  {c['gguf']} = {fmt(c['gguf_value'], 60)}")
        w("  graph branches:")
        for b in r["branches"]:
            ex = b["exercised_by_tiny"]
            w(f"    [{'tiny OK' if ex else 'NOT IN TINY' if ex is not None else 'tiny n/a'}] {b['id']}: {b['what']}")
            w(f"       real: {b['real']}  {b['real_detail']}")
            if b["tiny"]:
                w(f"       tiny: " + "; ".join(f"{k}: {v}" for k, v in b["tiny"].items()))
            for n, c in b["anchors"].items():
                w(f"       [{n}] " + "; ".join(c))
        render_keys(r["required_keys"], w)
        w(f"  routed-expert types per projection (layers): {json.dumps(r['experts'])}")
        a = r["audit"]
        w(f"\n(a) tools/gguf_quant_audit.py on the header-only shards (files total {a['file_bytes']} B = "
          f"{a['file_bytes'] / 2**30:.3f} GiB)")
        for ln in a["report"].splitlines():
            w("  " + ln)
        w(f"(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): {a['verdict']}; lowest-bpw "
          "matrices: " + ", ".join(f"{x['tensor']} {x['type']} {x['bpw']:.3f}" for x in a["lowest_bpw_matrices"][:6]))
    return "\n".join(out)


def render_keys(rk, w):
    w("  keys read without a default (llama_model_base::load_hparams body, load_arch_hparams, the vocabulary):")
    for pin, k in rk.items():
        if "error" in k:
            w(f"    [{pin}] {k['error']}")
            continue
        unc = [x for x in k["rows"] if not x["conditional"]]
        con = [x for x in k["rows"] if x["conditional"]]
        w(f"    [{pin}] {len(unc)} required, missing in the file: {k['missing']}; " +
          (f"{len(con)} only inside a nested block (conditional), missing: {k['conditional_missing']} "
           f"({', '.join(x['key'] + ' ' + x['at'] for x in con if not x['present'])})" if con else "none conditional"))


def render_mtp(r, w):
    w(f"mode mtp_files; NextN reference: {r['mtp_reference']}; trunk request: {r['trunk']}")
    for f in r["files"]:
        w(f"  {f['path']}: {f['size']} B, LFS sha256 {f['lfs_sha256']}, header {f['header_bytes']} B "
          f"(sha256 {f['header_sha256']}), fetched {f['bytes_fetched']} B, committed {f['header_gz_bytes']} B gz, "
          f"{f['n_tensors']} tensors, size check {f['size_check_recomputed']}, offsets {f['tensor_offsets']}")
    w("\nhow llama.cpp attaches a separate MTP file:")
    for pin, rows in r["attach"].items():
        for x in rows:
            w(f"  [{pin}] {x['at']}: {x['what']}")
    for x in r["mtp_files"]:
        w(f"\n--- {x['path']}: arch {x['arch']}, block_count {x['block_count']}, nextn_predict_layers "
          f"{x['nextn_predict_layers']}, n_layer {x['n_layer']}; layout: {x['layout']}")
        t = x["tokenizer"]
        w(f"  tokenizer: model {t['model']}, pre {t['pre']}, {t['tokens']} tokens; embedded chat template: "
          f"{x['chat_template']}")
        w("  non-layer tensors: " + (", ".join(f"{n} {ty} {sh}" for n, (ty, sh) in sorted(x["globals"].items()))
                                     or "none"))
        w(f"  layers present: {x['layers_present']}")
        for i, names in x["mtp_layers"].items():
            w(f"  blk.{i}: {len(names)} tensors: {names}")
        for i, d in (x.get("reference") or {}).items():
            w(f"  vs the reference NextN layer: blk.{i} missing {d['missing']}, missing optional "
              f"{d['missing_optional']}, extra {d['extra']}" +
              (f"; fused/split: {'; '.join(d['alternatives'])}" if d["alternatives"] else ""))
        if x.get("trunk"):
            tr = x["trunk"]
            w(f"  vs the trunk ({tr['arch']}): arch keys that differ: " +
              ("; ".join(f"{k}: trunk {fmt(a, 40)} / this {fmt(b, 40)}" for k, a, b in tr["keys_differ"]) or "none") +
              f"; token list equal: {tr['tokens_equal']}; token_embd trunk {tr['token_embd'][0]} / this "
              f"{tr['token_embd'][1]}")
        if x.get("config"):
            bad = [c for c in x["config"] if not c["match"]]
            w(f"  config.json cross-check: {sum(1 for c in x['config'] if c['match'])} of {len(x['config'])} OK" +
              ("".join(f"; {'ABSENT' if c['match'] is None else 'DIFF'} {c['config']} = {fmt(c['config_value'], 30)} -> "
                       f"{c['gguf']} = {fmt(c['gguf_value'], 30)}" for c in bad)))
        render_keys(x["required_keys"], w)
        w(f"  routed-expert types per projection (layers): {json.dumps(x['experts'])}")
        a = x["audit"]
        w(f"  audit: {a['verdict']}, {a['total_gib']:.3f} GiB, {a['total_bpw']:.3f} bpw; " +
          ", ".join(f"{k} {v['bpw']:.3f} bpw {v['types']}" for k, v in a["classes"].items()))
        for pin, v in x["loader"].items():
            w(f"  loader [{pin}]: {v['verdict']}")
            for h in v["hard"]:
                w(f"      - {h}")
            for c in v["check"]:
                w(f"      ? {c}")
            for g, c in v["anchors"].items():
                if c:
                    w(f"      {g}: {'; '.join(c)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "third_party", "gguf-headers"))
    ap.add_argument("--name", action="append", help="only these request names")
    ap.add_argument("--src", action="append", default=[], help="NAME=PATH of a llama.cpp-family checkout")
    ap.add_argument("--gguf-py", default=os.environ.get("LLAMA_CPP_GGUF_PY"))
    ap.add_argument("--json", help="write the full result as JSON here")
    a = ap.parse_args(argv)
    srcs = dict(s.split("=", 1) for s in a.src)
    names = a.name
    if not names:
        names = sorted(n for n in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, n))) \
            if os.path.isdir(a.root) else []
    work = tempfile.mkdtemp(prefix="gguf-analyze-")
    try:
        results = {n: analyze_dir(os.path.join(a.root, n), srcs, a.gguf_py, os.path.join(work, n)) for n in names}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(render(results, srcs))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"sources": {n: {"path": r, "head": git_head(r)} for n, r in srcs.items()},
                       "results": results}, f, indent=1, default=str)
            f.write("\n")
    bad = [n for n, r in results.items() for _, ok in r.get("integrity", []) if not ok]
    if bad:
        print(f"INTEGRITY FAIL: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
