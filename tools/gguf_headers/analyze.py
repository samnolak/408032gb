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
             branches.py: the labels the real file and the tiny models take, "exercised by tiny" or NOT, with the
             file:line of every anchor at every --src checkout;
  (a)+(f) tools/gguf_quant_audit.py on the header-only shards rebuilt from the committed bytes (it reads headers
             only), i.e. the ADR-012 D5 floor: routed experts and non-expert matrices >= 4.0 bpw.
A discover request's LISTING.json is printed as a table. Exit 1 if an integrity check fails.

Usage: python3 tools/gguf_headers/analyze.py [--root third_party/gguf-headers] [--name NAME ...]
         [--src ec7630a=<llama.cpp checkout>] [--src 79e2e74=<checkout>] [--src ik=<ik_llama.cpp>] [--json OUT]
"""
import argparse
import difflib
import gzip
import hashlib
import importlib
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
import branches  # noqa: E402
import gguf_header as G  # noqa: E402
import gguf_quant_audit as QA  # noqa: E402

TINY = {  # request "analysis.tiny" -> (tools/tinygen module, variants: (label, build kwargs, ARCH override))
    "glm5next": ("glm5next_tiny", [("default", {}, "glm5-next"), ("--mtp", {"mtp": True}, "glm5-next"),
                                   ("--quant-dims", {"quant": True}, "glm5-next"),
                                   ("--dialect ik", {"dialect": "ik"}, "glm5next")]),
    "qwen4exp": ("qwen4exp_tiny", [("default", {}, None), ("--mtp", {"mtp": True}, None),
                                   ("--quant-dims", {"quant": True}, None)]),
}

# config.json (text_config) key -> GGUF key suffix; "layers" = block_count - nextn_predict_layers
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
                 ("rope_parameters.rope_theta", "rope.freq_base"), ("ple_layer_ids", "ple.layers"),
                 ("ngram_size", "ple.ngram_size"), ("heads_per_ngram", "ple.heads_per_ngram"),
                 ("ple_conv_kernel_size", "ple.conv_kernel"), ("vocab_size", "@vocab")],
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
    out = {}
    for label, kw, arch in variants:
        kw = dict(kw)
        cfg = dict(mod.QUANT if kw.pop("quant", False) else mod.DEFAULT)
        if "n_layer" in cfg and modname == "glm5next_tiny":
            cfg["n_layer"] = 12            # the CLI default the parity drivers use
        path = os.path.join(workdir, f"{tiny}-{len(out)}.gguf")
        saved = mod.ARCH
        if arch:
            mod.ARCH = arch
        try:
            mod.build(path, cfg=cfg, **kw)
        finally:
            mod.ARCH = saved
        h, hb = G.read_file_header(path)
        out[label] = Model([G.summarize(h, hb)])
    return out, None


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
    for ck, gk in CONFIG_MAP[graph]:
        cv = _cfg_get(cfg, ck)
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
        rows.append({"config": ck, "config_value": cv, "gguf": gk, "gguf_value": gv,
                     "match": ok if gv is not None else None})
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
    m = Model([f["summary"] for f in req["files"]])
    res["files"] = [{k: f["rec"][k] for k in ("path", "size", "lfs_sha256", "header_bytes", "header_sha256",
                                              "bytes_fetched", "header_gz_bytes", "size_check", "tensor_offsets")}
                    for f in req["files"]]
    res["identity"] = section_identity(m, srcs)
    res["groups"] = section_groups(m, srcs)
    res["chat_template"] = section_template(m, an.get("hf_chat_template"))
    tiny, why = build_tiny(an.get("tiny"), gguf_py, workdir) if an.get("tiny") else (None, "no tiny model named")
    res["tiny_note"] = why
    res["hparams"] = section_hparams(m, tiny)
    res["config"] = section_config(m, an.get("hf_config"), an.get("graph"))
    res["branches"] = section_branches(m, an.get("graph"), tiny, srcs)
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
        for f in r["files"]:
            w(f"  {f['path']}: {f['size']} B, LFS sha256 {f['lfs_sha256']}, header {f['header_bytes']} B "
              f"(sha256 {f['header_sha256']}), fetched {f['bytes_fetched']} B, committed {f['header_gz_bytes']} B gz, "
              f"size check {f['size_check']}, offsets {f['tensor_offsets']}")
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
                w(f"    {'OK  ' if c['match'] else 'n/a ' if c['match'] is None else 'DIFF'} {c['config']} = "
                  f"{fmt(c['config_value'], 40)}  ->  {c['gguf']} = {fmt(c['gguf_value'], 60)}")
        w("  graph branches:")
        for b in r["branches"]:
            ex = b["exercised_by_tiny"]
            w(f"    [{'tiny OK' if ex else 'NOT IN TINY' if ex is not None else 'tiny n/a'}] {b['id']}: {b['what']}")
            w(f"       real: {b['real']}  {b['real_detail']}")
            if b["tiny"]:
                w(f"       tiny: " + "; ".join(f"{k}: {v}" for k, v in b["tiny"].items()))
            for n, c in b["anchors"].items():
                w(f"       [{n}] " + "; ".join(c))
        a = r["audit"]
        w(f"\n(a) tools/gguf_quant_audit.py on the header-only shards (files total {a['file_bytes']} B = "
          f"{a['file_bytes'] / 2**30:.3f} GiB)")
        for ln in a["report"].splitlines():
            w("  " + ln)
        w(f"(f) ADR-012 D5 floor (>= 4.0 bpw, routed experts and non-expert matrices): {a['verdict']}; lowest-bpw "
          "matrices: " + ", ".join(f"{x['tensor']} {x['type']} {x['bpw']:.3f}" for x in a["lowest_bpw_matrices"][:6]))
    return "\n".join(out)


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
