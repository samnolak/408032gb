#!/usr/bin/env python3
"""N23: what llama-server @79e2e74 actually reuses when Qwen's empty think block re-tokenizes (<think> Ċ | Ċ </think>
held vs <think> ĊĊ </think> in the next prompt), with the REAL Qwen3.8-Flash-Next vocabulary.

Model: N11's tiny qwen4exp generator (random weights, hybrid: recurrent + attention layers, so the server uses context
checkpoints) with the tokenizer keys of N15's committed Unsloth header (tokens, types, merges, pre qwen35, bos/eos/pad,
add_bos false) instead of its byte vocabulary. Template: the official Qwen file (--chat-template-file).
Per request k of a session:
  (a) POST /v1/chat/completions conv[:k] (max_tokens 1): the server renders and tokenizes the prompt; timings.cache_n is
      what it reused from the slot (the measurement for the previous request's held tokens)
  (b) POST /completion with the token ids T(prompt k) + T(turn k) (the canonical tokens of the turn as the official
      template renders it; /tokenize of this server, parse_special): the slot now holds exactly those tokens, as if the
      model had sampled them (plus 1 sampled token)
Predicted from tokens: reuse = lcp(held, T(prompt k+1)); a full extend gives cache_n = len(held) (or +1 if the extra
sampled token happens to match); a break gives the newest checkpoint at or before the first differing token.

Usage: <this> --tinygen <N11 tools/tinygen> --gguf-py <llama.cpp gguf-py> --n15 <dir with tools/gguf_headers,
       tools/gguf_quant_audit.py and the Qwen header> --server-bin <llama-server> --out <scratch dir>
"""
import argparse
import gzip
import importlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402
import hf_render  # noqa: E402
import run_n14 as R  # noqa: E402
import server as S  # noqa: E402

QWEN_HEADER = "third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf.header.gz"
SESSIONS = {"default": lambda: C.AGENT_SESSION, "empty_reasoning_turn": lambda: C.AGENT_SESSION_EMPTY_REASONING,
            "agent_loop_mixed": lambda: C.AGENT_LOOP_MIXED}


def build_model(a, path):
    sys.path.insert(0, a.gguf_py)
    sys.path.insert(0, a.tinygen)
    sys.path.insert(0, os.path.join(a.n15, "tools", "gguf_headers"))
    sys.path.insert(0, os.path.join(a.n15, "tools"))
    import gguf_header as G
    with open(os.path.join(a.n15, QWEN_HEADER), "rb") as f:
        h = G.parse(gzip.decompress(f.read()))
    tokens, types, merges = (h.get("tokenizer.ggml.tokens"), h.get("tokenizer.ggml.token_type"),
                             h.get("tokenizer.ggml.merges"))
    eos = h.get("tokenizer.ggml.eos_token_id")
    gen = importlib.import_module("qwen4exp_tiny")

    def write_real_vocab(w, bos=1, eos=2):
        w.add_tokenizer_model(h.get("tokenizer.ggml.model"))
        w.add_tokenizer_pre(h.get("tokenizer.ggml.pre"))
        w.add_token_list(tokens)
        w.add_token_types(types)
        w.add_token_merges(merges)
        w.add_bos_token_id(h.get("tokenizer.ggml.bos_token_id"))
        w.add_eos_token_id(h.get("tokenizer.ggml.eos_token_id"))
        w.add_pad_token_id(h.get("tokenizer.ggml.padding_token_id"))
        w.add_add_bos_token(bool(h.get("tokenizer.ggml.add_bos_token")))
        return len(tokens)

    gen.write_byte_bpe_vocab = write_real_vocab
    cfg = dict(gen.DEFAULT, n_vocab=len(tokens), ctx=16384, ple_eos=eos)
    gen.build(path, cfg=cfg, seed=0, text_vocab=True)
    return len(tokens), len(merges)


def lcp(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("tinygen", "gguf_py", "n15", "server_bin", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = os.path.join(a.out, "qwen4exp_realvocab.gguf")
    if not os.path.exists(model):
        nv, nm = build_model(a, model)
        print("wrote %s: n_vocab %d, merges %d, %d B" % (model, nv, nm, os.path.getsize(model)))
    tmpl = hf_render.Template(R.TEMPLATES["qwen"]["dir"])
    res = {}
    for sname, conv_fn in SESSIONS.items():
        conv = conv_fn()
        wd = os.path.join(a.out, "live-" + sname)
        rows = []
        with S.Server(a.server_bin, model, str(R.template_path("qwen")), wd, extra=["-c", "16384", "-lv", "4"]) as srv:
            def tok(text):
                st, body = S.post(srv.base, "/tokenize", {"content": text, "add_special": False, "parse_special": True})
                assert st == 200, body
                return body["tokens"]
            pts = A.request_points(conv)
            prev = None
            for idx, k in enumerate(pts):
                body = {"messages": C.to_openai(conv[:k]), "tools": C.TOOLS, "tool_choice": "none", "max_tokens": 1,
                        "temperature": 0, "cache_prompt": True}
                st, r = S.post(srv.base, "/v1/chat/completions", body, timeout=600)
                assert st == 200, r
                t = r["timings"]
                st, ap_ = srv.apply_template(C.to_openai(conv[:k]), tools=C.TOOLS, add_generation_prompt=True)
                p = ap_["prompt"]
                tp = tok(p)
                row = {"request": idx, "k": k, "prompt_tokens": len(tp), "cache_n": t.get("cache_n"),
                       "prompt_n": t.get("prompt_n")}
                if prev is not None:
                    held = prev
                    n = lcp(held, tp)
                    row.update({"held_tokens": len(held), "predicted_reuse": n, "token_extends": n == len(held)})
                rows.append(row)
                ref_p = tmpl.render(conv[:k], tools=C.TOOLS, add_generation_prompt=True)
                hist = tmpl.render(conv[:k + 1], tools=C.TOOLS, add_generation_prompt=False)
                assert ref_p == p and hist.startswith(p), (sname, k)
                turn = hist[len(p):]
                turn = turn[:turn.index("<|im_end|>") + len("<|im_end|>")] if "<|im_end|>" in turn else turn
                held = tp + tok(turn)
                st, r2 = S.post(srv.base, "/completion", {"prompt": held, "n_predict": 1, "temperature": 0,
                                                          "cache_prompt": True}, timeout=600)
                assert st == 200, r2
                prev = held
            log = srv.log_text()
        ck = [ln.split("| ", 2)[-1].strip() for ln in log.splitlines()
              if re.search(r"restored context checkpoint|forcing full prompt", ln)]
        res[sname] = {"rows": rows, "restores": ck}
    with open(os.path.join(a.out, "qwen-newline-live.json"), "w") as f:
        json.dump(res, f, indent=1)
    for sname, v in res.items():
        print(sname)
        for r in v["rows"]:
            print("  request %d k=%d prompt %d tok | held %s predicted reuse %s extends %s | cache_n %s prompt_n %s" % (
                r["request"], r["k"], r["prompt_tokens"], r.get("held_tokens"), r.get("predicted_reuse"),
                r.get("token_extends"), r["cache_n"], r["prompt_n"]))
        for c in v["restores"]:
            print("    | " + c)


if __name__ == "__main__":
    main()
