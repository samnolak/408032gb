#!/usr/bin/env python3
"""N15 stretch: N14's agent-loop prefix stability (B3) at token level, with the real GLM-5.3-Flash / Qwen3.8-Flash-Next
vocabularies and llama.cpp's own tokenizer.

For every request of N14's agent sessions (tools/chattmpl/run_n14.py PREFIX_SESSIONS, loop view of
tools/chattmpl/analysis.py loop_checks): p = llama-server's prompt for conv[:k] (/apply-template, the template embedded
in the Unsloth GGUF, i.e. what an R1 run with --jinja and no --chat-template-file renders), out = the turn a model
trained on the template writes (jinja2 render of conv[:k+1] after its own prompt), nxt = the server's prompt for the
next request. Byte level (N14): does nxt start with p + out? Token level (here): with T = llama-tokenize @79e2e74 on a
vocab-only GGUF built from the committed header (tools/gguf_headers/vocab_gguf.py; add_special and parse_special as
llama-server's handle_completions_impl, tools/server/server-context.cpp:4808), does T(nxt) start with
T(p) + T_nobos(out)? T(p) + T_nobos(out) stands in for what the slot holds: the prompt tokens plus the sampled tokens,
approximated by the canonical tokenization of the turn (a model may sample other tokens for the same bytes).
Also reported: T(nxt) starts with T(p) (reuse of the previous prompt alone), and T(p + out) == T(p) + T(out) (no BPE
merge across the prompt / turn boundary).
What it does not exercise: llama-server's own tokenization path and its cache (only the same tokenizer function through
llama-tokenize), real sampling, any model.

Usage: python3 <this> --tokenize-bin <llama-tokenize @79e2e74> --server-bin <llama-server @79e2e74>
       --glm-gguf <N14 glm5next_bytes.gguf> --qwen-gguf <N14 qwen4exp_bytes.gguf> --out <scratch dir>
"""
import argparse
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
sys.path.insert(0, str(ROOT / "tools" / "gguf_headers"))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402
import gguf_header as G  # noqa: E402
import run_n14 as R  # noqa: E402
import server as S  # noqa: E402
import vocab_gguf as V  # noqa: E402

HEADERS = {
    "glm": "third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf.header.gz",
    "qwen": "third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl/"
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf.header.gz",
}


TAGS = {
    "glm": ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>",
            "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>",
            "</tool_response>"],
    "qwen": ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
             "</tool_response>", "<function=", "</function>", "<parameter=", "</parameter>"],
}


class Tok:
    def __init__(self, binary, model, workdir):
        self.binary, self.model, self.workdir, self.cache = binary, model, workdir, {}
        os.makedirs(workdir, exist_ok=True)

    def __call__(self, text, bos=True):
        key = (hashlib.sha256(text.encode()).hexdigest(), bos)
        if key not in self.cache:
            p = os.path.join(self.workdir, key[0] + ".txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            r = subprocess.run([self.binary, "-m", self.model, "-f", p, "--ids", "--log-disable"] +
                               ([] if bos else ["--no-bos"]), capture_output=True, text=True, cwd=self.workdir,
                               timeout=300)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])
            self.cache[key] = json.loads(r.stdout.strip().splitlines()[-1])
        return self.cache[key]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(name, a, out):
    with open(ROOT / HEADERS[name], "rb") as f:
        hb = gzip.decompress(f.read())
    h = G.parse(hb)
    vocab = h.get("tokenizer.ggml.tokens")
    vpath = os.path.join(out, f"vocab-{name}.gguf")
    with open(vpath, "wb") as f:
        f.write(V.build(hb))                     # glm5-next and qwen4exp are known at 79e2e74: no rename
    tdir = pathlib.Path(out) / f"embedded-{name}"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "chat_template.jinja").write_bytes(h.get("tokenizer.chat_template").encode("utf-8", "surrogateescape"))
    shutil.copy(R.TEMPLATES[name]["dir"] / "tokenizer_config.json", tdir / "tokenizer_config.json")
    tok = Tok(a.tokenize_bin, vpath, os.path.join(out, f"tok-{name}"))
    piece = lambda ids: [vocab[i] if 0 <= i < len(vocab) else i for i in ids]  # noqa: E731
    res = {"vocab_gguf_sha256": hashlib.sha256(open(vpath, "rb").read()).hexdigest(), "n_vocab": len(vocab),
           "template_sha256": hashlib.sha256((tdir / "chat_template.jinja").read_bytes()).hexdigest(), "sessions": {}}
    # N14 section 8: are the template's tags single tokens? (no BOS; parse_special as the server; token type from the
    # header's tokenizer.ggml.token_type: 1 normal, 3 control, 4 user-defined)
    types = h.get("tokenizer.ggml.token_type")
    res["tags"] = {}
    for t in TAGS[name]:
        ids = tok(t, bos=False)
        res["tags"][t] = {"ids": ids, "pieces": piece(ids), "types": [types[i] for i in ids]}
    vend = R.TEMPLATES[name]["dir"]
    R.TEMPLATES[name]["dir"] = tdir
    try:
        model = a.glm_gguf if name == "glm" else a.qwen_gguf
        with S.Server(a.server_bin, model, str(tdir / "chat_template.jinja"), os.path.join(out, f"srv-{name}"),
                      extra=["--no-prefill-assistant"]) as srv:
            Rn = R.Renderers(name, srv)
            for sname, (vname, conv_fn) in R.PREFIX_SESSIONS.items():
                hkw, skw, extra = R.VARIANTS[name][vname]
                conv = conv_fn()
                sfn, hfn = Rn.server(skw, extra, C.TOOLS), Rn.hf(hkw, C.TOOLS)
                pts = A.request_points(conv)
                rows = []
                for idx, k in enumerate(pts[:-1]):
                    row = {"request": idx, "k": k}
                    try:
                        p = sfn(conv[:k], add_generation_prompt=True)
                        p_out = hfn(conv[:k], add_generation_prompt=True)
                        hist = hfn(conv[:k + 1], add_generation_prompt=False)
                        nxt = sfn(conv[:pts[idx + 1]], add_generation_prompt=True)
                    except Exception as e:  # noqa: BLE001
                        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                        rows.append(row)
                        continue
                    if not hist.startswith(p_out):
                        row["turn_not_defined"] = "the reference render of conv[:k+1] does not extend its own prompt"
                        rows.append(row)
                        continue
                    turn = hist[len(p_out):]
                    tp, tt, tn, th = tok(p), tok(turn, bos=False), tok(nxt), tok(p + turn)
                    held = tp + tt
                    n = lcp(held, tn)
                    row.update(prompt_bytes=len(p.encode()), turn_bytes=len(turn.encode()),
                               byte_extends=nxt.startswith(p + turn), byte_reuse=A.lcp((p + turn).encode(), nxt.encode()),
                               held_tokens=len(held), next_tokens=len(tn), token_reuse=n,
                               token_extends=n == len(held), prompt_token_reuse=lcp(tp, tn),
                               prompt_extends=lcp(tp, tn) == len(tp), concat_is_canonical=th == held)
                    if n < len(held):
                        row["at"] = {"held": piece(held[max(0, n - 3):n + 3]), "next": piece(tn[max(0, n - 3):n + 3]),
                                     "index": n}
                    if th != held:
                        m = lcp(th, held)
                        row["canonical_vs_concat"] = {"index": m, "canonical": piece(th[max(0, m - 2):m + 3]),
                                                      "concat": piece(held[max(0, m - 2):m + 3])}
                    rows.append(row)
                res["sessions"][sname] = rows
    finally:
        R.TEMPLATES[name]["dir"] = vend
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenize-bin", required=True)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--glm-gguf", required=True)
    ap.add_argument("--qwen-gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = {name: run(name, a, a.out) for name in ("glm", "qwen")}
    with open(os.path.join(a.out, "token-prefix.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    for name, res in results.items():
        print(f"{name}: vocab {res['n_vocab']}, vocab-only GGUF sha256 {res['vocab_gguf_sha256'][:16]}, embedded template "
              f"sha256 {res['template_sha256'][:16]}")
        print("  tags: " + "; ".join(f"{t} -> {v['ids']} (type {v['types']})" for t, v in res["tags"].items()))
        for sname, rows in res["sessions"].items():
            ok = [r for r in rows if "token_extends" in r]
            print(f"  {sname}: {len(rows)} request pairs, {len(ok)} measured; byte-extends {sum(r['byte_extends'] for r in ok)}, "
                  f"token-extends {sum(r['token_extends'] for r in ok)}, prompt-token-extends "
                  f"{sum(r['prompt_extends'] for r in ok)}, concat canonical {sum(r['concat_is_canonical'] for r in ok)}")
            for r in rows:
                if "error" in r or "turn_not_defined" in r:
                    print(f"    request {r['request']}: {r.get('error') or r.get('turn_not_defined')}")
                elif not r["token_extends"] or not r["byte_extends"] or not r["concat_is_canonical"]:
                    print("    request %d: bytes %s (reuse %d of %d), tokens %s (reuse %d of %d)%s%s" % (
                        r["request"], r["byte_extends"], r["byte_reuse"], r["prompt_bytes"] + r["turn_bytes"],
                        r["token_extends"], r["token_reuse"], r["held_tokens"],
                        (" at " + json.dumps(r["at"], ensure_ascii=False)) if "at" in r else "",
                        (" canonical!=concat " + json.dumps(r["canonical_vs_concat"], ensure_ascii=False))
                        if "canonical_vs_concat" in r else ""))


if __name__ == "__main__":
    main()
