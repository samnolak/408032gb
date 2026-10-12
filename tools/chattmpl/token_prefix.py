#!/usr/bin/env python3
"""N23: agent-loop prefix stability (B3) at token level, with the real tokenizers, for the official GLM-5.3-Flash
template, our corrected one (templates/glm-5.3-flash/chat_template.jinja) and the official Qwen3.8-Flash-Next template.

Two tokenizers, either or both:
  hf     the Hugging Face tokenizer.json of the pinned revision (third_party/hf/*-tokenizer/PIN.json, fetched and
         sha256-checked by scripts/hf_lfs_fetch.py), through the `tokenizers` library, add_special_tokens=False as
         transformers' apply_chat_template(tokenize=True) does (the template writes the special tokens itself)
  llama  llama.cpp's tokenizer: llama-tokenize @79e2e74 on a vocab-only GGUF built from the committed Unsloth GGUF header
         (N15, tools/gguf_headers/vocab_gguf.py), --no-bos --no-escape, special tokens parsed (as llama-server's
         tokenize_input_prompts / common_tokenize with parse_special = true)
Both: are the ids of every rendered prompt and turn equal (tokenizer parity)?

Per request of each agent session (as N14 / N15: p = prompt k, turn = the assistant turn as the reference template
renders it after p, nxt = prompt of the next request; jinja2 renders, i.e. what llama-server sends with our GLM file or
the Qwen file, CONFIRMED byte-equal in N14 / N23):
  token_extends       T(nxt) starts with T(p) + T(turn)   (the slot: prompt tokens + the sampled tokens, approximated by
                                                           the canonical tokenization of the turn)
  byte_extends        nxt starts with p + turn
  concat_canonical    T(p + turn) == T(p) + T(turn)       (no BPE merge across the prompt / turn boundary)
  at                  where the token prefix breaks (pieces around it)

Usage: token_prefix.py [--hf-glm tokenizer.json] [--hf-qwen tokenizer.json]
                       [--llama-tokenize BIN --vocab-glm GGUF --vocab-qwen GGUF] --out DIR
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402
import hf_render  # noqa: E402
import run_n14 as R  # noqa: E402

GLM_DIR = ROOT / "third_party/hf/glm-5.3-flash-chat"
QWEN_DIR = ROOT / "third_party/hf/qwen3.8-flash-next-chat"
OURS = ROOT / "templates/glm-5.3-flash/chat_template.jinja"

# name -> (vendored directory, template file or None = the directory's, model, variants key of run_n14.VARIANTS)
TEMPLATES = {
    "glm_hf": (GLM_DIR, None, "glm", "glm"),
    "glm_n23": (GLM_DIR, OURS, "glm", "glm"),
    "qwen_hf": (QWEN_DIR, None, "qwen", "qwen"),
}
SESSIONS = dict(R.PREFIX_SESSIONS, **{
    "agent_loop_mixed": ("default", lambda: C.AGENT_LOOP_MIXED),
    "agent_loop_mixed_preserve_off": ("preserve_off", lambda: C.AGENT_LOOP_MIXED),
})
TAGS = {
    "glm": ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>",
            "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>",
            "</tool_response>"],
    "qwen": ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
             "</tool_response>", "<function=", "</function>", "<parameter=", "</parameter>"],
}


class HFTok:
    kind = "hf"

    def __init__(self, path):
        import tokenizers
        self.path = path
        self.tok = tokenizers.Tokenizer.from_file(str(path))
        self.version = tokenizers.__version__

    def __call__(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def piece(self, i):
        return self.tok.id_to_token(i)


class LlamaTok:
    kind = "llama"

    def __init__(self, binary, model, workdir):
        # absolute: the subprocess runs in workdir, a relative path would resolve there (N23 round 2: the first CI
        # run passed ../build-tok/bin/llama-tokenize and every llama call failed with FileNotFoundError)
        self.binary, self.model = os.path.abspath(binary), os.path.abspath(model)
        # the work directory too: the text files are written there and passed to -f from inside it (round 2b: the
        # second CI run used --out tok-out and llama-tokenize could not open tok-out/tok-glm/<sha>.txt)
        self.workdir, self.cache = os.path.abspath(workdir), {}
        os.makedirs(workdir, exist_ok=True)

    def __call__(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.cache:
            p = os.path.join(self.workdir, key + ".txt")
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            r = subprocess.run([self.binary, "-m", self.model, "-f", p, "--ids", "--log-disable", "--no-bos",
                                "--no-escape"], capture_output=True, text=True, cwd=self.workdir, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])
            self.cache[key] = json.loads(r.stdout.strip().splitlines()[-1])
        return self.cache[key]

    def piece(self, i):
        return i


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def renderer(name):
    d, t, _model, _v = TEMPLATES[name]
    return hf_render.Template(d, t)


def requests_of(name):
    """(session, request, p, turn, nxt) for every request pair of every session, rendered by jinja2."""
    tmpl = renderer(name)
    vkey = TEMPLATES[name][3]
    out = []
    for sname, (vname, conv_fn) in SESSIONS.items():
        hkw = R.VARIANTS[vkey][vname][0]
        conv = conv_fn()

        def r(m, add_generation_prompt):
            return tmpl.render(m, tools=C.TOOLS, add_generation_prompt=add_generation_prompt, **hkw)
        pts = A.request_points(conv)
        for idx, k in enumerate(pts[:-1]):
            p = r(conv[:k], True)
            hist = r(conv[:k + 1], False)
            nxt = r(conv[:pts[idx + 1]], True)
            if not hist.startswith(p):
                raise RuntimeError("%s %s k=%d: the history render does not extend the prompt" % (name, sname, k))
            out.append((sname, idx, k, p, hist[len(p):], nxt))
    return out


def check(name, tok):
    rows = []
    for sname, idx, k, p, turn, nxt in requests_of(name):
        tp, tt, tn, th = tok(p), tok(turn), tok(nxt), tok(p + turn)
        held = tp + tt
        n = lcp(held, tn)
        row = {"session": sname, "request": idx, "k": k, "prompt_tokens": len(tp), "turn_tokens": len(tt),
               "held_tokens": len(held), "next_tokens": len(tn), "token_reuse": n, "token_extends": n == len(held),
               "byte_extends": nxt.startswith(p + turn), "byte_reuse": A.lcp((p + turn).encode(), nxt.encode()),
               "held_bytes": len((p + turn).encode()), "prompt_token_extends": lcp(tp, tn) == len(tp),
               "concat_canonical": th == held}
        if n < len(held):
            row["at"] = {"index": n, "held": [tok.piece(i) for i in held[max(0, n - 3):n + 3]],
                         "next": [tok.piece(i) for i in tn[max(0, n - 3):n + 3]]}
        rows.append(row)
    return rows


def texts_of(name):
    seen = {}
    for _s, _i, _k, p, turn, nxt in requests_of(name):
        for t in (p, turn, nxt):
            seen[hashlib.sha256(t.encode()).hexdigest()] = t
    return seen


def parity(name, hf, llama):
    """Token ids of every distinct rendered text under both tokenizers."""
    texts = texts_of(name)
    diffs = []
    for key, t in texts.items():
        a, b = hf(t), llama(t)
        if a != b:
            n = lcp(a, b)
            diffs.append({"text_sha256": key, "len_hf": len(a), "len_llama": len(b), "index": n,
                          "hf": [hf.piece(i) for i in a[max(0, n - 2):n + 3]], "llama": b[max(0, n - 2):n + 3]})
    return {"texts": len(texts), "equal": len(texts) - len(diffs), "diffs": diffs,
            "texts_with_backslash": sum(1 for t in texts.values() if "\\" in t)}


def tags(model, tok):
    return {t: tok(t) for t in TAGS[model]}


def summarize(results):
    lines = []
    for kind, per in results.items():
        if kind == "parity":
            for name, v in per.items():
                lines.append("parity %s: %d/%d distinct texts tokenized to equal ids by hf and llama (texts with a "
                             "backslash: %d)" % (name, v["equal"], v["texts"], v["texts_with_backslash"]))
                for d in v["diffs"][:5]:
                    lines.append("  DIFF " + json.dumps(d, ensure_ascii=False))
            continue
        for name, v in per.items():
            lines.append("%s %s: tags %s" % (kind, name, json.dumps(v["tags"], ensure_ascii=False)))
            by = {}
            for r in v["rows"]:
                by.setdefault(r["session"], []).append(r)
            for sname, rows in by.items():
                lines.append("  %s: %d request pairs; byte-extends %d, token-extends %d, prompt-token-extends %d, "
                             "concat canonical %d" % (sname, len(rows), sum(r["byte_extends"] for r in rows),
                                                      sum(r["token_extends"] for r in rows),
                                                      sum(r["prompt_token_extends"] for r in rows),
                                                      sum(r["concat_canonical"] for r in rows)))
                for r in rows:
                    if not (r["token_extends"] and r["byte_extends"] and r["concat_canonical"]):
                        lines.append("    request %d (k=%d): bytes %s (reuse %d of %d), tokens %s (reuse %d of %d, "
                                     "prompt %d)%s" % (r["request"], r["k"], r["byte_extends"], r["byte_reuse"],
                                                       r["held_bytes"], r["token_extends"], r["token_reuse"],
                                                       r["held_tokens"], r["prompt_tokens"],
                                                       (" at " + json.dumps(r["at"], ensure_ascii=False))
                                                       if "at" in r else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-glm")
    ap.add_argument("--hf-qwen")
    ap.add_argument("--llama-tokenize")
    ap.add_argument("--vocab-glm")
    ap.add_argument("--vocab-qwen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    toks = {"hf": {}, "llama": {}}
    if a.hf_glm:
        toks["hf"]["glm"] = HFTok(a.hf_glm)
    if a.hf_qwen:
        toks["hf"]["qwen"] = HFTok(a.hf_qwen)
    if a.llama_tokenize:
        if a.vocab_glm:
            toks["llama"]["glm"] = LlamaTok(a.llama_tokenize, a.vocab_glm, os.path.join(a.out, "tok-glm"))
        if a.vocab_qwen:
            toks["llama"]["qwen"] = LlamaTok(a.llama_tokenize, a.vocab_qwen, os.path.join(a.out, "tok-qwen"))
    results = {}
    for kind, by_model in toks.items():
        if not by_model:
            continue
        results[kind] = {}
        for name, (_d, _t, model, _v) in TEMPLATES.items():
            if model in by_model:
                results[kind][name] = {"tags": tags(model, by_model[model]), "rows": check(name, by_model[model])}
    results["parity"] = {}
    for name, (_d, _t, model, _v) in TEMPLATES.items():
        if model in toks["hf"] and model in toks["llama"]:
            results["parity"][name] = parity(name, toks["hf"][model], toks["llama"][model])
    with open(os.path.join(a.out, "token-prefix.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(summarize(results))


if __name__ == "__main__":
    main()
