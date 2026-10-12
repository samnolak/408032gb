#!/usr/bin/env python3
"""N15: N14's render and prefix sections (tools/chattmpl/run_n14.py, unchanged) on the chat templates embedded in the
Unsloth GGUFs, next to the vendored HF templates, plus jinja2(vendored) vs jinja2(embedded) on the same inputs.

The R1 configs start llama-server with --jinja and no --chat-template-file (scripts/r1/configs/llamacpp-server.json),
so the server uses the template embedded in the GGUF: this is what an R1 run renders. The embedded templates are read
from the committed headers (third_party/gguf-headers/*/<first shard>.header.gz); the tokenizer_config.json of the
vendored HF directory supplies the special tokens for jinja2 (as in N14). Tiny models: N14's byte-vocabulary models.

Usage: python3 <this> --server-bin <llama-server @79e2e74> --glm-gguf <glm5next_bytes.gguf> --qwen-gguf
       <qwen4exp_bytes.gguf> --out <scratch dir>        (run from a scratch directory: llama-server writes logs to cwd)
Writes <out>/embedded-templates.json and prints a summary.
"""
import argparse
import gzip
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
sys.path.insert(0, str(ROOT / "tools" / "gguf_headers"))
import convs as C  # noqa: E402
import gguf_header as G  # noqa: E402
import hf_render  # noqa: E402
import run_n14 as R  # noqa: E402

HEADERS = {
    "glm": "third_party/gguf-headers/glm-5.3-flash-ud-q4_k_xl/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf.header.gz",
    "qwen": "third_party/gguf-headers/qwen3.8-flash-next-ud-q4_k_xl/"
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf.header.gz",
}
CASE_FILTER = ("reasoning", "agent_session", "reasoning_mixed", "reasoning_empty")   # as run_n14.section_render


def embedded_dir(name, vendored, out):
    with open(ROOT / HEADERS[name], "rb") as f:
        hb = gzip.decompress(f.read())
    t = G.parse(hb).get("tokenizer.chat_template")
    d = pathlib.Path(out) / ("embedded-" + name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chat_template.jinja").write_bytes(t.encode("utf-8", errors="surrogateescape"))
    shutil.copy(vendored / "tokenizer_config.json", d / "tokenizer_config.json")
    return d


def jinja2_pair(name, vend, emb):
    tv, te = hf_render.Template(vend), hf_render.Template(emb)
    rows = []
    for vname, (hkw, _skw, _extra) in R.VARIANTS[name].items():
        for cname, case in C.CASES.items():
            if vname != "default" and cname not in CASE_FILTER:
                continue
            msgs = case["messages"] if vname != "enable_thinking_false" else R.no_reasoning(case["messages"])
            tools = C.TOOLS if case.get("tools") else None
            for agp in (True, False):
                a, ea = R.try_render(lambda m, add_generation_prompt: tv.render(
                    m, tools=tools, add_generation_prompt=add_generation_prompt, **hkw), msgs, agp)
                b, eb = R.try_render(lambda m, add_generation_prompt: te.render(
                    m, tools=tools, add_generation_prompt=add_generation_prompt, **hkw), msgs, agp)
                row = R.compare_row(name, vname, cname, agp, a, ea, b, eb)
                # compare_row names its sides after N14's (jinja2 = reference, server = other): here they are the
                # vendored and the embedded template, both rendered by jinja2
                rename = {"jinja2_error": "vendored_error", "server_error": "embedded_error", "jinja2": "vendored",
                          "server": "embedded", "len_jinja2": "len_vendored", "len_server": "len_embedded"}
                rows.append({rename.get(k, k): v for k, v in row.items()})
    return rows


def summarize(results):
    lines = []
    for key, res in results.items():
        if key.endswith("/jinja2_vendored_vs_embedded"):
            rows = res
            bad = [r for r in rows if not r["equal"]]
            lines.append(f"{key}: {len(rows) - len(bad)} of {len(rows)} renders byte-equal")
            for r in bad:
                lines.append("  differ: %s/%s agp=%s %s" % (r["variant"], r["case"], r["add_generation_prompt"], json.dumps(
                    {k: v for k, v in r.items() if k not in ("template", "variant", "case", "add_generation_prompt",
                                                             "equal")}, ensure_ascii=False)[:400]))
            continue
        rows = res["render"]["rows"]
        bad = [r for r in rows if not r["equal"]]
        lines.append(f"{key} render (server vs jinja2): {len(rows) - len(bad)} of {len(rows)} equal")
        for r in bad:
            what = r.get("server_error") or r.get("jinja2_error")
            what = ("error: " + what[:160]) if what else json.dumps(
                {k: v for k, v in r.items() if k in ("first_diff", "server", "jinja2", "ref", "got")}, ensure_ascii=False)[:300]
            lines.append("  %s/%s agp=%s %s" % (r["variant"], r["case"], r["add_generation_prompt"], what))
        for sname, v in res["prefix"].items():
            if sname in ("switches", "server_warnings"):
                continue
            hb = [h for h in v["history"] if h.get("stable") is False]
            herr = [h for h in v["history"] if h.get("stable") is None]
            lb = [x for x in v["loop"] if x.get("extends") is False or x.get("gen_prompt_is_prefix") is False]
            le = [x for x in v["loop"] if x.get("error")]
            lines.append("  prefix %s: history breaks %d (k=%s), refused prefixes %d; loop breaks %d of %d requests%s" % (
                sname, len(hb), [h["k"] for h in hb], len(herr), len(lb), len(v["loop"]),
                (", errors %d: %s" % (len(le), le[0]["error"][:160])) if le else ""))
            for x in lb:
                lines.append("    break at request %s: byte %s, %s -> %s" % (x["request"], x.get("first_diff_bytes"),
                                                                          json.dumps(x.get("old"), ensure_ascii=False),
                                                                          json.dumps(x.get("new"), ensure_ascii=False)))
        for sw, v in res["prefix"]["switches"].items():
            lines.append("  switch %s: %s" % (sw, json.dumps(v, ensure_ascii=False)[:200]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--glm-gguf", required=True)
    ap.add_argument("--qwen-gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    args = argparse.Namespace(server_bin=a.server_bin, glm_gguf=a.glm_gguf, qwen_gguf=a.qwen_gguf, harness=None)
    results = {}
    for name in ("glm", "qwen"):
        vend = R.TEMPLATES[name]["dir"]
        emb = embedded_dir(name, vend, a.out)
        try:
            for src, d in (("vendored", vend), ("embedded", emb)):
                R.TEMPLATES[name]["dir"] = d
                out = os.path.join(a.out, src)
                results[f"{name}/{src}"] = {"render": R.section_render(args, name, out),
                                            "prefix": R.section_prefix(args, name, out)}
        finally:
            R.TEMPLATES[name]["dir"] = vend
        results[f"{name}/jinja2_vendored_vs_embedded"] = jinja2_pair(name, vend, emb)
    with open(os.path.join(a.out, "embedded-templates.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False, default=str)
    print(summarize(results))


if __name__ == "__main__":
    main()
