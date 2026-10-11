#!/usr/bin/env python3
"""N23 driver: the corrected GLM-5.3-Flash chat template (templates/glm-5.3-flash/chat_template.jinja) against the
official HF template (third_party/hf/glm-5.3-flash-chat/chat_template.jinja @ eb9eb208), on CPU.

Sections:
  equal   jinja2 configured as transformers 5.16.0 (hf_render): official vs corrected, byte for byte, on every
          conversation of convs.CASES + convs.N23_CASES, every GLM template switch, add_generation_prompt true/false.
          If transformers is importable, its own renderer (transformers.utils.chat_template_utils.render_jinja_template)
          is compared with hf_render for both templates as well.
  render  llama-server (--jinja --chat-template-file <corrected>) /apply-template vs jinja2 rendering the OFFICIAL
          template, on the same conversations and switches (run_n14.section_render with name glm_n23); and, for
          contrast, the official template through the server (name glm) on the same conversations.
  prefix  run_n14.section_prefix for glm_n23: N14's sessions plus the mixed agent loop (convs.AGENT_LOOP_MIXED).
  parse   run_n14.section_parse for glm_n23 (format detection, delimiters, parser round trip).
  live    run_n14.section_live for glm_n23 and, for contrast, glm: N14's configurations plus the mixed agent loop;
          timings.cache_n per request.
Writes <out>/results.json and prints a summary (run_n14.summary plus the equal section).

Usage: run_n23.py --server-bin <llama-server @79e2e74> --glm-gguf <glm5next_bytes.gguf> [--harness <chat-parse-harness>]
                  --out <dir> [--sections equal,render,prefix,parse,live]
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402
import hf_render  # noqa: E402
import run_n14 as R  # noqa: E402

GLM_DIR = ROOT / "third_party/hf/glm-5.3-flash-chat"
OURS = ROOT / "templates/glm-5.3-flash/chat_template.jinja"
ALL_CASES = {**C.CASES, **C.N23_CASES}

PREFIX_SESSIONS = dict(R.PREFIX_SESSIONS, **{
    "agent_loop_mixed": ("default", lambda: C.AGENT_LOOP_MIXED),
    "agent_loop_mixed_preserve_off": ("preserve_off", lambda: C.AGENT_LOOP_MIXED),
})
LIVE_CONFIGS = R.LIVE_CONFIGS + [("agent_loop_mixed", "default", lambda: C.AGENT_LOOP_MIXED, [])]


def section_equal():
    official = hf_render.Template(GLM_DIR)
    ours = hf_render.Template(GLM_DIR, OURS)
    rows = []
    for cname, case in ALL_CASES.items():
        tools = C.TOOLS if case.get("tools") else None
        for vname, (hkw, _skw, _extra) in R.VARIANTS["glm"].items():
            msgs = case["messages"] if vname != "enable_thinking_false" else R.no_reasoning(case["messages"])
            for agp in (True, False):
                a, ea = R.try_render(lambda m, add_generation_prompt: official.render(
                    m, tools=tools, add_generation_prompt=add_generation_prompt, **hkw), msgs, agp)
                b, eb = R.try_render(lambda m, add_generation_prompt: ours.render(
                    m, tools=tools, add_generation_prompt=add_generation_prompt, **hkw), msgs, agp)
                row = {"case": cname, "variant": vname, "add_generation_prompt": agp, "equal": ea is None and a == b}
                if not row["equal"]:
                    row.update({"official_error": ea, "ours_error": eb})
                    if a is not None and b is not None:
                        d = A.lcp(a, b)
                        row.update({"first_diff": d, "official": A.snippet(a, d), "ours": A.snippet(b, d)})
                rows.append(row)
    res = {"rows": rows}
    try:
        import transformers
        from transformers.utils.chat_template_utils import render_jinja_template
    except ImportError as e:
        res["transformers"] = {"skipped": "transformers not importable: %s" % e}
        return res
    tr_rows = []
    for tname, tmpl in (("official", official), ("ours", ours)):
        for cname, case in ALL_CASES.items():
            tools = C.TOOLS if case.get("tools") else None
            for vname, (hkw, _skw, _extra) in R.VARIANTS["glm"].items():
                msgs = case["messages"] if vname != "enable_thinking_false" else R.no_reasoning(case["messages"])
                for agp in (True, False):
                    ref = tmpl.render(msgs, tools=tools, add_generation_prompt=agp, **hkw)
                    kw = dict(tmpl.special)
                    kw.update(hkw)
                    out, _ = render_jinja_template(conversations=[msgs], tools=tools, chat_template=tmpl.source,
                                                   add_generation_prompt=agp, **kw)
                    tr_rows.append({"template": tname, "case": cname, "variant": vname, "add_generation_prompt": agp,
                                    "equal": out[0] == ref})
    res["transformers"] = {"version": transformers.__version__, "rows": tr_rows}
    return res


def summary_equal(sec):
    rows = sec["rows"]
    lines = ["equal jinja2 official vs corrected: %d/%d byte-equal" % (sum(r["equal"] for r in rows), len(rows))]
    for r in rows:
        if not r["equal"]:
            lines.append("  DIFF %s" % json.dumps(r, ensure_ascii=False))
    tr = sec.get("transformers", {})
    if "rows" in tr:
        for t in ("official", "ours"):
            rr = [r for r in tr["rows"] if r["template"] == t]
            lines.append("equal transformers %s render_jinja_template vs hf_render (%s): %d/%d byte-equal" % (
                tr["version"], t, sum(r["equal"] for r in rr), len(rr)))
    else:
        lines.append("equal transformers: %s" % tr.get("skipped"))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-bin")
    ap.add_argument("--glm-gguf")
    ap.add_argument("--harness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="equal,render,prefix,parse,live")
    args = ap.parse_args()
    args.qwen_gguf = None
    os.makedirs(args.out, exist_ok=True)
    secs = args.sections.split(",")
    results, lines = {}, []
    if "equal" in secs:
        results["equal"] = section_equal()
        lines += summary_equal(results["equal"])
    names = ("glm_n23", "glm")
    if "render" in secs:
        results["render"] = {n: R.section_render(args, n, args.out, cases=ALL_CASES, variant_cases=tuple(ALL_CASES))
                             for n in names}
    if "prefix" in secs:
        results["prefix"] = {n: R.section_prefix(args, n, args.out, sessions=PREFIX_SESSIONS) for n in names}
    if "parse" in secs:
        results["parse"] = {"glm_n23": R.section_parse(args, "glm_n23", args.out)}
    if "live" in secs:
        results["live"] = {n: R.section_live(args, n, args.out, configs=LIVE_CONFIGS) for n in names}
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    lines += R.summary({k: v for k, v in results.items() if k != "equal"}).splitlines()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
