#!/usr/bin/env python3
"""N14 driver (gap 6 of docs/reassessment/CORRECTNESS.md): the vendored GLM-5.3-Flash and Qwen3.8-Flash-Next chat
templates through llama-server's own Jinja engine and chat code at the R1 pin, against a jinja2 reference renderer.

Sections (all on CPU, tiny random-weight hybrid models with a bytes-only vocabulary, bytes_vocab_model.py; the
weights only have to load, nothing here depends on what they compute):
  render  POST /apply-template for every conversation of convs.CASES, with and without the generation prompt, under
          every template switch (variants below), byte-compared with hf_render (transformers-style jinja2)
  prefix  analysis.history_checks / loop_checks on the agent session, rendered by the server and by jinja2
  parse   tools/chattmpl/chat_parse_harness: detected chat format, generation prompt, message delimiters (the
          checkpoint positions), and the parser on each assistant turn written exactly as the template renders it;
          the parsed message is rendered back (server) and compared with the original bytes
  live    the agent session through /v1/chat/completions, every assistant turn forced byte for byte by a grammar
          (tool_choice none), timings.cache_n / prompt_n per request and the server's checkpoint trace (-lv 4)
Writes <out>/results.json and prints a summary.

Usage: run_n14.py --server-bin <llama-server> --glm-gguf <glm5next_bytes.gguf> --qwen-gguf <qwen4exp_bytes.gguf>
                  [--harness <chat-parse-harness>] --out <dir> [--sections render,prefix,parse,live]
Run it from a scratch directory: llama-server writes its log next to it (<out>/<section>-<template>/srv.log).
"""
import argparse
import copy
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402
import hf_render  # noqa: E402
import server as S  # noqa: E402

TEMPLATES = {
    "glm": {"dir": ROOT / "third_party/hf/glm-5.3-flash-chat", "model_arg": "glm_gguf",
            # the model's end-of-turn token is the next message's role token (<|observation|> / <|user|>), so the
            # forced output stops before it; EOS ends the tiny model's turn instead
            "turn_end": ""},
    "qwen": {"dir": ROOT / "third_party/hf/qwen3.8-flash-next-chat", "model_arg": "qwen_gguf",
             "turn_end": "<|im_end|>"},
    # N23: our corrected GLM template (templates/glm-5.3-flash/chat_template.jinja, the loop-scope fix) is what the
    # server loads; the jinja2 reference that defines the format ("reference": None = <dir>/chat_template.jinja) stays
    # the official HF template, so every section compares the server with our file against the official format
    "glm_n23": {"dir": ROOT / "third_party/hf/glm-5.3-flash-chat", "template": ROOT / "templates/glm-5.3-flash/chat_template.jinja",
                "reference": None, "model_arg": "glm_gguf", "turn_end": ""},
}


def template_path(name):
    """The template file of TEMPLATES[name]: its own "template" entry, else <dir>/chat_template.jinja."""
    t = TEMPLATES[name]
    return pathlib.Path(t["template"]) if t.get("template") else pathlib.Path(t["dir"]) / "chat_template.jinja"


def reference_template(name):
    """The template file the jinja2 reference renders for TEMPLATES[name]: its "reference" entry if present (None =
    <dir>/chat_template.jinja), else the server's template_path(name)."""
    t = TEMPLATES[name]
    if "reference" in t:
        return pathlib.Path(t["reference"]) if t["reference"] else pathlib.Path(t["dir"]) / "chat_template.jinja"
    return template_path(name)


# name -> (jinja2 kwargs, server chat_template_kwargs, extra request body fields). The server maps the request kwarg
# preserve_reasoning to preserve_thinking / clear_thinking / ... (common/jinja/caps.cpp:22-27) and the body field
# reasoning_effort to the template variable (tools/server/server-common.cpp:1417-1426, common/chat.cpp:916-919).
VARIANTS = {
    "glm": {
        "default": ({}, None, None),
        "preserve_off": ({"clear_thinking": True}, {"preserve_reasoning": False}, None),
        "enable_thinking_false": ({"enable_thinking": False}, {"enable_thinking": False}, None),
        "effort_low": ({"reasoning_effort": "low"}, None, {"reasoning_effort": "low"}),
        "effort_high": ({"reasoning_effort": "high"}, None, {"reasoning_effort": "high"}),
        "effort_medium": ({"reasoning_effort": "medium"}, None, {"reasoning_effort": "medium"}),
        # "none" is not passed to the template: llama-server sets enable_thinking=false instead
        # (tools/server/server-common.cpp:1420-1422), so the jinja2 side is enable_thinking=False
        "effort_none": ({"enable_thinking": False}, None, {"reasoning_effort": "none"}),
    },
    "qwen": {
        "default": ({}, None, None),
        "preserve_off": ({"preserve_thinking": False}, {"preserve_reasoning": False}, None),
        "enable_thinking_false": ({"enable_thinking": False}, {"enable_thinking": False}, None),
        "effort_low": ({"reasoning_effort": "low"}, None, {"reasoning_effort": "low"}),
        "effort_medium": ({"reasoning_effort": "medium"}, None, {"reasoning_effort": "medium"}),
        "effort_high": ({"reasoning_effort": "high"}, None, {"reasoning_effort": "high"}),
        "effort_none": ({"enable_thinking": False}, None, {"reasoning_effort": "none"}),
    },
}


VARIANTS["glm_n23"] = VARIANTS["glm"]


def no_reasoning(conv):
    out = copy.deepcopy(conv)
    for m in out:
        m.pop("reasoning_content", None)
    return out


def session_for(variant):
    return no_reasoning(C.AGENT_SESSION) if variant == "enable_thinking_false" else C.AGENT_SESSION


def first_diff(a, b):
    d = A.lcp(a, b)
    return {"first_diff": d, "server": A.snippet(a, d), "jinja2": A.snippet(b, d), "len_server": len(a),
            "len_jinja2": len(b)}


def warnings_of(log):
    return [ln for ln in log.splitlines() if re.match(r"^\S+ [WE] ", ln) and "security: no API key" not in ln]


class Renderers:
    def __init__(self, name, srv):
        self.tmpl = hf_render.Template(TEMPLATES[name]["dir"], reference_template(name))
        self.srv = srv

    def hf(self, hkw, tools):
        def r(msgs, add_generation_prompt=False):
            return self.tmpl.render(msgs, tools=tools, add_generation_prompt=add_generation_prompt, **hkw)
        return r

    def server(self, skw, extra, tools):
        def r(msgs, add_generation_prompt=False):
            st, body = self.srv.apply_template(C.to_openai(msgs), tools=tools,
                                               add_generation_prompt=add_generation_prompt, kwargs=skw, extra=extra)
            if st != 200:
                raise RuntimeError("HTTP %d %s" % (st, json.dumps(body, ensure_ascii=False)[:400]))
            return body["prompt"]
        return r


def try_render(fn, msgs, agp):
    try:
        return fn(msgs, add_generation_prompt=agp), None
    except Exception as e:  # noqa: BLE001
        return None, "%s: %s" % (type(e).__name__, " ".join(str(e).split())[:600])


SERVER_FLAG_RUNS = [  # (row variant, extra server flags, variant of VARIANTS whose jinja2 kwargs it must equal)
    ("server_flag_no_reasoning_preserve", ["--no-reasoning-preserve"], "preserve_off"),
    ("server_flag_reasoning_off", ["--reasoning", "off"], "enable_thinking_false"),
]


def compare_row(name, vname, cname, agp, ref, eref, got, egot):
    row = {"template": name, "variant": vname, "case": cname, "add_generation_prompt": agp}
    if eref or egot:
        row.update({"equal": False, "jinja2_error": eref, "server_error": egot})
    elif got == ref:
        row.update({"equal": True, "len": len(ref)})
    else:
        row.update({"equal": False}, **first_diff(got, ref))
    return row


VARIANT_CASES = ("reasoning", "agent_session", "reasoning_mixed", "reasoning_empty")


def section_render(args, name, out, cases=None, variant_cases=VARIANT_CASES):
    """Rows: server vs jinja2 per variant, case and add_generation_prompt. Rows of the variants that turn thinking off
    (effort_none, server_flag_reasoning_off) also carry same_as_server_default: the server's prompt equals its prompt
    with the defaults (GLM's template does not read enable_thinking). cases: dict like convs.CASES (default it);
    variant_cases: the cases rendered under every variant, the others under the default only."""
    cases = C.CASES if cases is None else cases
    rows = []
    default_prompt = {}
    with S.Server(args.server_bin, getattr(args, TEMPLATES[name]["model_arg"]), str(template_path(name)),
                  os.path.join(out, "render-" + name)) as srv:
        R = Renderers(name, srv)
        for vname, (hkw, skw, extra) in VARIANTS[name].items():
            for cname, case in cases.items():
                if vname != "default" and cname not in variant_cases:
                    continue
                msgs = case["messages"] if vname != "enable_thinking_false" else no_reasoning(case["messages"])
                tools = C.TOOLS if case.get("tools") else None
                for agp in (True, False):
                    ref, eref = try_render(R.hf(hkw, tools), msgs, agp)
                    got, egot = try_render(R.server(skw, extra, tools), msgs, agp)
                    row = compare_row(name, vname, cname, agp, ref, eref, got, egot)
                    if vname == "default":
                        default_prompt[(cname, agp)] = got
                    if vname == "effort_none":
                        row["same_as_server_default"] = got is not None and got == default_prompt.get((cname, agp))
                    rows.append(row)
        logs = {"default": warnings_of(srv.log_text())}
    for vname, flags, like in SERVER_FLAG_RUNS:
        with S.Server(args.server_bin, getattr(args, TEMPLATES[name]["model_arg"]), str(template_path(name)),
                      os.path.join(out, "render-%s-%s" % (name, vname)), extra=flags) as srv:
            R = Renderers(name, srv)
            hkw = VARIANTS[name][like][0]
            for cname in ("reasoning", "agent_session"):
                case = C.CASES[cname]  # always the N14 cases: the flag runs compare two server configurations
                tools = C.TOOLS if case.get("tools") else None
                ref, eref = try_render(R.hf(hkw, tools), case["messages"], True)
                got, egot = try_render(R.server(None, None, tools), case["messages"], True)
                row = compare_row(name, vname, cname, True, ref, eref, got, egot)
                if vname == "server_flag_reasoning_off":
                    row["same_as_server_default"] = got is not None and got == default_prompt.get((cname, True))
                rows.append(row)
            logs[vname] = warnings_of(srv.log_text())
    return {"rows": rows, "server_warnings": logs}


PREFIX_SESSIONS = {  # session name -> (variant of VARIANTS, conversation)
    "default": ("default", lambda: C.AGENT_SESSION),
    "preserve_off": ("preserve_off", lambda: C.AGENT_SESSION),
    "enable_thinking_false": ("enable_thinking_false", lambda: no_reasoning(C.AGENT_SESSION)),
    "empty_reasoning_turn": ("default", lambda: C.AGENT_SESSION_EMPTY_REASONING),
}


def section_prefix(args, name, out, sessions=None):
    """history_checks and loop_checks: jinja2 alone, the server alone, and the server's prompts with the turns a model
    trained on the reference template writes (server_prompts+jinja2_turns). The server runs with
    --no-prefill-assistant so that a history ending with an assistant turn renders as history (with the default
    --prefill-assistant it is a continuation, and one with tool calls is refused)."""
    res = {}
    with S.Server(args.server_bin, getattr(args, TEMPLATES[name]["model_arg"]), str(template_path(name)),
                  os.path.join(out, "prefix-" + name), extra=["--no-prefill-assistant"]) as srv:
        R = Renderers(name, srv)
        for sname, (vname, conv_fn) in (PREFIX_SESSIONS if sessions is None else sessions).items():
            hkw, skw, extra = VARIANTS[name][vname]
            conv = conv_fn()
            sfn, hfn = R.server(skw, extra, C.TOOLS), R.hf(hkw, C.TOOLS)
            res["%s/jinja2" % sname] = {"history": A.history_checks(hfn, conv), "loop": A.loop_checks(hfn, conv)}
            res["%s/server" % sname] = {"history": A.history_checks(sfn, conv), "loop": A.loop_checks(sfn, conv)}
            res["%s/server_prompts+jinja2_turns" % sname] = {"history": [],
                                                              "loop": A.loop_checks(sfn, conv, render_out=hfn)}
        # a switch between two requests of one session: the same history rendered under another setting; the shared
        # byte prefix is all the server can reuse (the first context checkpoint lies after it, or none does)
        conv = C.AGENT_SESSION
        k = A.request_points(conv)[-1]
        base = R.server(None, None, C.TOOLS)(conv[:k], add_generation_prompt=True)
        sw = {}
        for vname, (hkw, skw, extra) in VARIANTS[name].items():
            if vname == "default":
                continue
            other, err = try_render(R.server(skw, extra, C.TOOLS), conv[:k], True)
            sw[vname] = {"error": err} if err else {"shared_bytes": A.lcp(base.encode(), other.encode()),
                                                    "prompt_bytes": len(base.encode()),
                                                    "at": A.snippet(other, A.lcp(base, other))}
        extra_tool = C.TOOLS + [{"type": "function", "function": {"name": "web_search", "description": "Search.",
                                 "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        other = R.server(None, None, extra_tool)(conv[:k], add_generation_prompt=True)
        sw["one_tool_added"] = {"shared_bytes": A.lcp(base.encode(), other.encode()),
                                "prompt_bytes": len(base.encode()), "at": A.snippet(other, A.lcp(base, other))}
        res["switches"] = sw
        res["server_warnings"] = warnings_of(srv.log_text())
    return res


def ideal_outputs(render, conv, turn_end):
    """(k, prompt, output) for every request point: output = the assistant turn as the template renders it in history,
    cut after the model's end-of-turn text (turn_end) when the template writes one."""
    outs = []
    for k in A.request_points(conv):
        p = render(conv[:k], add_generation_prompt=True)
        hist = render(conv[:k + 1], add_generation_prompt=False)
        if not hist.startswith(p):
            outs.append((k, p, None))
            continue
        o = hist[len(p):]
        if turn_end and turn_end in o:
            o = o[:o.index(turn_end) + len(turn_end)]
        outs.append((k, p, o))
    return outs


def norm_msg(m):
    """Fields a client sends back, in a comparable form (arguments parsed)."""
    calls = []
    for c in m.get("tool_calls") or []:
        a = c["function"]["arguments"]
        calls.append((c["function"]["name"], json.loads(a) if isinstance(a, str) else a))
    return {"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
            "tool_calls": calls}


def section_parse(args, name, out):
    if not args.harness:
        return {"skipped": "no --harness"}
    tdir = TEMPLATES[name]["dir"]
    tmpl = hf_render.Template(tdir, reference_template(name))
    conv = C.AGENT_SESSION
    hf = lambda m, add_generation_prompt=False: tmpl.render(m, tools=C.TOOLS, add_generation_prompt=add_generation_prompt)  # noqa: E731
    turn_end = TEMPLATES[name]["turn_end"]
    reqs, meta = [], []
    for k, p, o in ideal_outputs(hf, conv, turn_end):
        text = o[:-len(turn_end)] if (o and turn_end and o.endswith(turn_end)) else o
        cands = {"ideal": text}
        if text is not None:
            # whitespace the template would not reproduce: around the reasoning, before a tool call
            r = conv[k].get("reasoning_content") or ""
            if r and r in text:
                cands["reasoning_padded"] = text.replace(r, "\n" + r + "\n", 1)
            c = conv[k].get("content") or ""
            if c and c in text:
                cands["content_padded"] = text.replace(c, "\n" + c + "\n", 1)
            calls = conv[k].get("tool_calls") or []
            if calls and not meta:  # first request point only: a call to a tool that is not in the list
                cands["unknown_tool"] = text.replace(calls[0]["function"]["name"], "no_such_tool", 1)
        reqs.append({"id": "%s-k%d" % (name, k), "messages": C.to_openai(conv[:k]), "tools": C.TOOLS,
                     "outputs": list(cands.values())})
        meta.append((k, p, o, list(cands.keys())))
    # format detection without tools, and with enable_thinking false
    reqs.append({"id": "%s-notools" % name, "messages": C.to_openai(C.CASES["system_user"]["messages"])})
    reqs.append({"id": "%s-nothink" % name, "messages": C.to_openai(C.CASES["system_user"]["messages"]),
                 "kwargs": {"enable_thinking": False}})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=out) as f:
        json.dump({"template_file": str(template_path(name)), "requests": reqs}, f, ensure_ascii=False)
        spec = f.name
    proc = subprocess.run([args.harness, spec], capture_output=True, text=True, cwd=out, timeout=600,
                          env=dict(os.environ, CHATTMPL_DEBUG="1"))
    if proc.returncode != 0:
        return {"error": "harness exit %d" % proc.returncode, "stderr": proc.stderr[-4000:]}
    lines = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    by_id = {x["id"]: x for x in lines}
    handler = sorted({ln.strip() for ln in proc.stderr.splitlines()
                      if "Using specialized template" in ln or "using differential autoparser" in ln})
    res = {"requests": [], "notools": by_id.get("%s-notools" % name), "nothink": by_id.get("%s-nothink" % name),
           "handler_log": handler}
    with S.Server(args.server_bin, getattr(args, TEMPLATES[name]["model_arg"]), str(template_path(name)),
                  os.path.join(out, "parse-" + name)) as srv:
        R = Renderers(name, srv)
        srender = R.server(None, None, C.TOOLS)
        for (k, p, o, labels), req in zip(meta, reqs):
            h = by_id[req["id"]]
            row = {"k": k, "format": h.get("format"), "generation_prompt": h.get("generation_prompt"),
                   "thinking_start_tag": h.get("thinking_start_tag"), "thinking_end_tags": h.get("thinking_end_tags"),
                   "delimiters": h.get("delimiters"), "harness_prompt_equals_jinja2": h.get("prompt") == p,
                   "tool_grammar": h.get("grammar_nonempty"),
                   "error": h.get("error"), "candidates": []}
            want = norm_msg(C.to_openai([conv[k]])[0])
            for label, parsed in zip(labels, h.get("parsed", [])):
                cand = {"label": label}
                if "error" in parsed:
                    cand["parse_error"] = parsed["error"]
                    row["candidates"].append(cand)
                    continue
                msg = parsed["message"]
                cand["parsed"] = msg
                got = norm_msg(msg)
                cand["fields_equal"] = {f: got[f] == want[f] for f in want}
                # round trip: the parsed message back into the history (followed by the next message, so that the
                # server does not treat a trailing assistant message as a prefill), rendered by the server
                nxt = C.to_openai(conv[k + 1:k + 2])
                back = srender(C.to_openai(conv[:k]) + [msg] + nxt, add_generation_prompt=False)
                if label == "ideal":
                    orig = srender(C.to_openai(conv[:k + 2]), add_generation_prompt=False)
                    cand["round_trip_equal"] = back == orig
                    if back != orig:
                        d = A.lcp(back, orig)
                        cand.update({"round_trip_first_diff": d, "rendered": A.snippet(back, d),
                                     "original": A.snippet(orig, d)})
                else:
                    gen = p + req["outputs"][labels.index(label)]
                    cand["round_trip_reproduces_output"] = back.startswith(gen)
                    if not cand["round_trip_reproduces_output"]:
                        d = A.lcp(back, gen)
                        cand.update({"round_trip_first_diff": d, "rendered": A.snippet(back, d),
                                     "generated": A.snippet(gen, d)})
                row["candidates"].append(cand)
            res["requests"].append(row)
    return res


def gbnf_literal(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return 'root ::= "' + "".join(out) + '"'


LIVE_CONFIGS = [  # tag, variant of VARIANTS, conversation, extra server flags
    ("default", "default", lambda: C.AGENT_SESSION, []),
    ("preserve_off", "preserve_off", lambda: C.AGENT_SESSION, []),
    ("preserve_off-ctx-checkpoints-4", "preserve_off", lambda: C.AGENT_SESSION, ["--ctx-checkpoints", "4"]),
    ("empty_reasoning_turn", "default", lambda: C.AGENT_SESSION_EMPTY_REASONING, []),
]


def section_live(args, name, out, configs=None):
    """The agent session through /v1/chat/completions. Each request k sends conv[:k]; the assistant turn conv[k] is
    forced byte for byte (GBNF literal of the turn as the jinja2 reference renders it after the generation prompt,
    up to and including the template's end-of-turn text if it writes one, then EOS). With a bytes-only vocabulary
    (bytes_vocab_model.py) one token = one byte, so the text prediction below is in tokens:
      held   = bytes of the server's prompt k + the forced turn (all decoded: EOS is sampled after the last byte)
      shared = common byte prefix of held and the server's prompt k+1
    shared == held: the next request should reuse all of held (cache_n = held); otherwise the server restores the
    newest context checkpoint at or before `shared` (n_tokens in the trace) or re-processes from 0."""
    res = {}
    turn_end = TEMPLATES[name]["turn_end"]
    for tag, vname, conv_fn, xflags in (LIVE_CONFIGS if configs is None else configs):
        hkw, skw, extra = VARIANTS[name][vname]
        conv = conv_fn()
        flags = ["-c", "16384", "-lv", "4"] + xflags
        wd = os.path.join(out, "live-%s-%s" % (name, tag))
        rows = []
        with S.Server(args.server_bin, getattr(args, TEMPLATES[name]["model_arg"]), str(template_path(name)),
                      wd, extra=flags) as srv:
            R = Renderers(name, srv)
            hf, sv = R.hf(hkw, C.TOOLS), R.server(skw, extra, C.TOOLS)
            outs = ideal_outputs(hf, conv, turn_end)
            prev_held = None
            for i, (k, _p, o) in enumerate(outs):
                p = sv(conv[:k], add_generation_prompt=True)
                row = {"request": i, "k": k, "prompt_bytes": len(p.encode())}
                if prev_held is not None:
                    shared = A.lcp(prev_held, p.encode())
                    row.update({"prev_held_bytes": len(prev_held), "shared_bytes": shared,
                                "extends": shared == len(prev_held)})
                body = {"messages": C.to_openai(conv[:k]), "tools": C.TOOLS, "tool_choice": "none",
                        "grammar": gbnf_literal(o), "temperature": 0, "seed": 1, "max_tokens": len(o.encode()) + 16,
                        "cache_prompt": True}
                if skw:
                    body["chat_template_kwargs"] = skw
                if extra:
                    body.update(extra)
                st, r = S.post(srv.base, "/v1/chat/completions", body, timeout=600)
                row["status"] = st
                if st != 200:
                    row["error"] = json.dumps(r, ensure_ascii=False)[:600]
                    rows.append(row)
                    prev_held = None
                    continue
                t = r.get("timings", {})
                row.update({"cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
                            "predicted_n": t.get("predicted_n"), "forced_bytes": len(o.encode()),
                            "finish_reason": r["choices"][0].get("finish_reason")})
                rows.append(row)
                prev_held = (p + o).encode()
            log = srv.log_text()
        ck = [ln.split("| ", 2)[-1].strip() for ln in log.splitlines()
              if re.search(r"created context checkpoint|restored context checkpoint|forcing full prompt|"
                           r"erased invalidated|erasing", ln)]
        res[tag] = {"requests": rows, "checkpoint_log": ck, "flags": flags, "warnings": warnings_of(log)[:40]}
    return res


def summary(results):
    lines = []
    for name, sec in results.get("render", {}).items():
        eq = sum(1 for r in sec["rows"] if r.get("equal"))
        lines.append("render %s: %d/%d byte-equal (server vs jinja2)" % (name, eq, len(sec["rows"])))
        for r in sec["rows"]:
            if not r.get("equal"):
                if r.get("server_error") or r.get("jinja2_error"):
                    lines.append("  %s %s agp=%s server_error=%s | jinja2_error=%s" % (
                        r["variant"], r["case"], r["add_generation_prompt"], r.get("server_error"), r.get("jinja2_error")))
                else:
                    lines.append("  %s %s agp=%s first_diff=%d server=%r jinja2=%r" % (
                        r["variant"], r["case"], r["add_generation_prompt"], r["first_diff"], r["server"], r["jinja2"]))
        for r in sec["rows"]:
            if "same_as_server_default" in r:
                lines.append("  %s %s agp=%s equal_to_jinja2=%s same_as_server_default=%s" % (
                    r["variant"], r["case"], r["add_generation_prompt"], r["equal"], r["same_as_server_default"]))
        for run, ws in sec["server_warnings"].items():
            for w in ws:
                lines.append("  log (%s): %s" % (run, w))
    for name, sec in results.get("prefix", {}).items():
        for w in sec.get("server_warnings", []):
            lines.append("prefix %s log: %s" % (name, w))
        for key, v in sec.get("switches", {}).items():
            lines.append("prefix %s switch %s: %s" % (name, key, json.dumps(v, ensure_ascii=False)))
        for key, v in sec.items():
            if key in ("server_warnings", "switches"):
                continue
            hb = [h for h in v["history"] if h.get("stable") is False]
            herr = [h for h in v["history"] if h.get("stable") is None]
            lb = [x for x in v["loop"] if x.get("extends") is False or x.get("gen_prompt_is_prefix") is False]
            lines.append("prefix %s %s: history breaks %d (k=%s), refused prefixes %d; loop breaks %d of %d requests" % (
                name, key, len(hb), [h["k"] for h in hb], len(herr), len(lb), len(v["loop"])))
            for x in lb:
                lines.append("  request %d (k=%d): first_diff byte %s of %s held, in message %s (%s) old=%r new=%r" % (
                    x["request"], x["k"], x.get("first_diff_bytes"), x.get("held_bytes"), x.get("in_message"),
                    x.get("in_message_role"), x.get("old"), x.get("new")))
    for name, sec in results.get("parse", {}).items():
        if "requests" not in sec:
            lines.append("parse %s: %s" % (name, sec))
            continue
        fm = {r["format"] for r in sec["requests"]}
        lines.append("parse %s: format %s; notools %s; nothink gen prompt %r; handler %s" % (
            name, sorted(x for x in fm if x), sec["notools"].get("format") if sec["notools"] else None,
            sec["nothink"].get("generation_prompt") if sec["nothink"] else None, sec.get("handler_log")))
        r0 = sec["requests"][0]
        lines.append("  generation_prompt %r thinking %r %r delimiters %s" % (
            r0["generation_prompt"], r0["thinking_start_tag"], r0["thinking_end_tags"],
            [(d["role"], d["delimiter"]) for d in r0["delimiters"] or []]))
        for r in sec["requests"]:
            for c in r["candidates"]:
                lines.append("  k=%d %s: %s" % (r["k"], c["label"], json.dumps(
                    {x: c[x] for x in c if x not in ("label", "parsed")}, ensure_ascii=False)))
    for name, sec in results.get("live", {}).items():
        for tag, v in sec.items():
            lines.append("live %s %s (%s):" % (name, tag, " ".join(v["flags"])))
            for r in v["requests"]:
                lines.append("  request %d k=%d status %s prompt %s B | held %s B shared %s B extends %s | cache_n %s "
                             "prompt_n %s | forced %s B predicted_n %s finish %s %s" % (
                                 r["request"], r["k"], r["status"], r["prompt_bytes"], r.get("prev_held_bytes"),
                                 r.get("shared_bytes"), r.get("extends"), r.get("cache_n"), r.get("prompt_n"),
                                 r.get("forced_bytes"), r.get("predicted_n"), r.get("finish_reason"),
                                 r.get("error", "")))
            for c in v["checkpoint_log"]:
                lines.append("    | " + c)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--glm-gguf", required=True)
    ap.add_argument("--qwen-gguf", required=True)
    ap.add_argument("--harness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="render,prefix,parse,live")
    ap.add_argument("--templates", default="glm,qwen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fns = {"render": section_render, "prefix": section_prefix, "parse": section_parse, "live": section_live}
    results = {}
    for sec in args.sections.split(","):
        results[sec] = {}
        for name in args.templates.split(","):
            results[sec][name] = fns[sec](args, name, args.out)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(summary(results))


if __name__ == "__main__":
    main()
