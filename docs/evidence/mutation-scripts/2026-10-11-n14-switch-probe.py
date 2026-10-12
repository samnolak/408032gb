#!/usr/bin/env python3
"""N14 probe: a setting switched between two requests of one agent session (reasoning_effort, or one more tool in the
tools list) on llama-server, live. Requests 0 and 1 of the agent session run with the defaults (turns forced as in
tools/chattmpl/run_n14.py section live), request 2 with the switch; prints timings.cache_n / prompt_n per request.
Run from the repository root:
  python3 docs/evidence/mutation-scripts/2026-10-11-n14-switch-probe.py <llama-server> <glm5next_bytes.gguf> \
      <qwen4exp_bytes.gguf> <work dir>
"""
import json
import os
import sys

sys.path.insert(0, "tools/chattmpl")
import convs as C  # noqa: E402
import run_n14 as R  # noqa: E402
import server as S  # noqa: E402

srv_bin, glm, qwen, work = sys.argv[1:]
EXTRA_TOOL = {"type": "function", "function": {"name": "web_search", "description": "Search.",
                                               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
for name, model in (("glm", glm), ("qwen", qwen)):
    tdir = R.TEMPLATES[name]["dir"]
    for switch in ("reasoning_effort_low", "one_tool_added"):
        wd = os.path.join(work, "switch-%s-%s" % (name, switch))
        with S.Server(srv_bin, model, str(tdir / "chat_template.jinja"), wd, extra=["-c", "16384", "-lv", "4"]) as srv:
            hf = R.Renderers(name, srv).hf({}, C.TOOLS)
            outs = R.ideal_outputs(hf, C.AGENT_SESSION, R.TEMPLATES[name]["turn_end"])[:3]
            for i, (k, _p, o) in enumerate(outs):
                body = {"messages": C.to_openai(C.AGENT_SESSION[:k]), "tools": C.TOOLS, "tool_choice": "none",
                        "grammar": R.gbnf_literal(o), "temperature": 0, "seed": 1,
                        "max_tokens": len(o.encode()) + 16, "cache_prompt": True}
                if i == 2:
                    if switch == "reasoning_effort_low":
                        body["reasoning_effort"] = "low"
                    else:
                        body["tools"] = C.TOOLS + [EXTRA_TOOL]
                st, r = S.post(srv.base, "/v1/chat/completions", body, timeout=600)
                t = r.get("timings", {}) if st == 200 else {}
                print("%s %s request %d%s: status %d cache_n %s prompt_n %s" % (
                    name, switch, i, " (switched)" if i == 2 else "", st, t.get("cache_n"), t.get("prompt_n")))
            log = srv.log_text()
        for ln in log.splitlines():
            if "forcing full prompt" in ln or "restored context checkpoint" in ln:
                print("    | " + ln.split("| ", 2)[-1].strip())
