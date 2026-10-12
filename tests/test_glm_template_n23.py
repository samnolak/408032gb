"""N23: our corrected GLM-5.3-Flash chat template (templates/glm-5.3-flash/chat_template.jinja).

The official template (third_party/hf/glm-5.3-flash-chat/chat_template.jinja, zai-org/GLM-5.3-Flash @ eb9eb208) sets
reasoning_content inside the message loop only when a message has reasoning and tests "is defined"; llama.cpp's Jinja
engine at 79e2e74 keeps a {% set %} of one loop iteration in the next (N14, docs/evidence/2026-10-11-n14-chat-templates.md
finding 1(a)), so a turn without reasoning was rendered with the previous turn's. Ours resets the variable to none at
the top of every assistant iteration and tests "is not none"; nothing else changes.

Without anything built:
  CorrectedFile        the file is the source plus the header comment, one inserted line and one changed line
With jinja2 (installed by the `deps` step of ci.yml cpu-parity):
  JinjaEquality        ours renders byte for byte like the official template, jinja2 configured as transformers 5.16.0
                       (tools/chattmpl/hf_render.py), on every N14 and N23 conversation, switch and generation flag
With transformers (the token-level workflow installs it; skipped where it is missing):
  TransformersRender   transformers' own render_jinja_template gives the same bytes as hf_render, for both templates
With a llama-server build of the R1 pin (CHATTMPL_SERVER_BIN, CHATTMPL_GLM_GGUF; CHATTMPL_HARNESS for the parser;
tools/chattmpl/run_n23.sh):
  ServerRenderN23      llama-server with ours == jinja2 with the official template, except the two known non-template
                       differences; the official template through the server still leaks (the engine bug is still there)
  ParseHarnessN23      format detection, delimiters and the parser round trip are those of the official template (N14)
  PrefixN23            server prompts with ours extend every held prompt + turn except where past reasoning is dropped
  LiveB3N23            /v1/chat/completions: cache_n equals everything the slot holds at every request but the one
                       after a new user message with past reasoning dropped
"""
import argparse
import difflib
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402

GLM_DIR = ROOT / "third_party/hf/glm-5.3-flash-chat"
OURS = ROOT / "templates/glm-5.3-flash/chat_template.jinja"
SOURCE_SHA256 = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"
SOURCE_REV = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
ALL_CASES = {**C.CASES, **C.N23_CASES}
# the N23 conversations in which the official template leaks through llama-server @79e2e74 under the default switches,
# and the reasoning that leaks into the turn without one
LEAKS = {"reasoning_mixed": "R1", "reasoning_empty": "R1", "missing_after_two": "R2", "inline_then_missing": "R1",
         "reasoning_whitespace_then_missing": " \n ", "tools_reasoning_empty": "R1",
         "agent_session_empty_reasoning": "parse() splits on ';'"}

try:
    import jinja2  # noqa: F401
    HAVE_JINJA = True
except ImportError:
    HAVE_JINJA = False
try:
    from transformers.utils.chat_template_utils import render_jinja_template
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

SERVER = os.environ.get("CHATTMPL_SERVER_BIN")
GLM_GGUF = os.environ.get("CHATTMPL_GLM_GGUF")
HARNESS = os.environ.get("CHATTMPL_HARNESS")
HAVE_SERVER = bool(SERVER and GLM_GGUF and HAVE_JINJA)
SKIP_SERVER = "set CHATTMPL_SERVER_BIN, CHATTMPL_GLM_GGUF (tools/chattmpl/run_n23.sh)"


def args_ns():
    return argparse.Namespace(server_bin=SERVER, glm_gguf=GLM_GGUF, qwen_gguf=None, harness=HARNESS)


def workdir(name):
    base = os.environ.get("CHATTMPL_WORK")
    if base:
        d = os.path.join(base, "n23-" + name)
        os.makedirs(d, exist_ok=True)
        return d
    return tempfile.mkdtemp(prefix="chattmpl-n23-%s-" % name)


def strip_header(text):
    """The template without our leading {#- ... -#} comment (and the newline after it)."""
    m = re.match(r"\{#-.*?-#\}\n", text, re.S)
    assert m, "no header comment"
    return text[m.end():], m.group(0)


class CorrectedFile(unittest.TestCase):
    def test_source_is_the_pinned_template(self):
        src = (GLM_DIR / "chat_template.jinja").read_bytes()
        self.assertEqual(hashlib.sha256(src).hexdigest(), SOURCE_SHA256)
        self.assertEqual(json.loads((GLM_DIR / "PIN.json").read_text())["revision"], SOURCE_REV)

    def test_header_names_source_and_change(self):
        _, header = strip_header(OURS.read_text(encoding="utf-8"))
        for s in (SOURCE_REV, SOURCE_SHA256, "zai-org/GLM-5.3-Flash", "79e2e74", "is not none", "MIT"):
            self.assertIn(s, header)
        # the upstream MIT notice (copyright line and permission notice), verbatim from LICENSE of the same revision
        lic = (ROOT / "third_party/hf/glm-5.3-flash-tokenizer/LICENSE").read_text(encoding="utf-8").strip()
        self.assertIn("Copyright (c) 2026 Z.AI Co., Ltd", lic)
        self.assertIn(lic, header)
        # llama.cpp picks some chat handlers by substrings of the template source (common/chat.cpp:1082-1283
        # @79e2e74): the header must not add any tag
        self.assertNotRegex(header, r"[<\[]")

    def test_only_the_two_lines_differ(self):
        body, _ = strip_header(OURS.read_text(encoding="utf-8"))
        src = (GLM_DIR / "chat_template.jinja").read_text(encoding="utf-8")
        diff = [ln for ln in difflib.unified_diff(src.splitlines(), body.splitlines(), lineterm="", n=0)
                if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        self.assertEqual(diff, [
            "+{%- set reasoning_content = none %}",
            "-{%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}",
            "+{%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is not none -%}",
        ])
        # the reset is the first statement after content is computed, before any branch can set reasoning_content
        lines = body.splitlines()
        i = lines.index("{%- set reasoning_content = none %}")
        self.assertEqual(lines[i - 1], "{%- set content = visible_text(m.content) %}")
        self.assertEqual(lines[i - 3], "{%- elif m.role == 'assistant' -%}")


@unittest.skipUnless(HAVE_JINJA, "jinja2 is not installed")
class JinjaEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hf_render
        import run_n14
        cls.R = run_n14
        cls.official = hf_render.Template(GLM_DIR)
        cls.ours = hf_render.Template(GLM_DIR, OURS)

    def test_byte_equal_everywhere(self):
        n = 0
        for cname, case in ALL_CASES.items():
            tools = C.TOOLS if case.get("tools") else None
            for vname, (hkw, _skw, _extra) in self.R.VARIANTS["glm"].items():
                msgs = case["messages"] if vname != "enable_thinking_false" else self.R.no_reasoning(case["messages"])
                for agp in (True, False):
                    a = self.official.render(msgs, tools=tools, add_generation_prompt=agp, **hkw)
                    b = self.ours.render(msgs, tools=tools, add_generation_prompt=agp, **hkw)
                    self.assertEqual(a, b, (cname, vname, agp))
                    n += 1
        self.assertEqual(n, 294)  # 21 conversations x 7 switches x 2

    def test_turn_without_reasoning_renders_an_empty_think_block(self):
        out = self.ours.render(C.N23_CASES["missing_after_two"]["messages"])
        self.assertIn("<|user|>u3<|assistant|><think></think>a3<|user|>u4", out)
        out = self.ours.render(C.N23_CASES["inline_then_missing"]["messages"])
        self.assertIn("<|user|>u2<|assistant|><think></think>a2", out)

    def test_agent_loops_prefix_stable(self):
        r = lambda m, add_generation_prompt=False: self.ours.render(  # noqa: E731
            m, tools=C.TOOLS, add_generation_prompt=add_generation_prompt)
        for conv in (C.AGENT_LOOP_MIXED, C.AGENT_SESSION_EMPTY_REASONING):
            self.assertEqual([x for x in A.loop_checks(r, conv) if x.get("extends") is False
                              or x.get("gen_prompt_is_prefix") is False], [])
            self.assertEqual([h["k"] for h in A.history_checks(r, conv) if h["stable"] is False], [])


@unittest.skipUnless(HAVE_JINJA and HAVE_TRANSFORMERS, "transformers is not installed")
class TransformersRender(unittest.TestCase):
    def test_transformers_renders_like_hf_render(self):
        import hf_render
        import run_n14
        n = 0
        for tmpl in (hf_render.Template(GLM_DIR), hf_render.Template(GLM_DIR, OURS)):
            for cname, case in ALL_CASES.items():
                tools = C.TOOLS if case.get("tools") else None
                for vname, (hkw, _skw, _extra) in run_n14.VARIANTS["glm"].items():
                    msgs = case["messages"] if vname != "enable_thinking_false" else run_n14.no_reasoning(case["messages"])
                    for agp in (True, False):
                        kw = dict(tmpl.special, **hkw)
                        out, _ = render_jinja_template(conversations=[msgs], tools=tools, chat_template=tmpl.source,
                                                       add_generation_prompt=agp, **kw)
                        self.assertEqual(out[0], tmpl.render(msgs, tools=tools, add_generation_prompt=agp, **hkw),
                                         (tmpl.template_path.name, cname, vname, agp))
                        n += 1
        self.assertEqual(n, 588)


NULL_500 = "type must be string, but is null"  # common/chat.cpp:422-423 @79e2e74 reads reasoning_content as a string


def expected_server_difference(row):
    """The only rows where llama-server with our template differs from jinja2 with the official one."""
    if row["case"] == "tools_ends_with_assistant":   # --prefill-assistant: a trailing assistant message is continued
        return row["add_generation_prompt"] and "first_diff" in row
    if row["case"] == "reasoning_null":               # the server refuses a JSON null reasoning_content
        return NULL_500 in (row.get("server_error") or "") and row.get("jinja2_error") is None
    return False


@unittest.skipUnless(HAVE_SERVER, SKIP_SERVER)
class ServerRenderN23(unittest.TestCase):
    def test_server_with_ours_equals_official_format(self):
        import run_n14
        res = run_n14.section_render(args_ns(), "glm_n23", workdir("render"), cases=ALL_CASES,
                                     variant_cases=tuple(ALL_CASES))
        rows = res["rows"]
        self.assertEqual(len(rows), 298)  # 294 + 4 rows of the two server flag runs
        unexpected = [r for r in rows if not r["equal"] and not expected_server_difference(r)]
        self.assertEqual(unexpected, [])
        self.assertEqual(sum(1 for r in rows if not r["equal"]), 19)  # 7 prefill rows + 6 switches x 2 null rows
        for r in rows:
            if "same_as_server_default" in r and r["case"] != "reasoning_null":
                self.assertTrue(r["same_as_server_default"], r)   # GLM's template does not read enable_thinking
        for run, ws in res["server_warnings"].items():
            for w in ws:
                self.assertTrue("supports preserving reasoning" in w or NULL_500 in w, (run, w))

    def test_official_template_still_leaks_at_this_pin(self):
        import run_n14
        cases = {k: ALL_CASES[k] for k in LEAKS}
        res = run_n14.section_render(args_ns(), "glm", workdir("render-official"), cases=cases, variant_cases=())
        for r in res["rows"]:
            if r["variant"] != "default":
                continue
            self.assertFalse(r["equal"], r)
            # snippets: 30 characters before the first differing one, the marker, 50 after
            before, server_after = r["server"].split("‖", 1)
            self.assertTrue(before.endswith("<|assistant|><think>"), r)
            self.assertTrue(server_after.startswith(LEAKS[r["case"]]), r)
            self.assertTrue(r["jinja2"].split("‖", 1)[1].startswith("</think>"), r)


@unittest.skipUnless(HAVE_SERVER and HARNESS, SKIP_SERVER + " and CHATTMPL_HARNESS")
class ParseHarnessN23(unittest.TestCase):
    def test_parse_like_the_official_template(self):
        import run_n14
        res = run_n14.section_parse(args_ns(), "glm_n23", workdir("parse"))
        self.assertTrue(any("using differential autoparser" in h for h in res["handler_log"]), res["handler_log"])
        self.assertEqual(len(res["requests"]), 7)
        for r in res["requests"]:
            self.assertEqual(r["format"], "peg-native")
            self.assertTrue(r["harness_prompt_equals_jinja2"], r["k"])
            self.assertTrue(r["tool_grammar"], r["k"])
            self.assertEqual([[d["role"], d["delimiter"]] for d in r["delimiters"]],
                             [["assistant", "<|assistant|>"], ["user", "<|user|>"]])
            ideal = [c for c in r["candidates"] if c["label"] == "ideal"][0]
            self.assertTrue(ideal["round_trip_equal"], (r["k"], ideal))
            self.assertEqual(ideal["fields_equal"]["tool_calls"], r["k"] != 9, r["k"])  # N14: null -> "null"
            for c in r["candidates"]:
                if c["label"] == "reasoning_padded":
                    self.assertTrue(c["round_trip_reproduces_output"], r["k"])
                if c["label"] == "content_padded":
                    self.assertFalse(c["round_trip_reproduces_output"], r["k"])
                if c["label"] == "unknown_tool":
                    self.assertIn("does not match the expected", c["parse_error"])
        self.assertEqual(res["nothink"]["generation_prompt"], "<|assistant|><think>")


def n23_prefix_sessions():
    import run_n14
    return dict(run_n14.PREFIX_SESSIONS, **{
        "agent_loop_mixed": ("default", lambda: C.AGENT_LOOP_MIXED),
        "agent_loop_mixed_preserve_off": ("preserve_off", lambda: C.AGENT_LOOP_MIXED)})


@unittest.skipUnless(HAVE_SERVER, SKIP_SERVER)
class PrefixN23(unittest.TestCase):
    def test_server_prompts_extend_held_turns(self):
        import run_n14
        res = run_n14.section_prefix(args_ns(), "glm_n23", workdir("prefix"), sessions=n23_prefix_sessions())
        for sname in n23_prefix_sessions():
            loop = res["%s/server_prompts+jinja2_turns" % sname]["loop"]
            breaks = [x["k"] for x in loop if x.get("extends") is False or x.get("gen_prompt_is_prefix") is False]
            # past reasoning dropped: the request after the second user message re-renders the earlier turns
            want = {"preserve_off": [11], "agent_loop_mixed_preserve_off": [11]}.get(sname, [])
            self.assertEqual(breaks, want, sname)
            # the server's prompts with ours are the jinja2 prompts of the official template
            self.assertTrue(all(x.get("prompt_equals_reference") for x in loop if "error" not in x), sname)


@unittest.skipUnless(HAVE_SERVER, SKIP_SERVER)
class LiveB3N23(unittest.TestCase):
    def test_prompt_cache_reuse(self):
        import run_n14
        configs = run_n14.LIVE_CONFIGS + [("agent_loop_mixed", "default", lambda: C.AGENT_LOOP_MIXED, [])]
        res = run_n14.section_live(args_ns(), "glm_n23", workdir("live"), configs=configs)
        for tag, v in res.items():
            rows = v["requests"]
            self.assertTrue(all(r["status"] == 200 for r in rows), tag)
            for r in rows[1:]:
                if r["extends"]:
                    self.assertEqual(r["cache_n"], r["prev_held_bytes"], (tag, r))
                else:
                    self.assertLessEqual(r["cache_n"], r["shared_bytes"], (tag, r))
                    self.assertGreater(r["cache_n"], 0, (tag, r))
            breaks = [r["request"] for r in rows[1:] if not r["extends"]]
            want = {"preserve_off": [5], "preserve_off-ctx-checkpoints-4": [5]}.get(tag, [])
            self.assertEqual(breaks, want, tag)   # N14 with the official template: empty_reasoning_turn broke at [4]


if __name__ == "__main__":
    unittest.main()
