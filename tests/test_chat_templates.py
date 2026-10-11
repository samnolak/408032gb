"""N14 (gap 6 of docs/reassessment/CORRECTNESS.md): chat templates, tool calls and prefix stability (B3) of the vendored
GLM-5.3-Flash and Qwen3.8-Flash-Next chat templates (third_party/hf/*-chat, pinned revisions in PIN.json).

Without anything built (jinja2 only, which is in the CI image; skipped cleanly without it):
  VendoredFiles        the files match SHA256SUMS; the model facts of config.json quoted in the evidence
  TemplateProperties   the templates rendered the way transformers renders them (tools/chattmpl/hf_render.py):
                       prefix stability of the agent session (tools/chattmpl/analysis.py) with the template defaults
                       and with past reasoning dropped, the template switches, no date injection
With a llama-server build of the R1 pin (env CHATTMPL_SERVER_BIN, CHATTMPL_GLM_GGUF, CHATTMPL_QWEN_GGUF: tiny models
from tools/chattmpl/bytes_vocab_model.py; CHATTMPL_HARNESS for the parser; CHATTMPL_WORK for logs; driver
tools/chattmpl/run_n14.sh), each class skipped cleanly when its variables are missing:
  ServerRender         POST /apply-template against jinja2: byte-equal except the known differences listed here
  ParseHarness         llama.cpp's chat format detection and parser on the template's own tool-call format
  LiveB3               /v1/chat/completions on a hybrid tiny model: prompt-cache reuse per request (timings.cache_n)
The expectations are those of llama.cpp 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (docs/evidence/
2026-10-11-n14-chat-templates.md); a different pin may legitimately differ.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
import analysis as A  # noqa: E402
import convs as C  # noqa: E402

GLM_DIR = ROOT / "third_party/hf/glm-5.3-flash-chat"
QWEN_DIR = ROOT / "third_party/hf/qwen3.8-flash-next-chat"

try:
    import jinja2  # noqa: F401
    HAVE_JINJA = True
except ImportError:
    HAVE_JINJA = False

SERVER = os.environ.get("CHATTMPL_SERVER_BIN")
GLM_GGUF = os.environ.get("CHATTMPL_GLM_GGUF")
QWEN_GGUF = os.environ.get("CHATTMPL_QWEN_GGUF")
HARNESS = os.environ.get("CHATTMPL_HARNESS")
HAVE_SERVER = bool(SERVER and GLM_GGUF and QWEN_GGUF and HAVE_JINJA)
SKIP_SERVER = "set CHATTMPL_SERVER_BIN, CHATTMPL_GLM_GGUF, CHATTMPL_QWEN_GGUF (tools/chattmpl/run_n14.sh)"


def args_ns():
    return argparse.Namespace(server_bin=SERVER, glm_gguf=GLM_GGUF, qwen_gguf=QWEN_GGUF, harness=HARNESS)


def workdir(name):
    base = os.environ.get("CHATTMPL_WORK")
    if base:
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        return d
    return tempfile.mkdtemp(prefix="chattmpl-%s-" % name)


class VendoredFiles(unittest.TestCase):
    def test_sha256sums(self):
        for d in (GLM_DIR, QWEN_DIR):
            for line in (d / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split()
                self.assertEqual(hashlib.sha256((d / name).read_bytes()).hexdigest(), digest, d / name)

    def test_glm_config_facts(self):
        c = json.loads((GLM_DIR / "config.json").read_text())
        t = c["text_config"]
        self.assertEqual(c["architectures"], ["Glm5NextForConditionalGeneration"])
        self.assertIn("vision_config", c)
        self.assertEqual((t["n_group"], t["topk_group"]), (1, 1))
        self.assertEqual(t["swiglu_limit"], 10.0)
        self.assertEqual(t["index_topk"], 2048)
        self.assertEqual(t["first_k_dense_replace"], 3)
        self.assertEqual(t["num_hidden_layers"], 45)
        self.assertEqual(t["vocab_size"], 154880)
        self.assertEqual(t["max_position_embeddings"], 1048576)
        q = c["quantization_config"]
        self.assertEqual((q["quant_method"], q["fmt"], q["activation_scheme"]), ("fp8", "e4m3", "dynamic"))

    def test_qwen_config_facts(self):
        c = json.loads((QWEN_DIR / "config.json").read_text())
        self.assertEqual(c["architectures"], ["Qwen4ExpForConditionalGeneration"])
        self.assertIn("vision_config", c)
        self.assertIs(c["language_model_only"], False)


@unittest.skipUnless(HAVE_JINJA, "jinja2 is not installed")
class TemplateProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hf_render
        cls.glm = hf_render.Template(GLM_DIR)
        cls.qwen = hf_render.Template(QWEN_DIR)

    def renderer(self, t, **kw):
        return lambda m, add_generation_prompt=False: t.render(m, tools=C.TOOLS, add_generation_prompt=add_generation_prompt, **kw)

    def loop_breaks(self, t, conv, **kw):
        return [r for r in A.loop_checks(self.renderer(t, **kw), conv)
                if r.get("extends") is False or r.get("gen_prompt_is_prefix") is False]

    def test_defaults_prefix_stable_in_agent_loop(self):
        for t in (self.glm, self.qwen):
            self.assertEqual(self.loop_breaks(t, C.AGENT_SESSION), [])

    def test_history_breaks_default(self):
        # GLM: none. Qwen: only between two parallel tool results (<|im_end|> closes a tool block only after its
        # last result, template lines 154-157); the server never sees that state in an agent loop.
        glm = [h["k"] for h in A.history_checks(self.renderer(self.glm), C.AGENT_SESSION) if h["stable"] is False]
        qwen = [h["k"] for h in A.history_checks(self.renderer(self.qwen), C.AGENT_SESSION) if h["stable"] is False]
        self.assertEqual(glm, [])
        self.assertEqual(qwen, [6])
        self.assertEqual(C.AGENT_SESSION[6]["role"], "tool")

    def test_dropping_past_reasoning_breaks_at_new_user_message(self):
        for t, kw in ((self.glm, {"clear_thinking": True}), (self.qwen, {"preserve_thinking": False})):
            breaks = self.loop_breaks(t, C.AGENT_SESSION, **kw)
            self.assertEqual(len(breaks), 1, breaks)
            b = breaks[0]
            # the request after the second user message (index 12) re-renders the first assistant turn (index 2)
            self.assertEqual(b["k"], 11)
            self.assertEqual(b["in_message"], 2)
            self.assertEqual(C.AGENT_SESSION[12]["role"], "user")
            self.assertIn("I should look at the layout", b["old"])

    def test_glm_ignores_enable_thinking(self):
        conv = C.CASES["system_user"]["messages"]
        a = self.glm.render(conv, add_generation_prompt=True)
        b = self.glm.render(conv, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(a, b)
        self.assertTrue(a.endswith("<|assistant|><think>"))
        self.assertNotIn("enable_thinking", self.glm.source)

    def test_qwen_enable_thinking_false(self):
        conv = C.CASES["system_user"]["messages"]
        a = self.qwen.render(conv, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(a.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertNotIn("Reasoning effort", a)
        self.assertEqual(self.loop_breaks(self.qwen, [{k: v for k, v in m.items() if k != "reasoning_content"}
                                                      for m in C.AGENT_SESSION], enable_thinking=False), [])

    def test_reasoning_effort_values(self):
        conv = C.CASES["system_user"]["messages"]
        self.assertIn("Reasoning Effort: Max", self.glm.render(conv))
        self.assertIn("Reasoning Effort: Low", self.glm.render(conv, reasoning_effort="low"))
        self.assertIn("Reasoning Effort: Max", self.glm.render(conv, reasoning_effort="medium"))  # silently Max
        import jinja2
        with self.assertRaises(jinja2.exceptions.TemplateError):
            self.qwen.render(conv, reasoning_effort="high")

    def test_no_date_injection(self):
        for t in (self.glm, self.qwen):
            for word in ("strftime_now", "datetime", "date_string"):
                self.assertNotIn(word, t.source)

    def test_glm_loop_scope_jinja2(self):
        # jinja2 scopes {% set %} to one loop iteration: a turn without reasoning_content renders an empty think
        # block (llama.cpp's engine at 79e2e74 does not, see ServerRender)
        out = self.glm.render(C.CASES["reasoning_mixed"]["messages"])
        self.assertIn("<|user|>u2<|assistant|><think></think>a2", out)


# differences between llama-server's /apply-template and jinja2 at 79e2e74 (anything else fails)
LEAK = "<|assistant|><think>R1</think>a2"  # GLM: reasoning of the previous assistant turn leaks into the next one


def expected_render_difference(name, row):
    if row["case"] == "tools_ends_with_assistant" and row["variant"] == "default":
        # --prefill-assistant (default on): a trailing assistant message is continued, not closed
        return (name == "qwen") or row["add_generation_prompt"]
    if name == "glm" and row["case"] in ("reasoning_mixed", "reasoning_empty") and row["variant"] in (
            "default", "effort_low", "effort_high", "effort_medium", "effort_none"):
        return LEAK in row["server"].replace("\u2016", "")  # the snippet marks the first differing byte
    if name == "qwen" and row["variant"] == "effort_high":
        return "HTTP 500" in (row.get("server_error") or "") and "Unexpected reasoning effort high" in (
            row.get("jinja2_error") or "")
    return False


@unittest.skipUnless(HAVE_SERVER, SKIP_SERVER)
class ServerRender(unittest.TestCase):
    def test_render_server_vs_jinja2(self):
        import run_n14
        for name in ("glm", "qwen"):
            res = run_n14.section_render(args_ns(), name, workdir("render"))
            unexpected = [r for r in res["rows"] if not r["equal"] and not expected_render_difference(name, r)]
            self.assertEqual(unexpected, [], name)
            # every expected difference occurred
            n_diff = sum(1 for r in res["rows"] if not r["equal"])
            self.assertEqual(n_diff, {"glm": 21, "qwen": 10}[name], name)
            # thinking switched off (body reasoning_effort "none", server --reasoning off): GLM's template does not
            # read enable_thinking, so its prompt stays the default one; Qwen's changes
            off = [r for r in res["rows"] if "same_as_server_default" in r]
            self.assertEqual(len(off), 10, name)
            for r in off:
                self.assertEqual(r["same_as_server_default"], name == "glm", (name, r))
                if r["case"] in ("reasoning", "agent_session"):
                    self.assertTrue(r["equal"], (name, r))


@unittest.skipUnless(HAVE_SERVER and HARNESS, SKIP_SERVER + " and CHATTMPL_HARNESS")
class ParseHarness(unittest.TestCase):
    def test_parse_and_round_trip(self):
        import run_n14
        for name, handler, delims in (
                ("glm", "using differential autoparser", [["assistant", "<|assistant|>"], ["user", "<|user|>"]]),
                ("qwen", "Using specialized template: Qwen3-Coder",
                 [["assistant", "<|im_start|>assistant"], ["tool", "<|im_start|>user\n<tool_response>"],
                  ["tool", "<|im_start|>tool_response"], ["user", "<|im_start|>user"],
                  ["system", "<|im_start|>system"]])):
            res = run_n14.section_parse(args_ns(), name, workdir("parse"))
            self.assertTrue(any(handler in h for h in res["handler_log"]), res["handler_log"])
            for r in res["requests"]:
                self.assertEqual(r["format"], "peg-native")
                self.assertTrue(r["harness_prompt_equals_jinja2"], r["k"])
                self.assertTrue(r["tool_grammar"], r["k"])
                self.assertEqual([[d["role"], d["delimiter"]] for d in r["delimiters"]], delims)
                ideal = [c for c in r["candidates"] if c["label"] == "ideal"][0]
                self.assertTrue(ideal["round_trip_equal"], (name, r["k"], ideal))
                calls_ok = ideal["fields_equal"]["tool_calls"]
                if name == "glm" and r["k"] == 9:
                    # a null argument comes back as the string "null" (schema type ["string", "null"])
                    self.assertFalse(calls_ok)
                    args = json.loads(ideal["parsed"]["tool_calls"][0]["function"]["arguments"])
                    self.assertEqual(args["only"], "null")
                else:
                    self.assertTrue(calls_ok, (name, r["k"], ideal["parsed"]))
                for c in r["candidates"]:
                    if c["label"] == "reasoning_padded":
                        # GLM keeps whitespace inside <think>, Qwen trims it on re-render
                        self.assertEqual(c["round_trip_reproduces_output"], name == "glm", (name, r["k"]))
                    if c["label"] == "content_padded":
                        self.assertFalse(c["round_trip_reproduces_output"], (name, r["k"]))
                    if c["label"] == "unknown_tool":
                        if name == "glm":
                            self.assertIn("does not match the expected", c["parse_error"])
                        else:  # the call is dropped: no content, no tool_calls
                            self.assertEqual(c["parsed"].get("tool_calls"), None)
                            self.assertEqual(c["parsed"]["content"], "")
            self.assertEqual(res["nothink"]["generation_prompt"],
                             {"glm": "<|assistant|><think>",
                              "qwen": "<|im_start|>assistant\n<think>\n\n</think>\n\n"}[name])


@unittest.skipUnless(HAVE_SERVER, SKIP_SERVER)
class LiveB3(unittest.TestCase):
    def test_prompt_cache_reuse(self):
        import run_n14
        for name in ("glm", "qwen"):
            res = run_n14.section_live(args_ns(), name, workdir("live"))
            for tag, v in res.items():
                rows = v["requests"]
                self.assertTrue(all(r["status"] == 200 for r in rows), (name, tag))
                for r in rows[1:]:
                    if r["extends"]:
                        # everything the slot holds is reused
                        self.assertEqual(r["cache_n"], r["prev_held_bytes"], (name, tag, r))
                    else:
                        self.assertLessEqual(r["cache_n"], r["shared_bytes"], (name, tag, r))
                        self.assertGreater(r["cache_n"], 0, (name, tag, r))  # a checkpoint, not a full re-prefill
                breaks = [r["request"] for r in rows[1:] if not r["extends"]]
                want = {"default": [], "preserve_off": [5], "preserve_off-ctx-checkpoints-4": [5],
                        "empty_reasoning_turn": [4] if name == "glm" else []}[tag]
                self.assertEqual(breaks, want, (name, tag))


if __name__ == "__main__":
    unittest.main()
