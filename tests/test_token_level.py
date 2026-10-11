"""N23: agent-loop prefix stability at token level with the real tokenizers (tools/chattmpl/token_prefix.py).

Tokenizers (each class skips cleanly without its inputs; .github/workflows/token-level.yml provides all of them and
fails on any skip):
  hf     tokenizer.json of zai-org/GLM-5.3-Flash @ eb9eb208 and Qwen/Qwen3.8-Flash-Next @ de4b8e4d, pinned by sha256 in
         third_party/hf/*-tokenizer/PIN.json and found in the cache of scripts/hf_lfs_fetch.py ($HF_LFS_CACHE), through
         the `tokenizers` library
  llama  llama-tokenize of the R1 pin (TOKCHECK_LLAMA_TOKENIZE) on vocab-only GGUFs built from N15's committed Unsloth
         GGUF headers (TOKCHECK_VOCAB_GLM, TOKCHECK_VOCAB_QWEN; tools/gguf_headers/vocab_gguf.py of N15)
Expectations (the llama side measured in the N23 sandbox, docs/evidence/2026-10-11-n23-glm-template-tokenizer.md; the
hf side is the same hypothesis until the runner answers):
  GLM, official and corrected template (identical bytes under jinja2): token-level breaks only where bytes break (past
  reasoning dropped), every template tag one token.
  Qwen: besides the byte breaks, every assistant turn with empty or missing reasoning breaks at token level: the prompt
  ends "<think>\\n" (tokens <think> Ċ), the turn starts "\\n</think>" (Ċ </think>), the next prompt tokenizes the same
  bytes as <think> ĊĊ </think> (N15 finding); reuse stops at the prompt's last token.
  Parity: hf and llama give the same ids for every rendered prompt and turn.
"""
import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "chattmpl"))
sys.path.insert(0, str(ROOT / "scripts"))
import hf_lfs_fetch  # noqa: E402

PINS = {"glm": ROOT / "third_party/hf/glm-5.3-flash-tokenizer/PIN.json",
        "qwen": ROOT / "third_party/hf/qwen3.8-flash-next-tokenizer/PIN.json"}
CHAT_PINS = {"glm": ROOT / "third_party/hf/glm-5.3-flash-chat/PIN.json",
             "qwen": ROOT / "third_party/hf/qwen3.8-flash-next-chat/PIN.json"}
LLAMA_TOKENIZE = os.environ.get("TOKCHECK_LLAMA_TOKENIZE")
VOCAB = {"glm": os.environ.get("TOKCHECK_VOCAB_GLM"), "qwen": os.environ.get("TOKCHECK_VOCAB_QWEN")}

try:
    import jinja2  # noqa: F401
    HAVE_JINJA = True
except ImportError:
    HAVE_JINJA = False
try:
    import tokenizers  # noqa: F401
    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False


def hf_path(model):
    return hf_lfs_fetch.path_for(str(PINS[model]), "tokenizer.json")


HAVE_HF = HAVE_JINJA and HAVE_TOKENIZERS and all(hf_path(m) for m in PINS)
HAVE_LLAMA = HAVE_JINJA and bool(LLAMA_TOKENIZE and all(VOCAB.values()))
SKIP_HF = "needs jinja2, tokenizers and both tokenizer.json in the hf_lfs_fetch cache (PENDING runner)"
SKIP_LLAMA = "set TOKCHECK_LLAMA_TOKENIZE, TOKCHECK_VOCAB_GLM, TOKCHECK_VOCAB_QWEN"

# request indices whose next prompt does not extend prompt + turn (bytes or tokens), per template and session
BYTE_BREAKS = {"preserve_off": [4], "agent_loop_mixed_preserve_off": [4]}
QWEN_NEWLINE_BREAKS = {"empty_reasoning_turn": [3], "agent_loop_mixed": [1, 2, 5, 6],
                       "agent_loop_mixed_preserve_off": [1, 2, 5, 6]}
GLM_TAGS_SINGLE = 16
QWEN_TAGS_SINGLE = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                    "<tool_response>", "</tool_response>"]


class PinRecords(unittest.TestCase):
    @unittest.skipUnless(all(p.exists() for p in PINS.values()), "tokenizer PIN.json not vendored yet (PENDING runner)")
    def test_lfs_records_match_the_template_pins(self):
        for model, pin_path in PINS.items():
            pin = json.loads(pin_path.read_text())
            chat = json.loads(CHAT_PINS[model].read_text())
            self.assertEqual((pin["repo"], pin["revision"]), (chat["repo"], chat["revision"]))
            (e,) = pin["lfs_files"]
            self.assertEqual(e["path"], "tokenizer.json")
            self.assertEqual(e["sha256"], e["lfs_oid"])
            listing = {x["path"]: x for x in chat["upstream_listing"]}
            self.assertEqual(e["size"], listing["tokenizer.json"]["size"])
            self.assertIn("card_license", pin)
            self.assertIn("LICENSE", [f["path"] for f in pin["files"]])
            self.assertFalse((pin_path.parent / "tokenizer.json").exists())   # pinned, never committed


class Expectations:
    """Shared assertions for one tokenizer kind (self.tok(model) -> callable text -> ids)."""

    def rows(self, name, model):
        import token_prefix as T
        key = (name, model)
        if key not in self._rows:
            self._rows[key] = T.check(name, self.tok(model))
        return self._rows[key]

    def by_session(self, rows):
        out = {}
        for r in rows:
            out.setdefault(r["session"], []).append(r)
        return out

    def test_glm_tags_single_tokens(self):
        import token_prefix as T
        tok = self.tok("glm")
        self.assertEqual([t for t in T.TAGS["glm"] if len(tok(t)) == 1], T.TAGS["glm"])
        self.assertEqual(len(T.TAGS["glm"]), GLM_TAGS_SINGLE)

    def test_qwen_tags(self):
        import token_prefix as T
        tok = self.tok("qwen")
        self.assertEqual([t for t in T.TAGS["qwen"] if len(tok(t)) == 1], QWEN_TAGS_SINGLE)

    def test_glm_tokens_break_only_where_bytes_break(self):
        for name in ("glm_hf", "glm_n23"):
            for sname, rows in self.by_session(self.rows(name, "glm")).items():
                self.assertEqual([r["request"] for r in rows if not r["token_extends"]], BYTE_BREAKS.get(sname, []),
                                 (name, sname))
                self.assertEqual([r["request"] for r in rows if not r["byte_extends"]], BYTE_BREAKS.get(sname, []))
                self.assertTrue(all(r["concat_canonical"] for r in rows), (name, sname))

    def test_glm_official_and_corrected_identical(self):
        strip = lambda rows: [{k: v for k, v in r.items()} for r in rows]  # noqa: E731
        self.assertEqual(strip(self.rows("glm_hf", "glm")), strip(self.rows("glm_n23", "glm")))

    def test_qwen_newline_merge_after_empty_reasoning(self):
        tok = self.tok("qwen")
        (nl,), (nlnl,), (think,), (end_think,) = tok("\n"), tok("\n\n"), tok("<think>"), tok("</think>")
        for sname, rows in self.by_session(self.rows("qwen_hf", "qwen")).items():
            want = sorted(set(BYTE_BREAKS.get(sname, [])) | set(QWEN_NEWLINE_BREAKS.get(sname, [])))
            self.assertEqual([r["request"] for r in rows if not r["token_extends"]], want, sname)
            self.assertEqual([r["request"] for r in rows if not r["byte_extends"]], BYTE_BREAKS.get(sname, []))
            for r in rows:
                if r["request"] in QWEN_NEWLINE_BREAKS.get(sname, []):
                    self.assertTrue(r["byte_extends"], (sname, r))
                    self.assertFalse(r["concat_canonical"], (sname, r))
                    # reuse stops at the last token of the previous prompt
                    self.assertEqual(r["token_reuse"], r["prompt_tokens"] - 1, (sname, r))
                    held, nxt = r["at"]["held"], r["at"]["next"]
                    self.assertEqual(held[2:6], self.pieces("qwen", [think, nl, nl, end_think]), (sname, r))
                    self.assertEqual(nxt[2:5], self.pieces("qwen", [think, nlnl, end_think]), (sname, r))


@unittest.skipUnless(HAVE_HF, SKIP_HF)
class HFTokenLevel(Expectations, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import token_prefix as T
        cls._toks = {m: T.HFTok(hf_path(m)) for m in PINS}
        cls._rows = {}

    def tok(self, model):
        return self._toks[model]

    def pieces(self, model, ids):
        return [self._toks[model].piece(i) for i in ids]


@unittest.skipUnless(HAVE_LLAMA, SKIP_LLAMA)
class LlamaTokenLevel(Expectations, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        import token_prefix as T
        cls._dir = tempfile.TemporaryDirectory()
        cls._toks = {m: T.LlamaTok(LLAMA_TOKENIZE, VOCAB[m], os.path.join(cls._dir.name, m)) for m in VOCAB}
        cls._rows = {}

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def tok(self, model):
        return self._toks[model]

    def pieces(self, model, ids):
        return list(ids)


@unittest.skipUnless(HAVE_HF and HAVE_LLAMA, SKIP_HF + "; " + SKIP_LLAMA)
class Parity(unittest.TestCase):
    def test_same_ids(self):
        import tempfile
        import token_prefix as T
        with tempfile.TemporaryDirectory() as d:
            for name, model in (("glm_n23", "glm"), ("qwen_hf", "qwen")):
                res = T.parity(name, T.HFTok(hf_path(model)), T.LlamaTok(LLAMA_TOKENIZE, VOCAB[model],
                                                                         os.path.join(d, model)))
                self.assertEqual(res["diffs"], [], name)
                self.assertGreater(res["texts"], 30, name)


if __name__ == "__main__":
    unittest.main()
