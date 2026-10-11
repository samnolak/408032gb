"""tools/tinygen --text-vocab (N11): a byte-level BPE vocabulary so that a llama-server on a tiny model accepts text.

The pure-Python part always runs; writing GGUFs needs gguf-py (LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py, either pin).
The default outputs must stay byte-identical: their sha256 are the ones N7 recorded
(docs/evidence/2026-10-10-n7-server-hybrid-cpu.md, "Models"); gguf-py of ec7630a and of 79e2e74 write the same bytes.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINYGEN = os.path.join(ROOT, "tools", "tinygen")
sys.path.insert(0, TINYGEN)
import text_vocab  # noqa: E402

GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
N7_SHA256 = {("qwen4exp_tiny.py",): "0fe7442121108a67b7bb01d7bc432d308f43fff41107d56e7a08c99e1ec51951",
             ("glm5next_tiny.py",): "26c83267fe1a7588d40153c7b9baaea55ac9d437bb2df85e9ad34ad72acc2ae5",
             ("glm5next_tiny.py", "--layers", "4"): "6309ba52478a425e0bf6a87ac5ea1f283a584bba52b502ea56c62b44eba49a67"}


class Vocab(unittest.TestCase):
    def test_bytes_map_one_to_one(self):
        b2u = text_vocab.bytes_to_unicode()
        self.assertEqual(sorted(b2u), list(range(256)))
        self.assertEqual(len(set(b2u.values())), 256)
        self.assertTrue(all(len(c) == 1 for c in b2u.values()))

    def test_layout(self):
        toks, types, merges = text_vocab.byte_bpe_vocab()
        self.assertEqual(toks[:3], ["<unk>", "<s>", "</s>"])          # the ids of the default vocabulary
        self.assertEqual(types[:3], [2, 3, 3])
        b2u = text_vocab.bytes_to_unicode()
        self.assertEqual(toks[3:259], [b2u[b] for b in range(256)])
        self.assertEqual(len(toks), text_vocab.N_TEXT_VOCAB)
        self.assertEqual(len(set(toks)), len(toks))
        for m in merges:                                              # every merge result is a token
            a, b = m.split(" ")
            self.assertIn(a + b, toks)


@unittest.skipUnless(GGUF_PY, "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py")
class Generators(unittest.TestCase):
    def gen(self, argv, path):
        env = dict(os.environ, PYTHONPATH=GGUF_PY)
        subprocess.run([sys.executable, os.path.join(TINYGEN, argv[0]), path] + list(argv[1:]), check=True, env=env,
                       stdout=subprocess.DEVNULL)

    def test_defaults_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            for argv, want in N7_SHA256.items():
                p = os.path.join(d, "m.gguf")
                self.gen(argv, p)
                with open(p, "rb") as f:
                    self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want, argv)

    def test_text_vocab_and_ctx_are_written(self):
        sys.path.insert(0, GGUF_PY)
        from gguf import GGUFReader
        with tempfile.TemporaryDirectory() as d:
            for script, arch in (("qwen4exp_tiny.py", "qwen4exp"), ("glm5next_tiny.py", "glm5-next")):
                p = os.path.join(d, arch + ".gguf")
                self.gen((script, "--text-vocab", "--ctx", "4096"), p)
                r = GGUFReader(p)
                f = r.fields
                self.assertEqual(bytes(f["tokenizer.ggml.model"].parts[-1]).decode(), "gpt2")
                toks = [bytes(f["tokenizer.ggml.tokens"].parts[i]).decode() for i in f["tokenizer.ggml.tokens"].data]
                self.assertEqual(toks, text_vocab.byte_bpe_vocab()[0])
                self.assertEqual(int(f[arch + ".context_length"].parts[-1][0]), 4096)
                shapes = {t.name: t.shape.tolist() for t in r.tensors}
                self.assertEqual(shapes["token_embd.weight"][1], text_vocab.N_TEXT_VOCAB, arch)
                self.assertEqual(shapes["output.weight"][1], text_vocab.N_TEXT_VOCAB, arch)


if __name__ == "__main__":
    unittest.main()
