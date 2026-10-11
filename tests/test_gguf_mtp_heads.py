"""tools/gguf_mtp_heads.py (N21 workaround: a mimo2 GGUF with fewer NextN heads, docs/evidence/2026-10-11-n21-mimo-mtp-
heads-fix.md section 5) on the tiny models of tools/tinygen/mimo2_tiny.py: keeping one head of N17's 3-head model gives
N16's 1-head model byte for byte (the model N16's llama.cpp parity step runs: one MTP head, no chained heads); keeping
two heads gives the 2-head layout. Needs the pinned gguf-py (LLAMA_CPP_GGUF_PY, set by
tools/tinygen/run_parity_mimo2_n21.sh)."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
TINY = ["--fused-qkv", "--sinks", "swa", "--value-scale", "0.75"]


def py(*args):
    env = dict(os.environ, PYTHONPATH=GGUF_PY)
    p = subprocess.run([sys.executable, *map(str, args)], env=env, capture_output=True, text=True)
    return p


@unittest.skipUnless(GGUF_PY, "set LLAMA_CPP_GGUF_PY (the pinned llama.cpp's gguf-py)")
class GgufMtpHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = pathlib.Path(cls.tmp.name)
        for name, extra in (("n17.gguf", ["--mtp-layers", "3"]), ("n16.gguf", ["--mtp"])):
            p = py(ROOT / "tools/tinygen/mimo2_tiny.py", cls.d / name, *TINY, *extra)
            assert p.returncode == 0, p.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_keep_one_head_is_n16(self):
        out = self.d / "keep1.gguf"
        p = py(ROOT / "tools/gguf_mtp_heads.py", "--keep", "1", self.d / "n17.gguf", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("'heads': (3, 1)", p.stdout)
        self.assertEqual(out.read_bytes(), (self.d / "n16.gguf").read_bytes())

    def test_keep_two_heads(self):
        out = self.d / "keep2.gguf"
        p = py(ROOT / "tools/gguf_mtp_heads.py", "--keep", "2", self.d / "n17.gguf", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = py("-c", f"""
import gguf
r = gguf.GGUFReader({str(out)!r}); s = gguf.GGUFReader({str(self.d / 'n17.gguf')!r})
f = lambda k: r.fields[k].contents()
blk = sorted({{int(t.name.split('.')[1]) for t in r.tensors if t.name.startswith('blk.')}})
same = all((a.name, a.tensor_type, a.data.tobytes()) == (b.name, b.tensor_type, b.data.tobytes())
           for a, b in zip(r.tensors, [t for t in s.tensors if not t.name.startswith('blk.8.')]))
print(f('mimo2.block_count'), f('mimo2.nextn_predict_layers'), len(f('mimo2.attention.sliding_window_pattern')),
      len(f('mimo2.attention.head_count_kv')), blk[-1], len(r.tensors), same)
""")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.split(), ["8", "2", "8", "8", "7", "83", "True"])

    def test_refuses_more_heads_than_the_file_has(self):
        p = py(ROOT / "tools/gguf_mtp_heads.py", "--keep", "4", self.d / "n17.gguf", self.d / "bad.gguf")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("the file has 3 NextN heads", p.stderr + p.stdout)


if __name__ == "__main__":
    unittest.main()
