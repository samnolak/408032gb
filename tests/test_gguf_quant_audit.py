"""tools/gguf_quant_audit.py on GGUF headers written here (no gguf-py needed), plus a cross-check of its type table
against the pinned gguf-py when LLAMA_CPP_GGUF_PY is set."""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gguf_quant_audit as qa  # noqa: E402

TYPE_ID = {v[0]: k for k, v in qa.GGML_TYPES.items()}


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, arch="glm5-next", extra_kv=()):
    """tensors: list of (name, dims, type_name). Writes header only (no data): the auditor never reads data."""
    kv = [(b"general.architecture", 8, _s(arch))]
    kv += list(extra_kv)
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, vtype, payload in kv:
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", vtype) + payload
    off = 0
    for name, dims, tname in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", TYPE_ID[tname], off)
        off += 4096
    with open(path, "wb") as f:
        f.write(out)


GOOD = [("token_embd.weight", [4096, 1000], "Q8_0"), ("output.weight", [4096, 1000], "Q6_K"),
        ("blk.0.attn_q_a.weight", [4096, 1536], "Q8_0"), ("blk.0.attn_norm.weight", [4096], "F32"),
        ("blk.3.ffn_gate_exps.weight", [4096, 2048, 288], "Q4_K"), ("blk.3.ffn_up_exps.weight", [4096, 2048, 288], "Q4_K"),
        ("blk.3.ffn_down_exps.weight", [2048, 4096, 288], "Q5_K"), ("blk.3.ffn_up_shexp.weight", [4096, 2048], "Q8_0"),
        ("blk.0.ffn_up.weight", [4096, 12288], "Q6_K")]


class Audit(unittest.TestCase):
    def test_pass_and_classes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.gguf")
            write_gguf(p, GOOD, extra_kv=[(b"glm5-next.block_count", 4, struct.pack("<I", 46)),
                                          (b"tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _s("a") + _s("b"))])
            r = qa.audit(p)
            self.assertEqual(r["verdict"], "PASS")
            self.assertEqual(r["architecture"], "glm5-next")
            e = r["classes"]["routed_experts"]
            self.assertEqual(e["tensors"], 3)
            self.assertAlmostEqual(e["bpw"], (2 * 4.5 + 5.5) / 3, places=6)
            self.assertEqual(set(r["classes"]), {"routed_experts", "shared_experts", "attention", "dense_ffn",
                                                  "embedding", "output_head", "other"})

    def test_iq3_expert_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.gguf")
            bad = [t if t[0] != "blk.3.ffn_up_exps.weight" else (t[0], t[1], "IQ3_S") for t in GOOD]
            write_gguf(p, bad)
            r = qa.audit(p)
            self.assertEqual(r["verdict"], "FAIL")
            self.assertEqual([f["tensor"] for f in r["findings"] if f["level"] == "FAIL"], ["blk.3.ffn_up_exps.weight"])
            self.assertEqual(qa.main([p]), 1)

    def test_low_bit_dense_fails_unless_warn_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.gguf")
            low = [t if t[0] != "blk.0.attn_q_a.weight" else (t[0], t[1], "Q3_K") for t in GOOD]
            write_gguf(p, low)
            r = qa.audit(p)
            self.assertEqual(r["verdict"], "FAIL")
            self.assertEqual([f["level"] for f in r["findings"]], ["FAIL"])
            r = qa.audit(p, dense_warn_only=True)
            self.assertEqual(r["verdict"], "PASS")
            self.assertEqual([f["level"] for f in r["findings"]], ["WARN"])

    def test_split_files(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "M-00001-of-00002.gguf")
            b = os.path.join(d, "M-00002-of-00002.gguf")
            write_gguf(a, GOOD[:4])
            write_gguf(b, GOOD[4:])
            r = qa.audit(a)
            self.assertEqual(len(r["files"]), 2)
            self.assertEqual(r["classes"]["routed_experts"]["tensors"], 3)
            os.remove(b)
            with self.assertRaises(FileNotFoundError):
                qa.audit(a)

    def test_rejects_non_gguf_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.gguf")
            with open(p, "wb") as f:
                f.write(b"NOPE")
            with self.assertRaises(ValueError):
                qa.read_header(p)
            write_gguf(p, GOOD)
            with open(p, "r+b") as f:
                f.truncate(60)
            with self.assertRaises(ValueError):
                qa.read_header(p)

    @unittest.skipUnless(os.environ.get("LLAMA_CPP_GGUF_PY"), "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py")
    def test_type_table_matches_pinned_gguf_py(self):
        sys.path.insert(0, os.environ["LLAMA_CPP_GGUF_PY"])
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        ours = {k: (v[1], v[2]) for k, v in qa.GGML_TYPES.items()}
        theirs = {int(t): GGML_QUANT_SIZES[t] for t in GGMLQuantizationType}
        self.assertEqual(ours, theirs)
        names = {int(t): t.name for t in GGMLQuantizationType}
        self.assertEqual({k: v[0] for k, v in qa.GGML_TYPES.items()}, names)


if __name__ == "__main__":
    unittest.main()
