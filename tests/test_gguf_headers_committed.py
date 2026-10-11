"""Facts of the real GGUF headers committed by the gguf-headers workflow (bot commit 6cce416, run 38136140582), offline.

third_party/gguf-headers/: GLM-5.3-Flash UD-Q4_K_XL (unsloth/GLM-5.3-Flash-GGUF @a38483c, 6 shards), Qwen3.8-Flash-Next
UD-Q4_K_XL (unsloth/Qwen3.8-Flash-Next-GGUF @38bb39e, 4 shards), the discovery listing. Checked here, deterministically
and without network: integrity of the committed bytes (header sha256 of PIN.json, summaries re-derived byte for byte),
the architecture / tokenizer / hyperparameter facts docs/evidence/2026-10-11-n15-gguf-headers.md reports, the embedded
chat templates, the quant audit against ADR-012 D5, and the discovery listing. With LLAMA_CPP_GGUF_PY (+ numpy) also
the graph branches only the real geometry exercises; with the llama.cpp checkouts (LLAMA_CPP_GGUF_PY's at ec7630a,
LLAMA_CPP_SRC_79E2E74) also architecture / pre-tokenizer support at both pins.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = os.path.join(ROOT, "third_party", "gguf-headers")
GLM, QWEN = "glm-5.3-flash-ud-q4_k_xl", "qwen3.8-flash-next-ud-q4_k_xl"
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("gguf_headers_analyze", os.path.join(ROOT, "tools", "gguf_headers", "analyze.py"))


def _src_pins():
    want = {"ec7630a": ("ec7630a640789c393694fb194f1bbbf0369fc62d",
                        os.path.dirname(os.path.abspath(GGUF_PY)) if GGUF_PY else None),
            "79e2e74": ("79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6", os.environ.get("LLAMA_CPP_SRC_79E2E74"))}
    return {n: p for n, (sha, p) in want.items() if p and A.git_head(p) == sha}


class Committed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.req = {n: A.load_request(os.path.join(HDR, n)) for n in (GLM, QWEN)}
        cls.m = {n: A.Model([f["summary"] for f in r["files"]]) for n, r in cls.req.items()}

    def test_integrity_and_pins(self):
        for n, (repo, rev, nfiles, total) in {
                GLM: ("unsloth/GLM-5.3-Flash-GGUF", "a38483c8cd5df544f53d70fb281afe97369d5ab6", 6, 199707321472),
                QWEN: ("unsloth/Qwen3.8-Flash-Next-GGUF", "38bb39ee97821de2c9009abb7e93950eec396e66", 4, 111334654784)
        }.items():
            pin = self.req[n]["pin"]
            self.assertEqual((pin["status"], pin["repo"], pin["revision"]), ("ok", repo, rev))
            self.assertEqual([ok for _, ok in self.req[n]["integrity"]], [True] * nfiles)
            files = self.req[n]["files"]
            self.assertEqual(sum(f["rec"]["size"] for f in files), total)
            for f in files:
                self.assertTrue(A.G.size_check(f["summary"], f["rec"]["size"]).startswith("exact"), f["rec"]["path"])
                self.assertTrue(f["rec"]["etag_matches_lfs_sha256"])
                self.assertTrue(f["rec"]["repo_commit_matches"])
            # the first shard is metadata only: its whole file was read, every other shard only its header
            self.assertEqual(files[0]["summary"]["n_tensors"], 0)
            self.assertEqual(files[0]["rec"]["bytes_fetched"], files[0]["rec"]["size"])
            for f in files[1:]:
                self.assertEqual(f["rec"]["bytes_fetched"], 262144)
        self.assertEqual(self.req[GLM]["files"][0]["rec"]["header_sha256"],
                         "0c90bc17e2486d3ed599dfda7e2b244bdded27c48d0680569077d741c11b1016")

    def test_glm_facts(self):
        m = self.m[GLM]
        self.assertEqual(m.arch, "glm5-next")              # not "glm5next" (AGENTS.md says so of the host files)
        self.assertEqual((m.v("tokenizer.ggml.model"), m.v("tokenizer.ggml.pre")), ("gpt2", "glm4"))
        self.assertEqual((m.kv["tokenizer.ggml.tokens"]["count"], m.kv["tokenizer.ggml.merges"]["count"]),
                         (154880, 321649))
        self.assertEqual((m.a("expert_group_count"), m.a("expert_group_used_count")), (1, 1))
        self.assertEqual((m.a("block_count"), m.a("nextn_predict_layers"), m.a("leading_dense_block_count")), (46, 1, 3))
        self.assertEqual((m.a("expert_count"), m.a("expert_used_count"), m.a("expert_weights_scale")), (288, 8, 2.5))
        self.assertEqual((m.a("rope.dimension_count"), m.a("kda.gate_lower_bound"), m.a("context_length")),
                         (0, -5.0, 1048576))
        self.assertEqual(m.a("swiglu_clamp_exp"), [10.0] * 46)
        self.assertEqual(m.a("swiglu_clamp_shexp"), [10.0] * 46)
        self.assertAlmostEqual(m.a("attention.layer_norm_epsilon"), 1e-6, places=12)
        self.assertFalse(m.has("attention.indexer.kpool_select_tail"))   # llama.cpp default: true
        self.assertFalse(m.has("attention.indexer.types"))               # every DSA layer has a full indexer
        self.assertEqual(m.a("attention.indexer.index_share_mtp"), True)
        kv = m.a("attention.head_count_kv")
        self.assertEqual([i for i, x in enumerate(kv) if x], list(range(3, 44, 4)) + [45])
        self.assertEqual(len(m.tensors), 1412)
        self.assertEqual(sorted(n for n in m.names if ".nextn." in n),
                         ["blk.45.nextn.eh_proj.weight", "blk.45.nextn.enorm.weight", "blk.45.nextn.hnorm.weight",
                          "blk.45.nextn.shared_head_norm.weight"])

    def test_qwen_facts(self):
        m = self.m[QWEN]
        self.assertEqual(m.arch, "qwen4exp")
        self.assertEqual((m.v("tokenizer.ggml.model"), m.v("tokenizer.ggml.pre")), ("gpt2", "qwen35"))
        self.assertEqual((m.kv["tokenizer.ggml.tokens"]["count"], m.kv["tokenizer.ggml.merges"]["count"]),
                         (248320, 247587))
        self.assertEqual(m.a("block_count"), 48)
        self.assertFalse(m.has("nextn_predict_layers"))                   # no MTP block in this file
        self.assertEqual([n for n in m.names if "nextn" in n], [])
        self.assertFalse(m.has("expert_group_count"))
        self.assertEqual((m.a("ple.layers"), m.a("ple.image_token_id")), ([1], 248056))
        self.assertEqual((m.a("rope.dimension_sections"), m.a("rope.dimension_count"), m.a("rope.freq_base")),
                         ([11, 11, 10, 0], 64, 10000000.0))
        self.assertFalse(m.has("attention.recurrent_layers"))             # the full_attention_interval rule applies
        self.assertEqual(m.a("full_attention_interval"), 4)
        self.assertEqual(sorted(set(m.a("attention.compress_ratios"))), [0, 4])
        self.assertEqual(m.a("attention.compress_ratios").count(4), 12)
        self.assertEqual(len(m.tensors), 1224)

    def test_embedded_chat_templates(self):
        for n, rel, sha, size in (
                (GLM, "glm-5.3-flash-chat", "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724", 10648),
                (QWEN, "qwen3.8-flash-next-chat", "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce",
                 9993)):
            emb = self.m[n].v("tokenizer.chat_template").encode("utf-8", "surrogateescape")
            with open(os.path.join(ROOT, "third_party", "hf", rel, "chat_template.jinja"), "rb") as f:
                ven = f.read()
            self.assertEqual((hashlib.sha256(emb).hexdigest(), len(emb)), (sha, size))
            self.assertNotEqual(emb, ven)
        glm = self.m[GLM].v("tokenizer.chat_template")
        with open(os.path.join(ROOT, "third_party", "hf", "glm-5.3-flash-chat", "chat_template.jinja")) as f:
            ven = f.read()
        # the {% set %} inside the message loop that llama.cpp's engine carries into the next iteration (N14 1(a)):
        # the same lines in both templates
        block = ven.splitlines()[142:150]
        self.assertIn("{%- set reasoning_content = m.reasoning_content %}", block[1])
        self.assertIn("\n".join(block), glm)
        self.assertRegex(glm, r"\{%- for m in messages -%\}")
        qwen = self.m[QWEN].v("tokenizer.chat_template")
        self.assertIn("{%- if resolved_reasoning_effort == 'high' %}", qwen)
        self.assertNotIn("No user query found in messages.", qwen)

    def test_quant_audit_adr012_d5(self):
        with tempfile.TemporaryDirectory() as d:
            glm = A.section_audit(self.req[GLM]["files"], os.path.join(d, "glm"))
            qwen = A.section_audit(self.req[QWEN]["files"], os.path.join(d, "qwen"))
        for a, gib, bpw, exp_bpw in ((glm, 185.983, 4.981, 4.874), (qwen, 103.678, 5.033, 5.101)):
            self.assertEqual((a["verdict"], a["findings"], a["unknown_types"]), ("PASS", [], []))
            self.assertAlmostEqual(a["total_gib"], gib, places=3)
            self.assertAlmostEqual(a["total_bpw"], bpw, places=3)
            self.assertAlmostEqual(a["classes"]["routed_experts"]["bpw"], exp_bpw, places=3)
            self.assertEqual(a["lowest_bpw_matrices"][0]["bpw"], 4.5)
        self.assertEqual(sorted(glm["classes"]["routed_experts"]["types"]), ["Q4_K", "Q5_K", "Q6_K"])
        self.assertEqual(sorted(qwen["classes"]["routed_experts"]["types"]), ["Q4_K", "Q5_1", "Q5_K", "Q8_0"])
        self.assertEqual(sorted(qwen["classes"]["other"]["types"]), ["F32", "IQ4_NL", "Q8_0"])  # the PLE table

    def test_discovery_listing(self):
        with open(os.path.join(HDR, "discover-glm-mimo", "LISTING.json")) as f:
            lst = json.load(f)
        res = {r["id"]: r for s in lst["searches"] for r in s["results"]}
        self.assertEqual((res["unsloth/GLM-5.3-Flash-GGUF"]["gguf_architecture"], res["unsloth/GLM-5.3-Flash-GGUF"]["sha"]),
                         ("glm5-next", "a38483c8cd5df544f53d70fb281afe97369d5ab6"))
        g = res["ggml-org/GLM-5.3-Flash-GGUF"]
        with open(os.path.join(ROOT, "third_party", "hf", "glm-5.3-flash-chat", "chat_template.jinja"), "rb") as f:
            self.assertEqual(g["gguf_chat_template_sha256"], hashlib.sha256(f.read()).hexdigest())
        self.assertEqual(g["gguf_architecture"], "glm5-next")
        self.assertIn("GLM-5.3-Flash-Q4_K-00002-of-00002.gguf", [x["path"] for x in g["gguf_files"]])
        self.assertEqual(res["ggml-org/MiMo-V2.6-Flash-MOPD-GGUF"]["gguf_architecture"], "mimo2")
        self.assertEqual([s["n_results"] for s in lst["searches"]], [95, 30])

    @unittest.skipUnless(_src_pins(), "llama.cpp sources at the pins: LLAMA_CPP_GGUF_PY's checkout at ec7630a and/or "
                                      "LLAMA_CPP_SRC_79E2E74")
    def test_architecture_and_pre_tokenizer_known_at_the_pins(self):
        for pin, root in _src_pins().items():
            for kind, value in (("arch", "glm5-next"), ("arch", "qwen4exp"), ("pre", "glm4"), ("pre", "qwen35"),
                                ("model", "gpt2")):
                self.assertRegex(A.known(root, kind, value), r"^src/llama-(arch|vocab)\.cpp:\d+", (pin, value))
            self.assertEqual(A.known(root, "arch", "glm5next"), "not found")

    @unittest.skipUnless(GGUF_PY and HAVE_NUMPY, "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py (and install numpy)")
    def test_branches_only_the_real_geometry_exercises(self):
        with tempfile.TemporaryDirectory() as d:
            out = {}
            for n, tiny, graph in ((GLM, "glm5next", "glm5-next"), (QWEN, "qwen4exp", "qwen4exp")):
                os.makedirs(os.path.join(d, n))
                tm, why = A.build_tiny(tiny, GGUF_PY, os.path.join(d, n))
                self.assertIsNone(why)
                br = A.section_branches(self.m[n], graph, tm, {})
                out[n] = sorted(b["id"] for b in br if not b["exercised_by_tiny"])
        self.assertEqual(out[GLM], ["glm.layer_norm_eps", "glm.mtp_heads", "glm.shexp_clamp"])
        self.assertEqual(out[QWEN], ["qwen.ple_image_token", "qwen.recurrent_source"])


if __name__ == "__main__":
    unittest.main()
