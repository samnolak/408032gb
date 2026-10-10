"""tools/g0/predict_eval.py and scripts/traces/*: synthetic probe files (no GGUF, no llama.cpp needed)."""
import http.server
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "g0"))
import cachesim  # noqa: E402
import predict_eval as pe  # noqa: E402


def write_probe(path, layers, k, x, logits, topk, first, n_prompt):
    n_rec, n_layers, n_embd = x.shape
    n_expert = logits.shape[-1]
    with open(path, "wb") as f:
        f.write(b"G0PB")
        f.write(struct.pack("<9I", 1, n_layers, k, n_embd, n_expert, n_rec, first, n_prompt, first + n_rec))
        f.write(struct.pack(f"<{n_layers}I", *layers))
        for t in range(n_rec):
            for i in range(n_layers):
                f.write(topk[t, i].astype("<u2").tobytes())
                f.write(x[t, i].astype("<f2").tobytes())
                f.write(logits[t, i].astype("<f4").tobytes())


def synth(tmp, name, shared_input, gating=1, bias=False, groups=(0, 0), n_rec=40, layers=(1, 2, 3, 4), k=2,
          n_embd=16, n_expert=8, seed=0, first=10, n_prompt=30, W=None):
    rng = np.random.default_rng(seed)
    W = W or {l: rng.standard_normal((n_expert, n_embd)).astype(np.float32) for l in layers}
    b = {l: (0.3 * rng.standard_normal(n_expert)).astype(np.float32) for l in layers} if bias else {}
    x0 = rng.standard_normal((n_rec, n_embd)).astype(np.float16).astype(np.float32)
    xs = np.stack([x0 if shared_input else
                   rng.standard_normal((n_rec, n_embd)).astype(np.float16).astype(np.float32) for _ in layers], 1)
    logits = np.stack([xs[:, i] @ W[l].T for i, l in enumerate(layers)], 1)
    topk = np.stack([pe.topm(pe.selection_scores(logits[:, i], gating, b.get(l), groups), k)
                     for i, l in enumerate(layers)], 1)
    write_probe(os.path.join(tmp, name + ".probe"), list(layers), k, xs, logits, topk, first, n_prompt)
    return {"arch": "synth", "W": W, "b": b, "gating": gating, "groups": groups}, topk


class Selection(unittest.TestCase):
    def test_softmax_and_logits_rank_alike(self):
        z = np.random.default_rng(1).standard_normal((5, 8))
        np.testing.assert_array_equal(pe.topm(pe.selection_scores(z, 1), 3), pe.topm(z, 3))

    def test_sigmoid_bias_changes_selection(self):
        z = np.zeros((1, 4))
        b = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertEqual(int(pe.topm(pe.selection_scores(z, 2, b), 1)[0, 0]), 2)

    def test_groups_mask_like_brute_force(self):
        rng = np.random.default_rng(2)
        s = rng.standard_normal((50, 16))
        got = pe.topm(pe.selection_scores(s, 3, None, (4, 2)), 4)
        for row, sel in zip(s, got):
            g = row.reshape(4, 4)
            best = np.argsort(-np.sort(g, axis=1)[:, -2:].sum(axis=1), kind="stable")[:2]
            allowed = {e for gi in best for e in range(gi * 4, gi * 4 + 4)}
            self.assertTrue(set(sel.tolist()) <= allowed)
            masked = np.where(np.isin(np.arange(16), list(allowed)), row, -np.inf)
            self.assertEqual(set(sel.tolist()), set(np.argsort(-masked)[:4].tolist()))


class Evaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def load(self):
        return [pe.read_probe(os.path.join(self.tmp, f)) for f in sorted(os.listdir(self.tmp)) if f.endswith(".probe")]

    def test_shared_router_input_gives_perfect_pregate(self):
        router, _ = synth(self.tmp, "a", shared_input=True, gating=2, bias=True)
        res = pe.evaluate(self.load(), router, ds=(1, 2), ms=[2, 4])
        self.assertLess(res["checks"]["router_logits_max_rel_err"], 1e-6)
        self.assertEqual(res["checks"]["selection_rule_match"], 1.0)
        for r in res["results"]:
            self.assertAlmostEqual(r["pregate"], 1.0)

    def test_independent_inputs_give_about_random_recall(self):
        router, _ = synth(self.tmp, "a", shared_input=False, n_rec=400, n_embd=32)
        res = pe.evaluate(self.load(), router, ds=(1,), ms=[2])
        for r in res["results"]:
            self.assertLess(abs(r["pregate"] - r["random_baseline"]), 0.08)

    def test_wrong_router_weights_fail_the_self_check(self):
        router, _ = synth(self.tmp, "a", shared_input=True)
        router["W"] = {l: w + 0.5 for l, w in router["W"].items()}
        with self.assertRaises(SystemExit):
            pe.evaluate(self.load(), router)

    def test_wrong_selection_rule_fails_the_self_check(self):
        router, _ = synth(self.tmp, "a", shared_input=True, gating=2, bias=True)
        router["b"] = {}                                       # forget the selection bias
        with self.assertRaises(SystemExit):
            pe.evaluate(self.load(), router)

    def test_prompt_and_gen_parts_are_split_at_n_prompt(self):
        router, _ = synth(self.tmp, "a", shared_input=True, first=10, n_prompt=30, n_rec=40)
        res = pe.evaluate(self.load(), router, ds=(1,), ms=[2])
        pairs = {r["part"]: r["token_layer_pairs"] for r in res["results"]}
        self.assertEqual(pairs, {"prompt": 20 * 3, "gen": 20 * 3})   # tokens 10..29 and 30..49, 3 layer pairs

    def test_probe_rows_checked_against_trace(self):
        router, topk = synth(self.tmp, "a", shared_input=True, first=10, n_prompt=30, n_rec=40)
        layers, n_tok = [1, 2, 3, 4], 50
        rows = np.zeros((n_tok, 4, 2), dtype=np.int64)
        rows[10:] = topk
        cachesim.write_trace(os.path.join(self.tmp, "a.trace"), layers, 2,
                             [[[int(e) for e in rows[t, i]] for i in range(4)] for t in range(n_tok)], 30)
        probes = self.load()
        self.assertEqual(pe.check_against_traces(probes, self.tmp), 1.0)
        rows[45, 2] = [7, 6] if list(rows[45, 2]) != [7, 6] else [5, 4]
        cachesim.write_trace(os.path.join(self.tmp, "a.trace"), layers, 2,
                             [[[int(e) for e in rows[t, i]] for i in range(4)] for t in range(n_tok)], 30)
        self.assertLess(pe.check_against_traces(probes, self.tmp), 1.0)

    def test_truncated_probe_is_rejected(self):
        synth(self.tmp, "a", shared_input=True)
        p = os.path.join(self.tmp, "a.probe")
        with open(p, "rb") as f:
            data = f.read()
        with open(p, "wb") as f:
            f.write(data[:-3])
        with self.assertRaises(ValueError):
            pe.read_probe(p)


class Corpus(unittest.TestCase):
    def test_builder_is_deterministic_and_records_sources(self):
        outs = []
        for _ in range(2):
            d = tempfile.mkdtemp()
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "traces", "build_agent_prompts.py"),
                            "--root", ROOT, "--out", d, "--n", "4"], check=True, capture_output=True)
            files = {}
            for f in sorted(os.listdir(d)):
                with open(os.path.join(d, f), "rb") as fh:
                    files[f] = fh.read()
            outs.append(files)
        self.assertEqual(outs[0], outs[1])
        idx = json.loads(outs[0]["INDEX.json"])
        self.assertEqual(len(idx["conversations"]), 4)
        self.assertTrue(all(len(c["source_sha256"]) == 64 for c in idx["conversations"]))

    def test_renderer_uses_template_and_tokenizer_endpoints(self):
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, body))
                if self.path == "/apply-template":
                    out = {"prompt": "|".join(m["role"] + ":" + m["content"][:5] for m in body["messages"])}
                else:
                    out = {"tokens": [ord(c) % 50 for c in body["content"]]}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        d, ids = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "traces", "build_agent_prompts.py"),
                            "--root", ROOT, "--out", d, "--n", "2"], check=True, capture_output=True)
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "traces", "render_ids.py"), "--url",
                            f"http://127.0.0.1:{srv.server_port}", "--prompts", d, "--out", ids],
                           check=True, capture_output=True)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual([p for p, _ in seen], ["/apply-template", "/tokenize"] * 2)
        self.assertTrue(all(b["add_special"] and b["parse_special"] for p, b in seen if p == "/tokenize"))
        files = sorted(f for f in os.listdir(ids) if f.endswith(".ids"))
        self.assertEqual(len(files), 2)
        with open(os.path.join(ids, "RENDER.json")) as fh:
            render = json.load(fh)
        with open(os.path.join(ids, files[0])) as fh:
            self.assertEqual(render[0]["n_tokens"], len(fh.read().split(",")))


if __name__ == "__main__":
    unittest.main()
