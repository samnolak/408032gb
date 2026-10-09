"""tools/quality/logprob_compare.py against two stub OpenAI /v1/completions servers (no GPU, stdlib)."""
import hashlib
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "quality"))
import logprob_compare as lc  # noqa: E402

VOCAB = [f"tok{i}" for i in range(50)]


def dist(prompt, temp=1.0, flip=False):
    h = hashlib.sha256(prompt.encode()).digest()
    logits = [((h[i % 32] * (i + 7)) % 97) / 10.0 + i * 1e-4 for i in range(len(VOCAB))]   # no ties
    if flip:                                   # swap the two best tokens: a wrong-top-1 "engine"
        order = sorted(range(len(VOCAB)), key=lambda i: -logits[i])
        a, b = order[0], order[1]
        logits[a], logits[b] = logits[b], logits[a]
    z = [l / temp for l in logits]
    m = max(z)
    lse = m + math.log(sum(math.exp(x - m) for x in z))
    return {VOCAB[i]: z[i] - lse for i in range(len(VOCAB))}


def make_handler(temp=1.0, flip=False, no_logprobs=False):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            d = dist(body["prompt"], temp, flip)
            k = body.get("logprobs") or 5
            top = dict(sorted(d.items(), key=lambda x: -x[1])[:k])
            best = max(top, key=top.get)
            choice = {"text": best, "index": 0, "finish_reason": "length"}
            if not no_logprobs:
                choice["logprobs"] = {"tokens": [best], "token_logprobs": [top[best]], "top_logprobs": [top]}
            out = json.dumps({"choices": [choice]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    return H


class Servers:
    def __init__(self, *handlers):
        self.srv = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), h) for h in handlers]
        for s in self.srv:
            threading.Thread(target=s.serve_forever, daemon=True).start()
        self.urls = [f"http://127.0.0.1:{s.server_address[1]}" for s in self.srv]

    def close(self):
        for s in self.srv:
            s.shutdown()


class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "c.txt")
        with open(self.corpus, "w") as f:
            f.write(" ".join(f"word{i % 97}" for i in range(5000)))

    def tearDown(self):
        self.tmp.cleanup()

    def run_pair(self, h1, h2):
        s = Servers(h1, h2)
        try:
            out = os.path.join(self.tmp.name, "r.json")
            rc = lc.main(["--ref", s.urls[0], "--cand", s.urls[1], "--corpus", self.corpus, "--prefixes", "25",
                          "--top-k", "10", "--out", out])
            with open(out) as f:
                return rc, json.load(f)
        finally:
            s.close()

    def test_identical_engines(self):
        rc, r = self.run_pair(make_handler(), make_handler())
        self.assertEqual(rc, 0)
        self.assertEqual(r["summary"]["top1_agree"], 1.0)
        self.assertAlmostEqual(r["summary"]["kl_topk_max"], 0.0, places=9)

    def test_softer_candidate_has_kl_but_same_top1(self):
        rc, r = self.run_pair(make_handler(), make_handler(temp=1.5))
        self.assertEqual(r["summary"]["top1_agree"], 1.0)
        self.assertGreater(r["summary"]["kl_topk_mean"], 0.01)

    def test_wrong_top1_is_caught(self):
        rc, r = self.run_pair(make_handler(), make_handler(flip=True))
        self.assertEqual(r["summary"]["top1_agree"], 0.0)
        self.assertEqual(r["summary"]["ref_in_cand_topk"], 1.0)

    def test_engine_without_logprobs_is_an_error_not_a_number(self):
        rc, r = self.run_pair(make_handler(), make_handler(no_logprobs=True))
        self.assertEqual(rc, 1)
        self.assertIsNone(r["summary"])
        self.assertEqual(len(r["errors"]), 25)

    def test_compare_unit(self):
        ref = {"a": math.log(0.7), "b": math.log(0.2), "c": math.log(0.1)}
        self.assertAlmostEqual(lc.compare(ref, dict(ref))["kl_topk"], 0.0, places=12)
        m = lc.compare(ref, {"b": math.log(0.6), "a": math.log(0.3), "d": math.log(0.1)})
        self.assertFalse(m["top1_agree"])
        self.assertTrue(m["ref_in_cand"])
        self.assertGreater(m["kl_topk"], 0.0)

    def test_compare_exact_kl_direction_and_floor(self):
        # Pins KL(ref || cand) and the missing-token floor min(side) - log 2 from the module docstring
        # (mutations L1, L2 of docs/evidence/2026-10-09-tool-mutations.md survived the tests above).
        # ref {a: .6, b: .4}, cand {a: .5, c: .5}; floors: ref c -> .4/2 = .2, cand b -> .5/2 = .25;
        # renormalised on {a, b, c}: P = (.6, .4, .2)/1.2, Q = (.5, .25, .5)/1.25.
        ref = {"a": math.log(0.6), "b": math.log(0.4)}
        cand = {"a": math.log(0.5), "c": math.log(0.5)}
        P = [0.6 / 1.2, 0.4 / 1.2, 0.2 / 1.2]
        Q = [0.5 / 1.25, 0.25 / 1.25, 0.5 / 1.25]
        want = sum(p * math.log(p / q) for p, q in zip(P, Q))
        rev = sum(q * math.log(q / p) for p, q in zip(P, Q))
        self.assertGreater(abs(want - rev), 1e-3)          # this case tells KL(ref||cand) from KL(cand||ref)
        self.assertAlmostEqual(lc.compare(ref, cand)["kl_topk"], want, places=12)


if __name__ == "__main__":
    unittest.main()
