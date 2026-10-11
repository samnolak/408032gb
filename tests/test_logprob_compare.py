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

VOCAB = ["byte0", "byte1", "ctl2", "ctl3"] + [f"tok{i}" for i in range(4, 50)]   # names only; logits depend on the index


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


def llamacpp_entry(tok, lp, invalid_bytes=True):
    """One top_logprobs entry as llama-server @79e2e74 writes it (tools/server/server-task.cpp:256-293): id, token (the
    text after UTF-8 validation), bytes (the raw piece), logprob. With invalid_bytes, byte0/byte1 stand for single bytes
    >= 0x80, not valid UTF-8 alone: the server prints both as the replacement character, only "bytes" (and "id") differ;
    ctl2/ctl3 stand for control tokens (EOS and the like), which the server prints as an empty piece: only "id" differs."""
    i = VOCAB.index(tok)
    if invalid_bytes and tok.startswith("byte"):
        return {"id": i, "token": "\ufffd", "bytes": [0x80 + i], "logprob": lp}
    if invalid_bytes and tok.startswith("ctl"):
        return {"id": i, "token": "", "bytes": [], "logprob": lp}
    return {"id": i, "token": tok, "bytes": list(tok.encode()), "logprob": lp}


def make_handler(temp=1.0, flip=False, no_logprobs=False, dialect="openai", swap_bytes=False, invalid_bytes=True,
                 drop_logprobs_of=None, swap_ctl=False):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            d = dist(body["prompt"], temp, flip)
            if swap_bytes:                     # an engine that confuses two invalid-UTF-8 byte tokens
                d["byte0"], d["byte1"] = d["byte1"], d["byte0"]
            if swap_ctl:                       # an engine that confuses two control tokens (e.g. two end-of-turn ids)
                d["ctl2"], d["ctl3"] = d["ctl3"], d["ctl2"]
            k = body.get("logprobs") or 5
            top = dict(sorted(d.items(), key=lambda x: -x[1])[:k])
            best = max(top, key=top.get)
            choice = {"text": best, "index": 0, "finish_reason": "length"}
            out = {"choices": [choice], "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}}
            if no_logprobs:
                pass
            elif dialect == "llamacpp":         # choices[0].logprobs = {"content": [{..., "top_logprobs": [...]}]}
                e = llamacpp_entry(best, top[best], invalid_bytes)
                e["top_logprobs"] = [llamacpp_entry(t, v, invalid_bytes) for t, v in top.items()]
                choice["text"] = e["token"]
                choice["logprobs"] = {"content": [e]}
                if drop_logprobs_of and best in drop_logprobs_of:
                    # what llama-server sends when the token ends in an incomplete UTF-8 sequence (N11, observed)
                    choice["logprobs"], choice["text"] = None, "\ufffd"
            else:
                choice["logprobs"] = {"tokens": [best], "token_logprobs": [top[best]], "top_logprobs": [top]}
            out = json.dumps(out).encode()
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

    def test_llamacpp_logprobs_dialect(self):
        # N11 (docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md): llama-server @79e2e74 nests the top-K list under
        # choices[0].logprobs.content[0].top_logprobs; the tool read only the OpenAI legacy layout and every prefix failed
        rc, r = self.run_pair(make_handler(dialect="llamacpp"), make_handler(dialect="llamacpp"))
        self.assertEqual((rc, r["errors"]), (0, []))
        self.assertEqual(r["summary"]["n"], 25)
        self.assertEqual(r["summary"]["top1_agree"], 1.0)
        self.assertAlmostEqual(r["summary"]["kl_topk_max"], 0.0, places=9)

    def test_dialects_compare_with_each_other(self):
        # every token valid UTF-8: the text key of the OpenAI layout equals the bytes key of the llama.cpp layout
        rc, r = self.run_pair(make_handler(), make_handler(dialect="llamacpp", invalid_bytes=False))
        self.assertEqual((rc, r["errors"]), (0, []))
        self.assertEqual(r["summary"]["top1_agree"], 1.0)
        self.assertAlmostEqual(r["summary"]["kl_topk_max"], 0.0, places=9)

    def test_tokens_with_the_same_text_but_different_bytes_stay_apart(self):
        # two byte tokens that llama-server prints as the same "�": keyed by that text they merge and hide the swap
        rc, r = self.run_pair(make_handler(dialect="llamacpp", temp=0.2),
                              make_handler(dialect="llamacpp", temp=0.2, swap_bytes=True))
        self.assertEqual((rc, r["errors"]), (0, []))
        self.assertTrue([x for x in r["rows"] if x["kl_topk"] > 1e-3], "the byte0/byte1 swap must be visible")

    def test_control_tokens_with_an_empty_piece_stay_apart(self):
        # N11: on the real server <unk>, <s>, </s> all come back as token "" / bytes []; keyed by bytes they merged
        rc, r = self.run_pair(make_handler(dialect="llamacpp", temp=0.2),
                              make_handler(dialect="llamacpp", temp=0.2, swap_ctl=True))
        self.assertEqual((rc, r["errors"]), (0, []))
        self.assertTrue([x for x in r["rows"] if x["kl_topk"] > 1e-3], "the ctl2/ctl3 swap must be visible")
        self.assertEqual(sum(x["ref_merged"] + x["cand_merged"] for x in r["rows"]), 0)

    def test_answer_without_logprobs_is_an_error_that_says_why(self):
        # llama-server @79e2e74 sends "logprobs": null when the generated token ends in an incomplete UTF-8 sequence
        # (tools/server/server-context.cpp:1977-2009); such a prefix is an error, never a number
        greedy = {max(dist(p).items(), key=lambda x: x[1])[0] for p in self.prefixes()}
        drop = sorted(greedy)[:1]
        rc, r = self.run_pair(make_handler(dialect="llamacpp"), make_handler(dialect="llamacpp", drop_logprobs_of=drop))
        self.assertTrue(r["errors"])
        self.assertEqual(len(r["errors"]) + len(r["rows"]), 25)
        self.assertTrue(all("incomplete UTF-8" in e["error"] for e in r["errors"]), r["errors"][0])

    def prefixes(self):
        with open(self.corpus) as f:
            return lc.make_prefixes([f.read()], 25, 1)

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
