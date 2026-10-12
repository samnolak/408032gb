"""tools/kv/slot_persist.py against a stub llama-server (/completion, /slots save|restore|erase)."""
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "kv"))
import slot_persist as sp  # noqa: E402


class Stub:
    def __init__(self, broken_restore=False, nondeterministic=False, noop_restore=False, fail_restore_once=False,
                 same_text_other_tokens=False):
        self.cache = None          # prompt held by slot 0
        self.files = {}
        self.calls = 0
        self.restores = 0
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                out = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                u = urlparse(self.path)
                if u.path == "/completion":
                    n = len(body["prompt"].split())
                    reused = n if stub.cache == body["prompt"] else 0
                    stub.cache = body["prompt"]
                    stub.calls += 1
                    text = "answer-%d" % (stub.calls if nondeterministic else n)
                    # like llama-server with return_tokens; same_text_other_tokens: the warm answer (2nd call of a
                    # repeat) has other token ids that decode to the same text (N11: ids 200 and 201 both read "\ufffd")
                    toks = [ord(c) for c in text]
                    if same_text_other_tokens and stub.calls % 2 == 0:
                        toks[-1] += 1000
                    return self.reply(200, {"content": text, "tokens_cached": n,
                                            **({"tokens": toks} if body.get("return_tokens") else {}),
                                            "timings": {"prompt_n": n - reused, "prompt_ms": (n - reused) * 2.0,
                                                        "predicted_n": body["n_predict"], "predicted_ms": 10.0}})
                action = parse_qs(u.query).get("action", [""])[0]
                if action == "erase":
                    stub.cache = None
                    return self.reply(200, {"id_slot": 0, "n_erased": 1})
                if action == "save":
                    stub.files[body["filename"]] = stub.cache
                    return self.reply(200, {"id_slot": 0, "filename": body["filename"], "n_saved": 1, "n_written": 1234,
                                            "timings": {"save_ms": 1.0}})
                if action == "restore":
                    stub.restores += 1
                    if broken_restore or body["filename"] not in stub.files or (fail_restore_once and stub.restores == 1):
                        return self.reply(400, {"error": "file not found"})
                    if not noop_restore:           # noop_restore: answers 200 but loads nothing
                        stub.cache = stub.files[body["filename"]]
                    return self.reply(200, {"id_slot": 0, "n_restored": 1, "n_read": 1234, "timings": {"restore_ms": 0.5}})
                return self.reply(404, {"error": "?"})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


class SlotPersist(unittest.TestCase):
    def run_stub(self, **kw):
        stub = Stub(**kw)
        try:
            with tempfile.TemporaryDirectory() as d:
                pf = os.path.join(d, "p.txt")
                with open(pf, "w") as f:
                    f.write(" ".join(f"w{i}" for i in range(3000)))
                out = os.path.join(d, "r.json")
                rc = sp.main(["--url", stub.url, "--prompt-file", pf, "--gen", "8", "--repeats", "2", "--out", out])
                with open(out) as f:
                    return rc, json.load(f)
        finally:
            stub.close()

    def test_restore_makes_warm_prefill_free(self):
        rc, r = self.run_stub()
        s = r["summary"]
        self.assertEqual(rc, 0)
        self.assertEqual(s["cold_prompt_n_p50"], 3000)
        self.assertEqual(s["warm_prompt_n_p50"], 0)
        self.assertEqual(s["saved_bytes"], 1234)
        self.assertTrue(s["all_continuations_identical"])

    def test_failed_restore_is_an_error(self):
        rc, r = self.run_stub(broken_restore=True)
        self.assertEqual(rc, 1)
        self.assertIsNone(r["summary"])
        self.assertEqual(len(r["errors"]), 2)

    def test_divergent_continuation_is_reported(self):
        rc, r = self.run_stub(nondeterministic=True)
        self.assertFalse(r["summary"]["all_continuations_identical"])

    # mutations S1, S3 of docs/evidence/2026-10-09-tool-mutations.md survived the three tests above
    def test_restore_that_loads_nothing_shows_full_prefill(self):
        # the slot must be erased before restore, otherwise the warm request measures the live cache
        rc, r = self.run_stub(noop_restore=True)
        self.assertEqual(r["summary"]["warm_prompt_n_p50"], 3000)

    def test_same_text_from_other_tokens_is_not_the_same_continuation(self):
        # N11: on a real llama-server different tokens can decode to the same text (byte tokens that are not valid
        # UTF-8 alone all read "\ufffd"); the continuation check compares token ids when the server returns them
        rc, r = self.run_stub(same_text_other_tokens=True)
        self.assertFalse(r["summary"]["all_continuations_identical"])
        self.assertTrue(all(x["compared"] == "tokens" for x in r["runs"]))

    def test_partial_failure_is_a_nonzero_exit(self):
        rc, r = self.run_stub(fail_restore_once=True)
        self.assertEqual(len(r["errors"]), 1)
        self.assertEqual(r["summary"]["runs"], 1)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
