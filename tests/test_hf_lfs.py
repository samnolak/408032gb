"""N23: LFS files (tokenizer.json) pinned by sha256 instead of committed.

scripts/vendor_hf_files.py one() with "lfs_files": the object is downloaded once in CI, checked against the tree API's
LFS pointer (size, oid = sha256) and only recorded in PIN.json; scripts/hf_lfs_fetch.py fetches it on demand by that
record into a cache outside the repository (no network here: a fake hub in-process, a local HTTP server for the
fetcher). Also: the two tokenizer requests point at the revisions the chat templates are pinned at.
"""
import hashlib
import http.server
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_lfs_fetch  # noqa: E402
import vendor_hf_files  # noqa: E402

REV = "a" * 40
TOK = b'{"version": "1.0", "model": {"type": "BPE"}}\n' * 1000
LICENSE = b"MIT License\n"


def load(path):
    with open(path) as f:
        return json.load(f)


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeHub:
    """The three hub endpoints vendor_hf_files.one() reads."""

    def __init__(self, tok=TOK, lfs_oid=None, lfs_size=None, lfs=True):
        self.tok, self.calls = tok, []
        e = {"type": "file", "path": "tokenizer.json", "size": len(tok), "oid": "f" * 40}
        if lfs:
            e["lfs"] = {"oid": lfs_oid or hashlib.sha256(tok).hexdigest(), "size": lfs_size or len(tok),
                        "pointerSize": 134}
        self.tree = [e, {"type": "file", "path": "LICENSE", "size": len(LICENSE), "oid": git_blob(LICENSE)},
                     {"type": "file", "path": "model-00001-of-00002.safetensors", "size": 5 << 30, "oid": "e" * 40,
                      "lfs": {"oid": "d" * 64, "size": 5 << 30}}]

    def __call__(self, url):
        self.calls.append(url)
        if "/api/models/x/m/revision/" in url:
            return json.dumps({"sha": REV, "gated": False, "tags": ["license:mit", "text-generation"],
                               "cardData": {"license": "mit"}}).encode()
        if "/api/models/x/m/tree/" in url:
            return json.dumps(self.tree).encode()
        if url.endswith("/resolve/%s/tokenizer.json" % REV):
            return self.tok
        if url.endswith("/resolve/%s/LICENSE" % REV):
            return LICENSE
        raise AssertionError("unexpected URL " + url)


class VendorLfs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.req = os.path.join(self.d.name, "REQUEST.json")

    def tearDown(self):
        self.d.cleanup()

    def run_one(self, hub, **req):
        body = {"repo": "x/m", "revision": REV, "files": ["LICENSE"], "lfs_files": ["tokenizer.json"]}
        body.update(req)
        with open(self.req, "w") as f:
            json.dump(body, f)
        with mock.patch.object(vendor_hf_files, "http", side_effect=hub), redirect_stdout(io.StringIO()) as out:
            vendor_hf_files.one(self.req)
        return load(os.path.join(self.d.name, "PIN.json")), out.getvalue()

    def test_pinned_not_committed(self):
        pin, out = self.run_one(FakeHub())
        sha = hashlib.sha256(TOK).hexdigest()
        self.assertEqual(pin["lfs_files"], [{
            "path": "tokenizer.json", "size": len(TOK), "sha256": sha, "lfs_oid": sha, "pointer_git_blob": "f" * 40,
            "url": "https://huggingface.co/x/m/resolve/%s/tokenizer.json" % REV,
            "stored": "not committed: fetched on demand by scripts/hf_lfs_fetch.py"}])
        self.assertFalse(os.path.exists(os.path.join(self.d.name, "tokenizer.json")))
        self.assertEqual(pathlib.Path(self.d.name, "SHA256SUMS").read_text(),
                         "%s  LICENSE\n" % hashlib.sha256(LICENSE).hexdigest())
        self.assertEqual(pin["card_license"], {"license": "mit", "license_name": None, "license_link": None})
        self.assertEqual(pin["license_tags"], ["license:mit"])
        self.assertIn("LFS tokenizer.json %d B sha256 %s" % (len(TOK), sha), out)

    def test_sha256_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "LFS pointer says"):
            self.run_one(FakeHub(lfs_oid="0" * 64))

    def test_size_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "LFS pointer says"):
            self.run_one(FakeHub(lfs_size=len(TOK) - 1))

    def test_cap(self):
        with self.assertRaisesRegex(RuntimeError, "over the cap"):
            self.run_one(FakeHub(), lfs_max_bytes=len(TOK) - 1)
        self.assertEqual(vendor_hf_files.LFS_MAX_BYTES, 64 << 20)

    def test_never_weights(self):
        with self.assertRaisesRegex(RuntimeError, "not an allowed LFS file"):
            self.run_one(FakeHub(), lfs_files=["model-00001-of-00002.safetensors"])

    def test_not_lfs_upstream_fails(self):
        with self.assertRaisesRegex(RuntimeError, "not an LFS file"):
            self.run_one(FakeHub(lfs=False))

    def test_request_without_lfs_files_unchanged(self):
        body = {"repo": "x/m", "revision": REV, "files": ["LICENSE", "tokenizer.json"]}
        with open(self.req, "w") as f:
            json.dump(body, f)
        with mock.patch.object(vendor_hf_files, "http", side_effect=FakeHub()), redirect_stdout(io.StringIO()):
            vendor_hf_files.one(self.req)
        pin = load(os.path.join(self.d.name, "PIN.json"))
        self.assertNotIn("lfs_files", pin)
        self.assertNotIn("card_license", pin)
        self.assertEqual(pin["not_vendored"], [{"path": "tokenizer.json",
                                                "why": "%d bytes or LFS: not a small text file" % len(TOK)}])


class Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    hits = []

    def do_GET(self):  # noqa: N802
        Handler.hits.append(self.path)
        data = Handler.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class Fetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.endpoint = "http://127.0.0.1:%d" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.d.name, "cache")
        self.sha = hashlib.sha256(TOK).hexdigest()
        self.pin = os.path.join(self.d.name, "PIN.json")
        with open(self.pin, "w") as f:
            json.dump({"repo": "x/m", "revision": REV, "lfs_files": [
                {"path": "tokenizer.json", "size": len(TOK), "sha256": self.sha}]}, f)
        Handler.files = {"/x/m/resolve/%s/tokenizer.json" % REV: TOK}
        Handler.hits = []
        self.env = mock.patch.dict(os.environ, {"HF_ENDPOINT": self.endpoint})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.d.cleanup()

    def main(self, *args):
        with redirect_stdout(io.StringIO()) as out:
            rc = hf_lfs_fetch.main(["--cache", self.cache] + list(args))
        return rc, out.getvalue()

    def test_fetch_then_cached(self):
        self.assertIsNone(hf_lfs_fetch.path_for(self.pin, "tokenizer.json", self.cache))
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 0, out)
        p = os.path.join(self.cache, self.sha, "tokenizer.json")
        self.assertIn("%s  %s  x/m:tokenizer.json (downloaded)" % (self.sha, p), out)
        self.assertEqual(pathlib.Path(p).read_bytes(), TOK)
        self.assertEqual(hf_lfs_fetch.path_for(self.pin, "tokenizer.json", self.cache), p)
        rc, out = self.main(self.pin)
        self.assertIn("(cached)", out)
        self.assertEqual(len(Handler.hits), 1)
        rc, out = self.main("--offline", self.pin)
        self.assertEqual(rc, 0, out)

    def test_corrupt_cache_is_replaced(self):
        p = os.path.join(self.cache, self.sha, "tokenizer.json")
        os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f:
            f.write(TOK[:-1] + b"X")
        self.assertIsNone(hf_lfs_fetch.path_for(self.pin, "tokenizer.json", self.cache))
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 0, out)
        self.assertEqual(pathlib.Path(p).read_bytes(), TOK)

    def test_wrong_bytes_upstream_fail_and_leave_nothing(self):
        Handler.files = {"/x/m/resolve/%s/tokenizer.json" % REV: TOK[:-1] + b"X"}
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("::error::RuntimeError", out)
        self.assertEqual(os.listdir(os.path.join(self.cache, self.sha)), [])

    def test_longer_body_is_cut_off(self):
        Handler.files = {"/x/m/resolve/%s/tokenizer.json" % REV: TOK + b"more"}
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("more than the pinned", out)

    def test_offline_missing_fails(self):
        rc, out = self.main("--offline", self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("not in the cache", out)
        self.assertEqual(Handler.hits, [])

    def test_pin_without_lfs_files_fails(self):
        with open(self.pin, "w") as f:
            json.dump({"repo": "x/m", "revision": REV, "files": []}, f)
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("no lfs_files", out)

    def test_non_hex_sha256_refused(self):
        with open(self.pin, "w") as f:
            json.dump({"repo": "x/m", "revision": REV, "lfs_files": [
                {"path": "tokenizer.json", "size": len(TOK), "sha256": "g" * 64}]}, f)
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("64-hex sha256", out)
        self.assertEqual(Handler.hits, [])

    def test_short_revision_refused(self):
        with open(self.pin, "w") as f:
            json.dump({"repo": "x/m", "revision": "main", "lfs_files": [
                {"path": "tokenizer.json", "size": len(TOK), "sha256": self.sha}]}, f)
        rc, out = self.main(self.pin)
        self.assertEqual(rc, 1)
        self.assertIn("40-hex revision", out)


class Requests(unittest.TestCase):
    def test_tokenizer_requests_at_the_template_revisions(self):
        for name, chat in (("glm-5.3-flash-tokenizer", "glm-5.3-flash-chat"),
                           ("qwen3.8-flash-next-tokenizer", "qwen3.8-flash-next-chat")):
            req = json.loads((ROOT / "data/hf-requests" / name / "REQUEST.json").read_text())
            pin = json.loads((ROOT / "third_party/hf" / chat / "PIN.json").read_text())
            self.assertEqual(req["repo"], pin["repo"])
            self.assertEqual(req["revision"], pin["revision"])
            self.assertEqual(req["lfs_files"], ["tokenizer.json"])
            self.assertEqual(req["files"], ["LICENSE", "README.md"])
            listing = {e["path"]: e for e in pin["upstream_listing"]}
            self.assertTrue(listing["tokenizer.json"]["lfs"])
            self.assertLessEqual(listing["tokenizer.json"]["size"], req["lfs_max_bytes"])
            for f in req["files"]:
                self.assertFalse(listing[f]["lfs"])
                self.assertLess(listing[f]["size"], vendor_hf_files.MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
