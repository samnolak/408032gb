"""scripts/offline/weights.py: manifest building, offline verify, resumable download against a local Range server."""
import hashlib
import http.server
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(api_base=None):
    if api_base:
        os.environ["WEIGHTS_API_BASE"] = api_base
    spec = importlib.util.spec_from_file_location("weights", os.path.join(ROOT, "scripts", "offline", "weights.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.environ.pop("WEIGHTS_API_BASE", None)
    return mod


REV = "38bb39ee97821de2c9009abb7e93950eec396e66"


class Manifest(unittest.TestCase):
    def test_build_filters_and_uses_lfs_sha(self):
        w = load()
        entries = [{"type": "directory", "path": "UD-Q4_K_XL"},
                   {"type": "file", "path": "UD-Q4_K_XL/a-00001-of-00002.gguf", "size": 134, "oid": "g1",
                    "lfs": {"oid": "f" * 64, "size": 10, "pointerSize": 134}},
                   {"type": "file", "path": "UD-Q4_K_XL/a-00002-of-00002.gguf", "size": 134, "oid": "g2",
                    "lfs": {"oid": "e" * 64, "size": 20, "pointerSize": 134}},
                   {"type": "file", "path": "README.md", "size": 5, "oid": "g3"}]
        m = w.build_manifest("org/repo", REV, entries, "UD-Q4_K_XL/*")
        self.assertEqual([f["path"] for f in m["files"]], ["UD-Q4_K_XL/a-00001-of-00002.gguf", "UD-Q4_K_XL/a-00002-of-00002.gguf"])
        self.assertEqual(m["total_bytes"], 30)
        self.assertEqual(m["files"][0]["sha256"], "f" * 64)
        with self.assertRaises(ValueError):
            w.build_manifest("org/repo", "main", entries, "*")          # branch names are refused
        with self.assertRaises(ValueError):
            w.build_manifest("org/repo", REV, entries, "nothing/*")

    def test_committed_manifest_is_consistent(self):
        with open(os.path.join(ROOT, "models", "weights", "qwen3.8-flash-next-UD-Q4_K_XL.manifest.json")) as f:
            m = json.load(f)
        self.assertEqual(m["total_bytes"], sum(x["size"] for x in m["files"]))
        self.assertEqual(round(m["total_bytes"] / 1e9, 1), 111.3)          # the 111.3 GB quoted in research/R1
        for x in m["files"]:
            self.assertRegex(x["sha256"], r"^[0-9a-f]{64}$")


class Verify(unittest.TestCase):
    def test_ok_missing_size_and_corruption(self):
        w = load()
        with tempfile.TemporaryDirectory() as d:
            data = {"x/a.bin": b"hello" * 1000, "x/b.bin": b"world" * 999}
            files = []
            for p, b in data.items():
                os.makedirs(os.path.join(d, "x"), exist_ok=True)
                with open(os.path.join(d, p), "wb") as f:
                    f.write(b)
                files.append({"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()})
            man = {"files": files}
            log = []
            self.assertTrue(w.verify(man, d, log.append))
            with open(os.path.join(d, "x/b.bin"), "r+b") as f:
                f.write(b"W")
            self.assertFalse(w.verify(man, d, log.append))
            self.assertTrue(any(l.startswith("SHA256") for l in log))
            os.remove(os.path.join(d, "x/a.bin"))
            log.clear()
            self.assertFalse(w.verify(man, d, log.append))
            self.assertTrue(any(l.startswith("MISSING") for l in log))


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = self.payload
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            body = body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class NoRangeHandler(RangeHandler):
    """A server that ignores Range and always answers 200 with the whole file."""

    def do_GET(self):
        body = self.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Download(unittest.TestCase):
    def test_server_ignoring_range_is_refused(self):
        # appending a full 200 body to a partial file would corrupt it; the tool must refuse
        # (mutation W4 of docs/evidence/2026-10-09-tool-mutations.md survived without this test)
        payload = os.urandom(50_000)
        NoRangeHandler.payload = payload
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoRangeHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            w = load(f"http://127.0.0.1:{srv.server_address[1]}")
            with tempfile.TemporaryDirectory() as d:
                man = {"repo": "org/repo", "revision": REV,
                       "files": [{"path": "m/model.gguf", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
                os.makedirs(os.path.join(d, "m"))
                with open(os.path.join(d, "m/model.gguf"), "wb") as f:
                    f.write(payload[:12_345])
                with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                    w.download(man, d)
                self.assertEqual(os.path.getsize(os.path.join(d, "m/model.gguf")), 12_345)   # partial file untouched
        finally:
            srv.shutdown()

    def test_resume_and_verify(self):
        payload = os.urandom(300_000)
        RangeHandler.payload = payload
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            w = load(f"http://127.0.0.1:{srv.server_address[1]}")
            with tempfile.TemporaryDirectory() as d:
                man = {"repo": "org/repo", "revision": REV,
                       "files": [{"path": "m/model.gguf", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
                os.makedirs(os.path.join(d, "m"))
                with open(os.path.join(d, "m/model.gguf"), "wb") as f:
                    f.write(payload[:123_456])                                   # a partial earlier download
                buf = io.StringIO()
                with redirect_stdout(buf):
                    ok = w.download(man, d)
                self.assertTrue(ok, buf.getvalue())
                self.assertIn(f"+{len(payload) - 123_456} bytes", buf.getvalue())
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
