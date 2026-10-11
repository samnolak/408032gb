"""scripts/gguf_headers_fetch.py, tools/gguf_headers/gguf_header.py (and the workflow gguf-headers.yml) without network.

A local HTTP server plays the Hugging Face hub: the revision and tree APIs (tree paginated by Link headers), a resolve
endpoint that redirects to a Range-capable "CDN" with X-Linked-Etag / X-Repo-Commit / X-Linked-Size like the hub's,
and the model search API. It serves GGUFs generated here: written byte by byte by this file (stdlib only; a split model,
a header with a 60 000-entry token array that needs many Range steps, a 256-byte alignment) and, when
LLAMA_CPP_GGUF_PY points at a llama.cpp gguf-py and numpy imports, the tinygen glm5next (UD dialect, MTP) and qwen4exp
models, whose data offsets and tensor offsets are compared with gguf-py's GGUFReader. The server counts every byte it
sends, so the tests see that the fetcher stops at the header, never passes the cap and fetches nothing in --plan.
The workflow check needs PyYAML (installed in ci.yml); without it, or without gguf-py / numpy, those classes skip.
"""
import contextlib
import functools
import gzip
import hashlib
import http.server
import importlib
import io
import json
import os
import random
import re
import shutil
import struct
import sys
import tempfile
import threading
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools", "gguf_headers"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gguf_header as G  # noqa: E402
import gguf_headers_fetch as F  # noqa: E402
import gguf_quant_audit as QA  # noqa: E402

TYPE_ID = {v[0]: k for k, v in QA.GGML_TYPES.items()}
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
try:
    import yaml
except ImportError:
    yaml = None
SHA = "0123456789abcdef0123456789abcdef01234567"


# ------------------------------------------------------------------------------------------------ GGUF writer (stdlib)

def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _val(vtype, v):
    if vtype == G.T_STRING:
        return _str(v)
    if vtype == G.T_ARRAY:
        etype, items = v
        if etype == G.T_STRING:
            return struct.pack("<IQ", etype, len(items)) + b"".join(_str(x) for x in items)
        return struct.pack("<IQ", etype, len(items)) + struct.pack(f"<{len(items)}{G.SCALAR_FMT[etype]}", *items)
    return struct.pack("<" + G.SCALAR_FMT[vtype], v)


def make_gguf(kv, tensors, alignment=32, seed=0):
    """kv: [(key, vtype, value)]; tensors: [(name, dims, type name)]. Returns (file bytes, data offset)."""
    rng = random.Random(seed)
    out = bytearray(b"GGUF" + struct.pack("<Iqq", 3, len(tensors), len(kv)))
    for k, t, v in kv:
        out += _str(k) + struct.pack("<I", t) + _val(t, v)
    data, off = bytearray(), 0
    for name, dims, tname in tensors:
        nb = G.tensor_nbytes(dims, TYPE_ID[tname])
        out += _str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<q", d) for d in dims)
        out += struct.pack("<iQ", TYPE_ID[tname], off)
        pad = G.align_up(nb, alignment) - nb
        data += rng.randbytes(nb) + b"\0" * pad
        off += nb + pad
    if tensors:
        out += b"\0" * (G.align_up(len(out), alignment) - len(out))
    return bytes(out + data), len(out)


def arch_kv(arch="glm5next", extra=()):
    return [("general.architecture", G.T_STRING, arch), (f"{arch}.block_count", G.T_UINT32, 4),
            (f"{arch}.expert_group_count", G.T_UINT32, 1), *extra]


TENSORS = [("token_embd.weight", [256, 20000], "Q8_0"), ("blk.0.attn_norm.weight", [256], "F32"),
           ("blk.1.ffn_gate_exps.weight", [256, 32, 8], "Q4_K"), ("blk.1.ffn_down_exps.weight", [256, 256, 8], "Q5_K"),
           ("blk.1.ffn_up_exps.weight", [256, 32, 8], "Q4_K"), ("output.weight", [256, 64], "Q6_K")]


@functools.lru_cache(maxsize=None)
def split_model(n_tokens=60000):
    """Three shards like llama-gguf-split writes them; shard 1 carries all KV and a big token array."""
    toks = [f"tok{i:06d}" for i in range(n_tokens)]
    split = lambda i: [("split.no", G.T_UINT16, i), ("split.count", G.T_UINT16, 3),  # noqa: E731
                       ("split.tensors.count", G.T_INT32, len(TENSORS))]
    kv1 = arch_kv(extra=[("tokenizer.ggml.model", G.T_STRING, "gpt2"),
                         ("tokenizer.ggml.tokens", G.T_ARRAY, (G.T_STRING, toks)),
                         ("tokenizer.ggml.token_type", G.T_ARRAY, (G.T_INT32, [1] * n_tokens)),
                         ("tokenizer.chat_template", G.T_STRING, "{{ messages[0]['content'] }}é"),
                         ("glm5next.swiglu_clamp_exp", G.T_ARRAY, (G.T_FLOAT32, [10.0] * 4)),
                         ("glm5next.kda.gate_lower_bound", G.T_FLOAT32, float("-inf"))]) + split(0)
    return [make_gguf(kv1, TENSORS[:2], seed=1),
            make_gguf([("general.architecture", G.T_STRING, "glm5next")] + split(1), TENSORS[2:5], seed=2),
            make_gguf([("general.architecture", G.T_STRING, "glm5next")] + split(2), TENSORS[5:], seed=3)]


# ------------------------------------------------------------------------------------------------ fake hub

class FakeHub:
    def __init__(self):
        self.repos = {}          # repo -> {"sha": ..., "files": {path: bytes}}
        self.page_size = 3
        self.ignore_range = False
        self.wrong_etag = False
        self.status_missing = 404
        self.search = []         # items the search API returns
        self.log = []            # (path, range or None, bytes sent)
        self.lock = threading.Lock()
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body=b"", headers=()):
                self.send_response(code)
                for k, v in headers:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                with hub.lock:
                    hub.log.append((self.path, self.headers.get("Range"), len(body) if code in (200, 206) else 0))

            def do_GET(self):
                hub.handle(self)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def add(self, repo, files, sha=SHA):
        self.repos[repo] = {"sha": sha, "files": dict(files)}

    def blobs(self):
        return {hashlib.sha256(b).hexdigest(): b for r in self.repos.values() for b in r["files"].values()}

    def bytes_sent(self, prefix="/cdn/"):
        return sum(n for p, _, n in self.log if p.startswith(prefix))

    def handle(self, h):
        u = urllib.parse.urlsplit(h.path)
        q = urllib.parse.parse_qs(u.query)
        j = lambda o, hd=(): h.send(200, json.dumps(o).encode(), [("Content-Type", "application/json"), *hd])  # noqa
        m = re.fullmatch(r"/api/models/([^/]+/[^/]+)/revision/([^/]+)", u.path)
        if m:
            r = self.repos.get(m.group(1))
            if r is None:
                return h.send(self.status_missing, b'{"error":"Repository not found"}')
            return j({"id": m.group(1), "sha": r["sha"], "gated": False, "tags": ["gguf", "license:mit"]})
        m = re.fullmatch(r"/api/models/([^/]+/[^/]+)/tree/([0-9a-f]{40})", u.path)
        if m:
            r = self.repos[m.group(1)]
            assert m.group(2) == r["sha"] and q.get("recursive") == ["true"]
            entries = []
            for d in sorted({os.path.dirname(p) for p in r["files"] if os.path.dirname(p)}):
                entries.append({"type": "directory", "path": d, "oid": "0" * 40, "size": 0})
            for p, b in sorted(r["files"].items()):
                e = {"type": "file", "path": p, "size": len(b), "oid": hashlib.sha1(b).hexdigest()}
                if p.endswith(".gguf"):
                    e["lfs"] = {"oid": hashlib.sha256(b).hexdigest(), "size": len(b), "pointerSize": 134}
                entries.append(e)
            cur = int(q.get("cursor", ["0"])[0])
            page = entries[cur:cur + self.page_size]
            hd = []
            if cur + self.page_size < len(entries):
                nxt = f"{self.url}{u.path}?recursive=true&cursor={cur + self.page_size}"
                hd = [("Link", f'<{nxt}>; rel="next"')]
            return j(page, hd)
        m = re.fullmatch(r"/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)", u.path)
        if m:
            r = self.repos[m.group(1)]
            b = r["files"][urllib.parse.unquote(m.group(3))]
            sha256 = hashlib.sha256(b).hexdigest()
            etag = ("f" * 64) if self.wrong_etag else sha256
            return h.send(302, b"", [("Location", f"/cdn/{sha256}"), ("X-Linked-Etag", f'"{etag}"'),
                                     ("X-Linked-Size", str(len(b))), ("X-Repo-Commit", r["sha"])])
        m = re.fullmatch(r"/cdn/([0-9a-f]{64})", u.path)
        if m:
            b = self.blobs()[m.group(1)]
            rg = h.headers.get("Range")
            if self.ignore_range or not rg:
                return h.send(200, b)
            a, z = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", rg).groups())
            z = min(z, len(b) - 1)
            return h.send(206, b[a:z + 1], [("Content-Range", f"bytes {a}-{z}/{len(b)}")])
        if u.path == "/api/models":
            assert q.get("filter") == ["gguf"], q
            items = [it for it in self.search if q["search"][0].lower() in it["id"].lower()]
            return j(items)
        h.send(500, b"unexpected " + h.path.encode())


class HubCase(unittest.TestCase):
    def setUp(self):
        self.hub = FakeHub()
        self.addCleanup(self.hub.close)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self._sleep = F.SLEEP
        F.SLEEP = lambda s: None
        self.addCleanup(setattr, F, "SLEEP", self._sleep)
        self.addCleanup(setattr, F, "ENDPOINT", F.ENDPOINT)

    def request(self, name, **req):
        d = os.path.join(self.tmp, name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "REQUEST.json")
        with open(p, "w") as f:
            json.dump(req, f)
        return p

    def run_fetch(self, *paths, plan=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = F.main(["--endpoint", self.hub.url, *(["--plan"] if plan else []), *paths])
        return rc, out.getvalue()

    def load(self, p, name):
        with open(os.path.join(os.path.dirname(p), name)) as f:
            return json.load(f)


# ------------------------------------------------------------------------------------------------ tests: parser

class Parser(unittest.TestCase):
    def test_needmore_reports_the_exact_next_need(self):
        (b, off), = [make_gguf(arch_kv(extra=[("x.s", G.T_STRING, "a" * 1000)]), TENSORS)]
        h = G.parse(b[:off])
        self.assertEqual(h.data_offset, off)
        self.assertEqual(G.parse(b).data_offset, off)
        for cut in (0, 3, 20, 100, 600, off - 1):
            with self.assertRaises(G.NeedMore) as cm:
                G.parse(b[:cut])
            self.assertGreater(cm.exception.need, cut)
            self.assertLessEqual(cm.exception.need, off)
        # the 1000-byte string: one NeedMore jumps over all of it
        key_end = b.index(b"x.s") + 3 + 4 + 8
        with self.assertRaises(G.NeedMore) as cm:
            G.parse(b[:key_end + 10])
        self.assertEqual(cm.exception.need, key_end + 1000)

    def test_alignment_rules_of_gguf_cpp(self):
        kv = arch_kv(extra=[("general.alignment", G.T_UINT32, 256)])
        b, off = make_gguf(kv, TENSORS, alignment=256)
        h = G.parse(b)
        self.assertEqual(h.alignment, 256)
        self.assertEqual(h.data_offset % 256, 0)
        self.assertNotEqual(h.infos_end % 256, 0)
        self.assertEqual(h.data_offset, off)
        self.assertEqual(G.data_layout(h)[0], "contiguous")
        self.assertEqual(off + G.data_layout(h)[1], len(b))
        # no tensors: the data section starts right after the KV pairs, no padding (gguf.cpp:781)
        b0, off0 = make_gguf(kv, [])
        self.assertEqual(G.parse(b0).data_offset, len(b0))
        bad, _ = make_gguf(arch_kv(extra=[("general.alignment", G.T_UINT32, 48)]), TENSORS)
        with self.assertRaises(ValueError):
            G.parse(bad)
        bad, _ = make_gguf(arch_kv(extra=[("general.alignment", G.T_UINT64, 64)]), TENSORS)
        with self.assertRaises(ValueError):
            G.parse(bad)

    def test_rejects_what_gguf_cpp_rejects(self):
        b, _ = make_gguf(arch_kv(), TENSORS)
        for bad in (b"GGUX" + b[4:], b[:4] + struct.pack("<I", 1) + b[8:]):
            with self.assertRaises(ValueError):
                G.parse(bad)
        dup, _ = make_gguf(arch_kv() + [("general.architecture", G.T_STRING, "x")], [])
        with self.assertRaises(ValueError):
            G.parse(dup)
        G.parse(make_gguf(arch_kv(), [("t", [100], "F32")])[0])  # F32 has block 1: fine
        with self.assertRaises(ValueError):                         # a Q4_K row must hold whole 256-blocks
            G.parse(make_gguf(arch_kv(), [("t", [100], "Q4_K")])[0])

    def test_summary_digests_big_arrays_and_keeps_the_rest(self):
        b, off = split_model(n_tokens=5000)[0]
        h = G.parse(b)
        s = G.summarize(h, b[:off])
        kv = {e["key"]: e for e in s["kv"]}
        tok = kv["tokenizer.ggml.tokens"]
        self.assertEqual((tok["count"], tok["elem_type"]), (5000, "string"))
        self.assertNotIn("value", tok)
        span = next(e["span"] for e in h.kv if e["key"] == "tokenizer.ggml.tokens")
        self.assertEqual(tok["sha256"], hashlib.sha256(b[span[0]:span[1]]).hexdigest())
        self.assertEqual(kv["tokenizer.chat_template"]["value"], "{{ messages[0]['content'] }}é")
        self.assertEqual(kv["glm5next.kda.gate_lower_bound"]["value"], "-inf")
        self.assertEqual(kv["glm5next.swiglu_clamp_exp"]["value"], [10.0] * 4)
        self.assertEqual(s["header_sha256"], hashlib.sha256(b[:off]).hexdigest())
        self.assertEqual([t["name"] for t in s["tensors"]], [t[0] for t in TENSORS[:2]])
        json.loads(G.dumps(s))

    def test_rebuild_copies_values_byte_for_byte(self):
        b, off = split_model(n_tokens=300)[0]
        h = G.parse(b)
        nb = G.rebuild(h, b[:off], keep=lambda k: not k.startswith("split."),
                       rename=lambda k: k.replace("glm5next.", "glm5-next."))
        h2 = G.parse(nb)
        self.assertEqual(h2.n_tensors, 0)
        self.assertEqual(len(nb), h2.data_offset)
        self.assertEqual(h2.get("tokenizer.ggml.tokens"), h.get("tokenizer.ggml.tokens"))
        self.assertEqual(h2.get("glm5-next.block_count"), 4)
        self.assertIsNone(h2.get("split.count"))
        full = G.rebuild(h, b[:off], tensors=True)
        self.assertEqual(full, b[:off])


# ------------------------------------------------------------------------------------------------ tests: fetcher

class Fetch(HubCase):
    GLOB = "*M-UD-Q4_K_XL-0000?-of-00003.gguf"

    def setUp(self):
        super().setUp()
        self.shards = split_model()
        files = {f"UD-Q4_K_XL/M-UD-Q4_K_XL-0000{i + 1}-of-00003.gguf": b for i, (b, _) in enumerate(self.shards)}
        files["README.md"] = b"# card\n"
        files["Q8_0/M-Q8_0.gguf"] = make_gguf(arch_kv(), TENSORS)[0]
        self.hub.add("org/M-GGUF", files)

    def test_fetch_stops_at_the_header_and_pins_the_revision(self):
        p = self.request("m", repo="org/M-GGUF", revision="main", files=[self.GLOB], expect_count=3,
                         initial_bytes=4096)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        pin = self.load(p, "PIN.json")
        self.assertEqual((pin["status"], pin["revision"], pin["requested_revision"]), ("ok", SHA, "main"))
        self.assertEqual(len(pin["files"]), 3)
        for (b, off), rec in zip(self.shards, pin["files"]):
            self.assertEqual(rec["header_bytes"], off)
            self.assertEqual(rec["lfs_sha256"], hashlib.sha256(b).hexdigest())
            self.assertEqual((rec["size"], rec["lfs_size"]), (len(b), len(b)))
            self.assertEqual(rec["header_sha256"], hashlib.sha256(b[:off]).hexdigest())
            self.assertTrue(rec["etag_matches_lfs_sha256"])
            self.assertTrue(rec["repo_commit_matches"])
            self.assertEqual(rec["size_check"], "exact")
            # contiguous ranges, no byte fetched twice, the last one is the first that holds the whole header
            rr = rec["range_requests"]
            self.assertEqual(rr[0][0], 0)
            for (a0, z0), (a1, z1) in zip(rr, rr[1:]):
                self.assertEqual(a1, z0 + 1)
            self.assertEqual(rec["bytes_fetched"], rr[-1][1] + 1)
            self.assertLess(rr[-2][1] + 1 if len(rr) > 1 else 0, off)
            self.assertLessEqual(rec["bytes_fetched"], max(2 * off, 4096))
            with open(os.path.join(os.path.dirname(p), rec["header_gz"]), "rb") as f:
                hdr = gzip.decompress(f.read())
            self.assertEqual(hdr, b[:off])
        self.assertGreaterEqual(len(pin["files"][0]["range_requests"]), 8)   # 4 KiB doubling to ~1 MiB
        self.assertEqual(self.hub.bytes_sent(), pin["bytes_fetched"])         # nothing read beyond what was recorded
        self.assertLess(pin["bytes_fetched"], sum(len(b) for b, _ in self.shards) // 2)
        self.assertNotIn("Q8_0", "".join(x[0] for x in self.hub.log if x[0].startswith("/org")))

    def test_summary_round_trips_from_the_committed_header(self):
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, initial_bytes=65536)
        self.assertEqual(self.run_fetch(p)[0], 0)
        d = os.path.dirname(p)
        for rec in self.load(p, "PIN.json")["files"]:
            with open(os.path.join(d, rec["header_gz"]), "rb") as f:
                gz = f.read()
            self.assertEqual(hashlib.sha256(gz).hexdigest(), rec["header_gz_sha256"])
            hdr = gzip.decompress(gz)
            with open(os.path.join(d, rec["summary"])) as f:
                committed = f.read()
            self.assertEqual(G.dumps(G.summarize(G.parse(hdr), hdr)), committed)
            self.assertEqual(gzip.compress(hdr, compresslevel=9, mtime=0), gz)   # deterministic

    def test_audit_on_header_only_shards_equals_audit_on_the_files(self):
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3)
        self.assertEqual(self.run_fetch(p)[0], 0)
        full, hdr = os.path.join(self.tmp, "full"), os.path.join(self.tmp, "hdr")
        os.makedirs(full)
        os.makedirs(hdr)
        for (b, _), rec in zip(self.shards, self.load(p, "PIN.json")["files"]):
            base = os.path.basename(rec["path"])
            with open(os.path.join(full, base), "wb") as f:
                f.write(b)
            with open(os.path.join(os.path.dirname(p), rec["header_gz"]), "rb") as f, \
                    open(os.path.join(hdr, base), "wb") as g:
                g.write(gzip.decompress(f.read()))
        first = "M-UD-Q4_K_XL-00001-of-00003.gguf"
        a, b = QA.audit(os.path.join(full, first)), QA.audit(os.path.join(hdr, first))
        self.assertEqual(a, b)
        self.assertEqual(a["classes"]["routed_experts"]["tensors"], 3)

    def test_cap_exceeded_is_an_error_and_never_a_full_download(self):
        cap = 65536
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, max_header_bytes=cap,
                         initial_bytes=4096)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("cap exceeded", out)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(p), "PIN.json")))
        self.assertIn("cap exceeded", self.load(p, "ERROR.json")["error"])
        self.assertLessEqual(self.hub.bytes_sent(), cap)
        self.assertEqual(sorted(os.listdir(os.path.dirname(p))), ["ERROR.json", "REQUEST.json"])

    def test_plan_fails_before_any_request_on_caps(self):
        for req in ({"max_header_bytes": (64 << 20) + 1}, {"expect_count": 17}, {"initial_bytes": 1 << 21,
                                                                                  "max_header_bytes": 1 << 20}):
            p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], **{"expect_count": 3, **req})
            for plan in (True, False):
                rc, out = self.run_fetch(p, plan=plan)
                self.assertEqual(rc, 1, (req, out))
                self.assertIn("::error::", out)
        ps = [self.request(f"m{i}", repo="org/M-GGUF", files=[self.GLOB], expect_count=16) for i in range(2)]
        rc, out = self.run_fetch(*ps, plan=True)
        self.assertEqual(rc, 1)
        self.assertIn("over the per-run cap", out)
        self.assertEqual(self.hub.log, [])

    def test_plan_prints_the_worst_case_and_fetches_nothing(self):
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, max_header_bytes=1 << 20)
        rc, out = self.run_fetch(p, plan=True)
        self.assertEqual(rc, 0, out)
        self.assertIn(f"worst-case download {3 << 20} B", out)
        self.assertEqual(self.hub.log, [])

    def test_revision_mismatch_is_an_error(self):
        p = self.request("m", repo="org/M-GGUF", revision="f" * 40, files=[self.GLOB], expect_count=3)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("the hub answered", self.load(p, "ERROR.json")["error"])
        self.assertEqual(self.hub.bytes_sent(), 0)

    def test_missing_repo_is_recorded_not_fetched(self):
        for code in (404, 401):
            self.hub.status_missing = code
            p = self.request(f"x{code}", repo="nobody/GLM-GGUF", files=["*.gguf"], expect_count=6)
            rc, out = self.run_fetch(p)
            self.assertEqual(rc, 0, out)
            pin = self.load(p, "PIN.json")
            self.assertEqual((pin["status"], pin["http_status"], pin["files"]), ("repo_unavailable", code, []))
            self.assertIn("::warning::", out)
        self.assertEqual(self.hub.bytes_sent(), 0)

    def test_no_match_is_recorded_with_the_gguf_listing(self):
        p = self.request("m", repo="org/M-GGUF", files=["*UD-Q2_K_XL*.gguf"], expect_count=3)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        pin = self.load(p, "PIN.json")
        self.assertEqual(pin["status"], "no_match")
        self.assertEqual([x["path"] for x in pin["gguf_listing"]],
                         ["Q8_0/M-Q8_0.gguf"] + [f"UD-Q4_K_XL/M-UD-Q4_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        self.assertEqual(self.hub.bytes_sent(), 0)

    def test_count_mismatch_is_an_error(self):
        p = self.request("m", repo="org/M-GGUF", files=["*.gguf"], expect_count=3)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("4 files match", self.load(p, "ERROR.json")["error"])
        self.assertEqual(self.hub.bytes_sent(), 0)

    def test_a_server_that_ignores_range_is_rejected_unread(self):
        self.hub.ignore_range = True
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, initial_bytes=4096)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("instead of 206", self.load(p, "ERROR.json")["error"])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(p), "PIN.json")))

    def test_wrong_etag_is_recorded_and_warned(self):
        self.hub.wrong_etag = True
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        self.assertFalse(any(r["etag_matches_lfs_sha256"] for r in self.load(p, "PIN.json")["files"]))
        self.assertIn("X-Linked-Etag != LFS sha256", out)

    def test_done_requests_are_not_fetched_again_and_errors_retry(self):
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, max_header_bytes=65536)
        self.assertEqual(self.run_fetch(p)[0], 1)
        with open(p, "w") as f:
            json.dump({"repo": "org/M-GGUF", "files": [self.GLOB], "expect_count": 3}, f)
        self.assertEqual(self.run_fetch(p)[0], 0)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(p), "ERROR.json")))
        n = len(self.hub.log)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0)
        self.assertIn("nothing pending", out)
        self.assertEqual(len(self.hub.log), n)

    def test_tree_pagination(self):
        self.hub.page_size = 1
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3)
        self.assertEqual(self.run_fetch(p)[0], 0)
        self.assertGreaterEqual(sum(1 for x in self.hub.log if "/tree/" in x[0]), 6)

    def test_discover_lists_and_reads_no_file(self):
        self.hub.search = [
            {"id": "org/M-GGUF", "sha": SHA, "gguf": {"architecture": "glm5-next", "total": 1, "chat_template": "x"}},
            {"id": "org/GLM-5.3-Flash-other-GGUF", "sha": SHA, "gguf": {"architecture": "glm5next"}},
            {"id": "org/MiMo-V2.6-Flash-GGUF", "sha": SHA, "gguf": {"architecture": "mimo2"}}]
        self.hub.search[0]["id"] = "org/M-GGUF"
        self.hub.add("org/GLM-5.3-Flash-other-GGUF", {"a.gguf": b"GGUF"})
        self.hub.add("org/MiMo-V2.6-Flash-GGUF", {"b.gguf": b"GGUF", "c.txt": b""})
        p = self.request("d", kind="discover", list_files=True,
                         searches=[{"search": "GLM-5.3-Flash", "architectures": ["glm5-next"]},
                                   {"search": "M-GGUF", "architectures": ["glm5-next"]},
                                   {"search": "MiMo-V2.6-Flash", "architectures": ["mimo2"]}])
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        lst = self.load(p, "LISTING.json")
        self.assertEqual([s["n_results"] for s in lst["searches"]], [1, 1, 1])
        self.assertFalse(lst["searches"][0]["results"][0]["arch_match"])
        m = lst["searches"][1]["results"][0]
        self.assertTrue(m["arch_match"])
        self.assertEqual(len(m["gguf_files"]), 4)
        self.assertEqual(m["gguf_chat_template_sha256"], hashlib.sha256(b"x").hexdigest())
        self.assertTrue(lst["searches"][2]["results"][0]["arch_match"])
        self.assertEqual(lst["searches"][2]["results"][0]["gguf_files"], [{"path": "b.gguf", "size": 4}])
        self.assertEqual(self.hub.bytes_sent(), 0)
        self.assertFalse(any("/resolve/" in x[0] for x in self.hub.log))
        self.assertTrue(all("expand=gguf" in x[0] for x in self.hub.log if x[0].startswith("/api/models?")))


# ------------------------------------------------------------------------------------------------ tinygen models

@unittest.skipUnless(GGUF_PY and HAVE_NUMPY, "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py (and install numpy)")
class Tinygen(HubCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, GGUF_PY)
        sys.path.insert(0, os.path.join(ROOT, "tools", "tinygen"))
        cls.reader = importlib.import_module("gguf").GGUFReader
        cls.dir = tempfile.mkdtemp()
        glm = importlib.import_module("glm5next_tiny")
        qwen = importlib.import_module("qwen4exp_tiny")
        glm.ARCH = "glm5next"
        try:
            glm.build(os.path.join(cls.dir, "glm.gguf"), cfg=dict(glm.DEFAULT, n_layer=12), mtp=True, dialect="ik")
        finally:
            glm.ARCH = "glm5-next"
        qwen.build(os.path.join(cls.dir, "qwen.gguf"), mtp=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_offsets_agree_with_gguf_py(self):
        for name, arch in (("glm.gguf", "glm5next"), ("qwen.gguf", "qwen4exp")):
            with open(os.path.join(self.dir, name), "rb") as f:
                b = f.read()
            self.hub.add(f"tiny/{arch}", {name: b})
            p = self.request(arch, repo=f"tiny/{arch}", files=[name], expect_count=1, initial_bytes=4096)
            rc, out = self.run_fetch(p)
            self.assertEqual(rc, 0, out)
            rec = self.load(p, "PIN.json")["files"][0]
            r = self.reader(os.path.join(self.dir, name))
            self.assertEqual(rec["header_bytes"], r.data_offset)
            self.assertEqual(rec["size_check"], "exact")
            with open(os.path.join(os.path.dirname(p), rec["summary"])) as f:
                s = json.load(f)
            self.assertEqual(len(s["tensors"]), len(r.tensors))
            for t, rt in zip(s["tensors"], r.tensors):
                self.assertEqual(t["name"], rt.name)
                self.assertEqual(r.data_offset + t["offset"], rt.data_offset)
                self.assertEqual(t["nbytes"], rt.n_bytes)
                self.assertEqual(t["shape"], [int(x) for x in rt.shape])
            kv = {e["key"]: e for e in s["kv"]}
            self.assertEqual(kv["general.architecture"]["value"], arch)
            for k, field in r.fields.items():
                if k.startswith("GGUF."):
                    continue
                self.assertIn(k, kv)
            if arch == "glm5next":
                self.assertEqual(kv["glm5next.expert_group_count"]["value"], 1)


# ------------------------------------------------------------------------------------------------ workflow

WORKFLOW = os.path.join(ROOT, ".github", "workflows", "gguf-headers.yml")


@unittest.skipUnless(yaml is not None, "PyYAML is not installed")
class Workflow(unittest.TestCase):
    """The same safety properties as vendor-hf-branch.yml (AGENTS.md "Git and sync")."""

    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW) as f:
            cls.text = f.read()
        cls.wf = yaml.safe_load(cls.text)

    def test_triggers_and_guards(self):
        on = self.wf.get("on", self.wf.get(True))
        self.assertEqual(on["push"]["branches"], ["claude/**"])
        self.assertEqual(on["push"]["paths"], ["third_party/gguf-headers/*/REQUEST.json"])
        self.assertIn("workflow_dispatch", on)
        self.assertEqual(self.wf["permissions"], {"contents": "write"})
        (name, job), = self.wf["jobs"].items()
        self.assertEqual(job["if"], "startsWith(github.ref_name, 'claude/')")
        self.assertLessEqual(int(job["timeout-minutes"]), 30)
        runs = [s.get("run", "") for s in job["steps"]]
        for r in runs:
            self.assertNotIn("${{", r, "expressions reach scripts only through env")
            self.assertNotRegex(r, r"\bmain\b")
        pushes = [ln.strip() for r in runs for ln in r.splitlines() if re.search(r"\bpush\b", ln)]
        self.assertEqual(pushes, ['if git pull -q --rebase origin "$REF" && git push origin HEAD:"$REF"; then'])
        plan = [i for i, r in enumerate(runs) if "gguf_headers_fetch.py --plan" in r]
        fetch = [i for i, r in enumerate(runs) if re.search(r"gguf_headers_fetch\.py\s*$", r.strip())]
        self.assertEqual((len(plan), len(fetch)), (1, 1))
        self.assertLess(plan[0], fetch[0])
        commit = job["steps"][-1]
        self.assertEqual(commit["env"]["REF"], "${{ github.ref_name }}")
        self.assertIn('git pull -q --rebase origin "$REF"', commit["run"])
        self.assertIn('HEAD:"$REF"', commit["run"])
        self.assertIn("git add third_party/gguf-headers", commit["run"])
        self.assertEqual(commit.get("if"), "${{ !cancelled() }}")
        self.assertEqual(self.text.count("HEAD:"), 1)


if __name__ == "__main__":
    unittest.main()
