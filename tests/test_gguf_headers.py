"""scripts/gguf_headers_fetch.py, tools/gguf_headers/gguf_header.py (and the workflow gguf-headers.yml) without network.

A local HTTP server plays the Hugging Face hub: the revision and tree APIs (tree paginated by Link headers), a resolve
endpoint that redirects to a Range-capable "CDN" with X-Linked-Etag / X-Repo-Commit / X-Linked-Size like the hub's,
and the model search API. It serves GGUFs generated here: written byte by byte by this file (stdlib only; a split model,
a header with a 60 000-entry token array that needs many Range steps, a 256-byte alignment) and, when
LLAMA_CPP_GGUF_PY points at a llama.cpp gguf-py and numpy imports, the tinygen glm5next (UD dialect, MTP) and qwen4exp
models, whose data offsets and tensor offsets are compared with gguf-py's GGUFReader. The server counts every byte it
sends, so the tests see that the fetcher stops at the header, never passes the cap and fetches nothing in --plan.
The workflow check needs PyYAML (installed in ci.yml); without it, or without gguf-py / numpy, those classes skip.
N18: discover requests with named repos (Fetch), mimo2 support and separate MTP files (N18Analyzer: generated mimo2 /
MTP-only headers; fake llama.cpp source trees written from gguf_branches.MTP_LOAD / MTP_ATTACH stand in for the pins, so
the class needs no checkout), and BranchAnchors also checks the MTP_LOAD / MTP_ATTACH anchors at the real pins.
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
        self.wrong_total = False
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

    def add(self, repo, files, sha=SHA, gguf=None):
        self.repos[repo] = {"sha": sha, "files": dict(files), "gguf": gguf}

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
            return j({"id": m.group(1), "sha": r["sha"], "gated": False, "tags": ["gguf", "license:mit"],
                      **({"gguf": r["gguf"]} if r.get("gguf") else {})})
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
            total = len(b) + int(self.wrong_total)
            return h.send(206, b[a:z + 1], [("Content-Range", f"bytes {a}-{z}/{total}")])
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
        d = os.path.join(self.tmp, "req", name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "REQUEST.json")
        with open(p, "w") as f:
            json.dump(req, f)
        return p

    def run_fetch(self, *paths, plan=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = F.main(["--endpoint", self.hub.url, "--out", os.path.join(self.tmp, "out"),
                         *(["--plan"] if plan else []), *paths])
        return rc, out.getvalue()

    def dest(self, p):
        """where the results of request p go: <out>/<name>"""
        return os.path.join(self.tmp, "out", os.path.basename(os.path.dirname(p)))

    def load(self, p, name):
        with open(os.path.join(self.dest(p), name)) as f:
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

    def test_size_check_of_tensor_and_metadata_only_shards(self):
        b, off = make_gguf(arch_kv(), TENSORS)
        s = G.summarize(G.parse(b), b[:off])
        self.assertEqual(G.size_check(s, len(b)), "exact")
        self.assertTrue(G.size_check(s, len(b) + 32).startswith("mismatch"))
        # a metadata-only shard (llama-gguf-split --no-tensor-first-split): gguf.cpp's writer pads after the KV block,
        # its reader starts the data section without padding (the Unsloth first shards: 28 and 6 bytes)
        b0, off0 = make_gguf(arch_kv(extra=[("x.s", G.T_STRING, "abc")]), [])
        s0 = G.summarize(G.parse(b0), b0[:off0])
        pad = G.align_up(off0, 32) - off0
        self.assertGreater(pad, 0)
        self.assertEqual(G.size_check(s0, off0), "exact")
        self.assertEqual(G.size_check(s0, off0 + pad), "exact (no tensors: the writer's padding after the KV block)")
        self.assertTrue(G.size_check(s0, off0 + pad + 32).startswith("mismatch"))

    def test_tensor_offsets_must_be_contiguous_like_gguf_cpp(self):
        b, off = make_gguf(arch_kv(), TENSORS)
        h = G.parse(b)
        self.assertEqual(G.data_layout(h)[0], "contiguous")
        h.tensors[2]["offset"] += 32
        self.assertTrue(G.data_layout(h)[0].startswith("mismatch at blk.1.ffn_gate_exps.weight"))
        h.tensors[2]["offset"] -= 32
        h.tensors[3]["type"] = 99
        self.assertTrue(G.data_layout(h)[0].startswith("unknown type 99"))

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
        with open(p, "rb") as f:
            self.assertEqual((pin["request_path"], pin["request_sha256"]),
                             ("m/REQUEST.json", hashlib.sha256(f.read()).hexdigest()))
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
            with open(os.path.join(self.dest(p), rec["header_gz"]), "rb") as f:
                hdr = gzip.decompress(f.read())
            self.assertEqual(hdr, b[:off])
        self.assertGreaterEqual(len(pin["files"][0]["range_requests"]), 8)   # 4 KiB doubling to ~1 MiB
        self.assertEqual(self.hub.bytes_sent(), pin["bytes_fetched"])         # nothing read beyond what was recorded
        self.assertLess(pin["bytes_fetched"], sum(len(b) for b, _ in self.shards) // 2)
        self.assertNotIn("Q8_0", "".join(x[0] for x in self.hub.log if x[0].startswith("/org")))

    def test_summary_round_trips_from_the_committed_header(self):
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3, initial_bytes=65536)
        self.assertEqual(self.run_fetch(p)[0], 0)
        d = self.dest(p)
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
            with open(os.path.join(self.dest(p), rec["header_gz"]), "rb") as f, \
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
        self.assertFalse(os.path.exists(os.path.join(self.dest(p), "PIN.json")))
        self.assertIn("cap exceeded", self.load(p, "ERROR.json")["error"])
        self.assertLessEqual(self.hub.bytes_sent(), cap)
        self.assertEqual(sorted(os.listdir(self.dest(p))), ["ERROR.json"])

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
        self.assertFalse(os.path.exists(os.path.join(self.dest(p), "PIN.json")))

    def test_a_range_of_another_file_size_is_rejected(self):
        self.hub.wrong_total = True
        p = self.request("m", repo="org/M-GGUF", files=[self.GLOB], expect_count=3)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("does not match", self.load(p, "ERROR.json")["error"])

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
        self.assertFalse(os.path.exists(os.path.join(self.dest(p), "ERROR.json")))
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

    def test_discover_lists_named_repos_and_reads_no_file(self):
        """N18: a discover request may name repos; each is pinned like a headers request and its .gguf paths are
        listed with size and LFS sha256, before (and not again by) the searches; a missing repo is recorded."""
        sha2 = "89abcdef0123456789abcdef0123456789abcdef"
        self.hub.add("org/MiMo-V2.6-Flash-MOPD-GGUF", {"Q4_K/m-00001-of-00002.gguf": b"GGUF1",
                                                        "Q4_K/m-00002-of-00002.gguf": b"GGUF22",
                                                        "mtp-m-Q8_0.gguf": b"GGUF333", "README.md": b"x"},
                     sha=sha2, gguf={"architecture": "mimo2", "total": 7, "chat_template": "t"})
        self.hub.search = [{"id": "org/MiMo-V2.6-Flash-MOPD-GGUF", "sha": sha2, "gguf": {"architecture": "mimo2"}},
                           {"id": "org/M-GGUF", "sha": SHA, "gguf": {"architecture": "mimo2"}}]
        self.hub.status_missing = 404
        p = self.request("dn", kind="discover", list_files=True,
                         repos=[{"repo": "org/MiMo-V2.6-Flash-MOPD-GGUF", "revision": sha2},
                                {"repo": "org/M-GGUF"}, {"repo": "nobody/MiMo-GGUF", "revision": "main"}],
                         searches=[{"search": "GGUF", "architectures": ["mimo2"]}])
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        lst = self.load(p, "LISTING.json")
        a, b, c = lst["repos"]
        self.assertEqual((a["status"], a["revision"], a["gguf_architecture"]), ("ok", sha2, "mimo2"))
        self.assertEqual(a["gguf_chat_template_sha256"], hashlib.sha256(b"t").hexdigest())
        self.assertEqual(a["gguf_files"], [
            {"path": "Q4_K/m-00001-of-00002.gguf", "size": 5, "lfs_sha256": hashlib.sha256(b"GGUF1").hexdigest()},
            {"path": "Q4_K/m-00002-of-00002.gguf", "size": 6, "lfs_sha256": hashlib.sha256(b"GGUF22").hexdigest()},
            {"path": "mtp-m-Q8_0.gguf", "size": 7, "lfs_sha256": hashlib.sha256(b"GGUF333").hexdigest()}])
        self.assertEqual((b["status"], b["revision"], b["requested_revision"], len(b["gguf_files"])),
                         ("ok", SHA, "main", 4))
        self.assertEqual((c["status"], c["http_status"], c["revision"]), ("repo_unavailable", 404, None))
        # the search still reports both matches, but lists the files of neither again
        (s,) = lst["searches"]
        self.assertEqual([r["id"] for r in s["results"]], ["org/MiMo-V2.6-Flash-MOPD-GGUF", "org/M-GGUF"])
        self.assertTrue(all(r["arch_match"] and "gguf_files" not in r for r in s["results"]))
        self.assertEqual(sum(1 for x in self.hub.log if "/tree/" in x[0] and "MiMo" in x[0] and "cursor=" not in x[0]),
                         1)                                         # one listing (first pages only; pages of 3)
        self.assertEqual(self.hub.bytes_sent(), 0)
        self.assertFalse(any("/resolve/" in x[0] for x in self.hub.log))
        rc, out = self.run_fetch(self.request("dp", kind="discover", repos=[{"repo": "org/M-GGUF"}]), plan=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("discover, 0 searches, 1 named repos, reads no file", out)

    def test_discover_named_repo_revision_is_pinned_and_requests_are_checked(self):
        # malformed requests first, against an untouched hub (the hub logs a request after answering it, so a count
        # taken right after another fetch could miss its last entry)
        for bad in ({"kind": "discover"}, {"kind": "discover", "repos": []},
                    {"kind": "discover", "repos": [{"repo": "no-slash"}]},
                    {"kind": "discover", "repos": [{"repo": "a/b", "revision": "dev"}]},
                    {"kind": "discover", "repos": [{"repo": "a/b"}, {"repo": "a/b"}]},
                    {"kind": "discover", "repos": [{"repo": f"a/b{i}"} for i in range(F.MAX_LISTED_REPOS + 1)]}):
            rc, out = self.run_fetch(self.request("bad", **bad), plan=True)
            self.assertEqual(rc, 1, (bad, out))
            self.assertIn("::error::", out)
        self.assertEqual(self.hub.log, [])
        self.hub.add("org/M-GGUF", {"a.gguf": b"GGUF"})
        p = self.request("dx", kind="discover", repos=[{"repo": "org/M-GGUF", "revision": "f" * 40}])
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 1)
        self.assertIn("the hub answered", self.load(p, "ERROR.json")["error"])


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
            with open(os.path.join(self.dest(p), rec["summary"])) as f:
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
        self.assertEqual(on["push"]["paths"], ["data/gguf-headers/*/REQUEST.json"])
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


# ------------------------------------------------------------------------------------------------ analyzer

import importlib.util  # noqa: E402


def _load(name, path):
    """by path, under a unique name: `analyze` is also tools/g0/analyze.py (tests/test_g0_analyze.py), and unittest
    discover imports both test modules into one process"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("gguf_headers_analyze", os.path.join(ROOT, "tools", "gguf_headers", "analyze.py"))
import gguf_branches as B  # noqa: E402

GLM_TEMPLATE = os.path.join(ROOT, "third_party", "hf", "glm-5.3-flash-chat", "chat_template.jinja")


def _src_pins():
    """llama.cpp checkouts at the two pins, when available: the CI pin from LLAMA_CPP_GGUF_PY's checkout, the R1 pin
    from LLAMA_CPP_SRC_79E2E74. Only checkouts whose HEAD is the pin are returned."""
    want = {"ec7630a": ("ec7630a640789c393694fb194f1bbbf0369fc62d",
                        os.path.dirname(os.path.abspath(GGUF_PY)) if GGUF_PY else None),
            "79e2e74": ("79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6", os.environ.get("LLAMA_CPP_SRC_79E2E74"))}
    return {n: p for n, (sha, p) in want.items() if p and A.git_head(p) == sha}


class Analyzer(HubCase):
    def fetch_model(self, name, shards, analysis, n=None):
        files = {f"Q/{name}-0000{i + 1}-of-0000{len(shards)}.gguf": b for i, (b, _) in enumerate(shards)}
        self.hub.add(f"org/{name}", files)
        p = self.request(name, repo=f"org/{name}", files=[f"*{name}-*.gguf"], expect_count=len(shards),
                         analysis=analysis)
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)
        return self.dest(p)

    def analyze(self, *args):
        out = io.StringIO()
        js = os.path.join(self.tmp, "a.json")
        with contextlib.redirect_stdout(out):
            rc = A.main(["--root", os.path.join(self.tmp, "out"), "--json", js, *args])
        with open(js) as f:
            return rc, out.getvalue(), json.load(f)["results"]

    def glm_like(self, template, groups=(1, 1)):
        with open(GLM_TEMPLATE, encoding="utf-8") as f:
            tmpl = f.read() if template is None else template
        kv = arch_kv(extra=[("glm5next.expert_group_used_count", G.T_UINT32, groups[1]),
                            ("tokenizer.ggml.model", G.T_STRING, "gpt2"), ("tokenizer.ggml.pre", G.T_STRING, "glm4"),
                            ("tokenizer.ggml.tokens", G.T_ARRAY, (G.T_STRING, ["a", "b", "c"])),
                            ("tokenizer.ggml.eos_token_id", G.T_UINT32, 2),
                            ("tokenizer.chat_template", G.T_STRING, tmpl)])
        kv = [e if e[0] != "glm5next.expert_group_count" else (e[0], e[1], groups[0]) for e in kv]
        split = lambda i: [("split.no", G.T_UINT16, i), ("split.count", G.T_UINT16, 2),  # noqa: E731
                           ("split.tensors.count", G.T_INT32, len(TENSORS))]
        return [make_gguf(kv + split(0), TENSORS[:3], seed=4),
                make_gguf([("general.architecture", G.T_STRING, "glm5next")] + split(1), TENSORS[3:], seed=5)]

    def test_template_groups_identity_and_audit(self):
        an = {"graph": "glm5-next", "hf_chat_template": os.path.relpath(GLM_TEMPLATE, ROOT)}
        shards = self.glm_like(None)
        self.fetch_model("GLMa", shards, an)
        with open(GLM_TEMPLATE, encoding="utf-8") as f:
            changed = "{# changed #}" + f.read()
        self.fetch_model("GLMb", self.glm_like(changed, groups=(8, 4)), an)
        rc, out, res = self.analyze(*sum((["--src", f"{k}={v}"] for k, v in _src_pins().items()), []))
        self.assertEqual(rc, 0, out)
        a, b = res["GLMa"], res["GLMb"]
        self.assertEqual([ok for _, ok in a["integrity"]], [True, True])
        self.assertTrue(a["chat_template"]["byte_equal"])
        self.assertIn("BYTE-EQUAL", out)
        self.assertFalse(b["chat_template"]["byte_equal"])
        self.assertEqual(b["chat_template"]["first_difference_at_byte"], 0)
        self.assertEqual((a["groups"]["expert_group_count"], a["groups"]["expert_group_used_count"]), (1, 1))
        self.assertIn("no group step", a["groups"]["llama.cpp"])
        self.assertIn("group-limited", b["groups"]["llama.cpp"])
        ident = a["identity"]["identity"]
        self.assertEqual((ident["architecture"], ident["tokenizer.ggml.pre"], ident["vocab (tokenizer.ggml.tokens)"]),
                         ("glm5next", "glm4", 3))
        # the audit of the rebuilt header-only shards equals the audit of the full files
        d = os.path.join(self.tmp, "full")
        os.makedirs(d)
        for i, (bb, _) in enumerate(shards):
            with open(os.path.join(d, f"GLMa-0000{i + 1}-of-00002.gguf"), "wb") as f:
                f.write(bb)
        full = QA.audit(os.path.join(d, "GLMa-00001-of-00002.gguf"))
        self.assertEqual((a["audit"]["verdict"], a["audit"]["classes"]), (full["verdict"], json.loads(
            json.dumps(full["classes"]))))
        self.assertEqual(a["audit"]["lowest_bpw_matrices"][0]["type"], "Q4_K")
        for pin in _src_pins():
            sup = a["identity"]["support"][pin]
            self.assertEqual(sup["arch glm5next"], "not found")          # neither pin knows glm5next
            self.assertRegex(sup["pre-tokenizer glm4"], r"^src/llama-vocab\.cpp:\d+")

    def test_integrity_failure_is_reported(self):
        d = self.fetch_model("GLMc", self.glm_like(None), {})
        s = os.path.join(d, "GLMc-00001-of-00002.gguf.summary.json")
        with open(s) as f:
            txt = f.read()
        with open(s, "w") as f:
            f.write(txt.replace('"glm5next"', '"glm5-next"', 1))
        rc, out, res = self.analyze()
        self.assertEqual(rc, 1)
        self.assertEqual([ok for _, ok in res["GLMc"]["integrity"]], [False, True])

    def test_pending_and_unavailable_requests(self):
        p = self.request("gone", repo="nobody/X", files=["*.gguf"], expect_count=1)
        self.assertEqual(self.run_fetch(p)[0], 0)
        os.makedirs(os.path.join(self.tmp, "out", "later"))
        rc, out, res = self.analyze()
        self.assertEqual(rc, 0)
        self.assertEqual(res["gone"]["status"], "repo_unavailable")
        self.assertTrue(res["later"]["status"].startswith("PENDING"))

    @unittest.skipUnless(GGUF_PY and HAVE_NUMPY, "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py (and install numpy)")
    def test_tiny_models_exercise_their_own_branches(self):
        sys.path.insert(0, GGUF_PY)
        sys.path.insert(0, os.path.join(ROOT, "tools", "tinygen"))
        glm, qwen = importlib.import_module("glm5next_tiny"), importlib.import_module("qwen4exp_tiny")
        paths = {}
        for name, mod, kw, arch in (("glm", glm, {"mtp": True, "dialect": "ik"}, "glm5next"),
                                    ("qwen", qwen, {"mtp": True}, None)):
            paths[name] = os.path.join(self.tmp, f"{name}.gguf")
            saved = mod.ARCH
            mod.ARCH = arch or saved
            try:
                mod.build(paths[name], cfg=dict(mod.DEFAULT, **({"n_layer": 12} if name == "glm" else {})), **kw)
            finally:
                mod.ARCH = saved
            with open(paths[name], "rb") as f:
                self.fetch_model(name, [(f.read(), None)], {"graph": "glm5-next" if name == "glm" else "qwen4exp",
                                                            "tiny": "glm5next" if name == "glm" else "qwen4exp"})
        rc, out, res = self.analyze(*sum((["--src", f"{k}={v}"] for k, v in _src_pins().items()), []))
        self.assertEqual(rc, 0, out)
        for name, graph in (("glm", "glm5-next"), ("qwen", "qwen4exp")):
            br = res[name]["branches"]
            self.assertEqual({b["id"] for b in br}, {p.id for p in B.PROBES if graph in p.graphs})
            self.assertEqual([b["id"] for b in br if not b["exercised_by_tiny"]], [], out)
            keys = {row["key"] for row in res[name]["hparams"]}
            self.assertIn("block_count", keys)
            self.assertIsNone(res[name]["tiny_note"])
        self.assertIn("tiny OK", out)


# ------------------------------------------------------------------------------------------------ N18: mimo2, MTP files

def mimo2_kv(extra=(), drop=(), nextn=2, vocab=96):
    """a MiMo-V2.x-like split-shard KV block in miniature: 4 trunk layers (full at 0 and 3), nextn MTP layers (SWA)"""
    pat = [0, 1, 1, 0] + [1] * nextn
    kv = [("general.architecture", G.T_STRING, "mimo2"), ("mimo2.context_length", G.T_UINT32, 512),
          ("mimo2.embedding_length", G.T_UINT32, 64), ("mimo2.block_count", G.T_UINT32, 4 + nextn),
          ("mimo2.feed_forward_length", G.T_UINT32, 96), ("mimo2.attention.head_count", G.T_UINT32, 4),
          ("mimo2.attention.head_count_kv", G.T_ARRAY, (G.T_UINT32, [2 if x else 1 for x in pat])),
          ("mimo2.attention.key_length", G.T_UINT32, 24), ("mimo2.attention.value_length", G.T_UINT32, 16),
          ("mimo2.attention.layer_norm_rms_epsilon", G.T_FLOAT32, 1e-6), ("mimo2.rope.dimension_count", G.T_UINT32, 8),
          ("mimo2.rope.freq_base", G.T_FLOAT32, 1e7), ("mimo2.rope.freq_base_swa", G.T_FLOAT32, 1e4),
          ("mimo2.attention.value_scale", G.T_FLOAT32, 0.707), ("mimo2.attention.sliding_window", G.T_UINT32, 8),
          ("mimo2.attention.sliding_window_pattern", G.T_ARRAY, (G.T_BOOL, [bool(x) for x in pat])),
          ("mimo2.expert_count", G.T_UINT32, 8), ("mimo2.expert_used_count", G.T_UINT32, 2),
          ("mimo2.expert_feed_forward_length", G.T_UINT32, 32),
          ("tokenizer.ggml.model", G.T_STRING, "gpt2"), ("tokenizer.ggml.pre", G.T_STRING, "qwen2"),
          ("tokenizer.ggml.tokens", G.T_ARRAY, (G.T_STRING, [f"t{i}" for i in range(vocab)])), *extra]
    if nextn:
        kv.append(("mimo2.nextn_predict_layers", G.T_UINT32, nextn))
    return [e for e in kv if e[0] not in drop]


def mimo2_layer(i, swa, mtp=False, moe=True):
    """tensors of one mimo2 layer: fused qkv, sinks on SWA layers only, dense FFN on layer 0 and on MTP layers, MXFP4
    experts elsewhere; an MTP layer adds nextn.eh_proj / enorm / hnorm and layer_output_norm"""
    b, kv = f"blk.{i}", 2 if swa else 1
    t = [(f"{b}.attn_norm.weight", [64], "F32"), (f"{b}.ffn_norm.weight", [64], "F32"),
         (f"{b}.attn_qkv.weight", [64, 4 * 24 + kv * 24 + kv * 16], "Q8_0"), (f"{b}.attn_output.weight", [64, 64], "Q8_0")]
    if swa:
        t.append((f"{b}.attn_sinks.weight", [4], "F32"))
    if moe and not mtp:
        t += [(f"{b}.ffn_gate_inp.weight", [64, 8], "F32"), (f"{b}.exp_probs_b.bias", [8], "F32"),
              (f"{b}.ffn_gate_exps.weight", [64, 32, 8], "MXFP4"), (f"{b}.ffn_up_exps.weight", [64, 32, 8], "MXFP4"),
              (f"{b}.ffn_down_exps.weight", [32, 64, 8], "MXFP4")]
    else:
        t += [(f"{b}.ffn_gate.weight", [64, 96], "Q8_0"), (f"{b}.ffn_up.weight", [64, 96], "Q8_0"),
              (f"{b}.ffn_down.weight", [96, 64], "Q8_0")]
    if mtp:
        t += [(f"{b}.nextn.eh_proj.weight", [128, 64], "Q8_0"), (f"{b}.nextn.enorm.weight", [64], "F32"),
              (f"{b}.nextn.hnorm.weight", [64], "F32"), (f"{b}.layer_output_norm.weight", [64], "F32")]
    return t


def mimo2_tensors(nextn=2, trunk=True, mtp=True):
    g = [("token_embd.weight", [64, 96], "Q8_0"), ("output_norm.weight", [64], "F32"), ("output.weight", [64, 96], "Q8_0")]
    t = []
    if trunk:
        for i, swa in enumerate([0, 1, 1, 0]):
            t += mimo2_layer(i, swa, moe=i > 0)
    if mtp:
        for i in range(4, 4 + nextn):
            t += mimo2_layer(i, 1, mtp=True)
    return g + t


MIMO2_CONFIG = {"num_hidden_layers": 4, "num_nextn_predict_layers": 2, "hidden_size": 64, "intermediate_size": 96,
                "moe_intermediate_size": 32, "n_routed_experts": 8, "num_experts_per_tok": 2, "num_attention_heads": 4,
                "num_key_value_heads": 1, "swa_num_key_value_heads": 2, "hybrid_layer_pattern": [0, 1, 1, 0],
                "head_dim": 24, "v_head_dim": 16, "sliding_window": 8, "partial_rotary_factor": 0.334,
                "rope_theta": 10000000.0, "swa_rope_theta": 10000.0, "attention_value_scale": 0.707,
                "layernorm_epsilon": 1e-06, "max_position_embeddings": 512, "n_group": 1, "topk_group": 1,
                "vocab_size": 96}

# the load_arch_hparams of src/models/mimo2.cpp at ec7630a (lines 3-23; the same at 79e2e74), for a fake source tree
MIMO2_HPARAMS_SRC = """void llama_model_mimo2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa, false);

    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    float value_scale = 0.0f;
    if (ml.get_key(LLM_KV_ATTENTION_VALUE_SCALE, value_scale, false) && value_scale != 1.0f) {
        hparams.f_attn_value_scale = value_scale;
    }
}

void llama_model_mimo2::load_arch_tensors(llama_model_loader & ml) {
}
"""
GLM_HPARAMS_SRC = """void llama_model_glm5_next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
}

void llama_model_glm5_next::load_arch_tensors(llama_model_loader & ml) {
}
"""
FAKE_KV_NAMES = ["CONTEXT_LENGTH:context_length", "EMBEDDING_LENGTH:embedding_length", "BLOCK_COUNT:block_count",
                 "NEXTN_PREDICT_LAYERS:nextn_predict_layers",
                 "ATTENTION_LAYERNORM_RMS_EPS:attention.layer_norm_rms_epsilon",
                 "ATTENTION_LAYERNORM_EPS:attention.layer_norm_epsilon",
                 "EXPERT_FEED_FORWARD_LENGTH:expert_feed_forward_length",
                 "ATTENTION_SLIDING_WINDOW:attention.sliding_window", "ROPE_FREQ_BASE_SWA:rope.freq_base_swa",
                 "ATTENTION_SLIDING_WINDOW_PATTERN:attention.sliding_window_pattern",
                 "ATTENTION_VALUE_SCALE:attention.value_scale", "CLASSIFIER_ACTIVATION:classifier.activation"]


def fake_src(root, pin):
    """a llama.cpp-like source tree: the hparams functions above, the generic load_hparams in miniature, and every
    MTP_LOAD / MTP_ATTACH anchor whose pins include `pin`, one per line, in its file"""
    files = {"src/llama-arch.cpp": "".join(f'    {{ LLM_KV_{k}, "%s.{v}" }},\n' for k, v in
                                           (x.split(":") for x in FAKE_KV_NAMES)),
             "src/llama-model.cpp": "void llama_model_base::load_hparams(llama_model_loader & ml) {\n"
                                    "    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);\n"
                                    "    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);\n"
                                    "    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);\n"
                                    "    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,    hparams.n_layer_nextn,   false);\n"
                                    # an optional call in an if condition followed by a statement that ends in ");"
                                    # (src/llama-model.cpp:1350-1351@79e2e74): not a required key
                                    "    if (ml.get_key(LLM_KV_CLASSIFIER_ACTIVATION, act_cls, false)) {\n"
                                    "        const auto it = LLM_CLS_ACT_TYPES_FROM_STRING.find(act_cls);\n    }\n"
                                    "    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {\n"
                                    "        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);\n    }\n"
                                    "    load_arch_hparams(ml);\n}\n",
             "src/models/mimo2.cpp": MIMO2_HPARAMS_SRC, "src/models/glm5-next.cpp": GLM_HPARAMS_SRC,
             "src/llama-vocab.cpp": "        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);\n"}
    anchors = [a for r in B.MTP_LOAD.values() for k in ("detect", "no_detect", "no_graph", "graph", "fallbacks")
               for a in r.get(k, [])] + [a for r in B.MTP_LOAD.values() for _, a in r["globals"]] + \
        [e[:3] for e in B.MTP_ATTACH]
    for rel, needle, pins in anchors:
        if pin in pins:
            files[rel] = files.get(rel, "") + needle + "\n"
    for rel, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            f.write(text)
    return root


class N18Analyzer(HubCase):
    """mimo2 support and separate MTP files (N18) on generated headers; fake source trees stand in for the pins"""
    fetch_model = Analyzer.fetch_model          # the helpers only (inheriting Analyzer would run its tests twice)
    analyze = Analyzer.analyze

    def srcs(self):
        return {p: fake_src(os.path.join(self.tmp, "src-" + p), p) for p in ("ec7630a", "79e2e74")}

    def test_mimo2_config_branches_keys_experts_audit(self):
        cfg = os.path.join(self.tmp, "mimo2-config.json")
        with open(cfg, "w") as f:
            json.dump({"text_config": MIMO2_CONFIG}, f)
        an = {"graph": "mimo2", "hf_config": cfg}
        t = mimo2_tensors()
        split = lambda i: [("split.no", G.T_UINT16, i), ("split.count", G.T_UINT16, 2),  # noqa: E731
                           ("split.tensors.count", G.T_INT32, len(t))]
        self.fetch_model("MiMoA", [make_gguf(mimo2_kv() + split(0), [], seed=6),
                                   make_gguf([("general.architecture", G.T_STRING, "mimo2")] + split(1), t, seed=7)], an)
        self.fetch_model("MiMoB", [make_gguf(mimo2_kv(drop=("mimo2.attention.sliding_window",)), t, seed=8)], an)
        srcs = self.srcs()
        rc, out, res = self.analyze(*sum((["--src", f"{k}={v}"] for k, v in srcs.items()), []))
        self.assertEqual(rc, 0, out)
        a = res["MiMoA"]
        self.assertEqual([ok for _, ok in a["integrity"]], [True, True])
        rows = {(c["config"], c["gguf"]): c for c in a["config"]}
        self.assertEqual({k: c["match"] for k, c in rows.items() if c["match"] is not True},
                         {("n_group", "expert_group_count"): None, ("topk_group", "expert_group_used_count"): None})
        self.assertEqual(rows[("hybrid_layer_pattern", "attention.head_count_kv")]["config_value_compared"],
                         [1, 2, 2, 1, 2, 2])                          # + SWA per MTP layer of this file
        self.assertEqual(rows[("hybrid_layer_pattern", "attention.sliding_window_pattern")]["config_value_compared"],
                         [0, 1, 1, 0, 1, 1])
        self.assertEqual(rows[("partial_rotary_factor", "rope.dimension_count")]["config_value_compared"], 8)
        self.assertEqual(rows[("num_hidden_layers", "layers")]["gguf_value"], 4)
        br = {b["id"]: b for b in a["branches"]}
        self.assertEqual({p.id for p in B.PROBES if "mimo2" in p.graphs}, set(br))
        self.assertEqual(br["mimo.sinks"]["real"], ["SWA layers with sinks", "full layers without sinks"])
        self.assertIn("trunk layers with attn_sinks: 1-2 (2 of 4", br["mimo.sinks"]["real_detail"])
        self.assertIn("MTP layers with attn_sinks: [4, 5]", br["mimo.sinks"]["real_detail"])
        self.assertIn("full-attention trunk layers [0, 3]", br["mimo.swa_pattern"]["real_detail"])
        self.assertEqual(br["mimo.kv_heads"]["real"], ["per-layer KV heads, full != SWA"])
        self.assertEqual(br["mimo.head_dims"]["real"], ["K head dim != V head dim"])
        self.assertEqual(br["mimo.rope"]["real"], ["own SWA freq base (rope.freq_base_swa)",
                                                   "partial rotary (n_rot < K head dim)"])
        self.assertEqual(br["mimo.value_scale"]["real"], ["attn_out x value_scale"])
        self.assertEqual(br["mimo.qkv"]["real"], ["fused attn_qkv"])
        self.assertEqual(br["mimo.ffn"]["real"], ["MoE (sigmoid + exp_probs_b, renormalised top-k)", "dense FFN"])
        self.assertEqual(br["mimo.mtp"]["real"], ["nextn_predict_layers 2"])
        self.assertIn("blk.4: 12 tensors, nextn ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight']",
                      br["mimo.mtp"]["real_detail"])
        self.assertEqual(br["mimo.mtp_block"]["real"], ["MTP layer with a dense FFN", "MTP layer with fused attn_qkv"])
        self.assertEqual(br["mimo.mtp_head"]["real"], ["head norm layer_output_norm", "trunk output head",
                                                       "trunk token_embd"])
        self.assertEqual(br["mimo.file_layout"]["real"], ["trunk + MTP in one file"])
        self.assertEqual(a["experts"], {p: {"MXFP4": 3} for p in ("ffn_down_exps", "ffn_gate_exps", "ffn_up_exps")})
        self.assertEqual(a["audit"]["verdict"], "PASS")
        self.assertAlmostEqual(a["audit"]["classes"]["routed_experts"]["bpw"], 4.25)
        for pin in srcs:
            k = a["required_keys"][pin]
            self.assertEqual(sorted(x["key"] for x in k["rows"]), sorted(
                ["mimo2.context_length", "mimo2.embedding_length", "mimo2.block_count",
                 "mimo2.attention.layer_norm_rms_epsilon", "mimo2.expert_feed_forward_length",
                 "mimo2.attention.sliding_window", "mimo2.attention.sliding_window_pattern", "tokenizer.ggml.model"]))
            self.assertEqual(k["missing"], [])
            self.assertEqual(res["MiMoB"]["required_keys"][pin]["missing"], ["mimo2.attention.sliding_window"])
        self.assertIn("keys read without a default", out)
        self.assertIn('{"ffn_down_exps": {"MXFP4": 3}', out)
        self.assertIn("(c) chat template vs None: no embedded tokenizer.chat_template", out)
        tm = A.Model([G.summarize(G.parse(h), h) for h in [make_gguf(
            mimo2_kv(extra=[("tokenizer.chat_template", G.T_STRING, "{{ x }}")]), t)[0]]])
        self.assertEqual(A.section_template(tm, None), {"vendored": None, "embedded_present": True, "embedded_bytes": 7,
                                                        "embedded_sha256": hashlib.sha256(b"{{ x }}").hexdigest()})
        # the other side of the labels: NextN layer 5 without fused qkv and without ffn_gate, no value scale
        b, off = make_gguf(mimo2_kv(drop=("mimo2.attention.value_scale",)),
                           [x for x in t if x[0] not in ("blk.5.attn_qkv.weight", "blk.5.ffn_gate.weight")])
        mc = A.Model([G.summarize(G.parse(b[:off]), b[:off])])
        brc = {x["id"]: x["real"] for x in A.section_branches(mc, "mimo2", None, {})}
        self.assertEqual(brc["mimo.mtp_block"], ["MTP layer with a dense FFN", "MTP layer with fused attn_qkv",
                                                 "MTP layer without a dense FFN (graph_mtp asserts)",
                                                 "MTP layer without fused attn_qkv (graph_mtp asserts)"])
        self.assertEqual(brc["mimo.value_scale"], ["no value scale"])
        # an MTP-only file: the probes of trunk layers see no layers (not "SWA layers without sinks")
        b, off = make_gguf(mimo2_kv(), mimo2_tensors(trunk=False))
        mm = A.Model([G.summarize(G.parse(b[:off]), b[:off])])
        brm = {x["id"]: (x["real"], x["real_detail"]) for x in A.section_branches(mm, "mimo2", None, {})}
        self.assertEqual(brm["mimo.sinks"][0], [])
        self.assertIn("(0 of 0;", brm["mimo.sinks"][1])
        self.assertIn("MTP layers with attn_sinks: [4, 5]", brm["mimo.sinks"][1])
        self.assertEqual((brm["mimo.qkv"][0], brm["mimo.ffn"][0]), ([], []))
        self.assertEqual(brm["mimo.file_layout"][0], ["MTP-only file"])

    @unittest.skipUnless(GGUF_PY and HAVE_NUMPY, "set LLAMA_CPP_GGUF_PY=<llama.cpp>/gguf-py (and install numpy)")
    def test_mimo2_tiny_exercises_its_own_branches(self):
        tm, note = A.build_tiny("mimo2", GGUF_PY, os.path.join(self.tmp, "tiny"))
        self.assertIn("default", tm)
        takes_mtp = "mtp" in A.inspect.signature(importlib.import_module("mimo2_tiny").build).parameters
        self.assertEqual(note is None, takes_mtp, note)                   # the N16 generator builds both variants
        for lab, t in tm.items():
            br = A.section_branches(t, "mimo2", tm, {})
            self.assertEqual([b["id"] for b in br if not b["exercised_by_tiny"]], [], lab)
        d = {b["id"]: b["real"] for b in A.section_branches(tm["default"], "mimo2", None, {})}
        self.assertEqual(d["mimo.sinks"], ["SWA layers with sinks", "full layers with sinks"])
        self.assertEqual(d["mimo.qkv"], ["split attn_q/k/v"])

    def mtp_case(self):
        """GLM-like separate MTP files: a trunk-only request, a reference trunk + MTP request, and the MTP files"""
        base = [("general.architecture", G.T_STRING, "glm5-next"), ("glm5-next.context_length", G.T_UINT32, 512),
                ("glm5-next.embedding_length", G.T_UINT32, 64),
                ("glm5-next.attention.layer_norm_rms_epsilon", G.T_FLOAT32, 1e-5),
                ("tokenizer.ggml.model", G.T_STRING, "gpt2"),
                ("tokenizer.ggml.tokens", G.T_ARRAY, (G.T_STRING, ["a", "b", "c"]))]
        nx = [("glm5-next.block_count", G.T_UINT32, 4), ("glm5-next.nextn_predict_layers", G.T_UINT32, 1)]
        glob = [("token_embd.weight", [64, 3], "Q8_0"), ("output_norm.weight", [64], "F32"),
                ("output.weight", [64, 3], "Q8_0")]
        layer = lambda i: [(f"blk.{i}.attn_norm.weight", [64], "F32"), (f"blk.{i}.ffn_norm.weight", [64], "F32"),  # noqa
                           (f"blk.{i}.attn_q_a.weight", [64, 32], "Q8_0")]
        mtp = [("blk.3.attn_norm.weight", [64], "F32"), ("blk.3.ffn_norm.weight", [64], "F32"),
               ("blk.3.attn_q_a.weight", [64, 32], "Q8_0"), ("blk.3.nextn.eh_proj.weight", [128, 64], "Q4_0"),
               ("blk.3.nextn.enorm.weight", [64], "F32"), ("blk.3.nextn.hnorm.weight", [64], "F32")]
        trunk = glob + layer(0) + layer(1) + layer(2)
        self.fetch_model("trunk", [make_gguf(base + [("glm5-next.block_count", G.T_UINT32, 3)], trunk, seed=9)], {})
        self.fetch_model("ref", [make_gguf(base + nx, trunk + mtp + [("blk.3.nextn.shared_head_norm.weight", [64],
                                                                      "F32")], seed=10)], {})
        files = {"MTP/mtp-ok.gguf": make_gguf(base + nx, glob + mtp, seed=11)[0],
                 "MTP/mtp-shared.gguf": make_gguf(base + nx, [glob[1]] + mtp, seed=12)[0],
                 "MTP/mtp-nohnorm.gguf": make_gguf(base + nx, glob + mtp[:-1], seed=13)[0],
                 "MTP/mtp-extra.gguf": make_gguf(base + nx, glob + mtp + [("blk.3.foo.weight", [64], "F32")], seed=14)[0],
                 "MTP/mtp-novocab.gguf": make_gguf(base[:4] + nx, glob + mtp, seed=15)[0]}
        self.hub.add("org/GLM-MTP", files)
        p = self.request("mtp", repo="org/GLM-MTP", files=["MTP/mtp-*.gguf"], expect_count=len(files),
                         analysis={"mode": "mtp_files", "graph": "glm5-next", "trunk": "trunk",
                                   "mtp_reference": {"request": "ref"}})
        rc, out = self.run_fetch(p)
        self.assertEqual(rc, 0, out)

    def test_separate_mtp_files_loader_verdicts_per_pin(self):
        self.mtp_case()
        srcs = self.srcs()
        rc, out, res = self.analyze("--name", "mtp", *sum((["--src", f"{k}={v}"] for k, v in srcs.items()), []))
        self.assertEqual(rc, 0, out)
        r = res["mtp"]
        self.assertEqual((r["mode"], r["trunk"], r["mtp_reference"]), ("mtp_files", "trunk", "request ref, blk.3"))
        f = {os.path.basename(x["path"]): x for x in r["mtp_files"]}
        ok = f["mtp-ok.gguf"]
        self.assertEqual((ok["layout"], ok["n_layer"], ok["layers_present"]), ("MTP-only file", 3, [3]))
        self.assertEqual(ok["reference"], {"3": {"missing": [], "missing_optional": ["nextn.shared_head_norm.weight"],
                                                 "extra": [], "alternatives": []}})      # JSON: layer keys are strings
        self.assertEqual(ok["trunk"]["keys_differ"], [["block_count", 3, 4], ["nextn_predict_layers", "(absent)", 1]])
        self.assertTrue(ok["trunk"]["tokens_equal"])
        lo = {n: {pin: v for pin, v in x["loader"].items()} for n, x in f.items()}
        # the ec7630a-like tree: no MTP-only handling and no MTP graph for glm5-next
        for n in f:
            self.assertEqual(lo[n]["ec7630a"]["verdict"], "NOT loadable as a separate MTP model", n)
        self.assertTrue(any("no MTP-only handling" in h for h in lo["mtp-ok.gguf"]["ec7630a"]["hard"]))
        self.assertTrue(any("not implemented" in h for h in lo["mtp-ok.gguf"]["ec7630a"]["hard"]))
        # the 79e2e74-like tree: nextn_flags + graph_mtp
        v = lo["mtp-ok.gguf"]["79e2e74"]
        self.assertEqual((v["verdict"], v["hard"], v["check"]),
                         ("loadable as a separate MTP model (PROVISIONAL: by reading, nothing loaded)", [], []))
        self.assertEqual([h.split(" absent")[0] for h in lo["mtp-shared.gguf"]["79e2e74"]["hard"]],
                         ["token_embd.weight", "output.weight"])
        self.assertEqual(lo["mtp-nohnorm.gguf"]["79e2e74"]["hard"],
                         ["blk.3 lacks ['nextn.hnorm.weight'] (present in the reference NextN layer, not optional)"])
        self.assertEqual(lo["mtp-extra.gguf"]["79e2e74"]["verdict"], "CHECK")
        self.assertIn("tokenizer.ggml.model absent", " ".join(lo["mtp-novocab.gguf"]["79e2e74"]["hard"]))
        self.assertEqual(f["mtp-novocab.gguf"]["required_keys"]["79e2e74"]["missing"], ["tokenizer.ggml.model"])
        self.assertEqual(f["mtp-ok.gguf"]["audit"]["verdict"], "PASS")
        # every attach anchor of the 79e2e74-like tree is found and printed
        self.assertTrue(all(x["at"] != "not found" for x in r["attach"]["79e2e74"] if x["expected"]))
        self.assertIn("loader [79e2e74]: loadable as a separate MTP model", out)
        self.assertIn("how llama.cpp attaches a separate MTP file:", out)

    def test_reference_diff_takes_fused_and_split_forms_as_equal(self):
        ref = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate_up_exps.weight", "nextn.enorm.weight",
               "nextn.shared_head_norm.weight"}
        have = {"attn_qkv.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "nextn.hnorm.weight"}
        d = A.ref_diff(ref, have, {"nextn.shared_head_norm.weight"})
        self.assertEqual((d["missing"], d["missing_optional"], d["extra"]),
                         (["nextn.enorm.weight"], ["nextn.shared_head_norm.weight"], ["nextn.hnorm.weight"]))
        self.assertEqual(len(d["alternatives"]), 2)
        d = A.ref_diff(ref, {"attn_q.weight"}, set())                              # a partial split form: missing
        self.assertEqual([x for x in d["missing"] if x.startswith("attn_")], ["attn_k.weight", "attn_v.weight"])
        d = A.ref_diff(ref, {"attn_qkv.weight", "attn_q.weight"}, set())          # fused + a stray split one: extra
        self.assertEqual(d["extra"], ["attn_q.weight"])

    def test_listing_renders_named_repos(self):
        self.hub.add("org/M-GGUF", {"Q/m-00001-of-00002.gguf": b"GGUF1", "Q/m-00002-of-00002.gguf": b"GGUF22"},
                     gguf={"architecture": "mimo2"})
        p = self.request("dl", kind="discover", repos=[{"repo": "org/M-GGUF"}, {"repo": "nobody/X"}])
        self.assertEqual(self.run_fetch(p)[0], 0)
        rc, out, res = self.analyze("--name", "dl")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"named repo org/M-GGUF @ {SHA} (requested main): ok, arch=mimo2", out)
        self.assertIn("    Q/m-00002-of-00002.gguf  6 B", out)
        self.assertIn("named repo nobody/X @ None (requested main): repo_unavailable, HTTP 404", out)


import vocab_gguf as V  # noqa: E402

TOKENIZE_BIN = os.environ.get("LLAMA_TOKENIZE_BIN")


def _bytes_to_unicode():
    """GPT-2's byte -> printable character map (the byte-level BPE alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def bpe_header(arch="glm5next", pre="glm4"):
    """A GLM-like split shard header with a byte-level BPE vocabulary: <unk> <s> </s>, the 256 bytes, and one merge
    "h e" -> "he" (id 259), so llama.cpp tokenizes "hehe" to [259, 259]."""
    b2u = _bytes_to_unicode()
    toks = ["<unk>", "<s>", "</s>"] + [b2u[b] for b in range(256)] + ["he"]
    kv = arch_kv(arch, extra=[("tokenizer.ggml.model", G.T_STRING, "gpt2"), ("tokenizer.ggml.pre", G.T_STRING, pre),
                              ("tokenizer.ggml.tokens", G.T_ARRAY, (G.T_STRING, toks)),
                              ("tokenizer.ggml.token_type", G.T_ARRAY, (G.T_INT32, [3, 3, 3] + [1] * 257)),
                              ("tokenizer.ggml.merges", G.T_ARRAY, (G.T_STRING, ["h e"])),
                              ("tokenizer.ggml.bos_token_id", G.T_UINT32, 1),
                              ("tokenizer.ggml.eos_token_id", G.T_UINT32, 2),
                              ("tokenizer.ggml.add_bos_token", G.T_BOOL, False),
                              ("split.no", G.T_UINT16, 0), ("split.count", G.T_UINT16, 6),
                              ("split.tensors.count", G.T_INT32, 3)])
    b, off = make_gguf(kv, TENSORS[:3])
    return b[:off]


class VocabOnly(unittest.TestCase):
    def test_vocab_only_file_keeps_the_tokenizer_bytes_and_renames_the_arch(self):
        hdr = bpe_header()
        h = G.parse(hdr)
        out = V.build(hdr, arch="glm5-next")
        h2 = G.parse(out)
        self.assertEqual((h2.n_tensors, h2.data_offset), (0, len(out)))
        self.assertEqual(h2.get("general.architecture"), "glm5-next")
        self.assertEqual(h2.get("glm5-next.block_count"), 4)
        self.assertEqual([e["key"] for e in h2.kv if e["key"].startswith("split.")], [])
        for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.pre", "tokenizer.ggml.token_type"):
            s1, t1 = next(e["span"] for e in h.kv if e["key"] == k)
            s2, t2 = next(e["span"] for e in h2.kv if e["key"] == k)
            self.assertEqual(hdr[s1:t1], out[s2:t2], k)
        self.assertEqual(G.parse(V.build(hdr)).get("general.architecture"), "glm5next")

    @unittest.skipUnless(TOKENIZE_BIN, "set LLAMA_TOKENIZE_BIN=<llama.cpp build>/bin/llama-tokenize")
    def test_llama_tokenize_runs_on_the_vocab_only_file(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            for arch, ok in (("glm5-next", True), (None, False)):
                p = os.path.join(d, f"v-{arch}.gguf")
                with open(p, "wb") as f:
                    f.write(V.build(bpe_header(), arch=arch))
                r = subprocess.run([TOKENIZE_BIN, "-m", p, "-p", "hehe", "--ids", "--log-disable"], cwd=d,
                                   capture_output=True, text=True, timeout=120)
                if ok:
                    self.assertEqual((r.returncode, r.stdout.strip()), (0, "[259, 259]"), r.stderr[-2000:])
                else:
                    self.assertNotEqual(r.returncode, 0)       # the pins do not know "glm5next"


@unittest.skipUnless(_src_pins(), "llama.cpp sources at the pins: LLAMA_CPP_GGUF_PY's checkout at ec7630a and/or "
                                  "LLAMA_CPP_SRC_79E2E74")
class BranchAnchors(unittest.TestCase):
    def test_every_anchor_is_where_its_pins_say(self):
        bad = []
        load = [(f"MTP_LOAD.{arch}.{k}", a) for arch, r in B.MTP_LOAD.items()
                for k in ("detect", "no_detect", "no_graph", "graph", "fallbacks") for a in r.get(k, [])] + \
            [(f"MTP_LOAD.{arch}.{name}", a) for arch, r in B.MTP_LOAD.items() for name, a in r["globals"]] + \
            [(f"MTP_LOAD.{arch}.ple_table", r["ple_table"]) for arch, r in B.MTP_LOAD.items() if "ple_table" in r] + \
            [("MTP_ATTACH", e[:3]) for e in B.MTP_ATTACH]
        for pin, root in _src_pins().items():
            for pid, (rel, needle, pins) in [(p.id, a) for p in B.PROBES for a in p.anchors] + load:
                hits = A.locate(root, rel, needle)
                if bool(hits) != (pin in pins):
                    bad.append((pid, pin, rel, needle, hits))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
