"""tools/gateway: OpenAI-compatible gateway tested end-to-end against the stdlib mock server.

No GPU and no network: the mock (tools/bench/mock_server.py) and the gateway run on 127.0.0.1
with ephemeral ports in threads. The mock is extended here (subclass, not edited) to record the
raw request bytes it receives, to hold a generation until the test releases it (a deterministic
stand-in for a long prefill), to block mid-stream, to notice when its client (the gateway)
closes the upstream connection, and to answer llama.cpp's POST /apply-template.

These tests say nothing about any real engine; they check the gateway's own logic.
"""
import http.client
import json
import logging
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import shutil
import threading
import time
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "tools", "bench")
GWDIR = os.path.join(ROOT, "tools", "gateway")
sys.path.insert(0, BENCH)
sys.path.insert(0, GWDIR)
import mock_server  # noqa: E402
import gateway  # noqa: E402

assert os.path.dirname(os.path.abspath(mock_server.__file__)) == BENCH, mock_server.__file__
assert os.path.dirname(os.path.abspath(gateway.__file__)) == GWDIR, gateway.__file__

CHAT = "/v1/chat/completions"
COMP = "/v1/completions"
WAIT = 5.0


def setUpModule():
    gateway.LOG.addHandler(logging.NullHandler())
    gateway.LOG.propagate = False


# ---------------------------------------------------------------------------- mock extension
class RecHandler(mock_server.Handler):
    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        with self.server.rec_lock:
            self.server.raw_bodies.append((self.path, raw))
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    def do_POST(self):
        if self.path == "/apply-template" and self.server.apply_template:
            body = self._read_body()
            return self._send_json(200, {"prompt": mock_server.render_chat(body.get("messages") or [],
                                                                           body.get("tools"))})
        return super().do_POST()

    def _generate(self, body, chat):
        srv = self.server
        if chat and srv.null_reasoning_500 and any(isinstance(m, dict) and "reasoning_content" in m
                                                   and m["reasoning_content"] is None
                                                   for m in body.get("messages") or []):
            # llama.cpp llama-server 79e2e74 (common/chat.cpp:422-423) reads the field as a string: HTTP 500
            return self._send_json(500, {"error": {"code": 500, "type": "server_error", "message":
                                                   "Failed to parse messages: [json.exception.type_error.302] type "
                                                   "must be string, but is null"}})
        user = body.get("user")
        with srv.rec_lock:
            srv.gen_order.append(user)
        if user in srv.hold_users and not self._hold(user):
            self.close_connection = True
            return None
        return super()._generate(body, chat)

    def _hold(self, user):
        """Wait for srv.release; return False if the peer closed the connection meanwhile."""
        srv = self.server
        deadline = time.monotonic() + 15.0
        while not srv.release.is_set() and time.monotonic() < deadline:
            r, _, _ = select.select([self.connection], [], [], 0.02)
            if not r:
                continue
            try:
                data = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                with srv.rec_lock:
                    srv.peer_closed.append(user)
                srv.peer_closed_ev.set()
                return False
        return True

    def _chunk(self, obj):
        srv = self.server
        self._nchunks = getattr(self, "_nchunks", 0) + 1
        if srv.block_after is not None and self._nchunks == srv.block_after + 1:
            srv.blocked_ev.set()
            srv.mid_release.wait(10.0)
        try:
            super()._chunk(obj)
        except (BrokenPipeError, ConnectionResetError):
            srv.aborted_ev.set()
            raise


def make_mock(**cfg):
    cfg.setdefault("prefill_s_per_1k", 0.0)
    cfg.setdefault("decode_s_per_token", 0.0005)
    m = mock_server.MockServer(**cfg)
    h = m.httpd
    h.RequestHandlerClass = RecHandler
    h.rec_lock = threading.Lock()
    h.raw_bodies = []
    h.gen_order = []
    h.hold_users = set()
    h.release = threading.Event()
    h.peer_closed = []
    h.peer_closed_ev = threading.Event()
    h.aborted_ev = threading.Event()
    h.block_after = None
    h.blocked_ev = threading.Event()
    h.mid_release = threading.Event()
    h.apply_template = False
    h.null_reasoning_500 = False
    return m.start()


# ---------------------------------------------------------------------------- client helpers
def hostport(url):
    u = urllib.parse.urlsplit(url)
    return u.hostname, u.port


def request(url, method, path, body=None, headers=None, timeout=10.0):
    h, p = hostport(url)
    c = http.client.HTTPConnection(h, p, timeout=timeout)
    data = body if (body is None or isinstance(body, bytes)) else json.dumps(body).encode("utf-8")
    hd = {"Content-Type": "application/json", "Connection": "close"}
    hd.update(headers or {})
    try:
        c.request(method, path, body=data, headers=hd)
        r = c.getresponse()
        d = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, d
    finally:
        c.close()


def post(url, path, body, **kw):
    return request(url, "POST", path, body, **kw)


class Bg(threading.Thread):
    def __init__(self, fn, *a, **kw):
        super().__init__(daemon=True)
        self.fn, self.a, self.kw = fn, a, kw
        self.result = self.exc = None
        self.start()

    def run(self):
        try:
            self.result = self.fn(*self.a, **self.kw)
        except BaseException as e:  # noqa: BLE001
            self.exc = e

    def get(self, timeout=WAIT * 2):
        self.join(timeout)
        if self.is_alive():
            raise AssertionError("background call did not finish within %.1fs" % timeout)
        if self.exc is not None:
            raise self.exc
        return self.result


def wait_for(cond, timeout=WAIT, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    raise AssertionError("timed out after %.1fs waiting for %s" % (timeout, what))


def open_raw(url, path, body, headers=None):
    """Send a request on a raw socket (so the test can drop the connection at will)."""
    s = socket.create_connection(hostport(url), timeout=10)
    data = json.dumps(body).encode("utf-8")
    hd = "".join("%s: %s\r\n" % kv for kv in (headers or {}).items())
    s.sendall(("POST %s HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: %d\r\n%s\r\n"
               % (path, len(data), hd)).encode("ascii") + data)
    return s


def refused(url):
    try:
        socket.create_connection(hostport(url), timeout=1).close()
        return False
    except OSError:
        return True


def sse_events(raw):
    return [b[6:] for b in raw.split(b"\n\n") if b.startswith(b"data: ")]


def norm(raw):
    return re.sub(rb'"created": \d+', b'"created": 0', raw)


def chat_body(text="Hello there, gateway.", stream=False, max_tokens=8, **extra):
    b = {"model": "mock-model", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
         "temperature": 0, "stream": stream}
    b.update(extra)
    return b


def comp_body(prompt="Once upon a time", stream=False, max_tokens=8, **extra):
    b = {"model": "mock-model", "prompt": prompt, "max_tokens": max_tokens, "stream": stream}
    b.update(extra)
    return b


def metric_values(text):
    vals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, val = line.rsplit(" ", 1)
        vals[name] = float(val)
    return vals


class GwCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gw-test-")
        self.mock = make_mock()
        self.gws = []

    def tearDown(self):
        self.mock.httpd.release.set()
        self.mock.httpd.mid_release.set()
        for g in self.gws:
            g.shutdown(2.0)
        self.mock.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def gw(self, backend=None, **kw):
        kw.setdefault("listen_port", 0)
        g = gateway.Gateway(gateway.GatewayConfig(backend or self.mock.url, **kw)).start()
        self.gws.append(g)
        return g

    def mets(self, g):
        st, _, d = request(g.url, "GET", "/metrics")
        self.assertEqual(st, 200)
        return metric_values(d.decode("utf-8"))

    def wait_metric(self, g, key, value):
        # request counters/summaries are updated after the last response byte (like the log line)
        wait_for(lambda: self.mets(g).get(key) == value, what="%s == %s" % (key, value))

    def log_lines(self, path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def wait_log(self, path, n):
        # the gateway writes the log line after the last response byte, so the client can be faster
        wait_for(lambda: len(self.log_lines(path)) >= n, what="%d request log line(s)" % n)
        return self.log_lines(path)

    def gen_bodies(self):
        with self.mock.httpd.rec_lock:
            return [raw for p, raw in self.mock.httpd.raw_bodies if p in (CHAT, COMP)]


# ---------------------------------------------------------------------------- passthrough
class Passthrough(GwCase):
    def test_stream_passthrough_equals_direct_and_is_not_buffered(self):
        g = self.gw()
        body = chat_body(stream=True, max_tokens=10, ignore_eos=True, stream_options={"include_usage": True})
        st, _, direct = post(self.mock.url, CHAT, body)
        self.assertEqual(st, 200)
        self.mock.state.reset()
        self.mock.httpd.block_after = 3                # mock stops before its 4th SSE event
        h, p = hostport(g.url)
        c = http.client.HTTPConnection(h, p, timeout=WAIT)
        c.request("POST", CHAT, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIn("text/event-stream", r.getheader("Content-Type"))
        self.assertTrue(r.getheader("X-Request-Id", "").startswith("gw-"))
        head, n = b"", 0
        while n < 3:                                    # these arrive while the mock is blocked
            line = r.readline()
            self.assertTrue(line, "stream ended early")
            head += line
            n += line.startswith(b"data: ")
        self.assertTrue(self.mock.httpd.blocked_ev.wait(WAIT))  # the mock has not sent the rest yet
        self.assertFalse(self.mock.httpd.mid_release.is_set())
        self.mock.httpd.mid_release.set()
        rest = r.read()
        c.close()
        self.assertEqual(norm(head + rest), norm(direct))
        self.assertTrue(direct.rstrip().endswith(b"data: [DONE]"))

    def test_non_stream_chat_and_completions_equal_direct(self):
        g = self.gw()
        for path, body in ((CHAT, chat_body()), (COMP, comp_body())):
            self.mock.state.reset()
            st1, h1, d1 = post(self.mock.url, path, body)
            self.mock.state.reset()
            st2, h2, d2 = post(g.url, path, body)
            self.assertEqual((st1, st2), (200, 200))
            self.assertEqual(d1, d2)
            self.assertEqual(h2["content-type"], "application/json")
            self.assertTrue(h2["x-request-id"].startswith("gw-"))

    def test_models_and_health(self):
        g = self.gw()
        st, _, d = request(g.url, "GET", "/v1/models")
        self.assertEqual(st, 200)
        self.assertEqual(d, request(self.mock.url, "GET", "/v1/models")[2])
        st, _, d = request(g.url, "GET", "/health")
        self.assertEqual(st, 200)
        h = json.loads(d)
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["backend"]["reachable"])
        self.assertEqual((h["in_flight"], h["queued"], h["max_concurrency"]), (0, 0, 1))
        st, _, d = request(g.url, "GET", "/nope")
        self.assertEqual(st, 404)
        self.assertEqual(json.loads(d)["error"]["code"], "not_found")


# ---------------------------------------------------------------------------- auth
class Auth(GwCase):
    def test_auth_off_warns_and_allows(self):
        with self.assertLogs("gateway", level="WARNING") as cm:
            g = self.gw()
        self.assertTrue(any("auth is OFF" in m for m in cm.output), cm.output)
        self.assertEqual(post(g.url, CHAT, chat_body())[0], 200)

    def test_keys_file_good_bad_missing_and_never_logged(self):
        kf = os.path.join(self.tmp, "keys")
        with open(kf, "w") as f:
            f.write("# gateway keys\n\n  key-one-SECRET1  \nkey-two-SECRET2\n# key-three-COMMENTED\n")
        keys = gateway.parse_api_keys(kf)
        self.assertEqual(keys, ["key-one-SECRET1", "key-two-SECRET2"])
        logf = os.path.join(self.tmp, "req.jsonl")
        with self.assertLogs("gateway", level="DEBUG") as cm:
            g = self.gw(api_keys=keys, log_file=logf)
            st, h, d = post(g.url, CHAT, chat_body())
            self.assertEqual(st, 401)
            self.assertEqual(h.get("www-authenticate"), "Bearer")
            self.assertEqual(json.loads(d)["error"]["type"], "authentication_error")
            self.assertEqual(post(g.url, CHAT, chat_body(), headers={"Authorization": "Bearer nope"})[0], 401)
            self.assertEqual(post(g.url, CHAT, chat_body(), headers={"Authorization": "key-two-SECRET2"})[0], 401)
            self.assertEqual(post(g.url, CHAT, chat_body(),
                                  headers={"Authorization": "Bearer key-two-SECRET2"})[0], 200)
            self.assertEqual(request(g.url, "GET", "/v1/models")[0], 401)
            self.assertEqual(request(g.url, "GET", "/v1/models",
                                     headers={"Authorization": "Bearer key-one-SECRET1"})[0], 200)
            self.assertEqual(request(g.url, "GET", "/health")[0], 200)       # public
            self.assertEqual(request(g.url, "GET", "/metrics")[0], 200)      # public
        self.assertEqual(len(self.gen_bodies()), 1)                          # only the authorised one
        self.assertEqual(self.mets(g)["gateway_auth_failures_total"], 4)
        with open(logf) as f:
            logtext = f.read()
        self.assertNotIn("SECRET", logtext)
        self.assertNotIn("SECRET", "\n".join(cm.output))
        self.assertEqual(sorted(r["status"] for r in self.wait_log(logf, 6)), [200, 200, 401, 401, 401, 401])

    def test_empty_keys_file_refuses_to_start(self):
        kf = os.path.join(self.tmp, "keys")
        with open(kf, "w") as f:
            f.write("# nothing here\n")
        a = gateway.build_parser().parse_args(["--backend", self.mock.url, "--api-keys-file", kf])
        with self.assertRaises(ValueError):
            gateway.config_from_args(a)


# ---------------------------------------------------------------------------- admission
class Queueing(GwCase):
    def hold(self, g, user, path=CHAT, **kw):
        self.mock.httpd.hold_users.add(user)
        t = Bg(post, g.url, path, chat_body(user=user, **kw) if path == CHAT else comp_body(user=user, **kw))
        wait_for(lambda: user in self.mock.httpd.gen_order, what="%s at the backend" % user)
        return t

    def test_429_when_queue_full(self):
        g = self.gw(max_concurrency=1, max_queue=1, retry_after_s=7)
        a = self.hold(g, "A")
        b = Bg(post, g.url, CHAT, chat_body(user="B"))
        wait_for(lambda: g.admission.snapshot() == (1, 1), what="B queued")
        st, h, d = post(g.url, CHAT, chat_body(user="C"))
        self.assertEqual(st, 429)
        self.assertEqual(h.get("retry-after"), "7")
        err = json.loads(d)["error"]
        self.assertEqual((err["type"], err["code"]), ("rate_limit_error", "queue_full"))
        self.mock.httpd.release.set()
        self.assertEqual(a.get()[0], 200)
        self.assertEqual(b.get()[0], 200)
        self.assertEqual(self.mock.httpd.gen_order, ["A", "B"])
        m = self.mets(g)
        self.assertEqual(m["gateway_queue_rejected_total"], 1)
        self.wait_metric(g, 'gateway_requests_total{path="/v1/chat/completions",code="429"}', 1)

    def test_fifo_order(self):
        g = self.gw(max_concurrency=1, max_queue=8)
        a = self.hold(g, "A")
        ts = []
        for i, u in enumerate("BCDEF", start=1):
            ts.append(Bg(post, g.url, CHAT, chat_body(user=u)))
            wait_for(lambda i=i: g.admission.snapshot()[1] == i, what="%s queued" % u)
        self.mock.httpd.release.set()
        self.assertEqual(a.get()[0], 200)
        self.assertEqual([t.get()[0] for t in ts], [200] * 5)
        self.assertEqual(self.mock.httpd.gen_order, list("ABCDEF"))

    def test_queued_client_disconnect_leaves_queue(self):
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(max_concurrency=1, max_queue=4, log_file=logf)
        a = self.hold(g, "A")
        s = open_raw(g.url, CHAT, chat_body(user="B"))
        wait_for(lambda: g.admission.snapshot()[1] == 1, what="B queued")
        s.close()
        wait_for(lambda: g.admission.snapshot()[1] == 0, what="B removed from queue")
        self.mock.httpd.release.set()
        self.assertEqual(a.get()[0], 200)
        self.wait_log(logf, 2)
        self.assertEqual(self.mock.httpd.gen_order, ["A"])
        self.assertEqual(self.mets(g)['gateway_client_disconnects_total{phase="queued"}'], 1)
        b = [r for r in self.log_lines(logf) if r["status"] == 499]
        self.assertEqual(len(b), 1)
        self.assertTrue(b[0]["client_disconnected"])


# ---------------------------------------------------------------------------- disconnects
class Disconnect(GwCase):
    def test_stream_disconnect_closes_upstream(self):
        self.mock.configure(decode_s_per_token=0.01)          # 2000 tokens = 20 s if nobody stops it
        g = self.gw()
        h, p = hostport(g.url)
        c = http.client.HTTPConnection(h, p, timeout=WAIT)
        c.request("POST", COMP, body=json.dumps(comp_body(stream=True, max_tokens=2000, ignore_eos=True)),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        while b'"text": "' not in r.readline():
            pass
        t0 = time.monotonic()
        r.close()
        c.close()
        self.assertTrue(self.mock.httpd.aborted_ev.wait(WAIT), "the mock kept generating")
        self.assertLess(time.monotonic() - t0, WAIT)
        wait_for(lambda: self.mets(g)['gateway_client_disconnects_total{phase="streaming"}'] == 1,
                 what="disconnect metric")
        wait_for(lambda: g.admission.snapshot() == (0, 0), what="slot released")

    def test_stream_disconnect_detected_by_failed_write_when_watch_is_off(self):
        # Pipelined bytes make the client watch stop (it can no longer tell EOF from data), so only
        # the failed write to the client can notice the disconnect; it must still close upstream.
        self.mock.configure(decode_s_per_token=0.01)
        g = self.gw()
        s = open_raw(g.url, COMP, comp_body(stream=True, max_tokens=2000, ignore_eos=True))
        got = b""
        while b'"text": "' not in got:
            d = s.recv(65536)
            self.assertTrue(d, "stream ended early")
            got += d
        s.sendall(b"GET /health HTTP/1.1\r\nHost: t\r\n\r\n")   # never read by the gateway during the stream
        t0 = time.monotonic()
        s.close()                                              # unread data on our side -> RST
        self.assertTrue(self.mock.httpd.aborted_ev.wait(WAIT), "the mock kept generating")
        self.assertLess(time.monotonic() - t0, WAIT)
        wait_for(lambda: self.mets(g)['gateway_client_disconnects_total{phase="streaming"}'] == 1,
                 what="disconnect metric")

    def test_disconnect_while_engine_prefills_closes_upstream(self):
        g = self.gw()
        self.mock.httpd.hold_users.add("A")                  # the "engine" is busy before any byte
        s = open_raw(g.url, CHAT, chat_body(user="A", stream=True))
        wait_for(lambda: "A" in self.mock.httpd.gen_order, what="A at the backend")
        s.close()
        self.assertTrue(self.mock.httpd.peer_closed_ev.wait(WAIT), "gateway did not close the upstream socket")
        wait_for(lambda: self.mets(g)['gateway_client_disconnects_total{phase="upstream_wait"}'] == 1,
                 what="disconnect metric")
        wait_for(lambda: g.admission.snapshot() == (0, 0), what="slot released")


# ---------------------------------------------------------------------------- context guard
WORDS45 = " ".join("w%d" % i for i in range(45))              # 45 toy tokens in the mock tokenizer


class ContextGuard(GwCase):
    def test_reject_completion_over_limit_before_engine(self):
        g = self.gw(max_context_tokens=50)
        st, _, d = post(g.url, COMP, comp_body(WORDS45, max_tokens=10))
        self.assertEqual(st, 400)
        err = json.loads(d)["error"]
        self.assertEqual((err["type"], err["code"], err["param"]),
                         ("invalid_request_error", "context_length_exceeded", "prompt"))
        self.assertIn("45 in the prompt", err["message"])
        self.assertEqual(self.gen_bodies(), [])                 # never reached the engine
        self.assertEqual(post(g.url, COMP, comp_body(WORDS45, max_tokens=5))[0], 200)
        self.assertEqual(len(self.gen_bodies()), 1)
        m = self.mets(g)
        self.assertEqual(m['gateway_context_guard_total{result="rejected",reason="over_limit"}'], 1)
        self.assertEqual(m['gateway_context_guard_total{result="ok",reason="within_limit"}'], 1)

    def test_chat_guard_uses_apply_template(self):
        self.mock.httpd.apply_template = True
        g = self.gw(max_context_tokens=50)
        # rendered: "<|user|>\n" + 45 words + "\n<|assistant|>\n" = 5 + 45 + 5 = 55 toy tokens
        st, _, d = post(g.url, CHAT, chat_body(WORDS45, max_tokens=1))
        self.assertEqual(st, 400)
        self.assertIn("55 in the prompt", json.loads(d)["error"]["message"])
        self.assertEqual(post(g.url, CHAT, chat_body("short one", max_tokens=8))[0], 200)
        self.assertEqual(len(self.gen_bodies()), 1)

    def test_chat_without_apply_template_passes_unchecked(self):
        g = self.gw(max_context_tokens=50)                      # mock answers 404 to /apply-template
        self.assertEqual(post(g.url, CHAT, chat_body(WORDS45, max_tokens=10))[0], 200)
        self.assertEqual(self.mets(g)['gateway_context_guard_total{result="unchecked",reason="no_apply_template"}'], 1)

    def test_pass_through_when_backend_has_no_tokenize(self):
        self.mock.configure(tokenize_style="none")
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(max_context_tokens=50, log_file=logf)
        self.assertEqual(post(g.url, COMP, comp_body(WORDS45, max_tokens=10))[0], 200)
        self.assertEqual(len(self.gen_bodies()), 1)
        self.assertEqual(self.mets(g)['gateway_context_guard_total{result="unchecked",reason="no_tokenize"}'], 1)
        self.assertEqual(self.wait_log(logf, 1)[0]["guard"], "unchecked:no_tokenize")


# ---------------------------------------------------------------------------- failures
def dead_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Failures(GwCase):
    def test_dead_backend_502_and_health_503(self):
        g = self.gw(backend="http://127.0.0.1:%d" % dead_port())
        for body in (chat_body(), chat_body(stream=True)):
            st, _, d = post(g.url, CHAT, body)
            self.assertEqual(st, 502)
            self.assertEqual(json.loads(d)["error"]["code"], "backend_unreachable")
        st, _, d = request(g.url, "GET", "/health")
        self.assertEqual(st, 503)
        h = json.loads(d)
        self.assertEqual(h["status"], "backend_unavailable")
        self.assertFalse(h["backend"]["reachable"])
        self.assertEqual(self.mets(g)['gateway_backend_errors_total{kind="connect"}'], 2)
        wait_for(lambda: g.admission.snapshot() == (0, 0), what="slots released")

    def test_http500_before_first_byte_maps_to_502(self):
        self.mock.configure(fault="http500")
        g = self.gw()
        for body in (chat_body(), chat_body(stream=True)):
            st, _, d = post(g.url, CHAT, body)
            self.assertEqual(st, 502)
            err = json.loads(d)["error"]
            self.assertEqual((err["type"], err["code"]), ("upstream_error", "backend_error"))
            self.assertIn("injected fault", err["message"])
        self.assertEqual(self.mets(g)['gateway_backend_errors_total{kind="http_5xx"}'], 2)
        self.assertEqual(len(self.gen_bodies()), 2)                # no automatic retry

    def test_truncated_stream_ends_with_error_event_and_no_done(self):
        self.mock.configure(fault="truncate", fault_after_chunks=2)
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(log_file=logf)
        st, _, raw = post(g.url, CHAT, chat_body(stream=True, max_tokens=10, ignore_eos=True))
        self.assertEqual(st, 200)
        self.assertNotIn(b"[DONE]", raw)
        evs = sse_events(raw)
        self.assertGreaterEqual(len(evs), 2)
        last = json.loads(evs[-1])
        self.assertEqual(set(last), {"error"})
        self.assertEqual(last["error"]["code"], "backend_truncated")
        for e in evs[:-1]:
            self.assertIn("choices", json.loads(e))                 # upstream events were passed through
        self.assertTrue(raw.endswith(b"\n\n"))
        self.assertEqual(self.mets(g)['gateway_backend_errors_total{kind="truncated"}'], 1)
        rec = self.wait_log(logf, 1)[0]
        self.assertIn("truncated", rec["error"])
        self.assertEqual(len(self.gen_bodies()), 1)                # no automatic retry


# ---------------------------------------------------------------------------- log + metrics
SECRET_PROMPT = "PROMPT-TEXT-zx81 tell me nothing"


class LogAndMetrics(GwCase):
    def test_jsonl_fields_and_no_prompt_text(self):
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(log_file=logf)
        body = chat_body(SECRET_PROMPT, stream=True, stream_options={"include_usage": True})
        ids = []
        for _ in range(2):                                          # the 2nd hits the mock's prefix cache
            st, h, _ = post(g.url, CHAT, body)
            self.assertEqual(st, 200)
            ids.append(h["x-request-id"])
        st, h, d = post(g.url, COMP, comp_body(SECRET_PROMPT))
        ids.append(h["x-request-id"])
        usage = json.loads(d)["usage"]
        recs = self.wait_log(logf, 3)
        self.assertEqual([r["id"] for r in recs], ids)
        for k in ("ts", "id", "path", "model", "stream", "prompt_chars", "status", "ttft_s", "total_s",
                  "completion_tokens", "cached_tokens", "error"):
            for r in recs:
                self.assertIn(k, r)
        s1, s2, c = recs
        self.assertEqual((s1["path"], s1["model"], s1["stream"], s1["status"]), (CHAT, "mock-model", True, 200))
        self.assertEqual(s1["prompt_chars"], len(SECRET_PROMPT))
        self.assertGreater(s1["ttft_s"], 0)
        self.assertGreaterEqual(s1["total_s"], s1["ttft_s"])
        self.assertEqual(s1["completion_tokens"], 7)
        self.assertEqual(s1["cached_tokens"], 0)
        self.assertGreater(s2["cached_tokens"], 0)
        self.assertIsNone(s1["error"])
        self.assertEqual((c["stream"], c["ttft_s"]), (False, None))
        self.assertEqual(c["completion_tokens"], usage["completion_tokens"])
        self.assertEqual(c["cached_tokens"], usage["prompt_tokens_details"]["cached_tokens"])
        with open(logf) as f:
            self.assertNotIn("zx81", f.read())

    def test_log_prompts_opt_in(self):
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(log_file=logf, log_prompts=True)
        post(g.url, CHAT, chat_body(SECRET_PROMPT))
        rec = self.wait_log(logf, 1)[0]
        self.assertEqual(rec["prompt"], [{"role": "user", "content": SECRET_PROMPT}])

    def test_metrics_text_parses(self):
        g = self.gw()
        post(g.url, CHAT, chat_body(stream=True))
        post(g.url, CHAT, chat_body(stream=True))
        post(g.url, COMP, comp_body())
        self.wait_metric(g, "gateway_request_duration_seconds_count", 3)
        st, h, d = request(g.url, "GET", "/metrics")
        self.assertEqual(st, 200)
        self.assertTrue(h["content-type"].startswith("text/plain"))
        sample = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[a-zA-Z_][a-zA-Z0-9_]*="[^"]*"'
                            r'(,[a-zA-Z_][a-zA-Z0-9_]*="[^"]*")*\})? (NaN|[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?)$')
        typed = set()
        for line in d.decode().splitlines():
            if line.startswith("# TYPE "):
                typed.add(line.split()[2])
            elif line.startswith("# HELP "):
                continue
            else:
                self.assertRegex(line, sample)
                base = re.split(r"[{ ]", line)[0]
                self.assertTrue(base in typed or re.sub(r"_(sum|count)$", "", base) in typed, line)
        m = metric_values(d.decode())
        self.assertEqual(m['gateway_requests_total{path="/v1/chat/completions",code="200"}'], 2)
        self.assertEqual(m['gateway_requests_total{path="/v1/completions",code="200"}'], 1)
        self.assertEqual(m["gateway_in_flight"], 0)
        self.assertEqual(m["gateway_queue_length"], 0)
        self.assertEqual(m["gateway_ttft_seconds_count"], 2)
        self.assertEqual(m["gateway_request_duration_seconds_count"], 3)
        self.assertIn('gateway_ttft_seconds{quantile="0.9"}', m)


# ---------------------------------------------------------------------------- body fidelity
class BodyFidelity(GwCase):
    def test_set_default_injected_only_when_absent(self):
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(set_defaults=[("cache_prompt", True)], log_file=logf)
        raw1 = b'{"model":"mock-model",  "prompt": "caf\\u00e9 \xc3\xa9t\xc3\xa9",\n "max_tokens": 2 }\n'
        raw2 = b'{"cache_prompt": false, "prompt": "x y", "max_tokens": 2}'
        self.assertEqual(post(g.url, COMP, raw1)[0], 200)
        self.assertEqual(post(g.url, COMP, raw2)[0], 200)
        up1, up2 = self.gen_bodies()
        self.assertEqual(up1, b'{"model":"mock-model",  "prompt": "caf\\u00e9 \xc3\xa9t\xc3\xa9",\n "max_tokens": 2 '
                              b',"cache_prompt":true}\n')
        self.assertEqual(json.loads(up1), dict(json.loads(raw1), cache_prompt=True))
        self.assertEqual(up2, raw2)
        self.assertEqual([r["injected_defaults"] for r in self.wait_log(logf, 2)], [["cache_prompt"], []])

    def test_apply_defaults_unit(self):
        self.assertEqual(gateway.apply_defaults(b"{}", {}, [("a", 1), ("b", [1, "x"])]),
                         (b'{"a":1,"b":[1,"x"]}', ["a", "b"]))
        self.assertEqual(gateway.apply_defaults(b' { "a" : 2 } \r\n', {"a": 2}, [("a", 1)]),
                         (b' { "a" : 2 } \r\n', []))
        self.assertEqual(gateway.apply_defaults(b'{"a":2}  ', {"a": 2}, [("c", None)]),
                         (b'{"a":2,"c":null}  ', ["c"]))
        self.assertEqual(gateway.parse_set_default("cache_prompt=true"), ("cache_prompt", True))
        with self.assertRaises(ValueError):
            gateway.parse_set_default("cache_prompt=yes please")

    def test_byte_identical_forwarding_and_same_prefix_cache_hits(self):
        g = self.gw()
        conv, raws = [], []
        for i in range(4):                                     # agent-style growing conversation
            conv.append({"role": "user", "content": "Step %d: look at file_%d.py and report.é" % (i, i)})
            obj = {"messages": list(conv), "model": "mock-model", "max_tokens": 4, "temperature": 0,
                   "stream": False}
            raws.append(json.dumps(obj, indent=1 + i % 2, ensure_ascii=bool(i % 2)).encode("utf-8"))
            conv.append({"role": "assistant", "content": "ok %d" % i})

        def run(url):
            self.mock.state.reset()
            with self.mock.httpd.rec_lock:
                self.mock.httpd.raw_bodies.clear()
            out = []
            for raw in raws:
                st, _, d = post(url, CHAT, raw)
                self.assertEqual(st, 200)
                out.append(json.loads(d)["usage"]["prompt_tokens_details"]["cached_tokens"])
            return out, self.gen_bodies()

        direct_cached, direct_bodies = run(self.mock.url)
        gw_cached, gw_bodies = run(g.url)
        self.assertEqual(direct_bodies, raws)
        self.assertEqual(gw_bodies, raws)                      # byte for byte
        self.assertEqual(gw_cached, direct_cached)
        self.assertEqual(direct_cached[0], 0)
        self.assertTrue(all(c > 0 for c in direct_cached[1:]), direct_cached)


# ---------------------------------------------------------------------------- R1 safety rule 3
class NullReasoningContent(GwCase):
    """R1 safety rule 3 (docs/reassessment/PHASE-CLOSE-2026-10-11.md section (v)): the gateway drops
    `"reasoning_content": null` from chat messages; llama-server 79e2e74 answers it with HTTP 500 (the mock here
    imitates that answer when null_reasoning_500 is set)."""
    RAW = (b'{"model": "mock-model", "messages": [{"role": "user", "content": "caf\\u00e9 \xc3\xa9 Q1"},\n'
           b' {"role": "assistant", "reasoning_content": null, "content": "A1"},\n'
           b' {"role": "assistant", "content": "A2" , "reasoning_content" : null },\n'
           b' {"role": "assistant", "content": "A3", "reasoning_content": "R3"},\n'
           b' {"role": "user", "content": "Q2"}], "max_tokens": 4, "temperature": 0}')
    WANT = (b'{"model": "mock-model", "messages": [{"role": "user", "content": "caf\\u00e9 \xc3\xa9 Q1"},\n'
            b' {"role": "assistant", "content": "A1"},\n'
            b' {"role": "assistant", "content": "A2" },\n'
            b' {"role": "assistant", "content": "A3", "reasoning_content": "R3"},\n'
            b' {"role": "user", "content": "Q2"}], "max_tokens": 4, "temperature": 0}')

    def test_null_is_cut_out_and_every_other_byte_kept(self):
        self.mock.httpd.null_reasoning_500 = True
        logf = os.path.join(self.tmp, "req.jsonl")
        g = self.gw(log_file=logf)
        st, _, d = post(self.mock.url, CHAT, self.RAW)
        self.assertEqual(st, 500, d)                                   # the engine-side failure the rule avoids
        self.assertIn(b"must be string, but is null", d)
        with self.mock.httpd.rec_lock:
            self.mock.httpd.raw_bodies.clear()
        st, _, d = post(g.url, CHAT, self.RAW)
        self.assertEqual(st, 200, d)
        self.assertEqual(self.gen_bodies(), [self.WANT])
        want = json.loads(self.RAW)
        for m in want["messages"]:
            if m.get("reasoning_content", "") is None:
                del m["reasoning_content"]
        self.assertEqual(json.loads(self.WANT), want)
        rec = self.wait_log(logf, 1)[0]
        self.assertEqual(rec["dropped_null_fields"], ["messages[1].reasoning_content", "messages[2].reasoning_content"])

    def test_requests_without_null_are_forwarded_byte_for_byte(self):
        g = self.gw()
        raws = [self.WANT, json.dumps(chat_body(), indent=2).encode(),
                b'{"messages": [{"role": "assistant", "content": "x", "reasoning_content": ""}], "max_tokens": 2}',
                b'{"messages": [{"role": "user", "content": "null"}], "reasoning_content": null, "max_tokens": 2}']
        for raw in raws:
            self.assertEqual(post(g.url, CHAT, raw)[0], 200)
        self.assertEqual(self.gen_bodies(), raws)        # only chat messages are touched, only a null value

    def test_completions_untouched_and_switch_off(self):
        g = self.gw(drop_null_reasoning=False)
        self.assertEqual(post(g.url, CHAT, self.RAW)[0], 200)
        raw = b'{"prompt": "x", "messages": [{"reasoning_content": null}], "max_tokens": 2}'
        g2 = self.gw()
        self.assertEqual(post(g2.url, COMP, raw)[0], 200)
        self.assertEqual(self.gen_bodies(), [self.RAW, raw])
        a = gateway.build_parser().parse_args(["--backend", "http://x", "--keep-null-reasoning-content"])
        self.assertFalse(gateway.config_from_args(a).drop_null_reasoning)
        a = gateway.build_parser().parse_args(["--backend", "http://x"])
        self.assertTrue(gateway.config_from_args(a).drop_null_reasoning)

    def test_drop_unit(self):
        cases = [
            (b'{"messages":[{"reasoning_content":null}]}', b'{"messages":[{}]}'),
            (b'{"messages":[{"reasoning_content":null,"role":"user","content":"q"}]}',
             b'{"messages":[{"role":"user","content":"q"}]}'),
            (b'{"messages":[{"reasoning_content":null, "reasoning_content":null ,"role":"user","content":"q"}]}',
             b'{"messages":[{"role":"user","content":"q"}]}'),
            (b'{"messages":[{"role":"user","reasoning_content":null,"content":"q","reasoning_content":null}]}',
             b'{"messages":[{"role":"user","content":"q"}]}'),
            (b'{"messages":[{"role":"a","content":"q","reasoning_content":null}],"messages":[{"x":[1,{"y":null}],'
             b'"reasoning_content":null}]}', b'{"messages":[{"role":"a","content":"q"}],"messages":[{"x":[1,{"y":null}]}]}'),
            (b'{"messages":[{"role":"user","content":"q"}]}', None),
            (b'{"messages":"not a list"}', None),
            (b'{"messages":[{"role":"user","content":-1.5e3,"reasoning_content":false}]}', None),
        ]
        for raw, want in cases:
            with self.subTest(raw=raw):
                obj = json.loads(raw)
                body, obj2, dropped = gateway.drop_null_reasoning(raw, obj)
                self.assertIs(obj2, obj)
                self.assertEqual(body, raw if want is None else want)
                self.assertEqual(bool(dropped), want is not None)
                self.assertEqual(json.loads(body), obj)
        # a duplicate key whose LAST value is null: the textual cut would leave the first value, so the body is
        # re-serialised from the normalised object instead (still without the field)
        raw = b'{"messages":[{"role":"user","content":"q","reasoning_content":"R","reasoning_content":null}]}'
        body, obj, dropped = gateway.drop_null_reasoning(raw, json.loads(raw))
        self.assertEqual(json.loads(body), {"messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(dropped, ["messages[0].reasoning_content"])


# ---------------------------------------------------------------------------- shutdown
class Shutdown(GwCase):
    def test_graceful_shutdown_drains_in_flight(self):
        g = self.gw()
        self.mock.httpd.hold_users.add("A")
        a = Bg(post, g.url, CHAT, chat_body(user="A", stream=True))
        wait_for(lambda: "A" in self.mock.httpd.gen_order, what="A at the backend")
        sd = Bg(g.shutdown, 10.0)
        wait_for(lambda: refused(g.url), what="listening socket closed")
        self.assertTrue(sd.is_alive())                         # still draining A
        self.mock.httpd.release.set()
        self.assertTrue(sd.get())
        st, _, raw = a.get()
        self.assertEqual(st, 200)
        self.assertIn(b"data: [DONE]", raw)

    def test_drain_timeout_reports_false(self):
        g = self.gw()
        self.mock.httpd.hold_users.add("A")
        a = Bg(post, g.url, CHAT, chat_body(user="A"))
        wait_for(lambda: "A" in self.mock.httpd.gen_order, what="A at the backend")
        self.assertFalse(g.shutdown(0.2))
        self.mock.httpd.release.set()
        self.assertEqual(a.get()[0], 200)

    def test_sigterm_cli_drains_and_exits_zero(self):
        proc = subprocess.Popen([sys.executable, os.path.join(GWDIR, "gateway.py"), "--listen", "127.0.0.1:0",
                                 "--backend", self.mock.url, "--drain-s", "10"],
                                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        lines, url_box = [], []
        got_url = threading.Event()

        def reader():
            for line in proc.stderr:
                lines.append(line)
                m = re.search(r"listening on (http://\S+)", line)
                if m:
                    url_box.append(m.group(1))
                    got_url.set()

        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        try:
            self.assertTrue(got_url.wait(WAIT * 2), "".join(lines))
            url = url_box[0]
            self.mock.httpd.hold_users.add("A")
            a = Bg(post, url, CHAT, chat_body(user="A", stream=True))
            wait_for(lambda: "A" in self.mock.httpd.gen_order, what="A at the backend")
            proc.send_signal(signal.SIGTERM)
            wait_for(lambda: refused(url), what="CLI gateway stopped accepting")
            self.assertIsNone(proc.poll())
            self.mock.httpd.release.set()
            st, _, raw = a.get()
            self.assertEqual(st, 200)
            self.assertIn(b"data: [DONE]", raw)
            self.assertEqual(proc.wait(WAIT * 2), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        rt.join(2)
        proc.stderr.close()
        text = "".join(lines)
        self.assertIn("auth is OFF", text)
        self.assertIn("drained", text)


class CLIHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        r = subprocess.run([sys.executable, os.path.join(GWDIR, "gateway.py"), "--help"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--max-concurrency", r.stdout)


if __name__ == "__main__":
    unittest.main()
