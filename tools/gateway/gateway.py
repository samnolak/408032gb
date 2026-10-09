#!/usr/bin/env python3
"""Thin engine-agnostic OpenAI-compatible gateway (reverse proxy), Python 3 stdlib only.

Sits in front of ONE single-stream inference engine (llama.cpp llama-server, ik_llama.cpp,
Strata serve, TabbyAPI, ...) and adds what the engine does not reliably give us:

  * bearer-token auth (--api-keys-file), key values are never logged;
  * admission control: --max-concurrency requests in flight to the engine, --max-queue waiting
    in strict FIFO order, 429 + Retry-After when the queue is full;
  * client-disconnect propagation: a client that goes away while queued leaves the queue; a
    client that goes away while its request is at the engine gets the upstream TCP connection
    closed, so an engine that watches its socket stops generating;
  * optional context guard (--max-context-tokens) that counts prompt tokens with the engine's
    own /tokenize (and /apply-template for chat) before the request reaches a slot;
  * SSE passthrough chunk by chunk (no buffering), with an SSE error event when the engine
    drops the stream mid-way (never a fake [DONE]);
  * one JSONL line per request (--log-file), Prometheus text at /metrics, JSON at /health.

The request body is forwarded byte for byte (prefix caches in engines key on the rendered
prompt, and some on the raw request): the only permitted change is --set-default key=json,
inserted textually before the closing brace and only when the client did not send that key.

Generation requests are never retried automatically: they are not idempotent. A retry after
the engine accepted the request repeats prefill (minutes at 262K context on a single-stream
engine) and, for a stream, would splice two different generations into one response.

Endpoints: POST /v1/chat/completions, POST /v1/completions (queued, guarded);
GET /v1/models, POST /tokenize, POST /detokenize (proxied, not queued);
GET /health, GET /metrics (gateway's own, no auth).
"""
from __future__ import annotations

import argparse
import collections
import hmac
import http.client
import json
import logging
import math
import selectors
import signal
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["GatewayConfig", "Gateway", "apply_defaults", "parse_api_keys", "main"]

LOG = logging.getLogger("gateway")
VERSION = "408032gb-gateway/1"

GEN_PATHS = ("/v1/chat/completions", "/v1/completions")
PROXY_GET = ("/v1/models",)
PROXY_POST = ("/tokenize", "/detokenize")
OWN_GET = ("/health", "/metrics")
METRIC_PATHS = GEN_PATHS + PROXY_GET + PROXY_POST
BACKEND_ERROR_KINDS = ("connect", "timeout", "protocol", "http_5xx", "truncated")
DISCONNECT_PHASES = ("queued", "upstream_wait", "streaming")
MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens", "n_predict")
PROBE_TEXT = "The quick brown fox jumps over the lazy dog."
TOKENIZE_REPROBE_S = 60.0
MAX_ERROR_BODY = 4000


# ---------------------------------------------------------------------------- helpers
def oa_error(message, typ, code, param=None):
    """OpenAI-style error object."""
    return {"error": {"message": message, "type": typ, "param": param, "code": code}}


def utc_now():
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000)


def parse_api_keys(path):
    """One key per line; blank lines and lines starting with '#' are ignored."""
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                keys.append(s)
    return keys


def parse_set_default(spec):
    """'key=json' -> (key, value)."""
    if "=" not in spec:
        raise ValueError("--set-default expects key=json, got %r" % spec)
    k, v = spec.split("=", 1)
    k = k.strip()
    if not k:
        raise ValueError("--set-default: empty key in %r" % spec)
    try:
        return k, json.loads(v)
    except ValueError as e:
        raise ValueError("--set-default %s: value is not JSON: %s" % (k, e))


def apply_defaults(raw, obj, defaults):
    """Insert (key, value) pairs the client did not send, textually, before the closing brace.

    Everything the client sent stays byte-identical (key order, whitespace, escapes), so the
    engine sees the same prompt bytes as without the gateway. Returns (bytes, injected_keys).
    """
    missing = [(k, v) for k, v in defaults if k not in obj]
    if not missing:
        return raw, []
    stripped = raw.rstrip(b" \t\r\n")
    if not stripped.endswith(b"}"):          # cannot happen for a parsed JSON object
        raise ValueError("body is not a JSON object")
    cut = len(stripped) - 1
    parts = []
    need_comma = bool(obj)
    for k, v in missing:
        parts.append((b"," if need_comma else b"") + json.dumps(k).encode("utf-8") + b":"
                     + json.dumps(v, separators=(",", ":")).encode("utf-8"))
        need_comma = True
    return raw[:cut] + b"".join(parts) + raw[cut:], [k for k, _ in missing]


def _content_text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def prompt_chars(path, obj):
    if path == "/v1/chat/completions":
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            return None
        return sum(len(_content_text(m.get("content"))) for m in msgs if isinstance(m, dict))
    p = obj.get("prompt")
    if isinstance(p, str):
        return len(p)
    if isinstance(p, list) and all(isinstance(x, str) for x in p):
        return sum(len(x) for x in p)
    return None


def requested_max_tokens(obj):
    """First non-negative integer among max_tokens / max_completion_tokens / n_predict, else None."""
    for k in MAX_TOKEN_KEYS:
        v = obj.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    return None


# ---------------------------------------------------------------------------- config
class GatewayConfig:
    def __init__(self, backend, **kw):
        self.backend = backend
        self.listen_host = "127.0.0.1"
        self.listen_port = 8000
        self.api_keys = None                 # None = auth off; list of keys = auth on
        self.backend_api_key = None          # sent upstream as Bearer, never the client's key
        self.max_concurrency = 1
        self.max_queue = 8
        self.max_context_tokens = None
        self.tokenize_dialect = "auto"       # auto|llamacpp|vllm|off
        self.set_defaults = []               # [(key, value)]
        self.log_file = None                 # path, "-" = stdout, None = no request log
        self.log_prompts = False
        self.connect_timeout_s = 5.0
        self.backend_timeout_s = 3600.0      # per blocking read; non-stream waits for the whole generation
        self.guard_timeout_s = 30.0
        self.health_timeout_s = 2.0
        self.retry_after_s = 5
        self.drain_s = 30.0
        self.max_body_bytes = 64 << 20
        self.watch_poll_s = 0.05
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown gateway option %r" % k)
            setattr(self, k, v)
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_queue < 0:
            raise ValueError("max_queue must be >= 0")


# ---------------------------------------------------------------------------- metrics
class Summary:
    """Prometheus summary over a sliding window (quantiles) plus all-time sum/count."""

    def __init__(self, window=2048):
        self.vals = collections.deque(maxlen=window)
        self.count = 0
        self.sum = 0.0

    def observe(self, v):
        self.vals.append(v)
        self.count += 1
        self.sum += v

    def quantile(self, q):
        if not self.vals:
            return float("nan")
        s = sorted(self.vals)
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return repr(round(v, 6))
    return str(v)


class Metrics:
    QUANTILES = (0.5, 0.9, 0.99)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = collections.Counter()           # (path, code)
        self.backend_errors = collections.Counter({k: 0 for k in BACKEND_ERROR_KINDS})
        self.disconnects = collections.Counter({k: 0 for k in DISCONNECT_PHASES})
        self.guard = collections.Counter()              # (result, reason)
        self.auth_failures = 0
        self.queue_rejected = 0
        self.ttft = Summary()
        self.duration = Summary()
        self.queue_wait = Summary()

    def inc(self, attr, key=None, n=1):
        with self.lock:
            if key is None:
                setattr(self, attr, getattr(self, attr) + n)
            else:
                getattr(self, attr)[key] += n

    def observe(self, attr, v):
        with self.lock:
            getattr(self, attr).observe(v)

    def render(self, in_flight, queued, draining):
        out = []

        def head(name, typ, help_):
            out.append("# HELP %s %s" % (name, help_))
            out.append("# TYPE %s %s" % (name, typ))

        with self.lock:
            head("gateway_up", "gauge", "1 while the gateway process serves requests")
            out.append("gateway_up 1")
            head("gateway_draining", "gauge", "1 after SIGTERM while in-flight requests finish")
            out.append("gateway_draining %d" % (1 if draining else 0))
            head("gateway_requests_total", "counter", "requests by path and HTTP status")
            for (p, c), n in sorted(self.requests.items()):
                out.append('gateway_requests_total{path="%s",code="%s"} %d' % (p, c, n))
            head("gateway_in_flight", "gauge", "generation requests currently at the backend")
            out.append("gateway_in_flight %d" % in_flight)
            head("gateway_queue_length", "gauge", "generation requests waiting for admission")
            out.append("gateway_queue_length %d" % queued)
            head("gateway_queue_rejected_total", "counter", "requests rejected with 429 (queue full)")
            out.append("gateway_queue_rejected_total %d" % self.queue_rejected)
            head("gateway_auth_failures_total", "counter", "requests rejected with 401")
            out.append("gateway_auth_failures_total %d" % self.auth_failures)
            head("gateway_backend_errors_total", "counter", "backend failures by kind")
            for k, n in sorted(self.backend_errors.items()):
                out.append('gateway_backend_errors_total{kind="%s"} %d' % (k, n))
            head("gateway_client_disconnects_total", "counter", "client went away, by phase")
            for k, n in sorted(self.disconnects.items()):
                out.append('gateway_client_disconnects_total{phase="%s"} %d' % (k, n))
            head("gateway_context_guard_total", "counter", "context guard outcomes (unchecked = not counted)")
            for (r, why), n in sorted(self.guard.items()):
                out.append('gateway_context_guard_total{result="%s",reason="%s"} %d' % (r, why, n))
            for name, s, help_ in (
                    ("gateway_ttft_seconds", self.ttft,
                     "request receipt to first upstream SSE event carrying generated output"),
                    ("gateway_request_duration_seconds", self.duration,
                     "request receipt to end of response, generation requests"),
                    ("gateway_queue_wait_seconds", self.queue_wait, "time spent waiting for admission")):
                head(name, "summary", help_)
                for q in self.QUANTILES:
                    out.append('%s{quantile="%s"} %s' % (name, q, _fmt(s.quantile(q))))
                out.append("%s_sum %s" % (name, _fmt(float(s.sum))))
                out.append("%s_count %d" % (name, s.count))
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------- admission
class Admission:
    """At most max_concurrency admitted, at most max_queue waiting, strict FIFO."""

    ADMITTED, REJECTED, CANCELLED = "admitted", "rejected", "cancelled"

    def __init__(self, max_concurrency, max_queue):
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.cond = threading.Condition()
        self.in_flight = 0
        self.queue = collections.deque()

    def acquire(self, gone):
        """gone(): True when the client disconnected. Returns ADMITTED / REJECTED / CANCELLED."""
        with self.cond:
            if self.in_flight < self.max_concurrency and not self.queue:
                self.in_flight += 1
                return self.ADMITTED
            if len(self.queue) >= self.max_queue:
                return self.REJECTED
            token = object()
            self.queue.append(token)
            try:
                while True:
                    if self.queue[0] is token and self.in_flight < self.max_concurrency:
                        self.queue.popleft()
                        self.in_flight += 1
                        return self.ADMITTED
                    if gone():
                        self.queue.remove(token)
                        return self.CANCELLED
                    self.cond.wait(0.25)
            finally:
                self.cond.notify_all()           # the next head may now proceed

    def release(self):
        with self.cond:
            self.in_flight -= 1
            self.cond.notify_all()

    def poke(self):
        with self.cond:
            self.cond.notify_all()

    def snapshot(self):
        with self.cond:
            return self.in_flight, len(self.queue)


# ---------------------------------------------------------------------------- client watch
class ClientWatch(threading.Thread):
    """Detects that the client closed its connection while we wait for the backend.

    Polls the client socket; readable + MSG_PEEK returning b"" means EOF. Callbacks run once.
    A client that half-closes its write side after sending the request (rare) looks the same
    as a disconnect: this is a documented limitation.
    """

    def __init__(self, sock, poll_s):
        super().__init__(daemon=True, name="client-watch")
        self.sock = sock
        self.poll_s = poll_s
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.gone = False
        self.callbacks = []

    def run(self):
        try:
            sel = selectors.DefaultSelector()
            sel.register(self.sock, selectors.EVENT_READ)
        except (OSError, ValueError):
            return
        try:
            while not self.stop_ev.is_set():
                try:
                    ready = sel.select(self.poll_s)
                except (OSError, ValueError):
                    return
                if not ready:
                    continue
                try:
                    data = self.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    data = b""
                if data:
                    return                       # client sent more bytes (pipelining): stop watching
                self._fire()
                return
        finally:
            sel.close()

    def _fire(self):
        with self.lock:
            if self.gone:
                return
            self.gone = True
            cbs = list(self.callbacks)
        for cb in cbs:
            try:
                cb()
            except Exception:  # noqa: BLE001 - best effort
                LOG.debug("client-watch callback failed", exc_info=True)

    def add_callback(self, cb):
        with self.lock:
            if not self.gone:
                self.callbacks.append(cb)
                return
        cb()

    def stop(self):
        self.stop_ev.set()
        if self.is_alive():
            self.join(2.0)


# ---------------------------------------------------------------------------- SSE scan
class SSEScan:
    """Reads a copy of the passthrough bytes to find output events, [DONE], usage, errors."""

    MAX_BUF = 16 << 20

    def __init__(self):
        self.buf = b""
        self.data = []
        self.done = False
        self.finish_reason = None
        self.error_event = None
        self.first_content = None
        self.acc = {}
        self.broken = False

    def feed(self, chunk, now):
        if self.broken:
            return
        self.buf += chunk
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i], self.buf[i + 1:]
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                self._dispatch(now)
            elif line.startswith(b"data:"):
                v = line[5:]
                self.data.append(v[1:] if v.startswith(b" ") else v)
        if len(self.buf) > self.MAX_BUF:
            self.broken = True
            self.buf = b""

    def eof(self, now):
        if self.buf and not self.broken:
            self.feed(b"\n", now)
        self._dispatch(now)

    def _dispatch(self, now):
        if not self.data:
            return
        payload = b"\n".join(self.data)
        self.data = []
        if payload.strip() == b"[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(payload.decode("utf-8", errors="replace"))
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        if obj.get("error") is not None:
            self.error_event = obj["error"]
            return
        collect_usage(obj, self.acc)
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            out = False
            d = ch.get("delta")
            if isinstance(d, dict):
                for k in ("content", "reasoning_content", "reasoning"):
                    if isinstance(d.get(k), str) and d[k]:
                        out = True
                if d.get("tool_calls"):
                    out = True
            if isinstance(ch.get("text"), str) and ch["text"]:
                out = True
            if out and self.first_content is None:
                self.first_content = now
            if ch.get("finish_reason"):
                self.finish_reason = ch["finish_reason"]


def collect_usage(obj, acc):
    """OpenAI usage (+ prompt_tokens_details.cached_tokens) and llama.cpp timings.cache_n."""
    u = obj.get("usage")
    if isinstance(u, dict):
        for k in ("prompt_tokens", "completion_tokens"):
            if isinstance(u.get(k), int):
                acc["u_" + k] = u[k]
        d = u.get("prompt_tokens_details")
        if isinstance(d, dict) and isinstance(d.get("cached_tokens"), int):
            acc["u_cached_tokens"] = d["cached_tokens"]
    t = obj.get("timings")
    if isinstance(t, dict):
        for k in ("cache_n", "prompt_n", "predicted_n"):
            if isinstance(t.get(k), int):
                acc["t_" + k] = t[k]


def usage_fields(acc):
    out = {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None, "usage_src": None}
    if "u_completion_tokens" in acc or "u_prompt_tokens" in acc:
        out["usage_src"] = "usage"
        out["prompt_tokens"] = acc.get("u_prompt_tokens")
        out["completion_tokens"] = acc.get("u_completion_tokens")
    elif "t_predicted_n" in acc:
        out["usage_src"] = "timings"
        out["completion_tokens"] = acc.get("t_predicted_n")
        if "t_prompt_n" in acc and "t_cache_n" in acc:
            out["prompt_tokens"] = acc["t_prompt_n"] + acc["t_cache_n"]
    if "u_cached_tokens" in acc:
        out["cached_tokens"] = acc["u_cached_tokens"]
    elif "t_cache_n" in acc:
        out["cached_tokens"] = acc["t_cache_n"]
    return out


# ---------------------------------------------------------------------------- backend
class BackendError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


class Backend:
    def __init__(self, url, cfg):
        u = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
        if u.scheme not in ("http", "https"):
            raise ValueError("unsupported backend scheme %r" % u.scheme)
        self.scheme = u.scheme
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if u.scheme == "https" else 80)
        root = (u.path or "").rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.root = root
        self.url = "%s://%s:%d%s" % (self.scheme, self.host, self.port, self.root)
        self.cfg = cfg

    def connect(self, read_timeout):
        """Open a fresh connection (no pooling: closing it is how we stop the engine)."""
        if self.scheme == "https":
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.cfg.connect_timeout_s,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.cfg.connect_timeout_s)
        try:
            conn.connect()
        except (socket.timeout, TimeoutError) as e:
            conn.close()
            raise BackendError("connect", "backend connect timeout: %s" % e)
        except OSError as e:
            conn.close()
            raise BackendError("connect", "backend unreachable: %s: %s" % (type(e).__name__, e))
        conn.sock.settimeout(read_timeout)
        try:
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return conn

    def headers(self, rid, body=None, accept="*/*"):
        h = {"Host": "%s:%d" % (self.host, self.port), "Accept": accept, "User-Agent": VERSION,
             "X-Request-Id": rid, "Connection": "close"}
        if body is not None:
            h["Content-Type"] = "application/json"
            h["Content-Length"] = str(len(body))
        if self.cfg.backend_api_key:
            h["Authorization"] = "Bearer " + self.cfg.backend_api_key
        return h

    def send(self, conn, method, path, body, rid, accept="*/*"):
        conn.putrequest(method, self.root + path, skip_host=True, skip_accept_encoding=True)
        for k, v in self.headers(rid, body, accept).items():
            conn.putheader(k, v)
        conn.endheaders(body)
        return conn.getresponse()

    def simple(self, method, path, body=None, timeout=30.0, rid="-"):
        """Small buffered request. Returns (status, content_type, bytes); raises BackendError."""
        conn = self.connect(timeout)
        try:
            resp = self.send(conn, method, path, body, rid)
            data = resp.read()
            return resp.status, resp.getheader("Content-Type") or "application/json", data
        except (socket.timeout, TimeoutError) as e:
            raise BackendError("timeout", "backend timeout: %s" % e)
        except (OSError, http.client.HTTPException) as e:
            raise BackendError("protocol", "backend protocol error: %s: %s" % (type(e).__name__, e))
        finally:
            conn.close()

    def simple_json(self, path, obj, timeout, rid="-"):
        st, _, data = self.simple("POST", path, json.dumps(obj).encode("utf-8"), timeout, rid)
        try:
            return st, json.loads(data.decode("utf-8", errors="replace"))
        except ValueError:
            return st, None


# ---------------------------------------------------------------------------- context guard
class ContextGuard:
    """Counts prompt tokens with the backend's tokenizer. Never estimates.

    Dialects (same knowledge as tools/bench/oa_client.py):
      llama.cpp: POST /tokenize {"content": text, "add_special": true} -> {"tokens": [...]};
                 chat prompts are rendered first by POST /apply-template (same body) -> {"prompt": text}.
      vLLM:      POST /tokenize {"model", "prompt"} or {"model", "messages"} -> {"count": n}.
    llama.cpp answers {"tokens": []} to a body it does not understand, so a zero count on the
    probe is a miss. add_special=true matches how llama-server tokenizes generation prompts.
    """

    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg
        self.lock = threading.Lock()
        self.dialect = None if cfg.tokenize_dialect in ("auto", "off") else cfg.tokenize_dialect
        self.probed_at = None

    def _detect(self, rid):
        if self.cfg.tokenize_dialect == "off":
            return None
        with self.lock:
            if self.dialect is not None:
                return self.dialect
            if self.probed_at is not None and time.monotonic() - self.probed_at < TOKENIZE_REPROBE_S:
                return None
            self.probed_at = time.monotonic()
            for d in ("llamacpp", "vllm"):
                try:
                    n = self._count_text(d, PROBE_TEXT, None, rid)
                except BackendError:
                    n = None
                if n:
                    self.dialect = d
                    LOG.info("context guard: tokenizer dialect %s detected", d)
                    return d
            LOG.warning("context guard: backend has no usable /tokenize; requests pass unchecked")
            return None

    def _count_text(self, dialect, text, model, rid):
        if dialect == "llamacpp":
            st, obj = self.backend.simple_json("/tokenize", {"content": text, "add_special": True},
                                               self.cfg.guard_timeout_s, rid)
        else:
            body = {"prompt": text}
            if model:
                body["model"] = model
            st, obj = self.backend.simple_json("/tokenize", body, self.cfg.guard_timeout_s, rid)
        if st != 200 or not isinstance(obj, dict):
            return None
        if dialect == "vllm" and isinstance(obj.get("count"), int):
            return obj["count"]
        toks = obj.get("tokens")
        return len(toks) if isinstance(toks, list) else None

    def count(self, path, obj, body, rid):
        """Returns (n_tokens, None) or (None, reason)."""
        d = self._detect(rid)
        if d is None:
            return None, "no_tokenize"
        model = obj.get("model") if isinstance(obj.get("model"), str) else None
        try:
            if path == "/v1/completions":
                p = obj.get("prompt")
                if isinstance(p, list) and p and all(isinstance(x, int) and not isinstance(x, bool) for x in p):
                    return len(p), None
                if not isinstance(p, str):
                    return None, "prompt_not_string"
                n = self._count_text(d, p, model, rid)
                return (n, None) if n is not None else (None, "tokenize_error")
            if d == "llamacpp":
                st, _, data = self.backend.simple("POST", "/apply-template", body, self.cfg.guard_timeout_s, rid)
                if st == 404:
                    return None, "no_apply_template"
                try:
                    rendered = json.loads(data.decode("utf-8", errors="replace")).get("prompt")
                except (ValueError, AttributeError):
                    rendered = None
                if st != 200 or not isinstance(rendered, str):
                    return None, "apply_template_error"
                n = self._count_text(d, rendered, model, rid)
                return (n, None) if n is not None else (None, "tokenize_error")
            req = {"messages": obj.get("messages")}
            if model:
                req["model"] = model
            if obj.get("tools") is not None:
                req["tools"] = obj["tools"]
            st, res = self.backend.simple_json("/tokenize", req, self.cfg.guard_timeout_s, rid)
            if st == 200 and isinstance(res, dict) and isinstance(res.get("count"), int):
                return res["count"], None
            return None, "tokenize_error"
        except BackendError:
            return None, "tokenize_error"


# ---------------------------------------------------------------------------- request log
class RequestLog:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.f = None
        if path == "-":
            self.f = sys.stdout
        elif path:
            self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        if self.f is None:
            return
        line = json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n"
        with self.lock:
            self.f.write(line)
            self.f.flush()

    def close(self):
        if self.f is not None and self.f is not sys.stdout:
            self.f.close()


class ReqCtx:
    def __init__(self, method, path):
        self.rid = "gw-" + uuid.uuid4().hex
        self.t0 = time.monotonic()
        self.ts = utc_now()
        self.method = method
        self.path = path
        self.model = None
        self.stream = None
        self.prompt_chars = None
        self.prompt = None
        self.status = None
        self.queue_s = None
        self.first_byte_s = None
        self.ttft_s = None
        self.total_s = None
        self.acc = {}
        self.guard = None
        self.injected = []
        self.client_disconnected = False
        self.error = None
        self.reached_backend = False

    def record(self):
        r = {"ts": self.ts, "id": self.rid, "method": self.method, "path": self.path, "model": self.model,
             "stream": self.stream, "prompt_chars": self.prompt_chars, "status": self.status,
             "queue_s": _r(self.queue_s), "first_byte_s": _r(self.first_byte_s), "ttft_s": _r(self.ttft_s),
             "total_s": _r(self.total_s)}
        r.update(usage_fields(self.acc))
        r.update({"guard": self.guard, "injected_defaults": self.injected,
                  "client_disconnected": self.client_disconnected, "error": self.error})
        if self.prompt is not None:
            r["prompt"] = self.prompt
        return r


def _r(v):
    return None if v is None else round(v, 6)


# ---------------------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = VERSION

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, fmt, *args):
        LOG.debug("http: " + fmt, *args)

    @property
    def gw(self):
        return self.server.gw

    # -- plumbing ---------------------------------------------------------------
    def _send_bytes(self, ctx, code, data, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if ctx is not None:
            self.send_header("X-Request-Id", ctx.rid)
            ctx.status = code
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if self.gw.draining:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, ctx, code, obj, headers=None):
        self._send_bytes(ctx, code, json.dumps(obj).encode("utf-8"), "application/json", headers)

    def _err(self, ctx, code, msg, typ, ecode, headers=None, param=None):
        if ctx is not None and ctx.error is None:
            ctx.error = msg
        try:
            self._send_json(ctx, code, oa_error(msg, typ, ecode, param), headers)
        except OSError:
            self.close_connection = True

    def _route(self):
        return urllib.parse.urlsplit(self.path).path

    # -- routes -------------------------------------------------------------------
    def do_GET(self):
        p = self._route()
        if p == "/health":
            return self._health()
        if p == "/metrics":
            return self._metrics()
        if p in PROXY_GET:
            return self._wrapped(p, self._proxy)
        return self._wrapped(p, self._not_found)

    def do_POST(self):
        p = self._route()
        if p in GEN_PATHS:
            return self._wrapped(p, self._generation)
        if p in PROXY_POST:
            return self._wrapped(p, self._proxy)
        return self._wrapped(p, self._not_found)

    def _method_not_allowed(self):
        self._wrapped(self._route(), lambda ctx: self._err(ctx, 405, "method not allowed", "invalid_request_error",
                                                            "method_not_allowed"))

    do_PUT = do_DELETE = do_PATCH = do_HEAD = _method_not_allowed

    def _wrapped(self, path, fn):
        ctx = ReqCtx(self.command, path)
        self.gw.begin()
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001 - never let a handler bug kill the connection silently
            LOG.exception("request %s failed", ctx.rid)
            ctx.error = ctx.error or "gateway internal error: %s" % type(e).__name__
            if ctx.status is None:
                self._err(ctx, 500, "gateway internal error", "server_error", "gateway_error")
            self.close_connection = True
        finally:
            ctx.total_s = time.monotonic() - ctx.t0
            self.gw.finish(ctx)

    def _not_found(self, ctx):
        self._discard_body()
        self._err(ctx, 404, "not found: %s" % ctx.path, "invalid_request_error", "not_found")

    def _discard_body(self):
        n = self.headers.get("Content-Length")
        if n and n.isdigit() and int(n) <= self.gw.cfg.max_body_bytes:
            self.rfile.read(int(n))
        elif n:
            self.close_connection = True

    def _auth(self, ctx):
        keys = self.gw.cfg.api_keys
        if keys is None:
            return True
        h = self.headers.get("Authorization") or ""
        tok = h[7:].strip() if h[:7].lower() == "bearer " else ""
        ok = False
        for k in keys:                                  # no early exit: same work for every key
            ok = hmac.compare_digest(tok.encode("utf-8"), k.encode("utf-8")) or ok
        if ok and tok:
            return True
        self.gw.metrics.inc("auth_failures")
        self._discard_body()
        self._err(ctx, 401, "invalid or missing API key", "authentication_error", "invalid_api_key",
                  headers={"WWW-Authenticate": "Bearer"})
        return False

    def _read_body(self, ctx):
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self._err(ctx, 411, "chunked request bodies are not supported; send Content-Length",
                      "invalid_request_error", "length_required")
            return None
        n = self.headers.get("Content-Length") or "0"
        if not n.isdigit():
            self.close_connection = True
            self._err(ctx, 400, "bad Content-Length", "invalid_request_error", "bad_request")
            return None
        n = int(n)
        if n > self.gw.cfg.max_body_bytes:
            self.close_connection = True
            self._err(ctx, 413, "request body larger than %d bytes" % self.gw.cfg.max_body_bytes,
                      "invalid_request_error", "body_too_large")
            return None
        return self.rfile.read(n) if n else b""

    # -- own endpoints ------------------------------------------------------------
    def _health(self):
        self._send_json(None, *self.gw.health())

    def _metrics(self):
        self._send_bytes(None, 200, self.gw.render_metrics().encode("utf-8"), "text/plain; version=0.0.4")

    # -- simple proxy -------------------------------------------------------------
    def _proxy(self, ctx):
        if not self._auth(ctx):
            return
        body = None
        if self.command == "POST":
            body = self._read_body(ctx)
            if body is None:
                return
        try:
            st, ctype, data = self.gw.backend.simple(self.command, self.path, body,
                                                     self.gw.cfg.guard_timeout_s, ctx.rid)
        except BackendError as e:
            self.gw.metrics.inc("backend_errors", e.kind)
            return self._err(ctx, 502, e.message, "upstream_error", "backend_unreachable")
        if st >= 500:
            self.gw.metrics.inc("backend_errors", "http_5xx")
            return self._upstream_5xx(ctx, st, data)
        self._send_bytes(ctx, st, data, ctype)

    def _upstream_5xx(self, ctx, st, data):
        msg = data[:MAX_ERROR_BODY].decode("utf-8", errors="replace")
        try:
            e = json.loads(msg).get("error")
            msg = e.get("message", msg) if isinstance(e, dict) else (e if isinstance(e, str) else msg)
        except (ValueError, AttributeError):
            pass
        text = "backend returned HTTP %d: %s" % (st, msg)
        if st == 503:
            return self._err(ctx, 503, text, "unavailable_error", "backend_unavailable")
        return self._err(ctx, 502, text, "upstream_error", "backend_error")

    # -- generation -----------------------------------------------------------------
    def _generation(self, ctx):
        gw = self.gw
        if not self._auth(ctx):
            return
        if gw.draining:
            self._discard_body()
            self.close_connection = True
            return self._err(ctx, 503, "gateway is shutting down", "unavailable_error", "draining")
        raw = self._read_body(ctx)
        if raw is None:
            return
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            obj = None
        if not isinstance(obj, dict):
            return self._err(ctx, 400, "request body must be a JSON object", "invalid_request_error",
                             "invalid_json")
        ctx.model = obj.get("model") if isinstance(obj.get("model"), str) else None
        ctx.stream = obj.get("stream") is True
        ctx.prompt_chars = prompt_chars(ctx.path, obj)
        if gw.cfg.log_prompts:
            ctx.prompt = obj.get("messages") if ctx.path == "/v1/chat/completions" else obj.get("prompt")
        body, ctx.injected = apply_defaults(raw, obj, gw.cfg.set_defaults)

        watch = ClientWatch(self.connection, gw.cfg.watch_poll_s)
        watch.add_callback(gw.admission.poke)
        watch.start()
        try:
            if gw.guard is not None:
                if self._guard_rejects(ctx, obj, body):
                    return
            if watch.gone:
                return self._client_gone(ctx, "queued")
            t_q = time.monotonic()
            res = gw.admission.acquire(lambda: watch.gone)
            ctx.queue_s = time.monotonic() - t_q
            gw.metrics.observe("queue_wait", ctx.queue_s)
            if res == Admission.REJECTED:
                gw.metrics.inc("queue_rejected")
                _, q = gw.admission.snapshot()
                return self._err(ctx, 429, "server busy: %d request(s) waiting (max %d); retry later"
                                 % (q, gw.cfg.max_queue), "rate_limit_error", "queue_full",
                                 headers={"Retry-After": str(int(gw.cfg.retry_after_s))})
            if res == Admission.CANCELLED:
                gw.metrics.inc("disconnects", "queued")
                ctx.status, ctx.client_disconnected = 499, True
                ctx.error = "client disconnected while queued"
                self.close_connection = True
                return
            try:
                self._forward(ctx, body, watch)
            finally:
                gw.admission.release()
        finally:
            watch.stop()

    def _guard_rejects(self, ctx, obj, body):
        gw = self.gw
        limit = gw.cfg.max_context_tokens
        n, why = gw.guard.count(ctx.path, obj, body, ctx.rid)
        if n is None:
            gw.metrics.inc("guard", ("unchecked", why))
            ctx.guard = "unchecked:" + why
            return False
        mt = requested_max_tokens(obj)
        need = n + (mt or 0)
        if need > limit:
            gw.metrics.inc("guard", ("rejected", "over_limit"))
            ctx.guard = "rejected:%d+%s>%d" % (n, mt, limit)
            self._err(ctx, 400, "This gateway's maximum context length is %d tokens. However, you requested "
                      "%d tokens (%d in the prompt, %s for the completion). Please reduce the length."
                      % (limit, need, n, mt if mt is not None else "none requested"),
                      "invalid_request_error", "context_length_exceeded",
                      param="messages" if ctx.path == "/v1/chat/completions" else "prompt")
            return True
        gw.metrics.inc("guard", ("ok", "within_limit" if mt is not None else "no_max_tokens"))
        ctx.guard = "ok:%d+%s<=%d" % (n, mt, limit)
        return False

    def _forward(self, ctx, body, watch):
        gw = self.gw
        try:
            conn = gw.backend.connect(gw.cfg.backend_timeout_s)
        except BackendError as e:
            gw.metrics.inc("backend_errors", e.kind)
            return self._err(ctx, 502, e.message, "upstream_error", "backend_unreachable")
        raw_sock = conn.sock

        def kill_upstream():
            try:
                raw_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        watch.add_callback(kill_upstream)
        ctx.reached_backend = True
        try:
            accept = self.headers.get("Accept") or "*/*"
            try:
                resp = gw.backend.send(conn, "POST", self.path, body, ctx.rid, accept)
            except (socket.timeout, TimeoutError) as e:
                if watch.gone:
                    return self._client_gone(ctx, "upstream_wait")
                gw.metrics.inc("backend_errors", "timeout")
                return self._err(ctx, 504, "backend did not answer within %.0fs: %s"
                                 % (gw.cfg.backend_timeout_s, e), "upstream_error", "backend_timeout")
            except (OSError, http.client.HTTPException) as e:
                if watch.gone:
                    return self._client_gone(ctx, "upstream_wait")
                gw.metrics.inc("backend_errors", "protocol")
                return self._err(ctx, 502, "backend closed the connection before responding: %s: %s"
                                 % (type(e).__name__, e), "upstream_error", "backend_error")
            ctx.first_byte_s = time.monotonic() - ctx.t0
            ctype = resp.getheader("Content-Type") or "application/json"
            if resp.status >= 500:
                gw.metrics.inc("backend_errors", "http_5xx")
                try:
                    data = resp.read(MAX_ERROR_BODY)
                except (OSError, http.client.HTTPException):
                    data = b""
                return self._upstream_5xx(ctx, resp.status, data)
            if resp.status == 200 and "text/event-stream" in ctype.lower():
                return self._relay_stream(ctx, resp, ctype, watch, kill_upstream)
            try:
                data = resp.read()
            except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as e:
                if watch.gone:
                    return self._client_gone(ctx, "upstream_wait")
                gw.metrics.inc("backend_errors", "truncated")
                return self._err(ctx, 502, "backend response truncated: %s: %s" % (type(e).__name__, e),
                                 "upstream_error", "backend_truncated")
            if "json" in ctype.lower():
                try:
                    o = json.loads(data.decode("utf-8", errors="replace"))
                    if isinstance(o, dict):
                        collect_usage(o, ctx.acc)
                        if resp.status != 200 and o.get("error") is not None:
                            e = o["error"]
                            ctx.error = "backend %d: %s" % (resp.status, (e.get("message") if isinstance(e, dict)
                                                                          else e))
                except ValueError:
                    pass
            if resp.status != 200 and ctx.error is None:
                ctx.error = "backend %d" % resp.status
            try:
                self._send_bytes(ctx, resp.status, data, ctype)
            except OSError:
                self._client_gone(ctx, "upstream_wait", status=ctx.status)
        finally:
            conn.close()

    def _client_gone(self, ctx, phase, status=499):
        self.gw.metrics.inc("disconnects", phase)
        ctx.client_disconnected = True
        ctx.status = status
        ctx.error = ctx.error or "client disconnected (%s)" % phase
        self.close_connection = True

    def _write_chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def _relay_stream(self, ctx, resp, ctype, watch, kill_upstream):
        gw = self.gw
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Request-Id", ctx.rid)
        self.end_headers()
        ctx.status = 200
        scan = SSEScan()
        tail = b""
        upstream_exc = None
        client_lost = False
        while True:
            try:
                data = resp.read1(65536)
            except (socket.timeout, TimeoutError, OSError, http.client.HTTPException, ValueError) as e:
                upstream_exc = e
                break
            if not data:
                break
            scan.feed(data, time.monotonic())
            try:
                self._write_chunk(data)
            except OSError:
                client_lost = True
                break
            tail = (tail + data)[-4:]
        complete = upstream_exc is None and (scan.done or scan.finish_reason is not None)
        if client_lost or (watch.gone and not complete):
            kill_upstream()                          # the engine sees the connection close
            resp.close()
            self._client_gone(ctx, "streaming", status=200)
            self._finish_stream_timing(ctx, scan)
            return
        scan.eof(time.monotonic())
        self._finish_stream_timing(ctx, scan)
        truncated = upstream_exc is not None or not (scan.done or scan.finish_reason or scan.error_event)
        if scan.error_event is not None:
            ctx.error = "backend error event: %s" % json.dumps(scan.error_event)[:MAX_ERROR_BODY]
        try:
            if truncated and scan.error_event is None:
                gw.metrics.inc("backend_errors", "truncated")
                why = ("%s: %s" % (type(upstream_exc).__name__, upstream_exc)) if upstream_exc else "EOF"
                ctx.error = "backend stream truncated (%s)" % why
                ev = {"error": {"message": "backend stream ended before completion (%s); the response is "
                                           "incomplete" % why,
                                "type": "upstream_error", "param": None, "code": "backend_truncated"}}
                sep = b"" if tail.endswith(b"\n\n") or tail.endswith(b"\r\n\r\n") or not tail else b"\n\n"
                self._write_chunk(sep + b"data: " + json.dumps(ev).encode("utf-8") + b"\n\n")
                self.close_connection = True
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            if truncated:
                self._client_gone(ctx, "streaming", status=200)
            else:                                    # client left after [DONE]: response was complete
                self.close_connection = True

    @staticmethod
    def _finish_stream_timing(ctx, scan):
        ctx.acc.update(scan.acc)
        if scan.first_content is not None:
            ctx.ttft_s = scan.first_content - ctx.t0


# ---------------------------------------------------------------------------- server
class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = True


class Gateway:
    """In-process gateway: Gateway(cfg).start(); .url; .shutdown(drain_s) -> drained?; .stop()."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = Backend(cfg.backend, cfg)
        self.metrics = Metrics()
        self.admission = Admission(cfg.max_concurrency, cfg.max_queue)
        self.guard = ContextGuard(self.backend, cfg) if cfg.max_context_tokens else None
        self.reqlog = RequestLog(cfg.log_file)
        self.draining = False
        self.active = 0
        self.active_cond = threading.Condition()
        self.started_at = time.monotonic()
        self.httpd = _Server((cfg.listen_host, cfg.listen_port), Handler)
        self.httpd.gw = self
        self.thread = None
        self._closed = False

    @property
    def url(self):
        h, p = self.httpd.server_address[:2]
        return "http://%s:%d" % (h, p)

    def start(self):
        if self.cfg.api_keys is None:
            LOG.warning("auth is OFF: no --api-keys-file given; anyone who can reach %s can use the backend",
                        self.url)
        else:
            LOG.info("auth is on: %d API key(s) loaded", len(self.cfg.api_keys))
        if self.cfg.max_context_tokens:
            LOG.info("context guard: max %d tokens (tokenizer dialect %s)", self.cfg.max_context_tokens,
                     self.cfg.tokenize_dialect)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True, name="gateway-accept")
        self.thread.start()
        LOG.info("listening on %s -> backend %s (max_concurrency=%d, max_queue=%d)", self.url, self.backend.url,
                 self.cfg.max_concurrency, self.cfg.max_queue)
        return self

    # -- request accounting ----------------------------------------------------------
    def begin(self):
        with self.active_cond:
            self.active += 1

    def finish(self, ctx):
        if ctx.status is not None:
            p = ctx.path if ctx.path in METRIC_PATHS else "other"
            self.metrics.inc("requests", (p, str(ctx.status)))
        if ctx.path in GEN_PATHS and ctx.reached_backend:
            self.metrics.observe("duration", ctx.total_s)
            if ctx.ttft_s is not None:
                self.metrics.observe("ttft", ctx.ttft_s)
        try:
            self.reqlog.write(ctx.record())
        except Exception:  # noqa: BLE001 - logging must not break serving
            LOG.exception("request log write failed")
        with self.active_cond:
            self.active -= 1
            self.active_cond.notify_all()

    # -- own endpoints ---------------------------------------------------------------
    def health(self):
        b = {"url": self.backend.url, "reachable": False, "http_status": None, "latency_ms": None, "error": None}
        t = time.monotonic()
        try:
            st, _, _ = self.backend.simple("GET", "/health", None, self.cfg.health_timeout_s, "health")
            b.update(reachable=True, http_status=st)
        except BackendError as e:
            b["error"] = e.message
        b["latency_ms"] = round((time.monotonic() - t) * 1000.0, 3)
        in_flight, queued = self.admission.snapshot()
        if self.draining:
            status = "draining"
        elif b["reachable"] and b["http_status"] == 200:
            status = "ok"
        else:
            status = "backend_unavailable"
        obj = {"status": status, "draining": self.draining, "auth": self.cfg.api_keys is not None,
               "in_flight": in_flight, "queued": queued, "max_concurrency": self.cfg.max_concurrency,
               "max_queue": self.cfg.max_queue, "uptime_s": round(time.monotonic() - self.started_at, 3),
               "context_guard": {"max_context_tokens": self.cfg.max_context_tokens,
                                 "tokenize_dialect": self.guard.dialect if self.guard else None},
               "backend": b}
        return (200 if status == "ok" else 503), obj

    def render_metrics(self):
        in_flight, queued = self.admission.snapshot()
        return self.metrics.render(in_flight, queued, self.draining)

    # -- shutdown --------------------------------------------------------------------
    def shutdown(self, drain_s=None):
        """Stop accepting, wait for in-flight and queued requests up to drain_s. True if drained."""
        drain_s = self.cfg.drain_s if drain_s is None else drain_s
        self.draining = True
        if not self._closed:
            self._closed = True
            if self.thread is not None:
                self.httpd.shutdown()            # stop the accept loop
            self.httpd.server_close()            # close the listening socket: new connections are refused
        LOG.info("draining: waiting up to %.1fs for %d request(s)", drain_s, self.active)
        deadline = time.monotonic() + drain_s
        with self.active_cond:
            while self.active > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.active_cond.wait(min(left, 0.25))
            drained = self.active == 0
        if drained:
            LOG.info("drained")
        else:
            LOG.warning("drain timeout: %d request(s) still active", self.active)
        if drained:
            self.reqlog.close()
        return drained

    def stop(self):
        self.shutdown(0.0)


# ---------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8000", help="HOST:PORT to listen on (default %(default)s)")
    ap.add_argument("--backend", required=True, help="engine base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--api-keys-file", help="one bearer key per line, '#' comments; without it auth is OFF")
    ap.add_argument("--backend-api-key-file", help="file with the key the gateway sends to the engine")
    ap.add_argument("--max-concurrency", type=int, default=1, help="requests in flight to the engine")
    ap.add_argument("--max-queue", type=int, default=8, help="requests waiting; beyond that 429")
    ap.add_argument("--max-context-tokens", type=int, default=None,
                    help="reject prompt+max_tokens above this (needs backend /tokenize; else pass + count)")
    ap.add_argument("--tokenize-dialect", choices=("auto", "llamacpp", "vllm", "off"), default="auto")
    ap.add_argument("--set-default", action="append", default=[], metavar="KEY=JSON",
                    help="add KEY to generation bodies that lack it, e.g. cache_prompt=true (repeatable)")
    ap.add_argument("--log-file", help="JSONL request log ('-' = stdout)")
    ap.add_argument("--log-prompts", action="store_true", help="include prompt text/messages in the request log")
    ap.add_argument("--connect-timeout-s", type=float, default=5.0)
    ap.add_argument("--backend-timeout-s", type=float, default=3600.0,
                    help="per blocking read from the engine; must exceed the longest prefill / non-stream generation")
    ap.add_argument("--guard-timeout-s", type=float, default=30.0, help="timeout for /tokenize, /apply-template, proxies")
    ap.add_argument("--retry-after-s", type=int, default=5, help="Retry-After value sent with 429")
    ap.add_argument("--drain-s", type=float, default=30.0, help="on SIGTERM wait this long for in-flight requests")
    ap.add_argument("--max-body-mb", type=float, default=64.0)
    ap.add_argument("--verbose", action="store_true")
    return ap


def config_from_args(a):
    host, _, port = a.listen.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError("--listen expects HOST:PORT, got %r" % a.listen)
    keys = None
    if a.api_keys_file:
        keys = parse_api_keys(a.api_keys_file)
        if not keys:
            raise ValueError("--api-keys-file %s contains no keys (refusing to start with auth on and no keys)"
                             % a.api_keys_file)
    backend_key = None
    if a.backend_api_key_file:
        bk = parse_api_keys(a.backend_api_key_file)
        backend_key = bk[0] if bk else None
    return GatewayConfig(
        a.backend, listen_host=host.strip("[]"), listen_port=int(port), api_keys=keys, backend_api_key=backend_key,
        max_concurrency=a.max_concurrency, max_queue=a.max_queue, max_context_tokens=a.max_context_tokens,
        tokenize_dialect=a.tokenize_dialect, set_defaults=[parse_set_default(s) for s in a.set_default],
        log_file=a.log_file, log_prompts=a.log_prompts, connect_timeout_s=a.connect_timeout_s,
        backend_timeout_s=a.backend_timeout_s, guard_timeout_s=a.guard_timeout_s, retry_after_s=a.retry_after_s,
        drain_s=a.drain_s, max_body_bytes=int(a.max_body_mb * (1 << 20)))


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s gateway %(levelname)s %(message)s")
    try:
        cfg = config_from_args(a)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    gw = Gateway(cfg).start()
    stop = threading.Event()

    def on_signal(signum, _frame):
        LOG.info("signal %d: stop accepting, drain up to %.1fs", signum, cfg.drain_s)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while not stop.wait(0.5):
        pass
    drained = gw.shutdown(cfg.drain_s)
    return 0 if drained else 1


if __name__ == "__main__":
    sys.exit(main())
