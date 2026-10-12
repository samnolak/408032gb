"""Minimal OpenAI-compatible HTTP client, Python 3 stdlib only.

Talks to /v1/chat/completions and /v1/completions with SSE streaming and records the
arrival time (time.perf_counter_ns) of every streamed chunk. Nothing here invents a
number: a field the server did not send stays None, a transport problem becomes `error`.

Optional helpers fetch /v1/models, /health, /metrics as raw text and count tokens via
POST /tokenize in either of the two known dialects:
  llama.cpp : {"content": text, "add_special": false}        -> {"tokens": [...]}
  vLLM      : {"model": m, "prompt": text, "add_special_tokens": false}
                                                             -> {"count": n, "tokens": [...]}
"""
from __future__ import annotations

import http.client
import json
import socket
import ssl
import time
import urllib.parse

__all__ = ["OAClient", "StreamResult", "HTTPResult"]

_MAX_ERROR_BODY = 4000


class HTTPResult:
    """Outcome of a non-streaming request."""

    def __init__(self, status=None, text=None, elapsed_ms=None, error=None):
        self.status = status
        self.text = text
        self.elapsed_ms = elapsed_ms
        self.error = error

    def json(self):
        if self.text is None:
            return None
        try:
            return json.loads(self.text)
        except ValueError:
            return None

    @property
    def ok(self):
        return self.error is None and self.status == 200

    def to_dict(self, max_text=20000):
        t = self.text
        return {"status": self.status, "elapsed_ms": self.elapsed_ms, "error": self.error,
                "text": t if t is None or len(t) <= max_text else t[:max_text] + "...[truncated]"}


class StreamResult:
    """Everything observed while reading one streamed response.

    events: list of [t_us, content_chars, reasoning_chars, tool_call_deltas, text_chars]
            where t_us is microseconds since the request was sent; only chunks that carry
            generated output are recorded (role-only / empty deltas are not tokens).
    """

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.ok = False
        self.error = None
        self.http_status = None
        self.attempts = 1
        self.wall_start_utc = None
        self.connect_ms = None
        self.headers_ms = None
        self.t_end_us = None
        self.events = []
        self.n_sse_events = 0
        self.parse_errors = 0
        self.done_seen = False
        self.finish_reason = None
        self.usage = None
        self.server_timings = None
        self.content = ""
        self.reasoning = ""
        self.tool_calls = []
        self.notes = []
        self.request_keys = []

    def to_dict(self, keep_text=True, max_text=None):
        def cut(s):
            if max_text is None or s is None or len(s) <= max_text:
                return s
            return s[:max_text]
        d = {
            "ok": self.ok, "error": self.error, "http_status": self.http_status,
            "endpoint": self.endpoint, "attempts": self.attempts,
            "wall_start_utc": self.wall_start_utc, "connect_ms": self.connect_ms,
            "headers_ms": self.headers_ms, "t_end_us": self.t_end_us,
            "events": self.events, "n_sse_events": self.n_sse_events,
            "parse_errors": self.parse_errors, "done_seen": self.done_seen,
            "finish_reason": self.finish_reason, "usage": self.usage,
            "server_timings": self.server_timings, "tool_calls": self.tool_calls,
            "notes": self.notes, "request_keys": self.request_keys,
            "content_chars": len(self.content), "reasoning_chars": len(self.reasoning),
        }
        if keep_text:
            d["content"] = cut(self.content)
            d["reasoning"] = cut(self.reasoning)
        return d


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class OAClient:
    """OpenAI-compatible client. `url` may be http://host:port or http://host:port/v1."""

    def __init__(self, url, api_key=None, timeout=600.0, total_timeout=None, model=None,
                 stream_options="auto", tokenize="auto", extra_headers=None):
        u = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
        if u.scheme not in ("http", "https"):
            raise ValueError("unsupported URL scheme: %r" % u.scheme)
        self.scheme = u.scheme
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if u.scheme == "https" else 80)
        path = (u.path or "").rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        self.root = path                                   # prefix before /v1, usually ""
        self.url = "%s://%s:%d%s" % (self.scheme, self.host, self.port, self.root)
        self.api_key = api_key
        self.timeout = float(timeout)                     # per blocking socket operation
        self.total_timeout = total_timeout                 # whole request, None = unlimited
        self.model = model
        self.extra_headers = dict(extra_headers or {})
        # stream_options: "auto" (send, drop on 400/422), "on" (always send), "off" (never send)
        self.stream_options_mode = stream_options
        self.stream_options_supported = None if stream_options == "auto" else (stream_options == "on")
        # tokenize: "auto" (detect), "llamacpp", "vllm", "off"
        self.tokenize_mode = tokenize
        self.tokenize_style = None if tokenize == "auto" else (None if tokenize == "off" else tokenize)
        self.tokenize_probe_log = []

    # ------------------------------------------------------------------ transport
    def _conn(self, timeout=None):
        t = self.timeout if timeout is None else timeout
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=t,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=t)

    def _headers(self, json_body=True):
        h = {"Accept": "*/*", "Connection": "close", "User-Agent": "408032gb-bench/1"}
        if json_body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        h.update(self.extra_headers)
        return h

    def request(self, method, path, body=None, timeout=None):
        """Non-streaming request; returns HTTPResult (never raises for transport errors)."""
        t0 = time.perf_counter_ns()
        conn = None
        try:
            conn = self._conn(timeout)
            data = None if body is None else json.dumps(body).encode("utf-8")
            conn.request(method, self.root + path, body=data, headers=self._headers(data is not None))
            resp = conn.getresponse()
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return HTTPResult(resp.status, text, (time.perf_counter_ns() - t0) / 1e6)
        except (socket.timeout, TimeoutError) as e:
            return HTTPResult(None, None, (time.perf_counter_ns() - t0) / 1e6, "timeout: %s" % e)
        except Exception as e:  # noqa: BLE001 - every transport failure is data
            return HTTPResult(None, None, (time.perf_counter_ns() - t0) / 1e6,
                              "%s: %s" % (type(e).__name__, e))
        finally:
            if conn is not None:
                conn.close()

    def get_text(self, path, timeout=30.0):
        return self.request("GET", path, timeout=timeout)

    # ------------------------------------------------------------------ helpers
    def models(self):
        return self.get_text("/v1/models")

    def health(self):
        return self.get_text("/health")

    def metrics(self):
        return self.get_text("/metrics")

    def resolve_model(self):
        """Return self.model, or the first id from /v1/models. None if unavailable."""
        if self.model:
            return self.model
        r = self.models()
        obj = r.json() if r.ok else None
        try:
            self.model = obj["data"][0]["id"]
        except (TypeError, KeyError, IndexError):
            return None
        return self.model

    def _tokenize_style(self, style, text):
        if style == "llamacpp":
            body = {"content": text, "add_special": False}
        else:
            body = {"prompt": text, "add_special_tokens": False}
            if self.model:
                body["model"] = self.model
        r = self.request("POST", "/tokenize", body)
        if not r.ok:
            return None, r
        obj = r.json()
        if not isinstance(obj, dict):
            return None, r
        if style == "vllm" and isinstance(obj.get("count"), int):
            return obj["count"], r
        toks = obj.get("tokens")
        if isinstance(toks, list):
            return len(toks), r
        return None, r

    def tokenize_count(self, text):
        """Exact token count of `text` from the server, or None if /tokenize is unusable."""
        if self.tokenize_mode == "off":
            return None
        if self.tokenize_style is None:
            if self.tokenize_probe_log:          # detection already ran and failed
                return None
            probe = "The quick brown fox jumps over the lazy dog."
            for style in ("llamacpp", "vllm"):
                n, r = self._tokenize_style(style, probe)
                self.tokenize_probe_log.append({"style": style, "count": n, "http": r.to_dict(400)})
                # llama.cpp answers {"tokens": []} to a body without "content": a zero count is a miss
                if n is not None and n > 0:
                    self.tokenize_style = style
                    break
            if self.tokenize_style is None:
                return None
        n, _ = self._tokenize_style(self.tokenize_style, text)
        return n

    # ------------------------------------------------------------------ generation
    def chat_stream(self, messages, max_tokens, temperature=0.0, extra=None, tools=None,
                    tool_choice=None, response_format=None, seed=None, model=None):
        body = {"model": model or self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if seed is not None:
            body["seed"] = seed
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if extra:
            body.update(extra)
        return self.stream("/v1/chat/completions", body)

    def completion_stream(self, prompt, max_tokens, temperature=0.0, extra=None, seed=None, model=None):
        body = {"model": model or self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature}
        if seed is not None:
            body["seed"] = seed
        if extra:
            body.update(extra)
        return self.stream("/v1/completions", body)

    def chat(self, messages, max_tokens, temperature=0.0, extra=None, **kw):
        """Non-streaming chat completion; returns HTTPResult (body JSON in .json())."""
        body = {"model": kw.get("model") or self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        for k in ("tools", "tool_choice", "response_format", "seed"):
            if kw.get(k) is not None:
                body[k] = kw[k]
        if extra:
            body.update(extra)
        return self.request("POST", "/v1/chat/completions", body, timeout=self.timeout)

    def stream(self, path, body):
        body = dict(body)
        body["stream"] = True
        send_so = self.stream_options_supported is not False and self.stream_options_mode != "off"
        if send_so:
            body["stream_options"] = {"include_usage": True}
        res = self._stream_once(path, body)
        if send_so and self.stream_options_mode == "auto" and self.stream_options_supported is None:
            if res.http_status in (400, 422):
                body2 = dict(body)
                body2.pop("stream_options", None)
                res2 = self._stream_once(path, body2)
                if res2.http_status == 200:
                    self.stream_options_supported = False
                    res2.attempts = 2
                    res2.notes.append("stream_options rejected (HTTP %s); retried without it" % res.http_status)
                    return res2
                res.notes.append("retry without stream_options also failed (HTTP %s)" % res2.http_status)
            elif res.http_status == 200:
                self.stream_options_supported = True
        return res

    def _stream_once(self, path, body):
        r = StreamResult(path)
        r.request_keys = sorted(body.keys())
        r.wall_start_utc = _utc_now()
        conn = None
        t0 = time.perf_counter_ns()
        t_send = t0
        deadline = None if not self.total_timeout else t0 + int(self.total_timeout * 1e9)
        tc_by_index = {}
        try:
            conn = self._conn()
            conn.connect()
            t_send = time.perf_counter_ns()
            r.connect_ms = (t_send - t0) / 1e6
            payload = json.dumps(body).encode("utf-8")
            hdrs = self._headers(True)
            hdrs["Accept"] = "text/event-stream"
            conn.request("POST", self.root + path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            r.headers_ms = (time.perf_counter_ns() - t_send) / 1e6
            r.http_status = resp.status
            if resp.status != 200:
                txt = resp.read(_MAX_ERROR_BODY).decode("utf-8", errors="replace")
                r.error = "HTTP %d: %s" % (resp.status, txt)
                return r
            ctype = (resp.getheader("Content-Type") or "").lower()
            if "text/event-stream" not in ctype:
                txt = resp.read(_MAX_ERROR_BODY).decode("utf-8", errors="replace")
                r.error = "response is not text/event-stream (Content-Type %r): %s" % (ctype, txt[:500])
                return r
            data_lines = []
            t_data = None
            while True:
                if deadline is not None and time.perf_counter_ns() > deadline:
                    r.error = "total timeout %.1fs exceeded" % self.total_timeout
                    break
                line = resp.readline()
                now = time.perf_counter_ns()
                if not line:
                    break                                         # EOF
                s = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if s == "":
                    if data_lines:
                        stop = self._dispatch("\n".join(data_lines), t_data - t_send, r, tc_by_index)
                        data_lines = []
                        if stop:
                            break
                    continue
                if s.startswith(":"):
                    continue                                      # SSE comment / keep-alive
                if s.startswith("data:"):
                    if not data_lines:
                        t_data = now
                    data_lines.append(s[5:][1:] if s[5:6] == " " else s[5:])
                # "event:", "id:", "retry:" lines carry nothing we need
            if data_lines and r.error is None:
                self._dispatch("\n".join(data_lines), t_data - t_send, r, tc_by_index)
        except (socket.timeout, TimeoutError) as e:
            r.error = "timeout after %.1fs idle: %s" % (self.timeout, e)
        except Exception as e:  # noqa: BLE001 - transport/protocol failures are recorded, not raised
            r.error = "%s: %s" % (type(e).__name__, e)
        finally:
            r.t_end_us = (time.perf_counter_ns() - t_send) // 1000
            if conn is not None:
                conn.close()
        r.tool_calls = [tc_by_index[i] for i in sorted(tc_by_index)]
        if r.error is None and r.finish_reason is None:
            r.error = "stream ended without finish_reason (truncated?)"
        if r.error is None and not r.done_seen:
            r.notes.append("no [DONE] sentinel")
        r.ok = r.error is None
        return r

    @staticmethod
    def _dispatch(data, t_rel_ns, r, tc_by_index):
        """Handle one SSE event. Returns True when the stream is finished."""
        if data.strip() == "[DONE]":
            r.done_seen = True
            return True
        r.n_sse_events += 1
        try:
            obj = json.loads(data)
        except ValueError:
            r.parse_errors += 1
            return False
        if not isinstance(obj, dict):
            r.parse_errors += 1
            return False
        if obj.get("error") is not None:
            r.error = "server error event: %s" % json.dumps(obj.get("error"))[:_MAX_ERROR_BODY]
            return True
        if obj.get("usage"):
            r.usage = obj["usage"]
        if obj.get("timings"):
            r.server_timings = obj["timings"]                     # llama.cpp extension
        n_content = n_reason = n_tool = n_text = 0
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict) or ch.get("index", 0) != 0:
                continue
            delta = ch.get("delta")
            if isinstance(delta, dict):
                c = delta.get("content")
                if isinstance(c, str) and c:
                    r.content += c
                    n_content += len(c)
                for key in ("reasoning_content", "reasoning"):
                    rc = delta.get(key)
                    if isinstance(rc, str) and rc:
                        r.reasoning += rc
                        n_reason += len(rc)
                        break
                for tc in delta.get("tool_calls") or []:
                    n_tool += 1
                    idx = tc.get("index", 0)
                    cur = tc_by_index.setdefault(idx, {"id": None, "type": "function",
                                                       "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        cur["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if name:
                        if not cur["function"]["name"]:
                            cur["function"]["name"] = name
                        elif name != cur["function"]["name"]:
                            cur["function"]["name"] += name
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        cur["function"]["arguments"] += args
                    elif isinstance(args, dict):
                        cur["function"]["arguments"] += json.dumps(args)
            txt = ch.get("text")                                  # /v1/completions
            if isinstance(txt, str) and txt:
                r.content += txt
                n_text += len(txt)
            if ch.get("finish_reason"):
                r.finish_reason = ch["finish_reason"]
        if n_content or n_reason or n_tool or n_text:
            r.events.append([t_rel_ns // 1000, n_content, n_reason, n_tool, n_text])
        return False
