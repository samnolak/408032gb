#!/usr/bin/env python3
"""Mock OpenAI-compatible server for testing the bench harness (stdlib only, no GPU).

Implements GET /v1/models, /health, /metrics; POST /v1/chat/completions, /v1/completions
(streaming SSE with chunked transfer, and non-streaming), POST /tokenize (llama.cpp or vLLM
dialect). Timing is simulated with sleeps:

  TTFT   = uncached_prompt_tokens/1000 * prefill_s_per_1k
         + cached_prompt_tokens/1000 * cached_prefill_s_per_1k (+ slow_first_s when that fault is on)
  decode = one chunk every decode_s_per_token * tokens_per_chunk

A toy prefix cache remembers recent prompts (as toy-token lists); a new prompt's cached
tokens are its longest common prefix with any of them. A toy "model" answers the probe
questions by regex over the prompt (honest mode) or deliberately wrongly (wrong mode).
The model entry is owned_by "408032gb-bench-mock" so run summaries flag mock runs.

THIS SERVER PRODUCES NO EVIDENCE ABOUT ANY ENGINE. It exists so the harness logic is tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_OWNER = "408032gb-bench-mock"
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
PIECE_RE = re.compile(r"\s*(?:\w+|[^\w\s])")
FILLER_WORDS = ("the model keeps writing steady filler text so that decode speed can be measured "
                "over many tokens without stopping early").split()


def toy_tokens(text):
    return TOKEN_RE.findall(text)


def toy_count(text):
    return len(TOKEN_RE.findall(text))


class MockConfig:
    def __init__(self, **kw):
        self.model_id = "mock-model"
        self.prefill_s_per_1k = 0.05          # seconds per 1000 uncached prompt tokens
        self.cached_prefill_s_per_1k = 0.0    # seconds per 1000 cached prompt tokens
        self.decode_s_per_token = 0.002
        self.tokens_per_chunk = 1             # >1 imitates speculative decoding emitting several tokens at once
        self.prefix_cache = True
        self.cache_entries = 16
        self.honor_keys = "both"              # which "force length" keys work: vllm|llamacpp|both|none
        self.reject_unknown_keys = False      # 400 on body keys the emulated dialect does not know
        self.accept_stream_options = True
        self.accept_response_format = True
        self.tokenize_style = "llamacpp"      # llamacpp|vllm|none
        self.answer_mode = "honest"           # honest|wrong
        self.nondeterministic = False         # filler continuation varies between identical requests
        self.reasoning_tokens = 0             # emit this many reasoning_content tokens first
        self.max_context = None               # 400 when prompt + max_tokens exceeds this
        self.fault = "none"                   # none|http500|truncate|slow_first
        self.fault_every = 1                  # apply the fault to every k-th generation request
        self.fault_after_chunks = 3           # truncate: chunks sent before the connection is dropped
        self.slow_first_s = 1.0
        self.llamacpp_timings = True          # add llama.cpp-style "timings" with cache_n to the final chunk
        self.usage_in_stream = True           # honour stream_options.include_usage
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown mock option %r" % k)
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


KNOWN_KEYS_COMMON = {"model", "messages", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream",
                     "stream_options", "tools", "tool_choice", "response_format", "stop", "n",
                     "chat_template_kwargs", "max_completion_tokens"}
KNOWN_KEYS = {"vllm": KNOWN_KEYS_COMMON | {"ignore_eos", "min_tokens"},
              "llamacpp": KNOWN_KEYS_COMMON | {"ignore_eos", "n_predict", "cache_prompt"},
              "both": KNOWN_KEYS_COMMON | {"ignore_eos", "min_tokens", "n_predict", "cache_prompt"},
              "none": KNOWN_KEYS_COMMON}


# ---------------------------------------------------------------------------- toy model
def _content_text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)


def render_chat(messages, tools=None):
    parts = []
    if tools:
        parts.append("<|tools|>\n" + json.dumps(tools, sort_keys=True) + "\n")
    for m in messages:
        parts.append("<|%s|>\n" % m.get("role", "user"))
        txt = _content_text(m.get("content"))
        if txt:
            parts.append(txt + "\n")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parts.append("<|call|>%s %s\n" % (fn.get("name"), fn.get("arguments")))
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _question(text):
    i = text.rfind("Question:")
    return text[i:] if i >= 0 else text[-2000:]


def _wrong_num(s, width=None):
    v = (int(s) + 1) % (10 ** (width or len(s)))
    return str(v).zfill(width or len(s))


def toy_answer(prompt, messages, tools, cfg):
    """Returns ("text", str) or ("tool", (name, args_dict))."""
    wrong = cfg.answer_mode == "wrong"
    last = messages[-1] if messages else {"role": "user", "content": prompt}
    if tools and last.get("role") == "user":
        q = _content_text(last.get("content"))
        city = re.search(r"weather (?:like )?in ([A-Z][\w\- ]*?)(?: right now|\?|,|\.)", q)
        city = city.group(1) if city else "Nowhere"
        if wrong:
            return "tool", ("get_stock_price", {"ticker": "XXX"})
        return "tool", ("get_weather", {"city": city, "unit": "celsius"})
    if tools and last.get("role") == "tool":
        try:
            res = json.loads(_content_text(last.get("content")))
        except ValueError:
            res = {}
        if wrong:
            return "text", "I could not get the weather."
        return "text", "The current temperature in %s is %s degrees Celsius with %s." % (
            res.get("city"), res.get("temperature_c"), res.get("condition"))
    q = _question(prompt)
    asked = re.findall(r"`([^`]+)`", q)
    if "secret code" in q:
        found = dict(re.findall(r"The secret code for ([\w\-]+) is (\d{6})\.", prompt))
        lines = []
        for k in asked:
            v = found.get(k)
            if v is not None:
                lines.append("%s: %s" % (k, _wrong_num(v) if wrong else v))
        return "text", "\n".join(lines) or "I do not know."
    if "final numeric value" in q:
        assign = {}
        for name, rhs in re.findall(r"^(VAR_[A-Z0-9_]+) = (\d+|VAR_[A-Z0-9_]+)$", prompt, re.M):
            assign[name] = rhs
        lines = []
        for v in asked:
            cur, hops = v, 0
            while cur in assign and hops < 1000:
                cur, hops = assign[cur], hops + 1
            if cur.isdigit():
                lines.append("%s: %s" % (v, _wrong_num(cur) if wrong else cur))
        return "text", "\n".join(lines) or "Unknown."
    if "Which constant does" in q:
        lines = []
        for fn in asked:
            m = re.search(r"def %s\(\):\n\s+return (\d+)" % re.escape(fn), prompt)
            if not m:
                continue
            head = prompt.rfind("# ===== file: ", 0, m.start())
            path = re.match(r"# ===== file: (\S+) =====", prompt[head:]).group(1) if head >= 0 else "unknown.py"
            const = m.group(1)
            if wrong:
                const, path = _wrong_num(const), "src/wrong/place.py"
            lines.append("%s: %s in %s" % (fn, const, path))
        return "text", "\n".join(lines) or "Unknown."
    if "JSON object" in q or "JSON object" in prompt[-3000:]:
        t = prompt[-3000:]
        g = lambda rx: (re.search(rx, t).group(1) if re.search(rx, t) else None)  # noqa: E731
        tags = re.search(r"tags are ([\w\-]+), ([\w\-]+) and ([\w\-]+)", t)
        obj = {"name": g(r"name is ([A-Z][a-z]+ [A-Z][a-z]+)"),
               "age": int(g(r"is (\d+) years old") or 0),
               "city": g(r"lives in ([A-Z][\w\-]+)"),
               "tags": list(tags.groups()) if tags else [],
               "active": "account is active" in t}
        if wrong:
            return "text", "Sure! The name is %s and the age is %s." % (obj["name"], obj["age"])
        return "text", json.dumps(obj)
    if cfg.nondeterministic:                       # only the free-form answer varies
        return "text", "This is a short toy answer number %d." % random.getrandbits(48)
    return "text", "This is a short toy answer."


# ---------------------------------------------------------------------------- server
class MockState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.cache = []                 # recent prompts as toy-token lists
        self.gen_requests = 0
        self.requests = []              # (path, body keys) of every request, for assertions
        self.nd_counter = 0

    def reset(self):
        with self.lock:
            self.cache = []
            self.gen_requests = 0
            self.requests = []

    def lookup_and_store(self, toks):
        if not self.cfg.prefix_cache:
            return 0
        best = 0
        with self.lock:
            for old in self.cache:
                lo, hi = 0, min(len(old), len(toks))
                if hi <= best:
                    continue
                if old[:hi] == toks[:hi]:
                    best = hi
                    continue
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if old[:mid] == toks[:mid]:
                        lo = mid
                    else:
                        hi = mid
                best = max(best, lo)
            self.cache.append(toks)
            del self.cache[:-self.cfg.cache_entries]
        return min(best, max(len(toks) - 1, 0))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "bench-mock/1"

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, fmt, *args):  # quiet
        if getattr(self.server, "verbose", False):
            sys.stderr.write("mock: " + fmt % args + "\n")

    @property
    def st(self):
        return self.server.state

    # -- plumbing ---------------------------------------------------------------
    def _send_json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, code, text, ctype="text/plain; charset=utf-8"):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, msg, typ="invalid_request_error"):
        self._send_json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _chunk(self, obj):
        data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode("utf-8")
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return None

    # -- routes -------------------------------------------------------------------
    def do_GET(self):
        self.st.requests.append((self.path, []))
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": self.st.cfg.model_id, "object": "model", "owned_by": MOCK_OWNER,
                 "max_model_len": self.st.cfg.max_context}]})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/metrics":
            self._send_text(200, "# HELP mock_generation_requests_total generation requests\n"
                                 "mock_generation_requests_total %d\n" % self.st.gen_requests)
        else:
            self._error(404, "not found: " + self.path)

    def do_POST(self):
        body = self._read_body()
        self.st.requests.append((self.path, sorted(body) if isinstance(body, dict) else []))
        if body is None:
            return self._error(400, "invalid JSON body")
        if self.path == "/tokenize":
            return self._tokenize(body)
        if self.path in ("/v1/chat/completions", "/v1/completions"):
            return self._generate(body, chat=self.path.endswith("chat/completions"))
        self._error(404, "not found: " + self.path)

    def _tokenize(self, body):
        style = self.st.cfg.tokenize_style
        if style == "none":
            return self._error(404, "tokenize disabled")
        if style == "llamacpp":
            n = toy_count(body["content"]) if "content" in body else 0
            return self._send_json(200, {"tokens": [i % 997 for i in range(n)]})
        if "prompt" not in body:
            return self._error(400, "field 'prompt' required")
        n = toy_count(body["prompt"])
        return self._send_json(200, {"count": n, "max_model_len": self.st.cfg.max_context or 0,
                                     "tokens": [i % 997 for i in range(n)]})

    def _generate(self, body, chat):
        cfg = self.st.cfg
        with self.st.lock:
            self.st.gen_requests += 1
            k = self.st.gen_requests
        fault_now = cfg.fault != "none" and k % max(cfg.fault_every, 1) == 0
        if fault_now and cfg.fault == "http500":
            return self._error(500, "injected fault: internal error", "server_error")
        if cfg.reject_unknown_keys:
            unknown = sorted(set(body) - KNOWN_KEYS[cfg.honor_keys])
            if unknown:
                return self._error(400, "unknown fields: %s" % ", ".join(unknown))
        if "stream_options" in body and not cfg.accept_stream_options:
            return self._error(400, "stream_options is not supported")
        if "response_format" in body and not cfg.accept_response_format:
            return self._error(400, "response_format is not supported")
        messages = body.get("messages") or []
        tools = body.get("tools")
        prompt = render_chat(messages, tools) if chat else str(body.get("prompt", ""))
        toks = toy_tokens(prompt)
        n_prompt = len(toks)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        honor_v = cfg.honor_keys in ("vllm", "both")
        honor_l = cfg.honor_keys in ("llamacpp", "both")
        if honor_l and "n_predict" in body:
            max_tokens = int(body["n_predict"])
        if cfg.max_context and n_prompt + max_tokens > cfg.max_context:
            return self._error(400, "This model's maximum context length is %d tokens; requested %d"
                               % (cfg.max_context, n_prompt + max_tokens))
        ignore_eos = bool(body.get("ignore_eos")) and (honor_v or honor_l)
        min_tokens = int(body.get("min_tokens") or 0) if honor_v else 0
        cached = self.st.lookup_and_store(toks)
        uncached = n_prompt - cached

        kind, ans = toy_answer(prompt, messages, tools, cfg)
        if kind == "tool":
            name, args = ans
            arg_s = json.dumps(args)
            pieces = [("tool_name", name)] + [("tool_args", arg_s[i:i + 6]) for i in range(0, len(arg_s), 6)]
            finish = "tool_calls"
        else:
            pieces = [("content", p) for p in PIECE_RE.findall(ans)]
            finish = "stop"
        if chat:
            pieces = [("reasoning", " think") for _ in range(cfg.reasoning_tokens)] + pieces
        seed_src = prompt if not cfg.nondeterministic else prompt + str(time.time_ns())
        rng = random.Random(hashlib.sha256(seed_src.encode("utf-8")).hexdigest())
        target_len = max_tokens if ignore_eos else max(min_tokens, 0)
        while len(pieces) < min(target_len, max_tokens):
            pieces.append(("content", " " + rng.choice(FILLER_WORDS)))
        if len(pieces) > max_tokens:
            pieces = pieces[:max_tokens]
            finish = "length"
        elif len(pieces) == max_tokens and (ignore_eos or min_tokens >= max_tokens):
            finish = "length"
        n_out = len(pieces)

        prefill = uncached / 1000.0 * cfg.prefill_s_per_1k + cached / 1000.0 * cfg.cached_prefill_s_per_1k
        if fault_now and cfg.fault == "slow_first":
            prefill += cfg.slow_first_s
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        timings = {"cache_n": cached, "prompt_n": uncached, "prompt_ms": prefill * 1000.0,
                   "predicted_n": n_out, "predicted_ms": n_out * cfg.decode_s_per_token * 1000.0}
        cid = "mock-%d" % k
        model = body.get("model") or cfg.model_id

        if not body.get("stream"):
            time.sleep(prefill + n_out * cfg.decode_s_per_token)
            return self._send_json(200, self._final_object(cid, model, chat, pieces, finish, usage))

        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) and cfg.usage_in_stream
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        obj_type = "chat.completion.chunk" if chat else "text_completion"

        def chunk(choice_payload, extra=None):
            ch = {"index": 0, "finish_reason": None}
            ch.update(choice_payload)
            o = {"id": cid, "object": obj_type, "created": int(time.time()), "model": model, "choices": [ch]}
            if extra:
                o.update(extra)
            self._chunk(o)

        try:
            if chat:
                chunk({"delta": {"role": "assistant", "content": ""}})      # role-only: not a token
            time.sleep(prefill)
            tpc = max(int(cfg.tokens_per_chunk), 1)
            sent_chunks = 0
            for i in range(0, n_out, tpc):
                if i > 0:
                    time.sleep(cfg.decode_s_per_token * tpc)
                group = pieces[i:i + tpc]
                if fault_now and cfg.fault == "truncate" and sent_chunks >= cfg.fault_after_chunks:
                    self.close_connection = True
                    self.wfile.flush()
                    return                                       # drop the connection mid-stream
                if chat:
                    delta = {}
                    for kind_p, p in group:
                        if kind_p == "content":
                            delta["content"] = delta.get("content", "") + p
                        elif kind_p == "reasoning":
                            delta["reasoning_content"] = delta.get("reasoning_content", "") + p
                        elif kind_p == "tool_name":
                            delta["tool_calls"] = [{"index": 0, "id": "call_%s" % cid, "type": "function",
                                                    "function": {"name": p, "arguments": ""}}]
                        else:
                            tc = delta.setdefault("tool_calls", [{"index": 0, "function": {"arguments": ""}}])
                            tc[0]["function"]["arguments"] = tc[0]["function"].get("arguments", "") + p
                    chunk({"delta": delta})
                else:
                    chunk({"text": "".join(p for _, p in group)})
                sent_chunks += 1
            extra = {"timings": timings} if cfg.llamacpp_timings else None
            chunk({"delta": {}, "finish_reason": finish} if chat else {"text": "", "finish_reason": finish}, extra)
            if include_usage:
                self._chunk({"id": cid, "object": obj_type, "created": int(time.time()), "model": model,
                             "choices": [], "usage": usage})
            self._chunk("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    @staticmethod
    def _final_object(cid, model, chat, pieces, finish, usage):
        content = "".join(p for k, p in pieces if k == "content")
        reasoning = "".join(p for k, p in pieces if k == "reasoning")
        names = [p for k, p in pieces if k == "tool_name"]
        if not chat:
            return {"id": cid, "object": "text_completion", "model": model, "usage": usage,
                    "choices": [{"index": 0, "text": content, "finish_reason": finish}]}
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if names:
            msg["tool_calls"] = [{"id": "call_" + cid, "type": "function",
                                  "function": {"name": names[0],
                                               "arguments": "".join(p for k, p in pieces if k == "tool_args")}}]
        return {"id": cid, "object": "chat.completion", "model": model, "usage": usage,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}


class MockServer:
    """In-process mock: MockServer(**cfg).start(); .url; .cfg (mutable); .reset(); .stop()."""

    def __init__(self, host="127.0.0.1", port=0, verbose=False, **cfg):
        self.cfg = MockConfig(**cfg)
        self.state = MockState(self.cfg)
        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = self.state
        self.httpd.verbose = verbose
        self.thread = None

    @property
    def url(self):
        h, p = self.httpd.server_address[:2]
        return "http://%s:%d" % (h, p)

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def configure(self, **kw):
        for k, v in kw.items():
            if not hasattr(self.cfg, k):
                raise TypeError("unknown mock option %r" % k)
            setattr(self.cfg, k, v)

    def reset(self, **kw):
        self.cfg.__init__(**kw)
        self.state.reset()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--verbose", action="store_true")
    defaults = MockConfig()
    for k, v in defaults.as_dict().items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=v,
                            metavar="BOOL")
        elif isinstance(v, (int, float)):
            ap.add_argument(flag, type=type(v), default=v)
        elif v is None:
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, default=v)
    a = vars(ap.parse_args(argv))
    host, port, verbose = a.pop("host"), a.pop("port"), a.pop("verbose")
    srv = MockServer(host, port, verbose=verbose, **a)
    print("mock server on %s (config %s)" % (srv.url, json.dumps(srv.cfg.as_dict(), sort_keys=True)), flush=True)
    try:
        srv.httpd.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
