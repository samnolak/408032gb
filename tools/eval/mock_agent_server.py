#!/usr/bin/env python3
"""Scripted mock OpenAI-compatible server for testing tools/eval (stdlib only, no GPU).

THIS SERVER PRODUCES NO EVIDENCE ABOUT ANY ENGINE OR MODEL. Its /v1/models entry is owned_by the same
marker as tools/bench/mock_server.py (runlog.MOCK_OWNER), so run_eval.py flags runs against it as mock runs
exactly like the bench tools do.

Answers are scripted. A script is chosen by the conversation's FIRST user message:
  {"match": "<substring>"} or {"first_user_sha256": "<hex of the UTF-8 text>"}; first matching script wins.
The turn index is the number of assistant messages already in the request; the script's "turns"[i] is
returned, and "loop" for every turn past the list (a model that never stops calling tools).
A response is one of:
  {"content": "..."}                                          plain answer (finish_reason stop)
  {"tool_calls": [{"name": "f", "arguments": "{...}" | {...}}], "content"?: "..."}   (finish tool_calls)
  {"http_error": 500}                                         HTTP error with a JSON error body
  {"truncate": true}                                          stream cut before finish_reason / connection drop
  {"malformed": true}                                         HTTP 200 without choices (non-stream) or
                                                              non-JSON SSE data without finish_reason (stream)
  optional keys: "finish_reason" (override), "sleep_s" (delay before answering), "reasoning" (reasoning_content)
Unmatched requests get DEFAULT_RESPONSE.

CLI: python3 tools/eval/mock_agent_server.py --port 18090 --scripts scripts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from runlog import MOCK_OWNER  # noqa: E402

DEFAULT_RESPONSE = {"content": "mock: no script matched this conversation"}
PIECE = 12


def first_user(messages):
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class State:
    def __init__(self, scripts, owner, model_id):
        self.scripts = list(scripts or [])
        self.owner = owner
        self.model_id = model_id
        self.lock = threading.Lock()
        self.requests = []
        self.n = 0

    def pick(self, messages):
        fu = first_user(messages)
        h = sha256_text(fu)
        turn = sum(1 for m in messages or [] if isinstance(m, dict) and m.get("role") == "assistant")
        for s in self.scripts:
            if ("first_user_sha256" in s and s["first_user_sha256"] == h) or ("match" in s and s["match"] in fu):
                turns = s.get("turns") or []
                if turn < len(turns):
                    return turns[turn], turn
                if "loop" in s:
                    return s["loop"], turn
                return (turns[-1] if turns else DEFAULT_RESPONSE), turn
        return DEFAULT_RESPONSE, turn


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "eval-mock/1"

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, fmt, *args):
        pass

    @property
    def st(self):
        return self.server.state

    def _json(self, code, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _chunk(self, payload):
        data = ("data: " + (payload if isinstance(payload, str) else json.dumps(payload)) + "\n\n").encode("utf-8")
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.st.model_id, "object": "model",
                                                          "owned_by": self.st.owner}]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found", "code": 404}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except ValueError:
            return self._json(400, {"error": {"message": "invalid JSON", "code": 400}})
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found", "code": 404}})
        with self.st.lock:
            self.st.requests.append(body)
            self.st.n += 1
            k = self.st.n
        resp, turn = self.st.pick(body.get("messages"))
        if resp.get("sleep_s"):
            time.sleep(float(resp["sleep_s"]))
        if "http_error" in resp:
            code = int(resp["http_error"])
            return self._json(code, {"error": {"message": resp.get("message", "scripted error"), "code": code}})
        content = resp.get("content") or ""
        reasoning = resp.get("reasoning") or ""
        calls = []
        for i, c in enumerate(resp.get("tool_calls") or []):
            args = c.get("arguments", "")
            calls.append({"id": c.get("id", "call_%d_%d_%d" % (k, turn, i)), "name": c["name"],
                          "arguments": args if isinstance(args, str) else json.dumps(args)})
        finish = resp.get("finish_reason") or ("tool_calls" if calls else "stop")
        usage = {"prompt_tokens": max(1, len(json.dumps(body.get("messages")))) // 4,
                 "completion_tokens": max(1, (len(content) + len(reasoning) + sum(len(c["arguments"]) for c in calls)) // 4)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        cid = "evalmock-%d" % k
        model = body.get("model") or self.st.model_id
        if not body.get("stream"):
            if resp.get("truncate"):
                self.close_connection = True
                return
            if resp.get("malformed"):
                return self._json(200, {"id": cid, "object": "chat.completion", "model": model})
            msg = {"role": "assistant", "content": content or None}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            return self._json(200, {"id": cid, "object": "chat.completion", "model": model, "usage": usage,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(delta, fin=None):
            self._chunk({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
        try:
            chunk({"role": "assistant", "content": ""})
            if resp.get("malformed"):
                self._chunk("this is not json")
                self.wfile.write(b"0\r\n\r\n")
                return
            pieces = [reasoning[i:i + PIECE] for i in range(0, len(reasoning), PIECE)]
            for p in pieces:
                chunk({"reasoning_content": p})
            sent = 0
            for i in range(0, len(content), PIECE):
                if resp.get("truncate") and sent >= 2:
                    break
                chunk({"content": content[i:i + PIECE]})
                sent += 1
            if resp.get("truncate"):
                self.close_connection = True
                self.wfile.flush()
                return                               # no finish_reason, no terminating chunk
            for idx, c in enumerate(calls):
                chunk({"tool_calls": [{"index": idx, "id": c["id"], "type": "function",
                                       "function": {"name": c["name"], "arguments": ""}}]})
                for j in range(0, len(c["arguments"]), PIECE):
                    chunk({"tool_calls": [{"index": idx, "function": {"arguments": c["arguments"][j:j + PIECE]}}]})
            chunk({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                self._chunk({"id": cid, "object": "chat.completion.chunk", "model": model, "choices": [],
                             "usage": usage})
            self._chunk("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class MockAgentServer:
    """MockAgentServer(scripts).start(); .url; .set_scripts([...]); .requests; .stop()."""

    def __init__(self, scripts=None, host="127.0.0.1", port=0, owner=MOCK_OWNER, model_id="mock-agent"):
        self.state = State(scripts, owner, model_id)
        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = self.state
        self.thread = None

    @property
    def url(self):
        h, p = self.httpd.server_address[:2]
        return "http://%s:%d" % (h, p)

    @property
    def port(self):
        return self.httpd.server_address[1]

    @property
    def requests(self):
        return self.state.requests

    def set_scripts(self, scripts, owner=None):
        with self.state.lock:
            self.state.scripts = list(scripts)
            self.state.requests = []
            if owner is not None:
                self.state.owner = owner

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--scripts", default=None, help="JSON file with a list of scripts")
    a = ap.parse_args(argv)
    scripts = []
    if a.scripts:
        with open(a.scripts, encoding="utf-8") as f:
            scripts = json.load(f)
    srv = MockAgentServer(scripts, a.host, a.port)
    print("eval mock server on %s with %d scripts (NOT evidence about any engine)" % (srv.url, len(scripts)), flush=True)
    try:
        srv.httpd.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
