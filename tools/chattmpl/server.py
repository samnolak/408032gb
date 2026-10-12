"""Minimal llama-server client for the N14 chat-template probes (stdlib only).

POST /apply-template at llama.cpp 79e2e74 returns {"prompt": <rendered text>}: tools/server/server-context.cpp:5584-5598
runs oaicompat_chat_params_parse (tools/server/server-common.cpp:1268), i.e. the same template application as
/v1/chat/completions, without inference.
"""
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post(base, path, body, timeout=120):
    req = urllib.request.Request(base + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Server:
    """llama-server on a tiny model with --jinja --chat-template-file; the log goes to <workdir>/srv.log."""

    def __init__(self, binary, model, template_file, workdir, extra=(), port=None):
        self.port = port or free_port()
        self.base = "http://127.0.0.1:%d" % self.port
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.log_path = os.path.join(workdir, "srv.log")
        self.argv = [binary, "-m", model, "--jinja", "--chat-template-file", template_file,
                     "--host", "127.0.0.1", "--port", str(self.port), "-t", "2", "-np", "1"] + list(extra)
        self.proc = None

    def __enter__(self):
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.workdir)
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited with %s, see %s" % (self.proc.returncode, self.log_path))
            try:
                if get(self.base, "/health").get("status") == "ok":
                    return self
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become healthy, see %s" % self.log_path)

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()

    def apply_template(self, messages, tools=None, add_generation_prompt=True, kwargs=None, extra=None):
        body = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if tools:
            body["tools"] = tools
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        if extra:
            body.update(extra)
        return post(self.base, "/apply-template", body)

    def log_text(self):
        self.log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()
