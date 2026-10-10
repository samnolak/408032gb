#!/usr/bin/env python3
"""Stand-in for llama-server used by tests/test_srvcheck.py: same command line shape, same endpoints, a toy model.

Model: the next token and the top-k log-probs are a pure function of the token history the state encodes, so a
correct server gives the same answer warm and cold. Memory behaves like recurrent state: it cannot be cut at an
arbitrary position, so the cache is reused when its token list is a full prefix of the new prompt, otherwise only up to
a checkpoint; one checkpoint is kept 4 tokens before the end of every processed prompt (what llama-server @79e2e74
shows for the tiny models: a repeated prompt re-processes 4 tokens). The slot keeps `tokens` (what the server believes
is cached) and `state` (what the state really encodes); a correct server keeps them equal.
Faults (env STUB_FAULT) imitate the bugs the checks must catch:
  stale_reuse   a warm request continues from a state that lacks the last reused token
  noisy_warm    a warm request returns log-probs shifted by 1e-2 (tokens unchanged)
  restore_noop  /slots restore answers 200 but loads nothing
  restore_wrong /slots restore loads the state of a different prompt
  mtp_differs   with --spec-type the 5th generated token is changed
  mtp_crash     with --spec-type the process dies on the first completion
  mtp_nodraft   with --spec-type nothing is drafted (draft_n = 0)
  nondeterministic  every completion is shifted by a call counter
  shift_stale   a request with n_cache_reuse > 0 reuses its whole common prefix with a stale state (a KV shift
                that the recurrent state did not follow)
  crosstalk     a completion that runs while another slot is busy is computed from a disturbed history
Every slot (-np, default 1) has its own state; a completion takes 0.1 s, so two requests sent together overlap.
"""
import hashlib
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

FAULT = os.environ.get("STUB_FAULT", "")
V = 96


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


SPEC = arg("--spec-type") not in (None, "none")
SAVE = arg("--slot-save-path")
SLOTS = [{"tokens": [], "state": [], "ckpt": []} for _ in range(int(arg("-np", "1")))]
CALLS = {"n": 0}
BUSY = set()
LOCK = threading.Lock()


def scores(hist):
    h = hashlib.sha256(json.dumps(hist).encode()).digest()
    raw = [h[i % 32] ^ (i * 37 % 251) for i in range(V)]
    m = max(raw)
    z = math.log(sum(math.exp((r - m) / 16.0) for r in raw))
    return [(r - m) / 16.0 - z for r in raw]


def generate(prompt, n, n_probs, shift=0):
    hist, toks, probs = list(prompt), [], []
    for i in range(n):
        lp = scores(hist + [shift] if shift else hist)
        order = sorted(range(V), key=lambda k: -lp[k])
        tok = order[0]
        if SPEC and FAULT == "mtp_differs" and i == 4:
            tok = order[1]
        probs.append({"id": tok, "logprob": lp[tok],
                      "top_logprobs": [{"id": k, "logprob": lp[k]} for k in order[:n_probs]]})
        toks.append(tok)
        hist.append(tok)
    return toks, probs


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"status": "ok"})
        return self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        u = urlparse(self.path)
        if u.path == "/completion":
            if SPEC and FAULT == "mtp_crash":
                os._exit(139)
            k = body.get("id_slot", 0)
            STATE = SLOTS[k]
            with LOCK:
                CALLS["n"] += 1
                BUSY.add(k)
            time.sleep(0.1)
            others_busy = bool(BUSY - {k})
            p, cache = body["prompt"], STATE["tokens"]
            if cache and len(cache) < len(p) and p[:len(cache)] == cache:
                reuse = len(cache)
            else:
                lcp = next((i for i, (a, b) in enumerate(zip(cache, p)) if a != b), min(len(cache), len(p)))
                reuse = max([c for c in STATE["ckpt"] if c <= min(lcp, len(p) - 1)], default=0)
            if FAULT == "shift_stale" and body.get("n_cache_reuse", 0) > 0 and cache:
                reuse = min(len(p) - 1, len(cache))
            state = STATE["state"][:reuse]
            if reuse and FAULT == "stale_reuse":
                state = state[:-1]
            ctx = state + p[reuse:]
            shift = CALLS["n"] if FAULT == "nondeterministic" else 0
            if FAULT == "crosstalk" and others_busy:
                shift = 1
            toks, probs = generate(ctx, body["n_predict"], body.get("n_probs", 0), shift)
            if reuse and FAULT == "noisy_warm":
                for pr in probs:
                    for c in pr["top_logprobs"]:
                        c["logprob"] += 1e-2
            STATE["tokens"] = p + toks[:-1]
            STATE["state"] = ctx + toks[:-1]
            STATE["ckpt"] = [len(p) - 4] if len(p) > 4 else []
            with LOCK:
                BUSY.discard(k)
            timings = {"prompt_n": len(p) - reuse, "prompt_ms": 1.0, "predicted_n": len(toks), "predicted_ms": 1.0}
            if SPEC:
                timings.update({"draft_n": 0 if FAULT == "mtp_nodraft" else len(toks),
                                "draft_n_accepted": 0})
            return self.reply(200, {"content": "".join(f"t{t}" for t in toks), "tokens": toks,
                                    "completion_probabilities": probs, "timings": timings,
                                    "tokens_cached": len(STATE["tokens"])})
        if u.path.startswith("/slots/"):
            if not SAVE:
                return self.reply(501, {"error": {"code": 501, "message": "no --slot-save-path"}})
            action = parse_qs(u.query).get("action", [""])[0]
            STATE = SLOTS[int(u.path.split("/")[2])]
            path = os.path.join(SAVE, body.get("filename", "x"))
            if action == "erase":
                STATE.update(tokens=[], state=[], ckpt=[])
                return self.reply(200, {"id_slot": 0, "n_erased": 1})
            if action == "save":
                with open(path, "w") as f:
                    json.dump({k: STATE[k] for k in ("tokens", "state", "ckpt")}, f)
                return self.reply(200, {"id_slot": 0, "filename": body["filename"], "n_saved": len(STATE["tokens"]),
                                        "n_written": os.path.getsize(path)})
            if action == "restore":
                with open(path) as f:
                    saved = json.load(f)
                if FAULT == "restore_wrong":
                    saved["state"] = [(saved["state"][0] + 1) % V] + saved["state"][1:]
                STATE.update(tokens=saved["tokens"], ckpt=saved["ckpt"])
                STATE["state"] = [] if FAULT == "restore_noop" else saved["state"]
                return self.reply(200, {"id_slot": 0, "filename": body["filename"], "n_restored": len(saved["tokens"])})
        return self.reply(404, {})


ThreadingHTTPServer(("127.0.0.1", int(arg("--port"))), H).serve_forever()
