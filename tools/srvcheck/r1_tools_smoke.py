#!/usr/bin/env python3
"""N11: the measurement tools of R1 against a REAL llama-server on CPU (tiny random-weight model with a text vocabulary).

usage: r1_tools_smoke.py --server LLAMA_SERVER --model TEXT_VOCAB.gguf --workdir DIR [--out RESULT.json]
       r1_tools_smoke.py derive-config --model GGUF --prompt-file FILE --out CONFIG.json [--lengths smoke|evidence]

The model comes from tools/tinygen/qwen4exp_tiny.py --text-vocab --ctx 262144 (or glm5next_tiny.py): random weights,
byte-level BPE, so the server accepts text. Two servers are started on it: "cand" with the engine flags of
scripts/r1/configs/llamacpp-server.json (-fa on, f16 KV, --jinja, --metrics, --slot-save-path; the GPU flags -ngl/-sm/-ts
are kept: a CPU build ignores them with a warning) and "ref" with -fa off. Every tool then runs through its CLI, and the
checks are what a format bug would break - never a score or a speed:
  tokenize  prompts.py fits a prompt with the server's /tokenize (llama.cpp dialect detected)
  matrix    run_matrix.py: no failed request; usage and server cached tokens reported; a second run's cold cell is not
            served from the prompt cache (N11 finding: the cold nonce repeated across runs)
  probes    probes.py: every probe got an answer (scores are noise on random weights and are not checked)
  slot      slot_persist.py: save/erase/restore, identical continuation compared by token ids
  logprob   logprob_compare.py ref vs cand: answers parsed in llama-server's layout; every error is the known one
            (no logprobs for a token ending in an incomplete UTF-8 sequence)
  gateway   tools/gateway/gateway.py in front of cand: auth, proxying, stream equal to the direct one, context guard via
            /apply-template + /tokenize, client disconnect stops the server's task, queue full -> 429, request log
  r1        scripts/r1/run_r1.py on a config derived from llamacpp-server.json (derive_r1_config): the engine is started,
            health-checked and stopped by run_r1, every step ends as expected, the pack is built and verified, and its
            verdict is not PASS (a random-weight model on a CPU is not a measurement)
Wall times are not results; nothing here is evidence about speed or about any real model. Python 3 stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TEMPLATE = os.path.join(REPO, "scripts", "r1", "configs", "llamacpp-server.json")
PY = sys.executable or "python3"
KEY = "n11-smoke-key"
NULL_LOGPROBS = "incomplete UTF-8"           # logprob_compare's error text for llama-server's dropped probabilities


# ---------------------------------------------------------------------------- derived R1 config
def derive_r1_config(model, prompt_file, lengths="smoke", template=TEMPLATE):
    """llamacpp-server.json for a CPU run on a SYNTHETIC tiny model: same engine argv; no build-info (a CPU build has
    none, so engine_build_check is skipped and run_r1 starts the engine); no weights manifest (V2 NOT RUN; the model
    says why in weights_unverified_reason, which safety rule 8 of scripts/r1/safety_rules.py requires); shorter
    matrix/probes lengths; slot_persist on a prepared prompt file; logprob_compare with short prefixes."""
    with open(template, encoding="utf-8") as f:
        c = json.load(f, object_pairs_hook=collections.OrderedDict)
    c["_label"] = ("N11 DERIVED from scripts/r1/configs/llamacpp-server.json for a CPU run on a SYNTHETIC tiny random-weight "
                   "model. Changes: name, model (SYNTHETIC file, no manifest, runs marked unverified), engine.build_info removed (the CPU build "
                   "has no build-info.json), shorter matrix/probes lengths, slot_persist.prompt_file, logprob_compare "
                   "--max-chars. The engine argv is the template's. Nothing this config produces is a measurement.")
    c["name"] = "n11-llamacpp-cpu-tiny" + ("" if lengths == "evidence" else "-smoke")
    del c["engine"]["build_info"]
    c["model"] = {"desc": "SYNTHETIC tiny random-weight model %s (tools/tinygen --text-vocab), NOT the template's weights"
                          % os.path.basename(model),
                  "files": [os.path.abspath(model)], "weights_manifest": None, "weights_dest": None,
                  "weights_unverified_reason": "SYNTHETIC tiny random-weight model of a CPU smoke run: no sha256 "
                                               "manifest exists for it (R1 safety rule 8, scripts/r1/safety_rules.py)"}
    eb = ["--seed", "0", "--extra-body", "${EXTRA_BODY}"]
    if lengths == "evidence":
        matrix = [("smoke-2k", ["--inputs", "2048", "--outputs", "128", "--states", "cold,warm", "--concurrency", "1",
                                "--repeats", "1", "--scenarios", "matrix"]),
                  ("ctx-2k", ["--inputs", "2048", "--length-mode", "total", "--outputs", "128,512,2048", "--states",
                              "cold,warm", "--concurrency", "1", "--repeats", "3", "--scenarios", "matrix"]),
                  ("ctx-8k", ["--inputs", "8192", "--length-mode", "total", "--outputs", "128,512", "--states",
                              "cold,warm", "--concurrency", "1", "--repeats", "3", "--scenarios", "matrix"]),
                  ("filled-16k", ["--inputs", "16384", "--length-mode", "total", "--outputs", "512", "--states",
                                  "cold,warm", "--concurrency", "1", "--repeats", "3", "--scenarios", "matrix,agent",
                                  "--agent-target-tokens", "12000"])]
        probes = [("4k-16k", ["--contexts", "4096,16384", "--length-mode", "total", "--answer-tokens", "1024"])]
        lp = ["--prefixes", "200", "--top-k", "20", "--max-chars", "3000"]
    else:
        matrix = [("smoke-1k", ["--inputs", "1024", "--outputs", "32", "--states", "cold,warm", "--concurrency", "1",
                                "--repeats", "1", "--scenarios", "matrix"]),
                  ("filled-2k", ["--inputs", "2048", "--length-mode", "total", "--outputs", "64", "--states",
                                 "cold,warm", "--concurrency", "1", "--repeats", "3", "--scenarios", "matrix,agent",
                                 "--agent-target-tokens", "1500", "--agent-tool-tokens", "200,400",
                                 "--agent-max-tokens", "16"])]
        probes = [("2k", ["--contexts", "2048", "--length-mode", "total", "--answer-tokens", "64", "--det-tokens",
                          "512", "--det-max-tokens", "32"])]
        lp = ["--prefixes", "20", "--top-k", "20", "--max-chars", "1500"]
    c["steps"]["matrix"]["runs"] = [{"name": n, "args": a + eb} for n, a in matrix]
    c["steps"]["probes"]["runs"] = [{"name": n, "args": a + eb} for n, a in probes]
    c["steps"]["slot_persist"] = {"prompt_file": os.path.abspath(prompt_file),
                                  "args": ["--gen", "64" if lengths == "evidence" else "16", "--repeats", "3"]}
    c["steps"]["logprob_compare"]["args"] = lp
    return c


# ---------------------------------------------------------------------------- servers and HTTP
class Server:
    def __init__(self, name, binary, model, port, args, workdir):
        self.name, self.port = name, port
        self.url = "http://127.0.0.1:%d" % port
        self.log = os.path.join(workdir, "%s.log" % name)
        self.slots = os.path.join(workdir, "slots-%s" % name)
        os.makedirs(self.slots, exist_ok=True)
        self.argv = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port)] + args + \
            ["--slot-save-path", self.slots]
        self.p = None

    def start(self, timeout=120):
        self.fo = open(self.log, "wb")
        self.p = subprocess.Popen(self.argv, stdout=self.fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError("%s exited with %s, see %s" % (self.name, self.p.returncode, self.log))
            try:
                if http_req(self.url, "GET", "/health", timeout=2)[0] == 200:
                    return self
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError("%s not healthy after %s s" % (self.name, timeout))

    def stop(self):
        if self.p and self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        if self.p:
            self.fo.close()
        return None if self.p is None else self.p.returncode

    def log_text(self):
        with open(self.log, encoding="utf-8", errors="replace") as f:
            return f.read()


def http_req(url, method, path, body=None, key=None, timeout=300):
    u = urllib.parse.urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    try:
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        c.close()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tool(argv, env=None, timeout=1800):
    p = subprocess.run([PY] + argv, cwd=REPO, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, **(env or {})))
    return p.returncode, p.stdout, p.stderr


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def raw_rows(run_dir):
    with open(os.path.join(run_dir, "raw.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------- checks
class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, group, name, ok, observed):
        self.rows.append({"group": group, "check": name, "ok": bool(ok), "observed": observed})
        print("%-8s %-62s %s  %s" % (group, name, "PASS" if ok else "FAIL", observed), flush=True)
        return ok

    def guard(self, group, name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a crash of one group is a failed check, the others still run
            self(group, name, False, "%s: %s" % (type(e).__name__, e))


def check_tokenize(ck, cand, work):
    out = os.path.join(work, "prompt-fit.json")
    rc, so, se = tool(["tools/bench/prompts.py", "--url", cand.url, "--kind", "prose", "--targets", "2000",
                       "--save-text", os.path.join(work, "prompt"), "--out", out])
    rep = load(out)
    probe = rep["counter"].get("tokenize_probe") or [{}]
    ck("tokenize", "oa_client detects the llama.cpp /tokenize dialect",
       rep["counter"].get("tokenize_style") == "llamacpp" and probe[0].get("count", 0) > 0,
       "style %s, probe %s" % (rep["counter"].get("tokenize_style"), [(p.get("style"), p.get("count")) for p in probe]))
    r = rep["results"][0]
    # exactness depends on the vocabulary: the pad piece " the" is 3 tokens in the tiny one (1 in real BPE vocabularies)
    ck("tokenize", "prompts.py fits 2000 tokens with the server's /tokenize (<= target, within 3)",
       r["method"] == "server-tokenize" and 1997 <= r["tokens"] <= 2000, "tokens %s exit %s" % (r["tokens"], rc))
    return os.path.join(work, "prompt", "prose-2000.txt")


def check_matrix(ck, cand, work):
    args = ["--url", cand.url, "--no-nvidia-smi", "--inputs", "1024", "--outputs", "32", "--states", "cold,warm",
            "--concurrency", "1", "--repeats", "1", "--scenarios", "matrix,agent", "--agent-turns", "3",
            "--agent-tool-tokens", "200,400", "--agent-max-tokens", "16", "--seed", "0"]
    runs = []
    for name in ("m1", "m2"):                    # the same arguments twice against the same server
        d = os.path.join(work, name)
        rc, so, se = tool(["tools/bench/run_matrix.py", "--out-dir", d] + args)
        meta, s, raw = load(os.path.join(d, "meta.json")), load(os.path.join(d, "summary.json")), raw_rows(d)
        runs.append((rc, meta, s, raw))
        ck("matrix", "%s: exit 0, no failed request, not flagged as mock" % name,
           rc == 0 and not s["failures"] and meta["is_mock_server"] is False,
           "exit %s failures %d is_mock %s" % (rc, len(s["failures"]), meta["is_mock_server"]))
        meas = [r["metrics"] for r in raw if r.get("phase") in ("measure", "agent_turn", "agent_cold_control")]
        ck("matrix", "%s: usage and server cached tokens on every request" % name,
           meas and all(m["prompt_tokens"] and m["completion_tokens"] and m["cached_tokens"] is not None for m in meas),
           "%d requests, prompt_tokens %s" % (len(meas), sorted({m["prompt_tokens"] for m in meas})[:4]))
    cold = [r["metrics"] for r in runs[1][3] if r.get("phase") == "measure" and r.get("state") == "cold"]
    ck("matrix", "second run: the cold request is not served from the cache",
       cold and all(m["cached_tokens"] < 0.05 * m["prompt_tokens"] for m in cold),
       "cached %s of %s; salts %s / %s" % ([m["cached_tokens"] for m in cold], [m["prompt_tokens"] for m in cold],
                                           runs[0][1].get("nonce_salt"), runs[1][1].get("nonce_salt")))


def check_probes(ck, cand, work):
    d = os.path.join(work, "probes")
    rc, so, se = tool(["tools/bench/probes.py", "--url", cand.url, "--out-dir", d, "--no-nvidia-smi", "--contexts",
                       "2048", "--answer-tokens", "64", "--det-tokens", "512", "--det-max-tokens", "32"])
    s = load(os.path.join(d, "summary.json"))
    res = {r["probe"]: r for r in s["results"]}
    ck("probes", "every probe answered (no request error)",
       set(res) == {"needle", "multihop", "codeqa", "json", "tools", "determinism"} and not s["n_errors"],
       "probes %s errors %s exit %s (1 = scores below thresholds: random weights)" % (sorted(res), s["n_errors"], rc))
    ck("probes", "json: response_format accepted by the server",
       res.get("json", {}).get("details", {}).get("response_format_accepted") is True,
       "accepted %s" % res.get("json", {}).get("details", {}).get("response_format_accepted"))


def check_slot(ck, cand, work, prompt_file):
    out = os.path.join(work, "slot_persist.json")
    rc, so, se = tool(["tools/kv/slot_persist.py", "--url", cand.url, "--prompt-file", prompt_file, "--out", out,
                       "--gen", "16", "--repeats", "2"])
    d = load(out)
    s = d["summary"] or {}
    ck("slot", "save/erase/restore: identical continuation by token ids, warm re-processes few tokens",
       rc == 0 and s.get("all_continuations_identical") and all(r["compared"] == "tokens" for r in d["runs"])
       and (s.get("warm_prompt_n_p50") or 1e9) < 0.05 * (s.get("cold_prompt_n_p50") or 0),
       "exit %s identical %s compared %s warm prompt_n %s of %s" % (
           rc, s.get("all_continuations_identical"), sorted({r["compared"] for r in d["runs"]}),
           s.get("warm_prompt_n_p50"), s.get("cold_prompt_n_p50")))


def check_logprob(ck, ref, cand, work, corpus):
    out = os.path.join(work, "logprob_compare.json")
    rc, so, se = tool(["tools/quality/logprob_compare.py", "--ref", ref.url, "--cand", cand.url, "--corpus", corpus,
                       "--prefixes", "40", "--top-k", "20", "--max-chars", "1500", "--out", out])
    d = load(out)
    rows = d["rows"]
    other = [e for e in d["errors"] if NULL_LOGPROBS not in e["error"]]
    ck("logprob", "llama-server layout parsed; every error is the known incomplete-UTF-8 one",
       rc == 0 and rows and all(r["ref_layout"] == r["cand_layout"] == "llamacpp" for r in rows) and not other,
       "%d rows, %d errors (%d other), top1_agree %s, merged keys %d" % (
           len(rows), len(d["errors"]), len(other), (d["summary"] or {}).get("top1_agree"),
           sum(r["ref_merged"] + r["cand_merged"] for r in rows)))


def check_gateway(ck, cand, work):
    port = free_port()
    keys = os.path.join(work, "gateway-keys")
    with open(keys, "w") as f:
        f.write("# smoke\n%s\n" % KEY)
    glog = os.path.join(work, "gateway-requests.jsonl")
    gw_url = "http://127.0.0.1:%d" % port
    fo = open(os.path.join(work, "gateway.log"), "wb")
    p = subprocess.Popen([PY, os.path.join(REPO, "tools", "gateway", "gateway.py"), "--listen", "127.0.0.1:%d" % port,
                          "--backend", cand.url, "--api-keys-file", keys, "--max-concurrency", "1", "--max-queue", "2",
                          "--max-context-tokens", "4096", "--set-default", "cache_prompt=true", "--log-file", glog],
                         stdout=fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(100):
            try:
                if http_req(gw_url, "GET", "/health", timeout=2)[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        _gateway_session(ck, gw_url, cand, glog)
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            ck("gateway", "SIGTERM: drains and exits 0", p.wait(timeout=60) == 0, "exit %s" % p.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            ck("gateway", "SIGTERM: drains and exits 0", False, "killed after 60 s")
        fo.close()


def _norm_stream(data):
    out = []
    for line in data.decode("utf-8", "replace").split("\n"):
        if line.startswith("data: "):
            p = line[6:]
            if p == "[DONE]":
                out.append(p)
                continue
            o = json.loads(p)
            for k in ("id", "created", "timings"):
                o.pop(k, None)
            out.append(o)
    return out


def _launches(server):
    return len(re.findall(r"launch_slot_", server.log_text()))


def _gateway_session(ck, gw, cand, glog):
    g = "gateway"
    st, _, d = http_req(gw, "GET", "/health")
    ck(g, "/health without a key: 200, backend reachable", st == 200 and json.loads(d)["backend"]["http_status"] == 200,
       "HTTP %d" % st)
    st, _, _ = http_req(gw, "GET", "/v1/models")
    st2, _, d2 = http_req(gw, "GET", "/v1/models", key=KEY)
    ck(g, "/v1/models: 401 without the key, the server's list with it",
       st == 401 and st2 == 200 and json.loads(d2)["data"][0]["id"], "HTTP %d / %d" % (st, st2))
    st, _, d = http_req(gw, "POST", "/tokenize", {"content": "Hello world.", "add_special": False}, key=KEY)
    ck(g, "/tokenize proxied", st == 200 and json.loads(d)["tokens"], "HTTP %d %s" % (st, d[:60]))
    body = {"messages": [{"role": "user", "content": "Write one line about rivers."}], "max_tokens": 24,
            "temperature": 0, "stream": True, "ignore_eos": True, "cache_prompt": True}
    s1, _, d1 = http_req(cand.url, "POST", "/v1/chat/completions", body)
    s2, h2, d2 = http_req(gw, "POST", "/v1/chat/completions", body, key=KEY)
    e1, e2 = _norm_stream(d1), _norm_stream(d2)
    ck(g, "chat stream via the gateway == direct (minus id/created/timings)",
       s1 == s2 == 200 and e1 == e2 and e1 and e1[-1] == "[DONE]", "events %d / %d" % (len(e1), len(e2)))
    n0 = _launches(cand)
    long_text = "word " * 5000
    st, _, d = http_req(gw, "POST", "/v1/completions", {"prompt": long_text, "max_tokens": 8}, key=KEY)
    st2, _, d2 = http_req(gw, "POST", "/v1/chat/completions",
                          {"messages": [{"role": "user", "content": long_text}], "max_tokens": 8}, key=KEY)
    codes = [(json.loads(x).get("error") or {}).get("code") for x in (d, d2)]
    time.sleep(0.3)
    ck(g, "context guard: completion and chat over 4096 tokens -> 400, never reach a slot",
       st == st2 == 400 and codes == ["context_length_exceeded"] * 2 and _launches(cand) == n0,
       "HTTP %d/%d %s, slot launches %d -> %d" % (st, st2, codes, n0, _launches(cand)))
    n_cancel = len(re.findall(r"cancel task", cand.log_text()))
    body = json.dumps({"prompt": "Once upon a time", "max_tokens": 3500, "temperature": 0, "stream": True,
                       "ignore_eos": True}).encode()
    s = socket.create_connection(("127.0.0.1", urllib.parse.urlsplit(gw).port))
    s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nAuthorization: Bearer %s"
              b"\r\nContent-Length: %d\r\n\r\n%s" % (KEY.encode(), len(body), body))
    got = b""
    while got.count(b"data: ") < 5:
        got += s.recv(4096)
    s.close()
    idle = None
    t0 = time.time()
    while time.time() - t0 < 10:
        st, _, d = http_req(cand.url, "GET", "/slots")
        if st == 200 and not any(x.get("is_processing") for x in json.loads(d)):
            idle = time.time() - t0
            break
        time.sleep(0.1)
    time.sleep(0.2)
    cancel = re.findall(r".*cancel task.*", cand.log_text())
    ck(g, "client gone mid-stream: the gateway drops the backend, llama-server cancels the task",
       idle is not None and len(cancel) > n_cancel, "slot idle after %s s; server log %r" % (
           None if idle is None else round(idle, 2), cancel[-1].strip()[-60:] if cancel else None))
    codes = []

    def worker():
        st, hd, _ = http_req(gw, "POST", "/v1/completions", {"prompt": "Tell", "max_tokens": 2500, "ignore_eos": True},
                             key=KEY)
        codes.append((st, hd.get("Retry-After")))
    ths = [threading.Thread(target=worker) for _ in range(4)]
    for t in ths:
        t.start()
        time.sleep(0.1)
    for t in ths:
        t.join()
    ck(g, "queue: 1 running + 2 queued, the 4th request gets 429 + Retry-After",
       sorted(c for c, _ in codes) == [200, 200, 200, 429] and all(ra for c, ra in codes if c == 429), "%s" % codes)
    time.sleep(0.3)
    with open(glog, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    gen = [r for r in recs if r["status"] == 200 and r["path"].startswith("/v1/") and r["path"] != "/v1/models"
           and not r["client_disconnected"]]
    gone = [r for r in recs if r["client_disconnected"]]
    ck(g, "request log: usage from the server, the disconnect recorded, no prompt text",
       gen and all(r["completion_tokens"] for r in gen) and gone and not any("word word" in json.dumps(r) for r in recs),
       "%d records, %d generations (usage_src %s), %d disconnected" % (
           len(recs), len(gen), sorted({str(r["usage_src"]) for r in gen}), len(gone)))


EXPECTED_STEPS = {"weights_verify": "skipped", "quant_audit": "ok", "roofline_prediction": "ok",
                  "engine_build_check": "skipped", "engine_version": "ok", "engine_start": "ok", "health_wait": "ok",
                  "matrix": "ok", "probes": "failed", "slot_persist": "ok", "logprob_compare": "ok",
                  "vs_roofline": "ok", "engine_stop": "ok", "pack": "ok"}


def check_r1(ck, binary, model, ref, work, prompt_file, corpus):
    cfg = os.path.join(work, "r1-config.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(derive_r1_config(model, prompt_file, "smoke"), f, indent=2)
    root = os.path.join(work, "r1runs")
    rc, so, se = tool(["scripts/r1/run_r1.py", "--config", cfg, "--out", root,
                       "--set", "LLAMA_BIN=%s" % os.path.dirname(os.path.abspath(binary)),
                       "--set", "SLOT_DIR=%s" % os.path.join(work, "r1-slots"), "--set", "HEALTH_TIMEOUT_S=120",
                       "--set", "CTX=32768", "--set", "REFERENCE_URL=%s" % ref.url, "--set", "CORPUS=%s" % corpus])
    with open(os.path.join(work, "r1.stdout.txt"), "w") as f:
        f.write(so + se)
    runs = sorted(os.listdir(root))
    run = load(os.path.join(root, runs[-1], "run.json"))
    st = run["step_states"]
    bad = {k: (st.get(k), v) for k, v in EXPECTED_STEPS.items() if st.get(k) != v}
    ck("r1", "run_r1 started, health-checked and stopped the engine; steps as expected",
       not bad and rc == 1, "exit %s (1: probes fail on random weights); unexpected %s" % (rc, bad or "none"))
    pack = run.get("pack") or {}
    ck("r1", "pack built and verified; verdict not PASS",
       pack.get("overall") in ("FAIL", "NOT RUN") and st.get("pack") == "ok",
       "verdict %s (%s)" % (pack.get("overall"), (pack.get("overall_reason") or "")[:90]))
    ex = load(os.path.join(root, runs[-1], "artifacts", "engine", "exit.json"))
    ck("r1", "engine stopped by SIGTERM with exit code 0", ex.get("signals_sent") == ["SIGTERM"] and ex.get("exit_code") == 0,
       "signals %s exit %s" % (ex.get("signals_sent"), ex.get("exit_code")))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["derive-config"]:
        ap = argparse.ArgumentParser(prog="r1_tools_smoke.py derive-config")
        ap.add_argument("--model", required=True)
        ap.add_argument("--prompt-file", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--lengths", choices=["smoke", "evidence"], default="smoke")
        a = ap.parse_args(argv[1:])
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(derive_r1_config(a.model, a.prompt_file, a.lengths), f, indent=2)
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="llama-server binary (llama.cpp at tools/engines/pins.json)")
    ap.add_argument("--model", required=True, help="tiny text-vocab GGUF (tools/tinygen/*_tiny.py --text-vocab --ctx N)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default=None, help="write every check as JSON here")
    ap.add_argument("--threads", default="2")
    a = ap.parse_args(argv)
    work = os.path.abspath(a.workdir)
    os.makedirs(work, exist_ok=True)
    corpus = os.path.join(work, "corpus")
    os.makedirs(corpus, exist_ok=True)
    sys.path.insert(0, os.path.join(REPO, "tools", "bench"))
    from prompts import Filler
    for k in ("prose", "code", "logs"):
        with open(os.path.join(corpus, k + ".txt"), "w", encoding="utf-8") as f:
            f.write(Filler(k, "n11-corpus").text(400))
    common = ["-np", "1", "-t", a.threads, "--no-warmup"]
    cand = Server("cand", a.server, a.model, free_port(), ["-c", "32768", "-ngl", "all", "-sm", "layer", "-ts", "1,1,1,1",
                                                          "-fa", "on", "-ctk", "f16", "-ctv", "f16", "--jinja",
                                                          "--metrics"] + common, work)
    ref = Server("ref", a.server, a.model, free_port(), ["-c", "8192", "-fa", "off"] + common, work)
    ck = Checks()
    try:
        cand.start()
        ref.start()
        prompt_file = [None]
        ck.guard("tokenize", "prompts.py", lambda: prompt_file.__setitem__(0, check_tokenize(ck, cand, work)))
        ck.guard("matrix", "run_matrix.py", lambda: check_matrix(ck, cand, work))
        ck.guard("probes", "probes.py", lambda: check_probes(ck, cand, work))
        ck.guard("slot", "slot_persist.py", lambda: check_slot(ck, cand, work, prompt_file[0]))
        ck.guard("logprob", "logprob_compare.py", lambda: check_logprob(ck, ref, cand, work,
                                                                         os.path.join(corpus, "*.txt")))
        ck.guard("gateway", "gateway.py", lambda: check_gateway(ck, cand, work))
        cand.stop()                              # run_r1 starts its own engine
        ck.guard("r1", "run_r1.py", lambda: check_r1(ck, a.server, a.model, ref, work, prompt_file[0],
                                                     os.path.join(corpus, "*.txt")))
    finally:
        cand.stop()
        ref.stop()
    n_ok = sum(r["ok"] for r in ck.rows)
    print("r1_tools_smoke: %d/%d checks PASS" % (n_ok, len(ck.rows)))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"checks": ck.rows, "server": a.server, "model": a.model, "pass": n_ok == len(ck.rows)}, f,
                      indent=1)
    return 0 if ck.rows and n_ok == len(ck.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
