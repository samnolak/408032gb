#!/usr/bin/env python3
"""N11 demonstrations against a real llama-server on CPU (docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md).

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n11-demos.py LLAMA_SERVER SCRATCH_DIR ID_VOCAB.gguf TEXT_VOCAB.gguf
  ID_VOCAB.gguf    tools/tinygen/qwen4exp_tiny.py default output (minimal id-only vocabulary, context_length 512)
  TEXT_VOCAB.gguf  tools/tinygen/qwen4exp_tiny.py --text-vocab --ctx 262144
Run from the repository root. The tools "before" are taken with `git show c2f6c91:<path>` (the base of the N11 branch),
the tools "after" are the working tree. Prints literal server answers and tool outputs; nothing here is a measurement.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

BASE = "c2f6c91"
SERVER, SCRATCH, ID_GGUF, TEXT_GGUF = sys.argv[1:5]
PY = sys.executable


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(name, model, *args):
    p_ = port()
    log = open(os.path.join(SCRATCH, name + ".log"), "wb")
    p = subprocess.Popen([SERVER, "-m", model, "--host", "127.0.0.1", "--port", str(p_), "-np", "1", "-t", "2",
                          "--no-warmup"] + list(args), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = "http://127.0.0.1:%d" % p_
    for _ in range(300):
        try:
            if urllib.request.urlopen(url + "/health", timeout=2).status == 200:
                return p, url, log.name
        except OSError:
            time.sleep(0.1)
    raise SystemExit("%s did not start" % name)


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    p.wait(timeout=30)


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def old(path):
    d = os.path.join(SCRATCH, "base-" + BASE)
    out = os.path.join(d, path)
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(subprocess.check_output(["git", "show", "%s:%s" % (BASE, path)]))
    return out


def tool(argv):
    p = subprocess.run([PY] + argv, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).replace(SCRATCH, "<scratch>")


def head(title):
    print("\n== %s" % title, flush=True)


os.makedirs(SCRATCH, exist_ok=True)
for path in ("tools/bench/run_matrix.py", "tools/bench/oa_client.py", "tools/bench/prompts.py", "tools/bench/runlog.py",
             "tools/quality/logprob_compare.py"):
    old(path)

head("D1 id-only vocabulary (tinygen default): text is refused, the slot is capped at context_length 512")
p, url, log = start("d1", ID_GGUF, "-c", "1024")
for body in ({"content": "The quick brown fox.", "add_special": False}, {"prompt": "The quick", "add_special_tokens": False}):
    print("POST /tokenize %s -> %s %s" % (json.dumps(body), *post(url, "/tokenize", body)))
stop(p)
print([l.strip() for l in open(log, encoding="utf-8", errors="replace") if "capping" in l][0][-90:])

head("D2 text vocabulary (--text-vocab): /tokenize, logprobs layout of /v1/completions, two ids with one text")
p, url, log = start("d2", TEXT_GGUF, "-c", "16384", "-fa", "on")
print("POST /tokenize 'The quick brown fox.' -> %s %s" % post(url, "/tokenize", {"content": "The quick brown fox.",
                                                                               "add_special": False}))
print("POST /tokenize ' the' -> %s %s" % post(url, "/tokenize", {"content": " the", "add_special": False}))
st, txt = post(url, "/v1/completions", {"prompt": "The quick brown fox", "max_tokens": 1, "temperature": 0,
                                        "logprobs": 2, "stream": False})
ch = json.loads(txt)["choices"][0]
print("POST /v1/completions logprobs=2 -> %s; choices[0] keys %s; logprobs keys %s" % (
    st, sorted(ch), sorted(ch["logprobs"] or {})))
print("  choices[0].logprobs.content[0] = %s" % json.dumps(ch["logprobs"]["content"][0], ensure_ascii=True))
for ids in ([200], [201]):
    st, txt = post(url, "/detokenize", {"tokens": ids})
    print("POST /detokenize %s -> %s %s (content %s)" % (json.dumps({"tokens": ids}), st, ascii(txt),
                                                        ascii(json.loads(txt)["content"])))

head("D3 logprob_compare before (%s) and after, ref -fa off vs cand -fa on, 10 prefixes" % BASE)
p2, url2, _ = start("d3ref", TEXT_GGUF, "-c", "16384", "-fa", "off")
corpus = os.path.join(SCRATCH, "corpus")
os.makedirs(corpus, exist_ok=True)
sys.path.insert(0, "tools/bench")
from prompts import Filler  # noqa: E402
for k in ("prose", "code", "logs"):
    with open(os.path.join(corpus, k + ".txt"), "w") as f:
        f.write(Filler(k, "n11-corpus").text(400))
for label, lc in (("before", old("tools/quality/logprob_compare.py")), ("after", "tools/quality/logprob_compare.py")):
    out = os.path.join(SCRATCH, "lc-%s.json" % label)
    rc, text = tool([lc, "--ref", url2, "--cand", url, "--corpus", os.path.join(corpus, "*.txt"), "--prefixes", "10",
                     "--top-k", "20", "--max-chars", "1500", "--out", out])
    d = json.load(open(out))
    print("%s: exit %d: %s" % (label, rc, text.strip().splitlines()[-1]))
    print("  first error: %s" % (d["errors"][0]["error"][:150] if d["errors"] else None))
stop(p2)

head("D4 run_matrix before (%s) and after: the R1 template's smoke-2k, then ctx-2k's colliding cell, same server" % BASE)
for label, rm in (("before", old("tools/bench/run_matrix.py")), ("after", "tools/bench/run_matrix.py")):
    common = ["--url", url, "--no-nvidia-smi", "--concurrency", "1", "--repeats", "1", "--scenarios", "matrix",
              "--seed", "0", "--extra-body", "{}"]
    runs = {"smoke-2k": ["--inputs", "2048", "--outputs", "128", "--states", "cold,warm"],
            "ctx-2k": ["--inputs", "2048", "--length-mode", "total", "--outputs", "128", "--states", "cold"]}
    for name, a in runs.items():
        d = os.path.join(SCRATCH, "d4-%s-%s" % (label, name))
        shutil.rmtree(d, ignore_errors=True)
        tool([rm, "--out-dir", d] + common + a)
        for l in open(os.path.join(d, "raw.jsonl")):
            r = json.loads(l)
            if r.get("phase") == "measure" and r["state"] == "cold":
                print("%s %-8s cold 2048/128 rep 0: prompt_tokens %d, server cached_tokens %d" % (
                    label, name, r["metrics"]["prompt_tokens"], r["metrics"]["cached_tokens"]))
stop(p)
