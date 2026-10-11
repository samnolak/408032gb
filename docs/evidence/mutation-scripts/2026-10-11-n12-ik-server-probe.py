#!/usr/bin/env python3
"""N12 probe (not a mutation check): ik_llama.cpp's llama-server, started with the engine argv of
scripts/r1/configs/ik-llamacpp-server.json, on the tiny glm5next model (tools/tinygen/glm5next_tiny.py --layers 12
--dialect ik, id-only vocabulary), CPU only. Prints what the R1 tools rely on: /health, /props, /v1/models, /tokenize
and /detokenize, /completion (n_predict, ignore_eos, n_probs layout), /v1/completions logprobs layout, /slots
save / erase / restore (also of an empty slot); the server's greedy continuation of the N9 prompt with the
template's flags is written to <workdir>/server_tokens.txt (prompt + continuation) for the dump-tool comparison of
2026-10-11-n12-server-greedy.sh.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n12-ik-server-probe.py <repo> <ik bin dir> <ik.gguf>
       <gguf-py> <workdir> [port]
Substitutions in the template argv (CPU, 4 cores, tiny model): IK_BIN, MODEL, HOST=127.0.0.1, PORT; THREADS and
THREADS_BATCH = 4 (the template's 48 are the server's cores); everything else as the template's "vars" (CTX 262144,
TENSOR_SPLIT, KV_TYPE f16, NCMOE 17). The template has no --slot-save-path (supports_slots false); a second start
appends it for the /slots probe only. Wall times are not results."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


repo, ikbin, gguf, gguf_py, work = sys.argv[1:6]
work, ikbin, gguf = (os.path.abspath(x) for x in (work, ikbin, gguf))
port = int(sys.argv[6]) if len(sys.argv) > 6 else 18471
os.makedirs(work, exist_ok=True)
tpl = json.load(open(os.path.join(repo, "scripts/r1/configs/ik-llamacpp-server.json")))
V = dict(tpl["vars"], IK_BIN=ikbin, MODEL=gguf, HOST="127.0.0.1", PORT=str(port), THREADS="4", THREADS_BATCH="4")


def sub(s):
    for k, v in V.items():
        s = s.replace("${" + k + "}", v)
    return s


ARGV = [sub(a) for a in tpl["engine"]["argv"]]
GPU_ONLY = ("-ngl", "-sm", "-ts")        # a CPU build rejects -ts (llama_max_devices() = 1, common/common.cpp:2285-2287)
ARGV_CPU = [a for i, a in enumerate(ARGV) if a not in GPU_ONLY and (i == 0 or ARGV[i - 1] not in GPU_ONLY)]


def http(method, path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def port_busy():
    with socket.socket() as so:
        return so.connect_ex(("127.0.0.1", port)) == 0


STARTED = []


def start(argv, log):
    if port_busy():
        sys.exit(f"port {port} is already in use: stop that process first")
    p = subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=work)   # ik writes llama.log to cwd
    STARTED.append(p)
    t0 = time.time()
    while time.time() - t0 < 120:
        if p.poll() is not None:
            return p, None
        try:
            code, body = http("GET", "/health", timeout=5)
            if code == 200:
                return p, (code, body)
        except (urllib.error.URLError, ConnectionError, socket.timeout):
            pass
        time.sleep(0.5)
    return p, None


def stop(p):
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def keys(obj, depth=0):
    """Key layout of a JSON value (lists: the first element), values elided."""
    if isinstance(obj, dict):
        return {k: keys(v, depth + 1) for k, v in obj.items()} if depth < 4 else "{...}"
    if isinstance(obj, list):
        return [keys(obj[0], depth + 1)] if obj else []
    return type(obj).__name__


def show(title, code, body, layout=True):
    print(f"--- {title}: HTTP {code}")
    try:
        j = json.loads(body)
        print(json.dumps(keys(j) if layout else j)[:1500])
        return j
    except ValueError:
        print(body[:500])
        return None


import atexit  # noqa: E402
atexit.register(lambda: [stop(q) for q in STARTED])
print("argv (substituted):", " ".join(ARGV))
code, out = 0, subprocess.run([ARGV[0], "--version"], capture_output=True, text=True)
print("--version rc", out.returncode, "|", (out.stdout + out.stderr).strip().splitlines()[:2])
p, health = start(ARGV, os.path.join(work, "server-template.log"))
print("template argv: started" if health else f"template argv: did not become healthy (rc {p.poll()})", health)
log = open(os.path.join(work, "server-template.log")).read()
for w in ("warning: not compiled with GPU offload", "error:", "warning: llama.cpp was compiled without"):
    hits = [ln for ln in log.splitlines() if w in ln]
    if hits:
        print(f"log lines with '{w}':", hits[:4])
if health:
    stop(p)
print("CPU argv (the template's without -ngl/-sm/-ts):", " ".join(ARGV_CPU[1:]))
p, health = start(ARGV_CPU, os.path.join(work, "server-cpu.log"))
print("CPU argv: started" if health else f"CPU argv: did not become healthy (rc {p.poll()})", health)
log = open(os.path.join(work, "server-cpu.log")).read()
for w in ("error:", "warning:", "n_ctx", "DSA", "dsa", "merge", "muge"):
    hits = [ln for ln in log.splitlines() if w in ln]
    if hits:
        print(f"log lines with '{w}':", hits[:3])
if health:
    show("GET /props", *http("GET", "/props"))
    show("GET /v1/models", *http("GET", "/v1/models"))
    show("POST /tokenize {content: 't5t9'}", *http("POST", "/tokenize", {"content": "t5t9"}), layout=False)
    show("POST /tokenize with_pieces", *http("POST", "/tokenize", {"content": "t5t9", "with_pieces": True}), layout=False)
    show("POST /detokenize [5, 9]", *http("POST", "/detokenize", {"tokens": [5, 9]}), layout=False)
    toks = [int(x) for x in open(os.path.join(work, "tokens.txt")).read().split(",")]
    prompt = toks[:48]
    j = show("POST /completion n_predict 16 ignore_eos n_probs 3 (layout)", *http(
        "POST", "/completion", {"prompt": prompt, "n_predict": 16, "temperature": 0, "ignore_eos": True, "n_probs": 3,
                                "cache_prompt": False, "return_tokens": True}))
    if j:
        print("tokens_predicted", j.get("tokens_predicted"), "stop_type", j.get("stop_type"),
              "tokens", j.get("tokens"))
        cp = j.get("completion_probabilities") or []
        print("completion_probabilities[0]:", json.dumps(cp[0])[:600] if cp else None)
        # no token ids in this layout: the id-only vocabulary spells token n as "t<n>" (glm5next_tiny.py)
        srv = [int(e["content"][1:]) for e in cp]
        print("server greedy tokens:", ",".join(map(str, srv)))
        open(os.path.join(work, "server_tokens.txt"), "w").write(",".join(map(str, prompt + srv)))
    j = show("POST /completion logit_bias EOS +100, ignore_eos false", *http(
        "POST", "/completion", {"prompt": prompt, "n_predict": 16, "temperature": 0, "ignore_eos": False,
                                "logit_bias": [[2, 100.0]], "cache_prompt": False}))
    if j:
        print("tokens_predicted", j.get("tokens_predicted"), "stop_type", j.get("stop_type"))
    j = show("POST /completion logit_bias EOS +100, ignore_eos true", *http(
        "POST", "/completion", {"prompt": prompt, "n_predict": 16, "temperature": 0, "ignore_eos": True,
                                "logit_bias": [[2, 100.0]], "cache_prompt": False}))
    if j:
        print("tokens_predicted", j.get("tokens_predicted"), "stop_type", j.get("stop_type"))
    j = show("POST /v1/completions logprobs 3 max_tokens 2 (layout)", *http(
        "POST", "/v1/completions", {"prompt": prompt, "max_tokens": 2, "temperature": 0, "logprobs": 3}))
    if j:
        print("choices[0].logprobs:", json.dumps(j["choices"][0].get("logprobs"))[:700])
    show("GET /slots (no --slot-save-path)", *http("GET", "/slots"))
    show("POST /slots/0?action=save (no --slot-save-path)", *http("POST", "/slots/0?action=save",
                                                                   {"filename": "n12.bin"}), layout=False)
stop(p)

slotdir = os.path.join(work, "slots")
os.makedirs(slotdir, exist_ok=True)
p, health = start(ARGV_CPU + ["--slot-save-path", slotdir], os.path.join(work, "server-slots.log"))
print("CPU argv + --slot-save-path:", "started" if health else f"not healthy (rc {p.poll()})")
if health:
    toks = [int(x) for x in open(os.path.join(work, "tokens.txt")).read().split(",")]
    req = {"prompt": toks[:48], "n_predict": 8, "temperature": 0, "ignore_eos": True, "cache_prompt": True, "id_slot": 0}
    code, body = http("POST", "/completion", req)
    a = json.loads(body)
    print("completion before save: HTTP", code, a.get("content"), "tokens_cached", a.get("tokens_cached"))
    show("POST /slots/0?action=save", *http("POST", "/slots/0?action=save", {"filename": "n12.bin"}), layout=False)
    print("files in slot dir:", {f: os.path.getsize(os.path.join(slotdir, f)) for f in os.listdir(slotdir)})
    show("POST /slots/0?action=erase", *http("POST", "/slots/0?action=erase"), layout=False)
    show("POST /slots/0?action=restore", *http("POST", "/slots/0?action=restore", {"filename": "n12.bin"}),
         layout=False)
    b = json.loads(http("POST", "/completion", req)[1])
    print("before save:", a.get("content"), "| after restore:", b.get("content"), "| identical:",
          a.get("content") == b.get("content"), "| prompt_n after restore:", (b.get("timings") or {}).get("prompt_n"),
          "of", len(req["prompt"]), "| tokens_cached", b.get("tokens_cached"))
    # a save of an empty slot (after erase), then its restore
    show("POST /slots/0?action=erase", *http("POST", "/slots/0?action=erase"), layout=False)
    show("POST /slots/0?action=save (empty slot)", *http("POST", "/slots/0?action=save", {"filename": "empty.bin"}),
         layout=False)
    try:
        show("POST /slots/0?action=restore (empty save)",
             *http("POST", "/slots/0?action=restore", {"filename": "empty.bin"}), layout=False)
    except (urllib.error.URLError, ConnectionError, socket.timeout) as e:
        print("restore of the empty save: no HTTP answer:", type(e).__name__, e)
    time.sleep(1)
    print("server process after it: returncode", p.poll())
    tail = [ln for ln in open(os.path.join(work, "server-slots.log")).read().splitlines() if "GGML_ASSERT" in ln]
    print("server log:", tail[:2])
stop(p)
