#!/usr/bin/env python3
"""N7: functional checks of a real llama-server on hybrid-memory models (recurrent state + KV), CPU, no GPU.

Risks for R1/R5 that a CPU run on a tiny random-weight model can settle (docs/reassessment/ROADMAP.md, row N7):
  determinism    two cold runs of the same prompt give the same tokens and the same top-k log-probabilities;
  prefix_extend  agent loop: the next request = previous prompt + previous answer + a new tail. How many prompt
                 tokens the server re-processes (timings.prompt_n) and whether the answer equals a cold run (B3);
  prefix_diverge the next request shares only the first m tokens of the previous prompt (a re-rendered chat
                 history). Recurrent state cannot be cut at m, so the server either re-processes everything or
                 restores a context checkpoint; either way the answer must equal a cold run;
  cache_reuse    --cache-reuse 8 and a next prompt with 16 tokens deleted after a shared head (old tool output
                 dropped): the server may shift the KV chunk, but recurrent state cannot follow the shift, so the
                 answer must still equal a cold run; the log tells whether the chunk was shifted or the flag disabled;
  slot_persist   tools/kv/slot_persist.py run_once unchanged (save, erase, restore, same prompt again), plus a
                 restore into a freshly started server process (the K3 use: state on NVMe survives a restart);
  isolation      two slots (-np 2) decode two different prompts at the same time; each answer must equal the same
                 prompt run alone on that server (multi-sequence isolation, GLM53-TZ.md section 4); the server itself
                 must report both slots processing at once (GET /slots, is_processing), otherwise nothing proves that
                 the two sequences were decoded together;
  mtp            the same prompt with and without the in-model MTP head (--spec-type draft-mtp): the server must
                 not crash (docs/evidence/2026-10-10-qwen4exp-decode-parity.md finding 12) and greedy output must
                 equal plain decoding; rejected drafts exercise the rollback of recurrent state in the server.
A check passes only if every token is identical and every shared top-k log-probability differs by at most --tol
("not only top-1", GLM53-TZ.md section 4). Prompts are token-id lists, so the tiny models' vocabulary never matters.

Uses only documented endpoints of llama-server (tools/server/README.md): GET /health, POST /completion,
POST /slots/{id}?action=save|restore|erase. Python 3 stdlib only.

Usage:
  python3 tools/srvcheck/hybrid_server_checks.py --server <llama.cpp>/build/bin/llama-server \\
      --model tiny.gguf [--mtp-model tiny_mtp.gguf] --out result.json
Exit code 0 only if every check ran and passed.
"""
import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kv"))
import slot_persist  # noqa: E402  (tools/kv/slot_persist.py, used unchanged)

# reference precision contract of the CPU parity runs (f32 caches, no flash attention): N2/N6
BASE_ARGS = ["-c", "1024", "-np", "1", "-t", "2", "-fa", "off", "-ctk", "f32", "-ctv", "f32", "--no-warmup"]


def get(url, path, timeout=10):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.load(r)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One llama-server process; its log is kept for the evidence."""

    def __init__(self, binary, model, extra, log_path, timeout=120.0):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(self.port)] + list(extra)
        self.log_path = log_path
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t_end = time.time() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} before /health: {log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() > t_end:
                self.stop()
                raise RuntimeError(f"llama-server not healthy after {timeout} s: {log_path}")
            time.sleep(0.2)

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        """Stops the server; returns its exit code (negative = killed by that signal, before our SIGTERM)."""
        rc = self.proc.poll()
        if rc is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            rc = 0                       # stopped by us, not a crash
        self.log.close()
        return rc


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def complete(url, prompt, n_predict, n_probs=5, slot=0, **fields):
    """Greedy completion of a token-id prompt in one slot; returns ids, top-k log-probs per step and timings."""
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True, "id_slot": slot,
            "ignore_eos": True, "n_probs": n_probs, "return_tokens": True, "stream": False, **fields}
    r = post(url, "/completion", body)
    probs = r.get("completion_probabilities") or []
    steps = [{str(c["id"]): c["logprob"] for c in (p.get("top_logprobs") or [])} for p in probs]
    t = r.get("timings") or {}
    return {"tokens": r.get("tokens"), "top": steps, "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "cache_n": t.get("cache_n"),
            "predicted_n": t.get("predicted_n"), "draft_n": t.get("draft_n"),
            "draft_n_accepted": t.get("draft_n_accepted"), "tokens_cached": r.get("tokens_cached"),
            "n_prompt": len(prompt)}


def erase(url, slot=0):
    return post(url, f"/slots/{slot}?action=erase", {})


def compare(a, b, tol):
    """Same tokens, and every log-prob that both runs list in a step's top-k differs by at most tol."""
    out = {"same_tokens": a["tokens"] == b["tokens"], "first_diff": None, "max_abs_dlogprob": 0.0,
           "steps_compared": min(len(a["top"]), len(b["top"])), "top1_margin_at_first_diff": None}
    if not out["same_tokens"]:
        ta, tb = a["tokens"] or [], b["tokens"] or []
        out["first_diff"] = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        i = out["first_diff"]
        if i < len(a["top"]):        # a near tie explains a flip; a large margin does not
            v = sorted(a["top"][i].values(), reverse=True)
            out["top1_margin_at_first_diff"] = v[0] - v[1] if len(v) > 1 else None
    if out["steps_compared"] == 0 or len(a["top"]) != len(b["top"]):
        out["max_abs_dlogprob"] = None   # nothing to compare is not a pass
    else:
        for sa, sb in zip(a["top"], b["top"]):
            for k in set(sa) & set(sb):
                out["max_abs_dlogprob"] = max(out["max_abs_dlogprob"], abs(sa[k] - sb[k]))
    out["pass"] = bool(out["same_tokens"] and out["max_abs_dlogprob"] is not None and out["max_abs_dlogprob"] <= tol)
    return out


def ids(rng, n, lo=3, hi=96):
    return [rng.randrange(lo, hi) for _ in range(n)]


def check_determinism(url, prompt, gen, tol):
    erase(url)
    a = complete(url, prompt, gen)
    erase(url)
    b = complete(url, prompt, gen)
    return {"cold_1": a, "cold_2": b, "cmp": compare(a, b, tol)}


def check_prefix_extend(url, prompt, tail, gen, tol):
    erase(url)
    first = complete(url, prompt, gen)
    nxt = prompt + first["tokens"] + tail          # previous prompt + previous answer + new turn
    warm = complete(url, nxt, gen)
    erase(url)
    cold = complete(url, nxt, gen)
    reused = None if warm["prompt_n"] is None else len(nxt) - warm["prompt_n"]
    return {"first": first, "warm": warm, "cold": cold, "n_next": len(nxt), "reused_tokens": reused,
            "reusable_tokens": len(prompt) + len(first["tokens"]), "cmp": compare(warm, cold, tol)}


def check_prefix_diverge(url, prompt, keep, tail, gen, tol):
    erase(url)
    first = complete(url, prompt, gen)
    nxt = prompt[:keep] + tail                      # shares only the first `keep` tokens
    warm = complete(url, nxt, gen)
    erase(url)
    cold = complete(url, nxt, gen)
    reused = None if warm["prompt_n"] is None else len(nxt) - warm["prompt_n"]
    return {"first": first, "warm": warm, "cold": cold, "n_next": len(nxt), "shared_prefix": keep,
            "reused_tokens": reused, "cmp": compare(warm, cold, tol)}


def check_cache_reuse(binary, model, extra, head, body, cut, gen, tol, workdir):
    """--cache-reuse 8: first head + body, then head + body[cut:]; warm must equal cold."""
    log = os.path.join(workdir, "cache_reuse.log")
    # -lv 4: the chunk shift and the full re-processing are logged at TRACE (tools/server/server-context.cpp)
    srv = Server(binary, model, list(extra) + ["--cache-reuse", "8", "-lv", "4"], log)
    try:
        erase(srv.url)
        first = complete(srv.url, head + body, gen)
        nxt = head + body[cut:]
        warm = complete(srv.url, nxt, gen, n_cache_reuse=8)
        erase(srv.url)
        cold = complete(srv.url, nxt, gen)
    finally:
        rc = srv.stop()
    with open(log) as f:
        text = f.read()
    return {"first": first, "warm": warm, "cold": cold, "n_next": len(nxt), "deleted": cut, "exit": rc,
            "flag_disabled": "cache_reuse is not supported" in text, "chunk_shifted": "reusing chunk" in text,
            "forced_full_reprocess": "forcing full prompt re-processing" in text,
            "reused_tokens": None if warm["prompt_n"] is None else len(nxt) - warm["prompt_n"],
            "cmp": compare(warm, cold, tol)}


def check_isolation(binary, model, extra, prompts, gen, tol, workdir):
    """Each prompt alone in its slot, then both at once; the concurrent answers must equal the lone ones."""
    args = [a for a in extra]
    i = args.index("-np") if "-np" in args else None
    if i is not None:
        del args[i:i + 2]
    srv = Server(binary, model, args + ["-np", "2"], os.path.join(workdir, "isolation.log"))
    out = {"alone": [], "together": [None, None], "spans": [None, None], "max_busy_slots": 0, "slots_errors": []}
    try:
        for k, p in enumerate(prompts):
            erase(srv.url, 0)
            erase(srv.url, 1)
            out["alone"].append(complete(srv.url, p, gen, slot=k))
        erase(srv.url, 0)
        erase(srv.url, 1)
        errors = []

        def worker(k):
            t0 = time.perf_counter()
            try:
                out["together"][k] = complete(srv.url, prompts[k], gen, slot=k)
            except Exception as e:  # recorded below
                errors.append(repr(e))
            out["spans"][k] = (t0, time.perf_counter())

        done = threading.Event()

        def poll():                              # what the server says, not what the client sees
            while not done.is_set():
                try:
                    busy = sum(1 for s in get(srv.url, "/slots") if s.get("is_processing"))
                    out["max_busy_slots"] = max(out["max_busy_slots"], busy)
                except Exception as e:  # recorded; without /slots nothing proves concurrency
                    if len(out["slots_errors"]) < 3:
                        out["slots_errors"].append(repr(e))
                time.sleep(0.005)

        poller = threading.Thread(target=poll)
        poller.start()
        ts = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        done.set()
        poller.join()
        out["errors"] = errors
    finally:
        out["exit"] = srv.stop()
    (a0, a1), (b0, b1) = out["spans"]
    out["overlap_s"] = max(0.0, min(a1, b1) - max(a0, b0))   # client side: both requests in flight (not proof)
    out["cmp"] = [compare(out["together"][k], out["alone"][k], tol) if out["together"][k] else {"pass": False}
                  for k in range(2)]
    out["pass"] = not out["errors"] and out["exit"] == 0 and out["max_busy_slots"] >= 2 and \
        all(c["pass"] for c in out["cmp"])
    return out


def check_slot_persist(binary, model, extra, prompt, gen, tol, workdir):
    """slot_persist.run_once in one process, then save -> new process -> restore -> same prompt."""
    save_dir = extra[extra.index("--slot-save-path") + 1]
    args = list(extra)
    s1 = Server(binary, model, args, os.path.join(workdir, "slot_persist_1.log"))
    try:
        once = slot_persist.run_once(s1.url, prompt, gen, "n7-slot0.bin", 600)
        erase(s1.url)
        ref = complete(s1.url, prompt, gen)
        post(s1.url, "/slots/0?action=save", {"filename": "n7-restart.bin"})
    finally:
        rc1 = s1.stop()
    s2 = Server(binary, model, args, os.path.join(workdir, "slot_persist_2.log"))
    try:
        restored = post(s2.url, "/slots/0?action=restore", {"filename": "n7-restart.bin"})
        after = complete(s2.url, prompt, gen)
    finally:
        rc2 = s2.stop()
    size = os.path.getsize(os.path.join(save_dir, "n7-restart.bin"))
    return {"run_once": once, "reference": ref, "restore_response": restored, "after_restart": after,
            "file_bytes": size, "server_exit": [rc1, rc2],
            "cmp_restart": compare(after, ref, tol),
            "restart_reused_tokens": None if after["prompt_n"] is None else len(prompt) - after["prompt_n"]}


def check_mtp(binary, mtp_model, extra, mtp_args, prompt, gen, tol, workdir):
    """Plain greedy decode vs MTP speculative decode on the same GGUF (the MTP block is in the file)."""
    out = {"mtp_args": mtp_args}
    s0 = Server(binary, mtp_model, extra, os.path.join(workdir, "mtp_off.log"))
    try:
        erase(s0.url)
        out["off"] = complete(s0.url, prompt, gen)
    finally:
        out["off_exit"] = s0.stop()
    s1 = Server(binary, mtp_model, list(extra) + list(mtp_args), os.path.join(workdir, "mtp_on.log"))
    try:
        erase(s1.url)
        try:
            out["on"] = complete(s1.url, prompt, gen)
            erase(s1.url)
            out["on_repeat"] = complete(s1.url, prompt, gen)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            out["error"] = repr(e)
        out["alive_after"] = s1.alive()
    finally:
        out["on_exit"] = s1.stop()
    with open(os.path.join(workdir, "mtp_on.log")) as f:
        log = f.read()
    out["log_mentions_mtp"] = [ln for ln in log.splitlines() if "mtp" in ln.lower() or "draft" in ln.lower()][:40]
    if "on" in out:
        out["cmp"] = compare(out["on"], out["off"], tol)
        out["cmp_repeat"] = compare(out["on_repeat"], out["on"], tol)
        out["pass"] = out["cmp"]["pass"] and out["cmp_repeat"]["pass"] and out["on_exit"] == 0 \
            and bool(out["on"].get("draft_n"))       # MTP actually drafted: otherwise nothing was tested
    else:
        out["pass"] = False
    return out


def run_model(binary, model, mtp_model, mtp_args, extra, workdir, gen, tol, seed):
    rng = random.Random(seed)
    prompt = ids(rng, 200)
    tail = ids(rng, 24)
    save_dir = os.path.join(workdir, "slots")      # POST /slots/* answers 501 without --slot-save-path
    os.makedirs(save_dir, exist_ok=True)
    extra = list(extra) + ["--slot-save-path", save_dir]
    res = {"model": model, "server_args": list(extra)}
    srv = Server(binary, model, extra, os.path.join(workdir, "main.log"))
    try:
        res["determinism"] = check_determinism(srv.url, prompt, gen, tol)
        res["prefix_extend"] = check_prefix_extend(srv.url, prompt, tail, gen, tol)
        res["prefix_diverge"] = check_prefix_diverge(srv.url, prompt, 120, tail, gen, tol)
    finally:
        res["main_exit"] = srv.stop()
    res["slot_persist"] = check_slot_persist(binary, model, extra, prompt, gen, tol, workdir)
    res["cache_reuse"] = check_cache_reuse(binary, model, extra, prompt[:64], prompt[64:184], 16, gen, tol, workdir)
    res["isolation"] = check_isolation(binary, model, extra, [prompt, ids(rng, 180)], gen, tol, workdir)
    if mtp_model:
        res["mtp"] = check_mtp(binary, mtp_model, extra, mtp_args, prompt, gen, tol, workdir)
    res["verdict"] = verdict(res)
    return res


def verdict(res):
    v = {
        "determinism": res["determinism"]["cmp"]["pass"],
        "prefix_extend_correct": res["prefix_extend"]["cmp"]["pass"],
        "prefix_diverge_correct": res["prefix_diverge"]["cmp"]["pass"],
        "slot_persist_same_process": res["slot_persist"]["run_once"]["same_continuation"],
        "slot_persist_after_restart": res["slot_persist"]["cmp_restart"]["pass"],
        "cache_reuse_correct": res["cache_reuse"]["cmp"]["pass"] and res["cache_reuse"]["exit"] == 0,
        "isolation_two_slots": res["isolation"]["pass"],
        "server_exit_clean": res["main_exit"] == 0 and res["slot_persist"]["server_exit"] == [0, 0],
    }
    if "mtp" in res:
        v["mtp_equals_plain_decode"] = res["mtp"]["pass"]
    # reuse is reported, not judged: whether it is enough is the B3 decision on the server (EXPERIMENTS.md)
    v["all_pass"] = all(v.values())
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="llama-server binary")
    ap.add_argument("--model", required=True, help="GGUF without MTP block")
    ap.add_argument("--mtp-model", help="GGUF with the MTP block (nextn_predict_layers = 1)")
    ap.add_argument("--mtp-args", default="--spec-type draft-mtp --spec-draft-n-max 3",
                    help="llama-server flags that turn the in-model MTP head on")
    ap.add_argument("--extra", default=" ".join(BASE_ARGS), help="llama-server flags for every run")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-3, help="max |delta log-prob| over shared top-k entries")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workdir", help="keeps server logs and slot files (default: a temporary directory)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    workdir = a.workdir or tempfile.mkdtemp(prefix="n7-")
    os.makedirs(workdir, exist_ok=True)
    try:
        res = run_model(a.server, a.model, a.mtp_model, a.mtp_args.split(), a.extra.split(), workdir, a.gen, a.tol,
                        a.seed)
    except Exception as e:     # recorded, never turned into a pass
        res = {"model": a.model, "error": repr(e), "verdict": {"all_pass": False}}
    res["server_binary"] = a.server
    res["workdir"] = workdir
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(summary(res), indent=1))
    if not a.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if res["verdict"].get("all_pass") else 1


def summary(res):
    if "error" in res:
        return {"model": res["model"], "error": res["error"], "verdict": res["verdict"]}
    s = {"model": os.path.basename(res["model"]), "verdict": res["verdict"]}
    for k in ("determinism", "prefix_extend", "prefix_diverge"):
        c = res[k]["cmp"]
        s[k] = {"same_tokens": c["same_tokens"], "max_abs_dlogprob": c["max_abs_dlogprob"]}
    for k in ("prefix_extend", "prefix_diverge"):
        r = res[k]
        s[k].update({"n_next": r["n_next"], "warm_prompt_n": r["warm"]["prompt_n"], "warm_cache_n": r["warm"]["cache_n"],
                     "reused_tokens": r["reused_tokens"], "cold_prompt_n": r["cold"]["prompt_n"]})
    s["prefix_extend"]["reusable_tokens"] = res["prefix_extend"]["reusable_tokens"]
    s["prefix_diverge"]["shared_prefix"] = res["prefix_diverge"]["shared_prefix"]
    sp = res["slot_persist"]
    s["slot_persist"] = {"same_process_identical": sp["run_once"]["same_continuation"],
                         "same_process_warm_prompt_n": sp["run_once"]["warm"]["prompt_n"],
                         "restart_same_tokens": sp["cmp_restart"]["same_tokens"],
                         "restart_max_abs_dlogprob": sp["cmp_restart"]["max_abs_dlogprob"],
                         "restart_warm_prompt_n": sp["after_restart"]["prompt_n"], "file_bytes": sp["file_bytes"]}
    cr = res["cache_reuse"]
    s["cache_reuse"] = {"same_tokens": cr["cmp"]["same_tokens"], "max_abs_dlogprob": cr["cmp"]["max_abs_dlogprob"],
                        "n_next": cr["n_next"], "warm_prompt_n": cr["warm"]["prompt_n"],
                        "reused_tokens": cr["reused_tokens"], "flag_disabled": cr["flag_disabled"],
                        "chunk_shifted": cr["chunk_shifted"], "forced_full_reprocess": cr["forced_full_reprocess"]}
    iso = res["isolation"]
    s["isolation"] = {"pass": iso["pass"], "overlap_s": round(iso["overlap_s"], 3), "max_busy_slots": iso["max_busy_slots"],
                      "same_tokens": [c.get("same_tokens") for c in iso["cmp"]],
                      "max_abs_dlogprob": [c.get("max_abs_dlogprob") for c in iso["cmp"]]}
    if "mtp" in res:
        m = res["mtp"]
        s["mtp"] = {"pass": m["pass"], "on_exit": m.get("on_exit"), "error": m.get("error")}
        if "cmp" in m:
            s["mtp"].update({"same_tokens": m["cmp"]["same_tokens"], "max_abs_dlogprob": m["cmp"]["max_abs_dlogprob"],
                             "first_diff": m["cmp"]["first_diff"], "draft_n": m["on"]["draft_n"],
                             "draft_n_accepted": m["on"]["draft_n_accepted"]})
    return s


if __name__ == "__main__":
    sys.exit(main())
