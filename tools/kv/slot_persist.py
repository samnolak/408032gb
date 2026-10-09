#!/usr/bin/env python3
"""Experiment K3: is restoring a saved long-context state from NVMe faster than prefilling it again? (llama-server)

Needs llama-server started with `--slot-save-path <dir on NVMe>` and `-np 1` (one slot, id 0), e.g. the GLM-5.3-Flash or
Qwen3.8-Flash-Next server of R1. Uses only documented endpoints (tools/server/README.md @10a60cf30): POST /completion,
POST /slots/{id}?action=save|restore|erase. Hybrid models (KV + recurrent state + DSA indexer cache) are saved and restored
as a whole by llama.cpp (src/llama-memory-hybrid-idx.cpp:256-288 @10a60cf30).

Procedure (all timings literal, nothing estimated):
  1. cold: send PROMPT (target ~N tokens) with n_predict=G, temperature 0 -> TTFT_cold, tokens A (reference continuation)
  2. save slot 0 -> file; record save time and file size (bytes on disk = the state's real size)
  3. erase slot 0, then restore from the file -> restore time
  4. warm: send the same PROMPT again, n_predict=G -> TTFT_warm and tokens B; the server should reuse the restored cache
  5. correctness: B == A (greedy continuation identical); prompt_n of step 4 shows how many tokens were re-processed
Repeat R times; report p50 and every raw value.

Usage:
  python3 tools/kv/slot_persist.py --url http://127.0.0.1:8080 --prompt-file long.txt --gen 64 --repeats 3 \
      --out docs/evidence/K3-<model>-<date>.json
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


def post(url, path, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp, time.perf_counter() - t0


def completion(url, prompt, n_gen, timeout):
    body = {"prompt": prompt, "n_predict": n_gen, "temperature": 0.0, "cache_prompt": True, "id_slot": 0,
            "stream": False}
    resp, dt = post(url, "/completion", body, timeout)
    t = resp.get("timings") or {}
    return {"seconds": dt, "content": resp.get("content", ""), "prompt_n": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
            "tokens_cached": resp.get("tokens_cached")}


def run_once(url, prompt, n_gen, fname, timeout):
    post(url, "/slots/0?action=erase", {}, timeout)
    cold = completion(url, prompt, n_gen, timeout)
    save, t_save = post(url, "/slots/0?action=save", {"filename": fname}, timeout)
    post(url, "/slots/0?action=erase", {}, timeout)
    restore, t_restore = post(url, "/slots/0?action=restore", {"filename": fname}, timeout)
    warm = completion(url, prompt, n_gen, timeout)
    return {"cold": cold, "save": {"seconds": t_save, "response": save},
            "restore": {"seconds": t_restore, "response": restore}, "warm": warm,
            "same_continuation": cold["content"] == warm["content"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--filename", default="k3-slot0.bin", help="file name inside the server's --slot-save-path")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.prompt_file, encoding="utf-8") as f:
        prompt = f.read()
    runs, errors = [], []
    for i in range(a.repeats):
        try:
            runs.append(run_once(a.url, prompt, a.gen, a.filename, a.timeout))
        except Exception as e:  # recorded, never replaced by a number
            errors.append({"repeat": i, "error": repr(e)})
    summary = None
    if runs:
        def med(xs):
            xs = [x for x in xs if x is not None]
            return statistics.median(xs) if xs else None
        summary = {
            "runs": len(runs), "errors": len(errors),
            "cold_prompt_ms_p50": med([r["cold"]["prompt_ms"] for r in runs]),
            "cold_prompt_n_p50": med([r["cold"]["prompt_n"] for r in runs]),
            "save_s_p50": med([r["save"]["seconds"] for r in runs]),
            "restore_s_p50": med([r["restore"]["seconds"] for r in runs]),
            "warm_prompt_ms_p50": med([r["warm"]["prompt_ms"] for r in runs]),
            "warm_prompt_n_p50": med([r["warm"]["prompt_n"] for r in runs]),
            "saved_bytes": med([(r["save"]["response"] or {}).get("n_written") for r in runs]),
            "server_restore_ms_p50": med([((r["restore"]["response"] or {}).get("timings") or {}).get("restore_ms")
                                          for r in runs]),
            "all_continuations_identical": all(r["same_continuation"] for r in runs),
        }
    res = {"url": a.url, "prompt_file": a.prompt_file, "prompt_chars": len(prompt), "gen": a.gen,
           "summary": summary, "runs": runs, "errors": errors,
           "_label": "CONFIRMED only as measured on the named server; restore vs prefill decides experiment K3"}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(summary, indent=1) if summary else f"no successful runs: {errors}")
    return 0 if runs and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
