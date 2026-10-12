"""Shared pieces of the bench CLIs: run directory, metadata, per-request metrics, statistics.

A run directory holds:
  meta.json     what ran, where, against what (UTC times, argv, server, /v1/models, git SHA, host, nvidia-smi)
  raw.jsonl     one JSON object per request, all timings; failures included as failures
  summary.json  aggregates computed only from measured values
  summary.md    the same, human-readable; anything not measured prints as "n/a"
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCHEMA = "408032gb-bench/1"
MOCK_OWNER = "408032gb-bench-mock"          # owned_by of the mock server's model entry

# Definitions written into every summary so a reader never has to guess.
DEFINITIONS = {
    "ttft_ms": "request sent -> first SSE chunk carrying generated output (content, reasoning or tool call); "
               "includes HTTP, queueing, tokenisation, prefill and the first decode step",
    "ttfc_ms": "request sent -> first chunk carrying content or a tool call (excludes reasoning-only chunks)",
    "prefill_tok_s": "usage.prompt_tokens / TTFT; includes server-cached tokens, so warm values are not compute rates",
    "uncached_prefill_tok_s": "(usage.prompt_tokens - server-reported cached tokens) / TTFT; only when the server reports cached tokens",
    "decode_tok_s": "(n - n/chunks) / (t_last_chunk - t_first_chunk), n = usage.completion_tokens (else chunk count); "
                    "equals (n-1)/(t_last-t_first) when every chunk carries one token",
    "itl_ms": "intervals between consecutive output-carrying SSE chunks (inter-token when 1 token/chunk)",
    "aggregate_output_tok_s": "sum(completion_tokens) / wall time of the wave (first send -> last finish); concurrency > 1 only",
    "percentiles": "linear interpolation between order statistics (numpy 'linear'); p95 of 3 samples is close to the max",
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------- statistics
def percentile(xs, q):
    """Linear-interpolation percentile of a non-empty list; q in [0, 100]."""
    if not xs:
        raise ValueError("percentile of empty list")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def describe(xs, pcts=(50, 95)):
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out = {"n": len(vals)}
    if not vals:
        return out
    for p in pcts:
        out["p%d" % p] = percentile(vals, p)
    out["min"] = min(vals)
    out["max"] = max(vals)
    out["mean"] = sum(vals) / len(vals)
    return out


def fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "n/a"
        return ("%%.%df" % nd) % x
    return str(x)


def fstat(d, key, nd=1):
    return fmt(d.get(key) if d else None, nd)


# ---------------------------------------------------------------------------- per-request metrics
def cached_tokens(usage, timings):
    """Server-reported prompt-cache hits: OpenAI/vLLM/llama.cpp usage.prompt_tokens_details.cached_tokens,
    else llama.cpp timings.cache_n. None when the server does not report it."""
    if isinstance(usage, dict):
        det = usage.get("prompt_tokens_details")
        if isinstance(det, dict) and isinstance(det.get("cached_tokens"), int):
            return det["cached_tokens"]
    if isinstance(timings, dict) and isinstance(timings.get("cache_n"), int):
        return timings["cache_n"]
    return None


def request_metrics(r):
    """Metrics of one StreamResult dict. Failed requests get counts only, every timing None."""
    usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
    pt = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
    ct = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
    gen = [e for e in (r.get("events") or []) if e[1] or e[2] or e[3] or e[4]]
    m = {
        "ok": bool(r.get("ok")), "prompt_tokens": pt, "completion_tokens": ct,
        "cached_tokens": cached_tokens(usage, r.get("server_timings")),
        "n_chunks": len(gen), "token_count_source": "usage" if ct is not None else "chunks",
        "ttft_ms": None, "ttfc_ms": None, "prefill_tok_s": None, "uncached_prefill_tok_s": None,
        "decode_tok_s": None, "tokens_per_chunk": None, "itl_ms_p50": None, "itl_ms_p95": None,
        "itl_ms_p99": None, "total_ms": None, "finish_reason": r.get("finish_reason"),
    }
    if not m["ok"]:
        return m
    m["total_ms"] = r["t_end_us"] / 1000.0 if r.get("t_end_us") is not None else None
    if not gen:
        return m
    ttft_s = gen[0][0] / 1e6
    m["ttft_ms"] = ttft_s * 1000.0
    cont = [e for e in gen if e[1] or e[3] or e[4]]
    if cont:
        m["ttfc_ms"] = cont[0][0] / 1000.0
    if pt is not None and ttft_s > 0:
        m["prefill_tok_s"] = pt / ttft_s
        if m["cached_tokens"] is not None:
            m["uncached_prefill_tok_s"] = max(pt - m["cached_tokens"], 0) / ttft_s
    n = ct if ct is not None else len(gen)
    m["tokens_per_chunk"] = n / len(gen) if gen else None
    if len(gen) >= 2 and n >= 2:
        span_s = (gen[-1][0] - gen[0][0]) / 1e6
        if span_s > 0:
            m["decode_tok_s"] = (n - n / len(gen)) / span_s
        itl = [(b[0] - a[0]) / 1000.0 for a, b in zip(gen, gen[1:])]
        m["itl_ms_p50"] = percentile(itl, 50)
        m["itl_ms_p95"] = percentile(itl, 95)
        m["itl_ms_p99"] = percentile(itl, 99)
    return m


def itl_list(r):
    gen = [e for e in (r.get("events") or []) if e[1] or e[2] or e[3] or e[4]]
    return [(b[0] - a[0]) / 1000.0 for a, b in zip(gen, gen[1:])]


def text_digest(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- environment
def _run(cmd, cwd=None, timeout=20):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-2000:]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"cmd": " ".join(cmd), "rc": None, "error": "%s: %s" % (type(e).__name__, e)}


def git_info():
    if not shutil.which("git"):
        return {"sha": None, "error": "git not found"}
    sha = _run(["git", "rev-parse", "HEAD"], cwd=REPO)
    st = _run(["git", "status", "--porcelain"], cwd=REPO)
    out = {"repo": REPO, "sha": (sha.get("stdout", "").strip() or None) if sha.get("rc") == 0 else None}
    if sha.get("rc") != 0:
        out["error"] = sha.get("stderr") or sha.get("error")
    if st.get("rc") == 0:
        lines = [l for l in st["stdout"].splitlines() if l.strip()]
        out["dirty_files"] = len(lines)
        out["status_porcelain"] = lines[:50]
    return out


def host_info():
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "cpu_count": os.cpu_count()}


def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {"available": False, "note": "nvidia-smi not found on the client host"}
    out = {"available": True, "note": "captured on the CLIENT host; equals the server only if both run on one machine"}
    out["query"] = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,"
                         "pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm",
                         "--format=csv"])
    out["full"] = _run(["nvidia-smi"])
    return out


def sanitize_args(d):
    d = dict(d)
    for k in list(d):
        if "key" in k.lower() and k.lower() != "api_key_env":
            d[k] = "<redacted>" if d[k] else None
    return d


def http_snapshot(res, max_text=200000):
    d = res.to_dict(max_text)
    try:
        d["json"] = json.loads(res.text) if res.text else None
        d.pop("text", None)
    except ValueError:
        pass
    return d


def is_mock(models_snapshot):
    js = (models_snapshot or {}).get("json") or {}
    try:
        return any(m.get("owned_by") == MOCK_OWNER for m in js.get("data", []))
    except AttributeError:
        return False


# ---------------------------------------------------------------------------- run directory
class RunDir:
    def __init__(self, out_dir, scenario, argv, args):
        self.path = os.path.abspath(out_dir)
        os.makedirs(self.path, exist_ok=True)
        if os.path.exists(os.path.join(self.path, "raw.jsonl")):
            raise SystemExit("refusing to overwrite an existing run: %s/raw.jsonl" % self.path)
        self.meta = {
            "schema": SCHEMA, "scenario": scenario, "utc_start": utc_now(), "utc_end": None,
            "argv": list(argv), "args": sanitize_args(args), "status": "running",
        }
        self._raw = open(os.path.join(self.path, "raw.jsonl"), "a", encoding="utf-8")
        self.n_raw = 0

    def collect_env(self, client, engine_desc=None, model_desc=None, smi=True):
        self.meta["server_url"] = client.url
        self.meta["engine_desc"] = engine_desc
        self.meta["model_desc"] = model_desc
        self.meta["models_response"] = http_snapshot(client.models())
        self.meta["health_response"] = http_snapshot(client.health(), 4000)
        self.meta["is_mock_server"] = is_mock(self.meta["models_response"])
        self.meta["git"] = git_info()
        self.meta["host"] = host_info()
        self.meta["nvidia_smi"] = nvidia_smi() if smi else {"available": False, "note": "skipped (--no-nvidia-smi)"}
        self.snapshot_metrics(client, "metrics_before.txt")
        self.write_meta()

    def snapshot_metrics(self, client, name):
        r = client.metrics()
        key = "metrics_" + name.split("_")[1].split(".")[0]
        if r.ok and r.text:
            self.write_text(name, r.text)
            self.meta[key] = {"status": r.status, "file": name}
        else:
            self.meta[key] = {"status": r.status, "error": r.error, "file": None}

    def write_meta(self):
        self.write_json("meta.json", self.meta)

    def raw(self, row):
        self._raw.write(json.dumps(row, sort_keys=True) + "\n")
        self._raw.flush()
        self.n_raw += 1

    def write_json(self, name, obj):
        tmp = os.path.join(self.path, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, os.path.join(self.path, name))

    def write_text(self, name, text):
        with open(os.path.join(self.path, name), "w", encoding="utf-8") as f:
            f.write(text)

    def finish(self, status, client=None):
        """Record the end of the run (call before writing the summary, then close())."""
        if client is not None:
            self.snapshot_metrics(client, "metrics_after.txt")
        self.meta["utc_end"] = utc_now()
        self.meta["status"] = status
        self.meta["raw_rows"] = self.n_raw
        self.write_meta()

    def close(self):
        self.meta["raw_rows"] = self.n_raw
        self.write_meta()
        if not self._raw.closed:
            self._raw.close()


def load_raw(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def md_header(meta, title, extra_lines=()):
    lines = ["# %s" % title, ""]
    if meta.get("is_mock_server"):
        lines += ["**MOCK SERVER RUN: these numbers test the harness only; they are not engine evidence.**", ""]
    if meta.get("fatal"):
        lines += ["**FATAL: %s**" % meta["fatal"], ""]
    git = meta.get("git") or {}
    lines += [
        "- scenario: `%s`, status: `%s`" % (meta.get("scenario"), meta.get("status")),
        "- UTC: %s .. %s" % (meta.get("utc_start"), meta.get("utc_end")),
        "- server: `%s`, model: `%s`" % (meta.get("server_url"), meta.get("model")),
        "- engine: %s" % (meta.get("engine_desc") or "(not given)"),
        "- model desc: %s" % (meta.get("model_desc") or "(not given)"),
        "- harness git SHA: `%s` (dirty files: %s)" % (git.get("sha"), git.get("dirty_files")),
        "- host: `%s`" % ((meta.get("host") or {}).get("hostname")),
    ]
    lines += list(extra_lines)
    lines.append("")
    return lines


def parse_int_list(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_json_arg(s):
    if not s:
        return {}
    if s.startswith("@"):
        with open(s[1:], encoding="utf-8") as f:
            return json.load(f)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return obj
