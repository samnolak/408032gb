#!/usr/bin/env python3
"""Engine-agnostic speed matrix against any OpenAI-compatible server (Python 3 stdlib only).

Matrix: input lengths x output lengths x state {cold, warm} x concurrency, `--repeats` each.
  cold  every request starts with a fresh random prefix (unique nonce in the first line),
        so no prompt cache can help
  warm  one priming request with a long prefix, then the same prefix with a different
        short suffix: the TTFT drop measures prefix-cache reuse
Agent loop: K turns; each appends a synthetic tool call + a 1-4K-token tool result to a
growing conversation (to 100K+ tokens) and records TTFT per turn; a final cold control
re-sends the last conversation under a new session nonce.

Decode length is forced, when the server allows it, with the extra body keys
  vllm      {"ignore_eos": true, "min_tokens": N}
  llamacpp  {"ignore_eos": true, "n_predict": N}
`--eos-mode auto` probes which one makes a short answer run to max_tokens and records it.

Output: a run directory with meta.json, raw.jsonl, summary.json, summary.md (see runlog.py).
Numbers that were not measured are printed as n/a; failed requests are recorded as failures.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402
from oa_client import OAClient, StreamResult  # noqa: E402
from prompts import CounterError, fit_filler, make_counter, refine_ratio  # noqa: E402

DEFAULT_INPUTS = "2048,32768,131072,262144"
DEFAULT_OUTPUTS = "128,512,2048"
EOS_MODES = {
    "vllm": lambda n: {"ignore_eos": True, "min_tokens": n},
    "llamacpp": lambda n: {"ignore_eos": True, "n_predict": n},
    "none": lambda n: {},
}
EOS_PROBE_TOKENS = 24
SUFFIX = "\n\n[Request %06d] Continue the document above with a few more paragraphs in the same style."

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository and return its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite and return the log.",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}},
]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("server")
    g.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8080 (a trailing /v1 is fine)")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var with the API key (optional)")
    g.add_argument("--model", default=None, help="model id (default: first id from /v1/models)")
    g.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    g.add_argument("--extra-body", default="", help="JSON object (or @file.json) merged into every request body, "
                   "e.g. '{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")
    g.add_argument("--stream-options", choices=["auto", "on", "off"], default="auto",
                   help="send stream_options.include_usage (auto: drop it if the server answers 400/422)")
    g.add_argument("--timeout", type=float, default=3600.0,
                   help="idle socket timeout per request, seconds (a 262K prefill can be silent for long)")
    g.add_argument("--total-timeout", type=float, default=None, help="wall limit per request, seconds")
    m = ap.add_argument_group("matrix")
    m.add_argument("--scenarios", default="matrix,agent", help="comma list of matrix,agent")
    m.add_argument("--inputs", default=DEFAULT_INPUTS, help="input lengths in tokens (comma list)")
    m.add_argument("--outputs", default=DEFAULT_OUTPUTS, help="decode lengths in tokens (comma list)")
    m.add_argument("--states", default="cold,warm", help="comma list of cold,warm")
    m.add_argument("--concurrency", default="1,4",
                   help="comma list; set N to the server's parallel slots; values >1 are reported separately")
    m.add_argument("--repeats", type=int, default=3)
    m.add_argument("--length-mode", choices=["prompt", "total"], default="prompt",
                   help="prompt: input length = prompt tokens; total: prompt = input - output (fits a window)")
    m.add_argument("--filler", choices=["prose", "code", "logs"], default="prose")
    m.add_argument("--seed", default="0", help="seed for every generated prompt")
    m.add_argument("--temperature", type=float, default=0.0)
    m.add_argument("--send-seed", type=int, default=None, help="also send this value as the request 'seed'")
    m.add_argument("--eos-mode", choices=["auto", "vllm", "llamacpp", "none"], default="auto")
    m.add_argument("--counter", choices=["auto", "server", "ratio"], default="auto",
                   help="token counting for prompt fitting: server /tokenize, else calibrated ratio")
    m.add_argument("--ratio-refine", type=int, default=1,
                   help="ratio counter only: full-size requests per new length to re-calibrate (each costs a prefill)")
    m.add_argument("--pause-s", type=float, default=0.0, help="sleep between waves")
    a = ap.add_argument_group("agent loop")
    a.add_argument("--agent-turns", type=int, default=48, help="maximum turns")
    a.add_argument("--agent-target-tokens", type=int, default=131072, help="stop once prompt_tokens reaches this")
    a.add_argument("--agent-tool-tokens", default="1024,4096", help="min,max tokens of each tool result")
    a.add_argument("--agent-max-tokens", type=int, default=64)
    a.add_argument("--agent-format", choices=["tool", "user"], default="tool",
                   help="tool: assistant tool_calls + role=tool results; user: plain user messages")
    a.add_argument("--no-agent-cold-control", action="store_true")
    o = ap.add_argument_group("output")
    o.add_argument("--out-dir", default=None, help="run directory (default: ./bench-runs/<UTC>-matrix)")
    o.add_argument("--engine-desc", default=None, help="free text: engine, version/commit, launch command")
    o.add_argument("--model-desc", default=None, help="free text: model, quant, files")
    o.add_argument("--save-text", action="store_true", help="store generated text in raw.jsonl (default: head + sha256)")
    o.add_argument("--no-nvidia-smi", action="store_true")
    return ap


class Runner:
    def __init__(self, args, argv):
        self.a = args
        self.extra_body = runlog.parse_json_arg(args.extra_body)
        self.client = OAClient(args.url, api_key=os.environ.get(args.api_key_env) or None, timeout=args.timeout,
                               total_timeout=args.total_timeout, model=args.model,
                               stream_options=args.stream_options)
        out = args.out_dir or os.path.join("bench-runs", runlog.utc_stamp() + "-matrix")
        self.rd = runlog.RunDir(out, "matrix", argv, vars(args))
        self.seq = 0
        self.lock = threading.Lock()
        self.counter = None
        self.overhead = 0
        self.eos_mode = "none"
        self.eos_info = {}
        self.refined = set()
        self.n_hint = {}
        self.cells = []        # measured rows (dicts) used by the summary
        self.waves = []
        self.agent_rows = []
        self.failures = []

    # ------------------------------------------------------------------ logging
    def log(self, row):
        with self.lock:
            self.seq += 1
            row = dict(row, seq=self.seq, utc=runlog.utc_now())
            self.rd.raw(row)
        return row

    def result_row(self, res, phase, requested_max, **fields):
        d = res.to_dict(keep_text=self.a.save_text)
        if not self.a.save_text:
            d["content_head"] = res.content[:200]
            d["content_sha256"] = runlog.text_digest(res.content)
        m = runlog.request_metrics(d)
        ct = m["completion_tokens"] if m["completion_tokens"] is not None else m["n_chunks"]
        forced = phase in ("measure", "prime")          # only these ask for a forced decode length
        row = dict(fields, phase=phase, requested_max_tokens=requested_max, result=d, metrics=m,
                   short_output=bool(forced and m["ok"] and requested_max and ct < requested_max))
        row = self.log(row)
        if not res.ok and phase in ("prime", "measure", "agent_turn", "agent_cold_control"):
            self.failures.append({"seq": row["seq"], "phase": phase, "error": res.error,
                                  **{k: v for k, v in fields.items() if k in ("input", "output", "state",
                                                                              "concurrency", "repeat", "turn")}})
        return row

    # ------------------------------------------------------------------ requests
    def _send(self, content, max_tokens, eos=True, messages=None, tools=None):
        extra = dict(self.extra_body)
        if eos:
            extra.update(EOS_MODES[self.eos_mode](max_tokens))
        try:
            if self.a.endpoint == "chat" or messages is not None:
                msgs = messages if messages is not None else [{"role": "user", "content": content}]
                return self.client.chat_stream(msgs, max_tokens, temperature=self.a.temperature, extra=extra,
                                               tools=tools, seed=self.a.send_seed)
            return self.client.completion_stream(content, max_tokens, temperature=self.a.temperature,
                                                 extra=extra, seed=self.a.send_seed)
        except Exception as e:  # noqa: BLE001 - a harness bug must not look like a measurement
            r = StreamResult("internal")
            r.error = "harness exception: %s: %s" % (type(e).__name__, e)
            r.notes.append(traceback.format_exc()[-2000:])
            return r

    def wave(self, contents, max_tokens):
        """Send len(contents) requests at once; returns [(StreamResult, t_start_ns, t_end_ns)]."""
        if len(contents) == 1:
            t0 = time.perf_counter_ns()
            r = self._send(contents[0], max_tokens)
            return [(r, t0, time.perf_counter_ns())]
        out = [None] * len(contents)
        barrier = threading.Barrier(len(contents))

        def work(i):
            try:
                barrier.wait(timeout=60)
            except threading.BrokenBarrierError:
                pass
            t0 = time.perf_counter_ns()
            r = self._send(contents[i], max_tokens)
            out[i] = (r, t0, time.perf_counter_ns())

        th = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(contents))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out

    # ------------------------------------------------------------------ setup
    def setup(self):
        a = self.a
        self.rd.collect_env(self.client, a.engine_desc, a.model_desc, smi=not a.no_nvidia_smi)
        model = self.client.resolve_model()
        self.rd.meta["model"] = model
        if not model:
            self.rd.meta["fatal"] = "cannot resolve a model id from /v1/models; pass --model"
            return False
        try:
            self.counter, overhead, info = make_counter(
                self.client, a.counter, kind=a.filler, seed=a.seed,
                endpoint=a.endpoint, extra=self.extra_body,
                log=lambda row: self.log(dict(row, scenario="setup")))
        except CounterError as e:
            self.rd.meta["fatal"] = str(e)
            return False
        self.overhead = overhead or 0
        self.rd.meta["token_counter"] = info
        self.detect_eos()
        self.rd.meta["eos"] = self.eos_info
        self.rd.meta["stream_options_supported"] = self.client.stream_options_supported
        self.rd.write_meta()
        return True

    def detect_eos(self):
        prompt = "Reply with the single word OK."
        n = EOS_PROBE_TOKENS

        def attempt(mode):
            saved, self.eos_mode = self.eos_mode, mode
            r = self._send(prompt, n)
            self.eos_mode = saved
            row = self.result_row(r, "eos_probe", n, scenario="setup", eos_mode=mode)
            m = row["metrics"]
            return {"mode": mode, "ok": r.ok, "http_status": r.http_status, "error": r.error,
                    "completion_tokens": m["completion_tokens"], "finish_reason": r.finish_reason,
                    "reached_max": bool(r.ok and m["completion_tokens"] == n and r.finish_reason == "length")}

        if self.a.eos_mode != "auto":
            self.eos_mode = self.a.eos_mode
            self.eos_info = {"mode": self.eos_mode, "selected_by": "--eos-mode", "verified": None,
                             "attempts": [attempt(self.eos_mode)]}
            return
        base = attempt("none")
        natural_stop = bool(base["ok"] and base["completion_tokens"] is not None and base["completion_tokens"] < n)
        attempts, chosen, verified = [base], None, False
        for mode in ("vllm", "llamacpp"):
            d = attempt(mode)
            attempts.append(d)
            if d["reached_max"] and natural_stop:
                chosen, verified = mode, True
                break
            if d["ok"] and not natural_stop and chosen is None:
                chosen = mode          # accepted, but cannot prove it forced the length
        self.eos_mode = chosen or "none"
        note = None
        if not natural_stop:
            note = "baseline request did not stop early (or failed); forcing could not be verified"
        if chosen is None:
            note = "no key set forced full-length output; rows with fewer tokens are flagged short_output"
        self.eos_info = {"mode": self.eos_mode, "selected_by": "auto", "verified": verified,
                         "probe_max_tokens": n, "note": note, "attempts": attempts}

    # ------------------------------------------------------------------ prompts
    def build(self, input_len, output_len, key, suffix_k):
        target_total = input_len - (output_len if self.a.length_mode == "total" else 0)
        target = max(target_total - self.overhead, 1)
        header = "Document %s.\n" % random.Random("nonce:%s:%r" % (self.a.seed, key)).getrandbits(64)
        fr = fit_filler(self.a.filler, "%s:%r" % (self.a.seed, key), target, self.counter, header=header,
                        suffix=SUFFIX % suffix_k, n_hint=self.n_hint.get(target))
        if not self.counter.exact and self.a.ratio_refine > 0 and target not in self.refined:
            self.refined.add(target)
            for _ in range(self.a.ratio_refine):
                refine_ratio(self.client, self.counter, fr, self.overhead, self.a.endpoint, self.extra_body,
                             log=lambda row: self.log(dict(row, scenario="setup")))
                fr = fit_filler(self.a.filler, "%s:%r" % (self.a.seed, key), target, self.counter,
                                header=header, suffix=SUFFIX % suffix_k)
        self.n_hint[target] = fr["n_units"]
        fit = fr.public()
        fit["target_total"] = target_total
        fit["overhead_assumed"] = self.overhead
        return fr["text"], fit

    # ------------------------------------------------------------------ matrix
    def run_matrix(self):
        a = self.a
        for inp in runlog.parse_int_list(a.inputs):
            for out in runlog.parse_int_list(a.outputs):
                for state in [s.strip() for s in a.states.split(",") if s.strip()]:
                    for conc in runlog.parse_int_list(a.concurrency):
                        self.run_cell(inp, out, state, conc)

    def run_cell(self, inp, out, state, conc):
        a = self.a
        cell = dict(scenario="matrix", input=inp, output=out, state=state, concurrency=conc)
        if state == "warm":
            base, fit = self.build(inp, out, ("warm", inp, out, conc), 0)
            stem = base[: len(base) - len(SUFFIX % 0)]
            r = self._send(base, out)
            self.result_row(r, "prime", out, fit=fit, **cell)
        for rep in range(a.repeats):
            contents, fits = [], []
            for j in range(conc):
                k = 1 + rep * conc + j
                if state == "warm":
                    contents.append(stem + SUFFIX % k)
                    fits.append(dict(fit, note="warm: prime prefix + suffix %d" % k))
                elif state == "cold":
                    c, f = self.build(inp, out, ("cold", inp, out, conc, rep, j), k)
                    contents.append(c)
                    fits.append(f)
                else:
                    raise SystemExit("unknown state %r" % state)
            res = self.wave(contents, out)
            rows = []
            for j, (r, t0, t1) in enumerate(res):
                row = self.result_row(r, "measure", out, fit=fits[j], repeat=rep, slot=j, **cell)
                rows.append(row)
                self.cells.append(row)
            if conc > 1:
                wall_s = (max(t1 for _, _, t1 in res) - min(t0 for _, t0, _ in res)) / 1e9
                ok = [rw for rw in rows if rw["metrics"]["ok"]]
                all_ok = len(ok) == len(rows)
                cts = [rw["metrics"]["completion_tokens"] for rw in ok]
                pts = [rw["metrics"]["prompt_tokens"] for rw in ok]
                agg = None
                if all_ok and wall_s > 0 and all(c is not None for c in cts):
                    agg = sum(cts) / wall_s
                wrow = self.log(dict(cell, phase="wave", repeat=rep, wall_ms=wall_s * 1000.0,
                                     n_requests=len(rows), n_ok=len(ok), all_ok=all_ok,
                                     sum_completion_tokens=sum(c for c in cts if c is not None),
                                     sum_prompt_tokens=sum(p for p in pts if p is not None),
                                     aggregate_output_tok_s=agg,
                                     note=None if agg is not None else "not computed: a request failed or usage missing"))
                self.waves.append(wrow)
            if a.pause_s:
                time.sleep(a.pause_s)

    # ------------------------------------------------------------------ agent loop
    def run_agent(self):
        a = self.a
        lo, hi = runlog.parse_int_list(a.agent_tool_tokens)
        rng = random.Random("agent:%s" % a.seed)
        nonce = rng.getrandbits(64)

        def system(n):
            return ("Session %d. You are a coding agent working in a repository. Use the tools when needed and "
                    "keep each answer short." % n)

        messages = [{"role": "system", "content": system(nonce)},
                    {"role": "user", "content": "The integration tests fail. Find the cause; start by reading "
                                                "the relevant source files and logs."}]
        tools = AGENT_TOOLS if a.agent_format == "tool" else None
        prev_pt, last_ok = None, None
        for turn in range(1, a.agent_turns + 1):
            r = self._send(None, a.agent_max_tokens, eos=False, messages=messages, tools=tools)
            row = self.result_row(r, "agent_turn", a.agent_max_tokens, scenario="agent", turn=turn,
                                  n_messages=len(messages))
            m = row["metrics"]
            row_s = {"turn": turn, "ok": m["ok"], "prompt_tokens": m["prompt_tokens"], "ttft_ms": m["ttft_ms"],
                     "cached_tokens": m["cached_tokens"],
                     "new_tokens": (m["prompt_tokens"] - prev_pt) if (m["prompt_tokens"] is not None
                                                                       and prev_pt is not None) else None}
            row_s["new_tokens_per_s"] = (row_s["new_tokens"] / (m["ttft_ms"] / 1000.0)
                                         if row_s["new_tokens"] is not None and m["ttft_ms"] else None)
            self.agent_rows.append(row_s)
            if not r.ok:
                break
            last_ok = copy.deepcopy(messages)
            prev_pt = m["prompt_tokens"]
            if prev_pt is not None and prev_pt >= a.agent_target_tokens:
                break
            if turn == a.agent_turns:
                break
            size = rng.randint(lo, hi)
            kind = "code" if turn % 2 else "logs"
            tool_text = fit_filler(kind, "agent:%s:%d" % (a.seed, turn), size, self.counter)["text"]
            call_id = "call_%04d" % turn
            path = "src/module_%03d.py" % turn if kind == "code" else "logs/run_%03d.jsonl" % turn
            if a.agent_format == "tool":
                messages.append({"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_text})
            else:
                messages.append({"role": "assistant", "content": "I will read %s." % path})
                messages.append({"role": "user", "content": "Tool output for %s:\n%s" % (path, tool_text)})
        if last_ok is not None and not a.no_agent_cold_control:
            ctrl = copy.deepcopy(last_ok)
            ctrl[0]["content"] = system(rng.getrandbits(64))
            r = self._send(None, a.agent_max_tokens, eos=False, messages=ctrl, tools=tools)
            row = self.result_row(r, "agent_cold_control", a.agent_max_tokens, scenario="agent",
                                  n_messages=len(ctrl))
            self.agent_control = row["metrics"]
        else:
            self.agent_control = None

    # ------------------------------------------------------------------ summary
    def summarize(self):
        cells = {}
        for row in self.cells:
            k = (row["input"], row["output"], row["state"], row["concurrency"])
            cells.setdefault(k, []).append(row)
        waves = {}
        for w in self.waves:
            waves.setdefault((w["input"], w["output"], w["state"], w["concurrency"]), []).append(w)
        out_cells = []
        for k in sorted(cells, key=lambda k: (k[3], k[0], k[1], k[2])):
            rows = cells[k]
            ok = [r for r in rows if r["metrics"]["ok"]]
            c = {"input": k[0], "output": k[1], "state": k[2], "concurrency": k[3], "n": len(rows),
                 "n_ok": len(ok), "n_failed": len(rows) - len(ok),
                 "n_short_output": sum(1 for r in ok if r["short_output"]), "metrics": {}}
            for key in ("ttft_ms", "ttfc_ms", "prefill_tok_s", "uncached_prefill_tok_s", "decode_tok_s",
                        "total_ms", "prompt_tokens", "completion_tokens", "cached_tokens", "tokens_per_chunk"):
                c["metrics"][key] = runlog.describe([r["metrics"][key] for r in ok])
            itl = []
            for r in ok:
                itl.extend(runlog.itl_list(r["result"]))
            c["metrics"]["itl_ms_pooled"] = runlog.describe(itl, pcts=(50, 95, 99))
            if k[3] > 1:
                ws = waves.get(k, [])
                c["waves"] = {"n": len(ws), "n_all_ok": sum(1 for w in ws if w["all_ok"]),
                              "aggregate_output_tok_s": runlog.describe([w["aggregate_output_tok_s"] for w in ws]),
                              "wall_ms": runlog.describe([w["wall_ms"] for w in ws if w["all_ok"]])}
            out_cells.append(c)
        summary = {"schema": runlog.SCHEMA, "definitions": runlog.DEFINITIONS, "cells": out_cells,
                   "agent_loop": {"turns": self.agent_rows, "cold_control": getattr(self, "agent_control", None)},
                   "eos": self.eos_info, "token_counter": self.rd.meta.get("token_counter"),
                   "stream_options_supported": self.client.stream_options_supported,
                   "failures": self.failures, "is_mock_server": self.rd.meta.get("is_mock_server")}
        self.rd.write_json("summary.json", summary)
        self.rd.write_text("summary.md", self.render_md(summary))
        return summary

    def render_md(self, s):
        meta = self.rd.meta
        tc = s.get("token_counter") or {}
        eos = s.get("eos") or {}
        lines = runlog.md_header(meta, "Bench matrix run", [
            "- token counting: `%s`, chat template overhead (measured): %s tokens" % (
                tc.get("method"), runlog.fmt(tc.get("overhead"))),
            "- forced decode length: eos mode `%s` (verified: %s)" % (eos.get("mode"), eos.get("verified")),
            "- stream_options accepted: %s; filler `%s`; length mode `%s`; temperature %s" % (
                s.get("stream_options_supported"), self.a.filler, self.a.length_mode, self.a.temperature),
        ])
        f = runlog.fstat
        single = [c for c in s["cells"] if c["concurrency"] == 1]
        lines += ["## Single stream (concurrency 1)", ""]
        if single:
            rows = []
            for c in single:
                m = c["metrics"]
                rows.append([c["input"], c["output"], c["state"], "%d/%d" % (c["n_ok"], c["n"]),
                             f(m["prompt_tokens"], "p50", 0), f(m["ttft_ms"], "p50"), f(m["ttft_ms"], "p95"),
                             f(m["prefill_tok_s"], "p50"), f(m["decode_tok_s"], "p50"), f(m["decode_tok_s"], "p95"),
                             f(m["decode_tok_s"], "min"), f(m["decode_tok_s"], "max"),
                             "%s / %s / %s" % (f(m["itl_ms_pooled"], "p50", 2), f(m["itl_ms_pooled"], "p95", 2),
                                               f(m["itl_ms_pooled"], "p99", 2)),
                             c["n_short_output"]])
            lines.append(runlog.md_table(
                ["input", "output", "state", "ok/n", "prompt tok p50", "TTFT ms p50", "TTFT ms p95",
                 "prefill tok/s p50", "decode tok/s p50", "decode p95", "decode min", "decode max",
                 "ITL ms p50/p95/p99", "short"], rows))
        else:
            lines.append("(no concurrency-1 cells)")
        lines.append("")
        pairs = {}
        for c in single:
            pairs.setdefault((c["input"], c["output"]), {})[c["state"]] = c
        cmp_rows = []
        for (i, o), d in sorted(pairs.items()):
            if "cold" in d and "warm" in d:
                tc_, tw = d["cold"]["metrics"]["ttft_ms"].get("p50"), d["warm"]["metrics"]["ttft_ms"].get("p50")
                cmp_rows.append([i, o, runlog.fmt(tc_), runlog.fmt(tw),
                                 runlog.fmt(tw / tc_ if tc_ and tw is not None else None, 3),
                                 f(d["warm"]["metrics"]["cached_tokens"], "p50", 0)])
        if cmp_rows:
            lines += ["## Prefix cache: warm vs cold TTFT (concurrency 1)", "",
                      runlog.md_table(["input", "output", "cold TTFT ms p50", "warm TTFT ms p50",
                                       "warm/cold (derived)", "warm cached tok p50 (server-reported)"], cmp_rows), ""]
        multi = [c for c in s["cells"] if c["concurrency"] > 1]
        if multi:
            rows = []
            for c in multi:
                m, w = c["metrics"], c.get("waves", {})
                ag = w.get("aggregate_output_tok_s", {})
                rows.append([c["input"], c["output"], c["state"], c["concurrency"], "%d/%d" % (c["n_ok"], c["n"]),
                             "%d/%d" % (w.get("n_all_ok", 0), w.get("n", 0)), f(ag, "p50"), f(ag, "min"),
                             f(ag, "max"), f(m["decode_tok_s"], "p50"), f(m["ttft_ms"], "p50"),
                             f(m["ttft_ms"], "p95")])
            lines += ["## Concurrency > 1 (aggregate throughput; NOT comparable with single-stream numbers)", "",
                      runlog.md_table(["input", "output", "state", "conc", "req ok/n", "waves ok/n",
                                       "aggregate out tok/s p50", "agg min", "agg max",
                                       "per-request decode tok/s p50", "TTFT ms p50", "TTFT ms p95"], rows), ""]
        ag = s["agent_loop"]
        if ag["turns"]:
            rows = [[t["turn"], "ok" if t["ok"] else "FAIL", runlog.fmt(t["prompt_tokens"]),
                     runlog.fmt(t["new_tokens"]), runlog.fmt(t["cached_tokens"]), runlog.fmt(t["ttft_ms"]),
                     runlog.fmt(t["new_tokens_per_s"])] for t in ag["turns"]]
            lines += ["## Agent loop (TTFT per turn)", "",
                      runlog.md_table(["turn", "status", "prompt tok", "new tok", "cached tok (server)",
                                       "TTFT ms", "new tok / TTFT s (derived)"], rows), ""]
            cc = ag.get("cold_control")
            last = next((t for t in reversed(ag["turns"]) if t["ok"]), None)
            if cc is not None and last is not None:
                lines.append("Cold control (same final conversation, new session nonce): prompt %s tok, TTFT %s ms "
                             "(last warm turn: %s ms; ratio warm/cold derived = %s)." % (
                                 runlog.fmt(cc.get("prompt_tokens")), runlog.fmt(cc.get("ttft_ms")),
                                 runlog.fmt(last["ttft_ms"]),
                                 runlog.fmt(last["ttft_ms"] / cc["ttft_ms"] if cc.get("ttft_ms") and last["ttft_ms"]
                                            else None, 3)))
                lines.append("")
        lines += ["## Failures", ""]
        if s["failures"]:
            lines.append(runlog.md_table(["seq", "phase", "cell / turn", "error"],
                                         [[x["seq"], x["phase"],
                                           ", ".join("%s=%s" % (k, x[k]) for k in ("input", "output", "state",
                                                                                   "concurrency", "repeat", "turn")
                                                     if k in x),
                                           (x["error"] or "")[:200].replace("|", "/").replace("\n", " ")]
                                          for x in s["failures"]]))
        else:
            lines.append("none")
        lines += ["", "## Definitions", ""] + ["- `%s`: %s" % kv for kv in sorted(s["definitions"].items())]
        return "\n".join(lines) + "\n"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    run = Runner(args, argv)
    status, code = "ok", 0
    try:
        if not run.setup():
            status, code = "fatal", 2
        else:
            scen = [s.strip() for s in args.scenarios.split(",") if s.strip()]
            if "matrix" in scen:
                run.run_matrix()
            if "agent" in scen:
                run.run_agent()
            measured = [r for r in run.cells if r["metrics"]["ok"]] + [t for t in run.agent_rows if t["ok"]]
            if run.failures or not measured:
                status, code = "completed_with_failures", 1
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    finally:
        run.rd.finish(status, run.client)
        run.summarize()
        run.rd.close()
    print("run dir: %s (status %s, %d raw rows, %d failures)" % (run.rd.path, status, run.rd.n_raw,
                                                                  len(run.failures)))
    return code


if __name__ == "__main__":
    sys.exit(main())
