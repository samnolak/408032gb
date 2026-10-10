#!/usr/bin/env python3
"""Agentic evaluation stand: run a task set against any OpenAI-compatible endpoint and grade it.

    python3 tools/eval/run_eval.py --url http://127.0.0.1:8080 --tasks DIR --out RUN_DIR \
        [--repeats N] [--temperature 0] [--max-tokens 4096] [--engine-desc ...] [--model-desc ...]
    python3 tools/eval/run_eval.py --tasks DIR --validate-only      # no server: check the task set itself

Categories: code_fix, code_write (answer applied to a copy of the fixture, tests run in a sandbox),
tool_use (multi-turn loop through a deterministic tool simulator), long_doc_qa, reasoning (answer
matching), structured_output (JSON Schema subset + value checks). Requests go through
tools/bench/oa_client.py; the run directory reuses tools/bench/runlog.py (meta.json with the mock-server
flag, git SHA, host; raw.jsonl). Files written:
  meta.json      what ran against what (engine/model descriptions, /v1/models, is_mock_server, git SHA,
                 task ids + sha256, sandbox capabilities, parameters)
  raw.jsonl      every request and response (with stream timings and usage), tool execution, sandbox run
  results.jsonl  one row per task per repeat: status pass | fail | request_error | harness_error
  summary.json   per-category pass rates with Wilson 95% CIs (attempt level and task level);
                 request and harness errors are counted separately and never enter a pass rate
  summary.md     the same for people
Exit code: 0 = completed without errors, 1 = completed but some attempts are request/harness errors,
2 = fatal (invalid task set, no model id), 130 = interrupted. The pass rate never changes the exit code.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench"))
sys.path.insert(0, HERE)
import runlog  # noqa: E402
from oa_client import OAClient  # noqa: E402

import ev_answers  # noqa: E402
import ev_sandbox  # noqa: E402
import ev_scoring  # noqa: E402
import ev_stats  # noqa: E402
import ev_tasks  # noqa: E402
import ev_toolsim  # noqa: E402

EVAL_SCHEMA = "408032gb-eval/1"
DEFINITIONS = {
    "status": "pass / fail = graded model outcome; request_error = the request failed (transport, HTTP status, "
              "truncated stream, malformed response) - not a model failure; harness_error = the stand could not "
              "grade (e.g. sandbox unavailable)",
    "attempt_pass_rate": "passes / valid attempts (task x repeat with status pass or fail); the Wilson interval "
                         "treats repeats as independent, so with repeats > 1 it is too narrow",
    "task_pass_rate": "per task one outcome from its valid repeats by the aggregate rule (majority: more than half "
                      "pass, a tie is a fail); passes / tasks with at least one valid repeat; Wilson 95% interval",
    "ci95": "Wilson score interval, z = 1.959964",
    "mean_score": "mean partial score over valid attempts (fraction of sub-checks passed); informational",
}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="OpenAI-compatible server, e.g. http://127.0.0.1:8080 (or .../v1)")
    ap.add_argument("--tasks", required=True, help="task directory (files *.task.json, searched recursively)")
    ap.add_argument("--out", help="run directory to create (refuses to overwrite an existing run)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096, help="default max_tokens (a task may override)")
    ap.add_argument("--seed", type=int, default=None, help="also send this value as the request 'seed'")
    ap.add_argument("--model", default=None, help="model id (default: first id of /v1/models)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--extra-body", default="", help="JSON object (or @file) merged into every request body")
    ap.add_argument("--stream", choices=["on", "off"], default="on",
                    help="on: SSE streaming with per-chunk timings (default); off: plain JSON responses")
    ap.add_argument("--stream-options", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--timeout", type=float, default=600.0, help="idle socket timeout per request, seconds")
    ap.add_argument("--total-timeout", type=float, default=None, help="whole-request limit, seconds")
    ap.add_argument("--only", default="", help="comma list of task ids, id globs or category names")
    ap.add_argument("--aggregate", choices=ev_stats.AGGREGATES, default="majority",
                    help="per-task outcome from repeats for the task-level pass rate")
    ap.add_argument("--weak-sandbox", action="store_true",
                    help="run code tests even without full isolation (recorded); default refuses")
    ap.add_argument("--work-dir", default=None, help="parent of the per-attempt temp directories")
    ap.add_argument("--keep-workdirs", action="store_true", help="keep code-task working copies (debugging)")
    ap.add_argument("--raw-prompt-chars", type=int, default=20000,
                    help="message contents longer than this are stored in raw.jsonl as head/tail + sha256 "
                         "(0 = store everything)")
    ap.add_argument("--engine-desc", default=None)
    ap.add_argument("--model-desc", default=None)
    ap.add_argument("--no-nvidia-smi", action="store_true")
    ap.add_argument("--validate-only", action="store_true",
                    help="no server: validate tasks, run reference answers and unchanged fixtures")
    ap.add_argument("--skip-invalid", action="store_true", help="run the valid tasks even if some are invalid")
    return ap


def select(tasks, only):
    if not only:
        return tasks
    pats = [p.strip() for p in only.split(",") if p.strip()]
    return [t for t in tasks if any(t.category == p or fnmatch.fnmatchcase(t.id, p) for p in pats)]


def compact_messages(messages, limit):
    if not limit:
        return messages
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) and len(c) > limit:
            m = dict(m)
            m["content"] = None
            m["content_compacted"] = {"chars": len(c), "sha256": runlog.text_digest(c),
                                      "head": c[:limit // 2], "tail": c[-(limit // 4):]}
        out.append(m)
    return out


def normalize_stream(res):
    calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name") or "",
              "arguments": (tc.get("function") or {}).get("arguments") or ""} for tc in res.tool_calls]
    return {"ok": bool(res.ok), "error": res.error, "http_status": res.http_status, "content": res.content or "",
            "reasoning": res.reasoning or "", "tool_calls": calls, "finish_reason": res.finish_reason,
            "usage": res.usage}


def normalize_json(r):
    """HTTPResult of a non-streaming chat completion -> normalized response."""
    out = {"ok": False, "error": None, "http_status": r.status, "content": "", "reasoning": "", "tool_calls": [],
           "finish_reason": None, "usage": None, "elapsed_ms": r.elapsed_ms}
    if r.error:
        out["error"] = r.error
        return out
    if r.status != 200:
        out["error"] = "HTTP %s: %s" % (r.status, (r.text or "")[:4000])
        return out
    obj = r.json()
    try:
        ch = obj["choices"][0]
        msg = ch["message"]
    except (TypeError, KeyError, IndexError):
        out["error"] = "malformed response (no choices[0].message): %s" % (r.text or "")[:500]
        return out
    out["content"] = msg.get("content") or ""
    out["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning") or ""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        out["tool_calls"].append({"id": tc.get("id"), "name": fn.get("name") or "",
                                  "arguments": args if isinstance(args, str) else json.dumps(args)})
    out["finish_reason"] = ch.get("finish_reason")
    out["usage"] = obj.get("usage")
    if out["finish_reason"] is None:
        out["error"] = "malformed response: finish_reason missing"
        return out
    out["ok"] = True
    return out


class EvalRunner:
    def __init__(self, args, argv, tasks):
        self.a = args
        self.tasks = tasks
        self.extra_body = runlog.parse_json_arg(args.extra_body)
        self.client = OAClient(args.url, api_key=os.environ.get(args.api_key_env) or None, timeout=args.timeout,
                               total_timeout=args.total_timeout, model=args.model, stream_options=args.stream_options)
        self.rd = runlog.RunDir(args.out, "eval", argv, vars(args))
        self.results_f = open(os.path.join(self.rd.path, "results.jsonl"), "a", encoding="utf-8")
        self.results = []
        self.seq = 0
        self.sandbox = None
        self.work_root = None

    # ------------------------------------------------------------------ logging
    def log(self, row):
        self.seq += 1
        row = dict(row, seq=self.seq, utc=runlog.utc_now())
        self.rd.raw(row)
        return row["seq"]

    # ------------------------------------------------------------------ requests
    def ask(self, task, rep, turn, messages):
        max_tokens = task.max_tokens or self.a.max_tokens
        kw = {"tools": task.tools, "tool_choice": task.tool_choice, "response_format": task.response_format,
              "seed": self.a.seed}
        t0 = time.perf_counter()
        stream_d = metrics = None
        try:
            if self.a.stream == "on":
                res = self.client.chat_stream(messages, max_tokens, temperature=self.a.temperature,
                                              extra=self.extra_body, **kw)
                resp = normalize_stream(res)
                stream_d = res.to_dict(keep_text=False)
                metrics = runlog.request_metrics(stream_d)
            else:
                r = self.client.chat(messages, max_tokens, temperature=self.a.temperature, extra=self.extra_body, **kw)
                resp = normalize_json(r)
        except Exception as e:  # noqa: BLE001 - a harness bug must not look like a model failure
            resp = {"ok": False, "error": "harness exception: %s: %s" % (type(e).__name__, e), "content": "",
                    "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None, "harness": True,
                    "traceback": traceback.format_exc()[-2000:]}
        resp["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        req = {"messages": compact_messages(messages, self.a.raw_prompt_chars), "n_messages": len(messages),
               "tools": [t["function"]["name"] for t in task.tools] if task.tools else None,
               "max_tokens": max_tokens, "temperature": self.a.temperature, "seed": self.a.seed,
               "response_format": task.response_format, "tool_choice": task.tool_choice, "stream": self.a.stream}
        seq = self.log({"kind": "request", "task_id": task.id, "category": task.category, "repeat": rep,
                        "turn": turn, "request": req, "response": resp, "stream": stream_d, "metrics": metrics})
        return resp, seq

    # ------------------------------------------------------------------ one task attempt
    def run_task(self, task, rep):
        t0 = time.perf_counter()
        seqs, usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "reported": 0, "requests": 0}
        row = {"task_id": task.id, "category": task.category, "repeat": rep, "task_sha256": task.sha256}

        def account(resp, seq):
            seqs.append(seq)
            usage["requests"] += 1
            u = resp.get("usage") if isinstance(resp.get("usage"), dict) else None
            if u:
                usage["reported"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    if isinstance(u.get(k), int):
                        usage[k] += u[k]

        def log_sub(r):
            seqs.append(self.log(dict(r, task_id=task.id, category=task.category, repeat=rep)))

        try:
            if task.category == "tool_use":
                g, last = self._tool_loop(task, rep, account, log_sub)
            else:
                resp, seq = self.ask(task, rep, 0, task.messages)
                account(resp, seq)
                last = resp
                if not resp["ok"]:
                    g = {"status": "harness_error" if resp.get("harness") else "request_error", "score": None,
                         "reason": resp["error"], "details": {}}
                else:
                    g = ev_scoring.grade(task, resp["content"], self.sandbox, log_sub)
        except Exception as e:  # noqa: BLE001
            g = {"status": "harness_error", "score": None, "reason": "harness exception: %s: %s" % (type(e).__name__, e),
                 "details": {"traceback": traceback.format_exc()[-2000:]}}
            last = {}
        fr = (last or {}).get("finish_reason")
        row.update({"status": g["status"], "pass": {"pass": True, "fail": False}.get(g["status"]),
                    "score": g["score"], "reason": g["reason"], "details": g["details"],
                    "n_requests": usage["requests"], "raw_seqs": seqs, "finish_reason": fr,
                    "truncated": fr == "length",
                    "usage": {k: (usage[k] if usage["reported"] else None) for k in ("prompt_tokens", "completion_tokens")},
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0, "utc": runlog.utc_now()})
        if row["truncated"] and row["status"] == "fail":
            row["reason"] += " (finish_reason=length: the answer hit max_tokens)"
        self.results_f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        self.results_f.flush()
        self.log({"kind": "result", **{k: v for k, v in row.items() if k != "details"}})
        self.results.append(row)
        return row

    def _tool_loop(self, task, rep, account, log_sub):
        sim = ev_toolsim.ToolSimulator(task.sim_spec, task.tools)
        msgs = [dict(m) for m in task.messages]
        calls = []
        last = None
        for turn in range(task.max_turns):
            resp, seq = self.ask(task, rep, turn, msgs)
            account(resp, seq)
            last = resp
            if not resp["ok"]:
                return {"status": "harness_error" if resp.get("harness") else "request_error", "score": None,
                        "reason": "turn %d: %s" % (turn, resp["error"]), "details": {"calls": calls}}, last
            if not resp["tool_calls"]:
                g = ev_scoring.grade_tool_use(task, calls, resp["content"], False)
                g["details"]["calls"] = calls
                return g, last
            tcs = []
            for i, tc in enumerate(resp["tool_calls"]):
                cid = tc.get("id") or "call_%d_%d" % (turn, i)
                tcs.append({"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
            msgs.append({"role": "assistant", "content": resp["content"] or "", "tool_calls": tcs})
            for tc in tcs:
                result, rec = sim.call(tc["function"]["name"], tc["function"]["arguments"])
                rec["turn"] = turn
                calls.append(rec)
                log_sub({"kind": "tool_exec", "turn": turn, "call_id": tc["id"], "name": rec["name"],
                         "arguments": tc["function"]["arguments"], "args_valid": rec["args_valid"],
                         "source": rec["source"], "result": result[:20000]})
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        g = ev_scoring.grade_tool_use(task, calls, None, True)
        g["details"]["calls"] = calls
        return g, last

    # ------------------------------------------------------------------ main flow
    def setup(self):
        self.rd.collect_env(self.client, self.a.engine_desc, self.a.model_desc, smi=not self.a.no_nvidia_smi)
        model = self.client.resolve_model()
        self.rd.meta["model"] = model
        code = any(t.category in ev_tasks.CODE_CATEGORIES for t in self.tasks)
        if code:
            self.work_root = tempfile.mkdtemp(prefix="eval-work-", dir=self.a.work_dir)
            self.sandbox = ev_scoring.CodeSandbox(self.work_root, weak=self.a.weak_sandbox, keep=self.a.keep_workdirs)
        self.rd.meta["eval"] = {
            "schema": EVAL_SCHEMA, "tasks_dir": os.path.abspath(self.a.tasks), "n_tasks": len(self.tasks),
            "task_set_sha256": ev_tasks.task_set_sha256(self.tasks), "tasks": [t.meta() for t in self.tasks],
            "synthetic_task_set": bool(self.tasks) and all(t.meta()["synthetic"] for t in self.tasks),
            "repeats": self.a.repeats, "temperature": self.a.temperature, "max_tokens_default": self.a.max_tokens,
            "seed": self.a.seed, "stream": self.a.stream, "aggregate": self.a.aggregate,
            "extra_body": self.extra_body, "weak_sandbox": self.a.weak_sandbox,
            "sandbox": self.sandbox.caps if self.sandbox else None,
            "sandbox_missing": self.sandbox.missing() if self.sandbox else None}
        if not model:
            self.rd.meta["fatal"] = "cannot resolve a model id from /v1/models; pass --model"
            self.rd.write_meta()
            return False
        self.rd.write_meta()
        return True

    def run(self):
        for rep in range(self.a.repeats):
            for t in self.tasks:
                row = self.run_task(t, rep)
                print("[%d/%d] %-40s %-14s %s" % (rep + 1, self.a.repeats, t.id, row["status"], row["reason"][:100]),
                      flush=True)
        self.rd.meta["stream_options_supported"] = self.client.stream_options_supported

    def close(self, status):
        self.rd.finish(status, self.client)
        summary = summarize(self.results, self.rd.meta, self.a.aggregate)
        self.rd.write_json("summary.json", summary)
        self.rd.write_text("summary.md", summary_md(summary, self.rd.meta))
        self.rd.close()
        self.results_f.close()
        if self.work_root and not self.a.keep_workdirs:
            shutil.rmtree(self.work_root, ignore_errors=True)
        return summary


# ---------------------------------------------------------------------------- summary
def _block(rows, aggregate):
    valid = [r for r in rows if r["status"] in ("pass", "fail")]
    k = sum(1 for r in valid if r["status"] == "pass")
    lo, hi = ev_stats.wilson(k, len(valid))
    outs = ev_stats.task_outcomes(rows, aggregate)
    scored = {tid: o for tid, o in outs.items() if o["pass"] is not None}
    tk = sum(1 for o in scored.values() if o["pass"])
    tlo, thi = ev_stats.wilson(tk, len(scored))
    scores = [r["score"] for r in valid if r["score"] is not None]
    return {
        "n_tasks": len(outs), "n_attempts": len(rows), "n_valid_attempts": len(valid), "attempt_passes": k,
        "attempt_pass_rate": k / len(valid) if valid else None, "attempt_ci95": [lo, hi],
        "n_tasks_scored": len(scored), "task_passes": tk, "task_pass_rate": tk / len(scored) if scored else None,
        "task_ci95": [tlo, thi],
        "n_request_errors": sum(1 for r in rows if r["status"] == "request_error"),
        "n_harness_errors": sum(1 for r in rows if r["status"] == "harness_error"),
        "n_truncated": sum(1 for r in rows if r.get("truncated")),
        "mean_score": sum(scores) / len(scores) if scores else None,
        "tasks_failed": sorted(tid for tid, o in scored.items() if not o["pass"]),
        "tasks_without_valid_attempt": sorted(tid for tid, o in outs.items() if o["pass"] is None),
    }


def summarize(rows, meta, aggregate="majority"):
    ev = meta.get("eval") or {}
    cats = [c for c in ev_tasks.CATEGORIES if any(r["category"] == c for r in rows)]
    out = {"schema": EVAL_SCHEMA, "aggregate": aggregate, "definitions": DEFINITIONS,
           "is_mock_server": meta.get("is_mock_server"), "synthetic_task_set": ev.get("synthetic_task_set"),
           "task_set_sha256": ev.get("task_set_sha256"), "repeats": ev.get("repeats"),
           "categories": {c: _block([r for r in rows if r["category"] == c], aggregate) for c in cats},
           "overall": _block(rows, aggregate)}
    out["n_request_errors"] = out["overall"]["n_request_errors"]
    out["n_harness_errors"] = out["overall"]["n_harness_errors"]
    return out


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


def _ci(ci):
    return "n/a" if ci[0] is None else "[%.1f, %.1f]" % (100.0 * ci[0], 100.0 * ci[1])


def summary_md(s, meta):
    extra = []
    if s.get("synthetic_task_set"):
        extra.append("- **SYNTHETIC TASK SET: tests the stand only; not evidence of model quality.**")
    ev = meta.get("eval") or {}
    extra.append("- tasks: %s (%s tasks, set sha256 `%s`), repeats %s, temperature %s, default max_tokens %s, "
                 "stream %s, aggregate `%s`" % (ev.get("tasks_dir"), ev.get("n_tasks"), (ev.get("task_set_sha256") or "")[:16],
                                               ev.get("repeats"), ev.get("temperature"), ev.get("max_tokens_default"),
                                               ev.get("stream"), s.get("aggregate")))
    if ev.get("sandbox") is not None:
        extra.append("- code sandbox: netns `%s`, Landlock ABI %s, weak mode %s, missing: %s" % (
            " ".join(ev["sandbox"].get("netns_prefix") or []) or "none", ev["sandbox"].get("landlock_abi"),
            ev.get("weak_sandbox"), ev.get("sandbox_missing") or "nothing"))
    lines = runlog.md_header(meta, "Agentic evaluation run", extra)
    rows = []
    for name, b in list(s["categories"].items()) + [("**overall**", s["overall"])]:
        rows.append([name, b["n_tasks"], b["n_attempts"], b["n_valid_attempts"], _pct(b["task_pass_rate"]),
                     _ci(b["task_ci95"]), _pct(b["attempt_pass_rate"]), _ci(b["attempt_ci95"]),
                     b["n_request_errors"], b["n_harness_errors"], b["n_truncated"]])
    lines.append(runlog.md_table(["category", "tasks", "attempts", "valid", "task pass", "task 95% CI",
                                  "attempt pass", "attempt 95% CI", "request errors", "harness errors",
                                  "truncated"], rows))
    lines += ["", "Request errors and harness errors are excluded from every pass rate (they are not model failures).",
              "Task-level outcome: aggregate `%s` over valid repeats. Wilson 95%% intervals." % s.get("aggregate"), ""]
    o = s["overall"]
    if o["tasks_failed"]:
        lines.append("Failed tasks: " + ", ".join("`%s`" % t for t in o["tasks_failed"]))
    if o["tasks_without_valid_attempt"]:
        lines.append("Tasks without a valid attempt (errors only): " +
                     ", ".join("`%s`" % t for t in o["tasks_without_valid_attempt"]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- validate-only
def validate_tasks(tasks, args):
    """Check the task set without a model. Returns (ok, report rows)."""
    sandbox = None
    work = None
    report = []
    if any(t.category in ev_tasks.CODE_CATEGORIES for t in tasks):
        work = tempfile.mkdtemp(prefix="eval-validate-", dir=args.work_dir)
        sandbox = ev_scoring.CodeSandbox(work, weak=args.weak_sandbox)
    try:
        for t in tasks:
            probs, notes = [], []
            if t.category in ev_tasks.CODE_CATEGORIES:
                base = sandbox.run_fixture(t, None, apply_answer=False)
                if base["status"] == "harness_error":
                    probs.append("unchanged fixture: %s" % base["reason"])
                elif t.check.get("expect_fixture_fails", True) and base["status"] == "pass":
                    probs.append("tests already pass on the unchanged fixture (task is trivial)")
                else:
                    notes.append("unchanged fixture: %s" % base["reason"])
            if t.category == "tool_use":
                sim = ev_toolsim.ToolSimulator(t.sim_spec, t.tools)
                for i, c in enumerate(t.check.get("required_calls") or []):
                    if "arguments" in c:
                        _, rec = sim.call(c["name"], json.dumps(c["arguments"]))
                        if rec["source"] == "default":
                            probs.append("required_calls[%d]: the simulator has no result for these arguments" % i)
                if t.reference_answer is not None and "final_answer" in t.check:
                    g = ev_answers.grade_text(t.reference_answer, t.check["final_answer"])
                    (notes if g["ok"] else probs).append("reference final answer: %s" % g["detail"])
            elif t.reference_answer is not None:
                g = ev_scoring.grade(t, t.reference_answer, sandbox)
                (notes if g["status"] == "pass" else probs).append("reference answer: %s %s" % (g["status"], g["reason"]))
            else:
                notes.append("no reference_answer (only static checks)")
            report.append({"id": t.id, "category": t.category, "ok": not probs, "problems": probs, "notes": notes})
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
    return all(r["ok"] for r in report), report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    tasks, errors = ev_tasks.load_dir(a.tasks)
    if errors:
        print("task set problems (%d):" % len(errors), file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        if not a.skip_invalid or not tasks:
            return 2
    tasks = select(tasks, a.only)
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    if a.validate_only:
        ok, report = validate_tasks(tasks, a)
        for r in report:
            print("%-8s %-40s %-18s %s" % ("OK" if r["ok"] else "PROBLEM", r["id"], r["category"],
                                           "; ".join(r["problems"] or r["notes"])[:300]))
        print("validated %d tasks: %s" % (len(report), "all OK" if ok else "PROBLEMS FOUND"))
        return 0 if ok else 1
    if not a.url or not a.out:
        print("--url and --out are required (or use --validate-only)", file=sys.stderr)
        return 2
    if a.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    run = EvalRunner(a, argv, tasks)
    status, code = "ok", 0
    try:
        if not run.setup():
            status, code = "fatal", 2
        else:
            run.run()
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    finally:
        s = run.close(status)
    if code == 0 and (s["n_request_errors"] or s["n_harness_errors"]):
        code = 1
    o = s["overall"]
    print("run dir: %s (status %s; task pass rate %s %s; request errors %d; harness errors %d%s)" % (
        run.rd.path, status, _pct(o["task_pass_rate"]), _ci(o["task_ci95"]), s["n_request_errors"],
        s["n_harness_errors"], "; MOCK SERVER" if s.get("is_mock_server") else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
