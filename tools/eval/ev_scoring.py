"""Per-category grading. Every grader returns dict(status, score, reason, details) with status in
{"pass", "fail", "harness_error"}; request errors are decided by the runner before grading and never
reach these functions. Only assistant content is graded (reasoning is ignored; see ev_answers.strip_think).
"""
from __future__ import annotations

import difflib
import os
import shutil
import tempfile

import ev_answers
import ev_patch
import ev_sandbox
import ev_schema
import ev_tasks
from ev_toolsim import args_match

MAX_DIFF_CHARS = 20000


def outcome(passed, score, reason, details=None):
    return {"status": "pass" if passed else "fail", "score": float(score), "reason": reason, "details": details or {}}


def harness_error(reason, details=None):
    return {"status": "harness_error", "score": None, "reason": reason, "details": details or {}}


# ---------------------------------------------------------------------------- text answers
def grade_text(task, content):
    g = ev_answers.grade_text(content, task.check)
    return outcome(g["ok"], 1.0 if g["ok"] else 0.0, g["detail"], {"extracted": g["extracted"], "how": g["how"]})


def grade_structured(task, content):
    chk = task.check
    obj, how = ev_answers.extract_json(content, chk.get("parse", "strict"))
    if how not in ev_answers.JSON_PARSE_OK:
        return outcome(False, 0.0, how, {"parse": how, "checks": []})
    subs = []
    errs = ev_schema.validate(obj, chk["schema"])
    subs.append({"check": "schema", "ok": not errs, "detail": errs[:20]})
    for v in chk.get("values") or []:
        ok, detail = ev_answers.check_value(obj, v)
        subs.append({"check": "value %s" % v["path"], "ok": ok, "detail": detail})
    n_ok = sum(1 for s in subs if s["ok"])
    passed = n_ok == len(subs)
    failed = [s["check"] for s in subs if not s["ok"]]
    reason = "all %d checks passed" % len(subs) if passed else "failed: %s" % "; ".join(
        failed[:5] + (["schema errors: %s" % "; ".join(errs[:3])] if errs else []))
    return outcome(passed, n_ok / len(subs), reason, {"parse": how, "checks": subs})


# ---------------------------------------------------------------------------- tool use
def _call_matches(call, spec):
    if call.get("name") != spec["name"]:
        return False
    if "arguments" not in spec:
        return True
    return args_match(call.get("args"), spec["arguments"], spec.get("match", "subset"))


def _assign_any_order(calls, req):
    """Distinct call index for every required call (backtracking), or a partial assignment with None."""
    best = [None] * len(req)

    def rec(i, used, cur):
        nonlocal best
        if i == len(req):
            return True
        for j, c in enumerate(calls):
            if j not in used and _call_matches(c, req[i]):
                cur[i] = j
                if rec(i + 1, used | {j}, cur):
                    return True
                cur[i] = None
        return False

    cur = [None] * len(req)
    if rec(0, frozenset(), cur):
        return cur
    # no complete assignment: report each requirement on its own (presence only)
    for i, spec in enumerate(req):
        best[i] = next((j for j, c in enumerate(calls) if _call_matches(c, spec)), None)
    return best


def _in_order(calls, req):
    """Greedy earliest subsequence match: True iff the required calls occur in this order."""
    j = 0
    for spec in req:
        while j < len(calls) and not _call_matches(calls[j], spec):
            j += 1
        if j == len(calls):
            return False
        j += 1
    return True


def grade_tool_use(task, calls, final_content, turn_limit_hit):
    chk = task.check
    req = chk.get("required_calls") or []
    subs = []
    assign = _assign_any_order(calls, req)
    for i, spec in enumerate(req):
        subs.append({"check": "required_call[%d] %s" % (i, spec["name"]), "ok": assign[i] is not None,
                     "detail": {"matched_call": assign[i], "want": spec}})
    if req and chk.get("order", "strict") == "strict":
        subs.append({"check": "order", "ok": _in_order(calls, req), "detail": [c.get("name") for c in calls]})
    for spec in chk.get("forbidden_calls") or []:
        hits = [j for j, c in enumerate(calls) if _call_matches(c, spec)]
        subs.append({"check": "forbidden %s" % spec["name"], "ok": not hits, "detail": {"calls": hits}})
    if "max_calls" in chk:
        subs.append({"check": "max_calls", "ok": len(calls) <= chk["max_calls"],
                     "detail": "%d calls, limit %d" % (len(calls), chk["max_calls"])})
    if chk.get("require_valid_arguments", True):
        bad = [j for j, c in enumerate(calls) if not c.get("args_valid")]
        subs.append({"check": "valid_arguments", "ok": not bad, "detail": {"invalid_calls": bad}})
    subs.append({"check": "turn_limit", "ok": not turn_limit_hit,
                 "detail": "turn limit %d reached without a final answer" % task.max_turns if turn_limit_hit else "ok"})
    if "final_answer" in chk:
        if turn_limit_hit or final_content is None:
            subs.append({"check": "final_answer", "ok": False, "detail": "no final answer"})
        else:
            g = ev_answers.grade_text(final_content, chk["final_answer"])
            subs.append({"check": "final_answer", "ok": g["ok"], "detail": g["detail"], "extracted": g["extracted"]})
    n_ok = sum(1 for s in subs if s["ok"])
    passed = n_ok == len(subs)
    failed = [s["check"] for s in subs if not s["ok"]]
    reason = "all %d checks passed" % len(subs) if passed else "failed: %s" % ", ".join(failed)
    if turn_limit_hit:
        reason = "turn limit %d reached; %s" % (task.max_turns, reason)
    return outcome(passed, n_ok / len(subs), reason, {"checks": subs, "n_calls": len(calls)})


# ---------------------------------------------------------------------------- code
def _lines(p):
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return f.read().splitlines(True)


def _diff(fixture, repo, edited, limit=MAX_DIFF_CHARS):
    out = []
    for e in edited:
        rel = e["path"]
        try:
            a = _lines(os.path.join(fixture, rel))
            b = _lines(os.path.join(repo, rel))
        except (OSError, UnicodeDecodeError) as ex:
            out.append("# %s: %s\n" % (rel, ex))
            continue
        out.extend(difflib.unified_diff(a, b, "a/" + rel, "b/" + rel))
    s = "".join(out)
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]\n"


class CodeSandbox:
    """Holds the sandbox policy for a run. weak=False refuses to run without full isolation."""

    def __init__(self, work_root=None, weak=False, keep=False, caps=None):
        self.caps = caps or ev_sandbox.capabilities()
        self.weak = weak
        self.keep = keep
        self.work_root = work_root

    def missing(self):
        return ev_sandbox.missing_isolation(self.caps)

    def run_fixture(self, task, answer, log=None, apply_answer=True):
        """Copy the fixture, apply `answer` (unless apply_answer=False), add the overlay, run the command."""
        miss = self.missing()
        if miss and not self.weak:
            return harness_error("sandbox unavailable: missing %s (use --weak-sandbox to run anyway)"
                                 % "; ".join(miss), {"capabilities": self.caps})
        chk = task.check
        attempt = tempfile.mkdtemp(prefix="eval-attempt-", dir=self.work_root)
        repo = os.path.join(attempt, "repo")
        try:
            shutil.copytree(task.fixture_dir, repo, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
            details = {"attempt_dir": attempt if self.keep else None}
            if apply_answer:
                applied = ev_patch.apply_answer(ev_answers.strip_think(answer),
                                                repo, chk.get("protected_paths", ev_patch.DEFAULT_PROTECTED),
                                                chk.get("allowed_paths"))
                details["apply"] = applied
                details["diff"] = _diff(task.fixture_dir, repo, applied["edited"])
                if not applied["ok"]:
                    if log:
                        log({"kind": "sandbox", "apply": applied, "run": None})
                    return outcome(False, 0.0, "answer not applied: %s" % applied["error"], details)
            if task.overlay_dir:
                shutil.copytree(task.overlay_dir, repo, symlinks=True, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            limits = dict(chk.get("limits") or {})
            limits["timeout_s"] = chk.get("timeout_s", ev_sandbox.DEFAULT_LIMITS["timeout_s"])
            run = ev_sandbox.run_command(attempt, chk["command"], limits, self.caps, env_extra=chk.get("env"))
            details["run"] = run
            if log:
                log({"kind": "sandbox", "apply": details.get("apply"), "diff": details.get("diff"), "run": run})
            if run["error"]:
                return harness_error("sandbox: %s" % run["error"], details)
            iso = run["isolation"] or {}
            if not self.weak and not (iso.get("landlock_fs") and (iso.get("netns") or iso.get("landlock_net"))):
                return harness_error("sandbox isolation was not applied: %s" % iso, details)
            if run["timed_out"]:
                return outcome(False, 0.0, "timeout: tests did not finish in %ss" % limits["timeout_s"], details)
            passed = run["rc"] == 0
            return outcome(passed, 1.0 if passed else 0.0,
                           "tests passed" if passed else "tests failed (exit code %s)" % run["rc"], details)
        finally:
            if not self.keep:
                shutil.rmtree(attempt, ignore_errors=True)


def grade_code(task, content, sandbox, log=None):
    return sandbox.run_fixture(task, content, log)


def grade(task, content, sandbox=None, log=None):
    """Grade a single-answer task (everything except tool_use)."""
    if task.category in ev_tasks.CODE_CATEGORIES:
        return grade_code(task, content, sandbox, log)
    if task.category == "structured_output":
        return grade_structured(task, content)
    return grade_text(task, content)
