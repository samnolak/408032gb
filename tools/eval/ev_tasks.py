"""Task files of the agentic evaluation stand: discovery, validation, rendering, hashing (stdlib only).

One task per file named "*.task.json" anywhere under the task directory (directories whose name starts
with "." or "_" are skipped). Paths inside a task are relative to the task file's directory and must stay
inside the task directory (checked after resolving symlinks). The full format is in README.md.

Placeholders in message contents (replaced once, inserted text is not re-scanned; other {{...}} stay):
  {{document}}        the task's document (document.file or document.text)
  {{file:REL}}        content of a fixture file (code tasks)
  {{fixture_tree}}    sorted list of fixture files (code tasks)
  {{answer_protocol}} the code answer protocol text (ev_patch.ANSWER_PROTOCOL)
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import ev_answers
import ev_patch
import ev_schema
import ev_toolsim

CATEGORIES = ("code_fix", "code_write", "tool_use", "long_doc_qa", "structured_output", "reasoning")
CODE_CATEGORIES = ("code_fix", "code_write")
TOP_KEYS = {"id", "category", "description", "synthetic", "messages", "tools", "tool_choice", "tool_simulator",
            "fixture", "document", "check", "max_tokens", "max_turns", "response_format", "reference_answer",
            "notes", "tags"}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PH_RE = re.compile(r"\{\{\s*(document|fixture_tree|answer_protocol|file:([^{}]+?))\s*\}\}")
SKIP_DIRS = {"__pycache__", ".git"}
TEXT_MATCHERS = {"long_doc_qa": ev_answers.MATCHERS, "reasoning": ("exact", "normalized", "number")}
DEFAULT_MAX_TURNS = 8
CODE_CHECK_KEYS = {"type", "command", "timeout_s", "protected_paths", "allowed_paths", "overlay", "limits", "env",
                   "expect_fixture_fails"}


class TaskError(Exception):
    pass


def resolve_rel(root, base_dir, rel, what):
    """Absolute real path of rel (relative to base_dir) that must stay under root."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\x00" in rel:
        raise TaskError("%s: path must be a non-empty relative path, got %r" % (what, rel))
    root_real = os.path.realpath(root)
    p = os.path.realpath(os.path.join(base_dir, rel))
    if p != root_real and not p.startswith(root_real + os.sep):
        raise TaskError("%s: %r points outside the task directory" % (what, rel))
    return p


def list_files(d):
    out = []
    for dp, dns, fns in os.walk(d):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS)
        for fn in sorted(fns):
            if fn.endswith(".pyc"):
                continue
            out.append(os.path.relpath(os.path.join(dp, fn), d).replace(os.sep, "/"))
    return sorted(out)


def _hash_tree(h, d, label):
    h.update(("\0dir:%s\0" % label).encode())
    for rel in list_files(d):
        h.update(rel.encode("utf-8") + b"\0")
        with open(os.path.join(d, rel), "rb") as f:
            h.update(f.read())


def check_matcher(spec, where, allowed=ev_answers.MATCHERS):
    errs = []
    if not isinstance(spec, dict):
        return ["%s: must be an object" % where]
    t = spec.get("type")
    if t not in allowed:
        return ["%s: type must be one of %s, got %r" % (where, list(allowed), t)]
    if t in ("exact", "normalized"):
        a = spec.get("answers")
        if not (isinstance(a, list) and a and all(isinstance(x, str) for x in a)):
            errs.append("%s: 'answers' must be a non-empty list of strings" % where)
    if t == "regex":
        try:
            re.compile(spec.get("pattern"))
        except (re.error, TypeError) as e:
            errs.append("%s: bad 'pattern': %s" % (where, e))
        if set(spec.get("flags", "")) - set("ims"):
            errs.append("%s: flags may only contain i, m, s" % where)
    if t in ("contains_all", "contains_any"):
        v = spec.get("values")
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            errs.append("%s: 'values' must be a non-empty list of strings" % where)
    if t == "number" and (isinstance(spec.get("value"), bool) or not isinstance(spec.get("value"), (int, float))):
        errs.append("%s: 'value' must be a number" % where)
    ex = spec.get("extract", "whole")
    if ex not in ev_answers.EXTRACTORS:
        errs.append("%s: extract must be one of %s" % (where, list(ev_answers.EXTRACTORS)))
    if ex == "regex":
        try:
            re.compile(spec.get("extract_regex"))
        except (re.error, TypeError) as e:
            errs.append("%s: bad extract_regex: %s" % (where, e))
    extra = set(spec) - {"type", "answers", "pattern", "flags", "fullmatch", "values", "value", "abs_tol", "rel_tol",
                         "extract", "extract_regex"}
    if extra:
        errs.append("%s: unknown keys %s" % (where, sorted(extra)))
    return errs


class Task:
    """A validated task. Attributes are filled by load_task(); messages are rendered."""

    def __init__(self, path, root):
        self.path = path
        self.root = root
        self.dir = os.path.dirname(path)
        self.raw = None
        self.id = None
        self.category = None
        self.messages = None
        self.tools = None
        self.tool_choice = None
        self.sim_spec = None
        self.fixture_dir = None
        self.overlay_dir = None
        self.document = None
        self.check = None
        self.max_tokens = None
        self.max_turns = DEFAULT_MAX_TURNS
        self.response_format = None
        self.reference_answer = None
        self.sha256 = None

    @property
    def rel_path(self):
        return os.path.relpath(self.path, self.root).replace(os.sep, "/")

    def meta(self):
        return {"id": self.id, "category": self.category, "file": self.rel_path, "sha256": self.sha256,
                "synthetic": bool(self.raw.get("synthetic"))}


def _render(task, content, errs, where):
    def sub(m):
        name = m.group(1)
        if name == "document":
            if task.document is None:
                errs.append("%s: {{document}} used but the task has no document" % where)
                return m.group(0)
            return task.document
        if name == "answer_protocol":
            return ev_patch.ANSWER_PROTOCOL
        if task.fixture_dir is None:
            errs.append("%s: {{%s}} used but the task has no fixture" % (where, name))
            return m.group(0)
        if name == "fixture_tree":
            return "\n".join(list_files(task.fixture_dir))
        rel = m.group(2).strip()
        try:
            p = resolve_rel(task.fixture_dir, task.fixture_dir, rel, where)
            with open(p, encoding="utf-8") as f:
                return f.read()
        except (TaskError, OSError, UnicodeDecodeError) as e:
            errs.append("%s: {{file:%s}}: %s" % (where, rel, e))
            return m.group(0)
    return PH_RE.sub(sub, content)


def load_task(path, root):
    """Returns (Task or None, [errors])."""
    t = Task(path, root)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        return None, ["%s: cannot read task JSON: %s" % (path, e)]
    where = os.path.relpath(path, root)
    if not isinstance(data, dict):
        return None, ["%s: a task must be a JSON object" % where]
    t.raw = data
    errs = []
    unknown = set(data) - TOP_KEYS
    if unknown:
        errs.append("%s: unknown top-level keys %s" % (where, sorted(unknown)))
    t.id = data.get("id")
    if not isinstance(t.id, str) or not ID_RE.match(t.id):
        errs.append("%s: 'id' must match %s" % (where, ID_RE.pattern))
    t.category = data.get("category")
    if t.category not in CATEGORIES:
        errs.append("%s: 'category' must be one of %s" % (where, list(CATEGORIES)))
        return None, errs
    h = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    # ------------------------------------------------------------------ referenced files
    try:
        if "fixture" in data:
            t.fixture_dir = resolve_rel(root, t.dir, data["fixture"], "%s: fixture" % where)
            if not os.path.isdir(t.fixture_dir):
                raise TaskError("%s: fixture %r is not a directory" % (where, data["fixture"]))
            _hash_tree(h, t.fixture_dir, "fixture")
        doc = data.get("document")
        if doc is not None:
            if not isinstance(doc, dict) or len(set(doc) & {"file", "text"}) != 1 or set(doc) - {"file", "text"}:
                raise TaskError("%s: document must be {\"file\": REL} or {\"text\": STR}" % where)
            if "text" in doc:
                if not isinstance(doc["text"], str):
                    raise TaskError("%s: document.text must be a string" % where)
                t.document = doc["text"]
            else:
                p = resolve_rel(root, t.dir, doc["file"], "%s: document.file" % where)
                with open(p, encoding="utf-8") as f:
                    t.document = f.read()
                h.update(b"\0document\0" + t.document.encode("utf-8"))
        ref = data.get("reference_answer")
        if ref is not None:
            if isinstance(ref, str):
                t.reference_answer = ref
            elif isinstance(ref, dict) and set(ref) == {"file"}:
                p = resolve_rel(root, t.dir, ref["file"], "%s: reference_answer.file" % where)
                with open(p, encoding="utf-8") as f:
                    t.reference_answer = f.read()
                h.update(b"\0reference\0" + t.reference_answer.encode("utf-8"))
            else:
                raise TaskError("%s: reference_answer must be a string or {\"file\": REL}" % where)
    except (TaskError, OSError, UnicodeDecodeError) as e:
        errs.append(str(e))
    # ------------------------------------------------------------------ messages
    msgs = data.get("messages")
    if not (isinstance(msgs, list) and msgs):
        errs.append("%s: 'messages' must be a non-empty list" % where)
        msgs = []
    rendered = []
    for i, m in enumerate(msgs):
        w = "%s: messages[%d]" % (where, i)
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") \
                or not isinstance(m.get("content"), str) or set(m) - {"role", "content"}:
            errs.append("%s: must be {\"role\": system|user|assistant, \"content\": STR}" % w)
            continue
        rendered.append({"role": m["role"], "content": _render(t, m["content"], errs, w)})
    if rendered and rendered[-1]["role"] != "user":
        errs.append("%s: the last message must have role 'user'" % where)
    t.messages = rendered
    # ------------------------------------------------------------------ request options
    if "max_tokens" in data and not (isinstance(data["max_tokens"], int) and data["max_tokens"] > 0):
        errs.append("%s: max_tokens must be a positive integer" % where)
    t.max_tokens = data.get("max_tokens")
    if "response_format" in data and not isinstance(data["response_format"], dict):
        errs.append("%s: response_format must be an object" % where)
    t.response_format = data.get("response_format")
    t.tool_choice = data.get("tool_choice")
    # ------------------------------------------------------------------ category specifics
    chk = data.get("check")
    if not isinstance(chk, dict):
        errs.append("%s: 'check' must be an object" % where)
        chk = {}
    t.check = chk
    cat = t.category
    if cat in CODE_CATEGORIES:
        if t.fixture_dir is None and not any("fixture" in e for e in errs):
            errs.append("%s: code tasks need a 'fixture' directory" % where)
        if chk.get("type") != "code_tests":
            errs.append("%s: check.type must be 'code_tests' for %s" % (where, cat))
        cmd = chk.get("command")
        if not (isinstance(cmd, list) and cmd and all(isinstance(x, str) and x for x in cmd)):
            errs.append("%s: check.command must be a non-empty list of strings (argv, no shell)" % where)
        for k in ("protected_paths", "allowed_paths"):
            if k in chk and not (isinstance(chk[k], list) and all(isinstance(x, str) for x in chk[k])):
                errs.append("%s: check.%s must be a list of glob strings" % (where, k))
        if "timeout_s" in chk and not (isinstance(chk["timeout_s"], (int, float)) and 0 < chk["timeout_s"] <= 3600):
            errs.append("%s: check.timeout_s must be in (0, 3600]" % where)
        lim = chk.get("limits", {})
        if not isinstance(lim, dict) or set(lim) - {"as_mb", "fsize_mb", "nofile", "nproc"} \
                or not all(isinstance(v, int) and v >= 0 for v in lim.values()):
            errs.append("%s: check.limits keys: as_mb, fsize_mb, nofile, nproc (integers >= 0; 0 = unlimited)" % where)
        if "env" in chk and not (isinstance(chk["env"], dict) and all(isinstance(v, str) for v in chk["env"].values())):
            errs.append("%s: check.env must map names to strings" % where)
        if "overlay" in chk:
            try:
                t.overlay_dir = resolve_rel(root, t.dir, chk["overlay"], "%s: check.overlay" % where)
                if not os.path.isdir(t.overlay_dir):
                    raise TaskError("%s: check.overlay is not a directory" % where)
                _hash_tree(h, t.overlay_dir, "overlay")
            except TaskError as e:
                errs.append(str(e))
        extra = set(chk) - CODE_CHECK_KEYS
        if extra:
            errs.append("%s: unknown check keys %s" % (where, sorted(extra)))
    elif cat == "tool_use":
        tools = data.get("tools")
        names = []
        if not (isinstance(tools, list) and tools):
            errs.append("%s: tool_use tasks need a non-empty 'tools' list" % where)
            tools = []
        for i, tl in enumerate(tools):
            fn = tl.get("function") if isinstance(tl, dict) else None
            if not (isinstance(tl, dict) and tl.get("type") == "function" and isinstance(fn, dict)
                    and isinstance(fn.get("name"), str)):
                errs.append("%s: tools[%d] must be {\"type\": \"function\", \"function\": {\"name\": ...}}" % (where, i))
                continue
            names.append(fn["name"])
            if "parameters" in fn:
                errs.extend("%s: tools[%d].parameters %s" % (where, i, p) for p in ev_schema.schema_problems(fn["parameters"]))
        t.tools = tools
        sim = data.get("tool_simulator")
        if not isinstance(sim, dict):
            errs.append("%s: tool_use tasks need a 'tool_simulator' object" % where)
            sim = {}
        errs.extend("%s: %s" % (where, p) for p in ev_toolsim.spec_problems(sim, tools))
        t.sim_spec = sim
        mt = data.get("max_turns", DEFAULT_MAX_TURNS)
        if not (isinstance(mt, int) and 1 <= mt <= 50):
            errs.append("%s: max_turns must be an integer in [1, 50]" % where)
        t.max_turns = mt
        if chk.get("type") != "tool_use":
            errs.append("%s: check.type must be 'tool_use'" % where)
        rc = chk.get("required_calls", [])
        if not isinstance(rc, list):
            errs.append("%s: check.required_calls must be a list" % where)
            rc = []
        for key in ("required_calls", "forbidden_calls"):
            for i, c in enumerate(chk.get(key) or []):
                if not (isinstance(c, dict) and c.get("name") in names and set(c) <= {"name", "arguments", "match"}
                        and isinstance(c.get("arguments", {}), dict) and c.get("match", "subset") in ("exact", "subset")):
                    errs.append("%s: check.%s[%d] must be {name (declared tool), arguments?: {}, match?: exact|subset}"
                                % (where, key, i))
        if chk.get("order", "strict") not in ("strict", "any"):
            errs.append("%s: check.order must be strict or any" % where)
        if "max_calls" in chk and not (isinstance(chk["max_calls"], int) and chk["max_calls"] >= 1):
            errs.append("%s: check.max_calls must be a positive integer" % where)
        if "final_answer" in chk:
            errs.extend(check_matcher(chk["final_answer"], "%s: check.final_answer" % where))
        extra = set(chk) - {"type", "required_calls", "forbidden_calls", "order", "max_calls",
                            "require_valid_arguments", "final_answer"}
        if extra:
            errs.append("%s: unknown check keys %s" % (where, sorted(extra)))
    elif cat in TEXT_MATCHERS:
        errs.extend(check_matcher(chk, "%s: check" % where, TEXT_MATCHERS[cat]))
    elif cat == "structured_output":
        if chk.get("type") != "json_schema":
            errs.append("%s: check.type must be 'json_schema'" % where)
        if "schema" not in chk:
            errs.append("%s: check.schema is required" % where)
        else:
            errs.extend("%s: check.schema %s" % (where, p) for p in ev_schema.schema_problems(chk["schema"]))
        if chk.get("parse", "strict") not in ("strict", "lenient"):
            errs.append("%s: check.parse must be strict or lenient" % where)
        for i, v in enumerate(chk.get("values") or []):
            ops = [k for k in ev_answers.VALUE_OPS if k in v] if isinstance(v, dict) else []
            if not (isinstance(v, dict) and isinstance(v.get("path"), str) and len(ops) == 1
                    and set(v) <= {"path", "tol"} | set(ops)):
                errs.append("%s: check.values[%d] must be {path, one of %s}" % (where, i, list(ev_answers.VALUE_OPS)))
        extra = set(chk) - {"type", "schema", "values", "parse"}
        if extra:
            errs.append("%s: unknown check keys %s" % (where, sorted(extra)))
    if cat != "tool_use" and any(k in data for k in ("tools", "tool_simulator", "max_turns")):
        errs.append("%s: tools / tool_simulator / max_turns are only for tool_use tasks" % where)
    t.sha256 = h.hexdigest()
    return (None if errs else t), errs


def discover(root):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith((".", "_")) and d not in SKIP_DIRS)
        for fn in sorted(fns):
            if fn.endswith(".task.json"):
                out.append(os.path.join(dp, fn))
    return sorted(out, key=lambda p: os.path.relpath(p, root))


def load_dir(root):
    """Returns (tasks, errors). Duplicate ids are errors."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return [], ["task directory not found: %s" % root]
    tasks, errors, seen = [], [], {}
    paths = discover(root)
    if not paths:
        errors.append("no *.task.json files under %s" % root)
    for p in paths:
        t, errs = load_task(p, root)
        errors.extend(errs)
        if t is None:
            continue
        if t.id in seen:
            errors.append("%s: duplicate id %r (also in %s)" % (t.rel_path, t.id, seen[t.id]))
            continue
        seen[t.id] = t.rel_path
        tasks.append(t)
    return tasks, errors


def task_set_sha256(tasks):
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda x: x.id):
        h.update(("%s %s\n" % (t.id, t.sha256)).encode())
    return h.hexdigest()
