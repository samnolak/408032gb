"""Apply a model's code answer to a working copy (stdlib only; no `patch`/`git` binary needed).

Answer protocol (the exact text the model sees is ANSWER_PROTOCOL; README.md documents it):

  1. Whole-file blocks. A line
         FILE: <relative/path>
     (optionally as a markdown heading "### FILE: path" or in bold "**FILE: path**"; the path may be
     wrapped in backticks) followed, after optional blank lines, by a fenced code block (``` or ~~~,
     any info string). The block content is the COMPLETE new content of that file (created if missing).
     A line "DELETE: <relative/path>" deletes a file.
  2. Unified diff. Fenced blocks with info string "diff" or "patch" holding `--- a/<path>` / `+++ b/<path>`
     file headers and `@@ -l,s +l,s @@` hunks; "/dev/null" creates or deletes a file. If the answer has
     no FILE/DELETE lines and no diff-fenced block, bare unified-diff text anywhere in the answer is used.
  Precedence: if any FILE:/DELETE: line exists, only (1) is applied and diff blocks are ignored (noted).

Hunk application (lenient like `git apply --recount`): the line counts in "@@" headers are ignored; a
hunk's old side (context + '-' lines) must match the file exactly except for trailing whitespace; the
search starts at the header's line number (or at the previous hunk for a bare "@@" header without
numbers) and moves outward, never before the end of the previous hunk of the same file; an empty line inside a hunk is a blank context line; "\\ No newline at end of file"
is honoured for the last line. Paths: "a/" and "b/" prefixes are stripped when both headers carry them.

Safety: every path must be relative, without "..", NUL or backslashes, must not resolve (through
symlinks) outside the working copy and must not be an existing symlink. Paths matching a protected
glob (fnmatch on the POSIX relative path) may not be created, changed or deleted; when allowed globs
are given, every edited path must match one of them.
"""
from __future__ import annotations

import fnmatch
import os
import posixpath
import re

ANSWER_PROTOCOL = """Answer format (machine-checked; prose outside these blocks is ignored):
- To create or replace a file, write a line `FILE: <path relative to the repository root>` and, right after it, a fenced code block with the COMPLETE new content of that file.
- Or give a unified diff in a ```diff fenced block with `--- a/<path>` and `+++ b/<path>` headers and @@ hunks.
- To delete a file, write a line `DELETE: <path>`.
- Do not modify the tests."""

FILE_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?(FILE|DELETE):[ \t]*`?([^`*\s]+)`?(?:\*\*)?[ \t]*$", re.I)
FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*([^`\s]*)[^`]*$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# fnmatch globs on the POSIX relative path; '*' also matches '/'
DEFAULT_PROTECTED = ("test_*.py", "*/test_*.py", "*_test.py", "tests/*", "*/tests/*", "test/*", "*/test/*",
                     "conftest.py", "*/conftest.py")


class PatchError(Exception):
    pass


# ---------------------------------------------------------------------------- paths
def check_rel_path(rel):
    """Normalized POSIX relative path or PatchError."""
    if not rel or "\x00" in rel or "\\" in rel:
        raise PatchError("unsafe path %r" % rel)
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise PatchError("absolute path not allowed: %r" % rel)
    parts = rel.split("/")
    if any(p == ".." for p in parts):
        raise PatchError("path with '..' not allowed: %r" % rel)
    norm = posixpath.normpath(rel)
    if norm in (".", "") or norm.startswith("../"):
        raise PatchError("unsafe path %r" % rel)
    return norm


def resolve_in(root, rel):
    """Absolute path of rel inside root; PatchError if it would leave root (also via symlinks)."""
    norm = check_rel_path(rel)
    root_real = os.path.realpath(root)
    target = os.path.join(root_real, *norm.split("/"))
    parent_real = os.path.realpath(os.path.dirname(target))
    if parent_real != root_real and not parent_real.startswith(root_real + os.sep):
        raise PatchError("path escapes the working copy: %r" % rel)
    if os.path.islink(target):
        raise PatchError("refusing to write through a symlink: %r" % rel)
    return target


def matches_any(rel, globs):
    return any(fnmatch.fnmatchcase(rel, g) for g in globs)


# ---------------------------------------------------------------------------- parsing
def _fenced_blocks(lines, start):
    """Parse a fenced block whose opener is at or after `start` (blank lines allowed before it).
    Returns (info, content_lines, index_after_block) or None."""
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None
    m = FENCE_OPEN_RE.match(lines[i])
    if not m:
        return None
    fence = m.group(1)
    info = m.group(2)
    body = []
    j = i + 1
    while j < len(lines):
        s = lines[j].strip()
        if s and s[0] == fence[0] and set(s) == {fence[0]} and len(s) >= len(fence):
            return info, body, j + 1
        body.append(lines[j])
        j += 1
    return info, body, j          # unclosed fence: the block runs to the end of the answer


def parse_file_blocks(text):
    """[(op, path, content_or_None)], op in {"write", "delete"}."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = FILE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        op, path = m.group(1).upper(), m.group(2)
        if op == "DELETE":
            out.append(("delete", path, None))
            i += 1
            continue
        blk = _fenced_blocks(lines, i + 1)
        if blk is None:
            raise PatchError("FILE: %s is not followed by a fenced code block" % path)
        _, body, nxt = blk
        out.append(("write", path, "\n".join(body) + ("\n" if body else "")))
        i = nxt
    return out


def diff_texts(text):
    """Diff payloads: the content of ```diff / ```patch blocks; else the whole text if it holds a diff."""
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        m = FENCE_OPEN_RE.match(lines[i])
        if m:
            blk = _fenced_blocks(lines, i)
            info, body, nxt = blk
            if info.lower() in ("diff", "patch", "udiff"):
                blocks.append("\n".join(body))
            i = nxt
            continue
        i += 1
    if blocks:
        return blocks
    if re.search(r"^--- \S.*\n\+\+\+ \S", text, re.M):
        return [text]
    return []


def _strip_ab(old, new):
    if old != "/dev/null" and new != "/dev/null":
        if old.startswith("a/") and new.startswith("b/"):
            return old[2:], new[2:]
        return old, new
    if old.startswith("a/"):
        old = old[2:]
    if new.startswith("b/"):
        new = new[2:]
    return old, new


def _header_path(line, prefix):
    p = line[len(prefix):].rstrip("\r")
    p = p.split("\t", 1)[0].strip()
    if p.startswith('"') and p.endswith('"') and len(p) > 1:
        p = p[1:-1]
    return p


def parse_unified(text):
    """[{"old": path, "new": path, "hunks": [{"old_start", "lines": [(tag, text)], "no_eol_new"}]}]"""
    lines = text.split("\n")
    files = []
    cur = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old, new = _strip_ab(_header_path(ln, "--- "), _header_path(lines[i + 1], "+++ "))
            cur = {"old": old, "new": new, "hunks": []}
            files.append(cur)
            i += 2
            continue
        m = HUNK_RE.match(ln)
        if cur is not None and (m or ln.startswith("@@")):
            # a bare "@@" header (no line numbers) is accepted: the context is searched from the previous hunk
            hunk = {"old_start": int(m.group(1)) if m else None, "lines": [], "no_eol_new": False}
            i += 1
            while i < len(lines):
                h = lines[i]
                if h.startswith("@@") or (h.startswith("--- ") and i + 1 < len(lines)
                                          and lines[i + 1].startswith("+++ ")) or h.startswith("diff --git"):
                    break
                if h.startswith("\\"):
                    prev = hunk["lines"][-1][0] if hunk["lines"] else None
                    if prev in ("+", " "):
                        hunk["no_eol_new"] = True
                elif h == "" or h == "\r":
                    hunk["lines"].append((" ", ""))
                elif h[0] in " +-":
                    hunk["lines"].append((h[0], h[1:].rstrip("\r")))
                else:
                    break
                i += 1
            while hunk["lines"] and hunk["lines"][-1] == (" ", ""):
                hunk["lines"].pop()                       # trailing blank lines are separators, not context
            if hunk["lines"]:
                cur["hunks"].append(hunk)
            continue
        i += 1
    return [f for f in files if f["hunks"] or f["new"] == "/dev/null"]


def _find(file_lines, old, hint, floor):
    """Index where `old` matches (trailing whitespace ignored), nearest to hint, >= floor; or None."""
    last = len(file_lines) - len(old)
    if last < floor:
        return None
    want = [x.rstrip() for x in old]
    hint = min(max(hint, floor), last)
    for delta in range(0, last - floor + 1):
        for pos in ((hint - delta, hint + delta) if delta else (hint,)):
            if floor <= pos <= last and all(file_lines[pos + j].rstrip() == want[j] for j in range(len(old))):
                return pos
    return None


def apply_hunks(content, hunks, path):
    """Apply parsed hunks to text; PatchError when a hunk's old side is not found."""
    eol = content.endswith("\n") if content else True
    file_lines = content.split("\n")
    if file_lines and file_lines[-1] == "":
        file_lines.pop()
    floor = 0
    offset = 0
    for hi, h in enumerate(hunks):
        old = [t for tag, t in h["lines"] if tag in (" ", "-")]
        new = [t for tag, t in h["lines"] if tag in (" ", "+")]
        if not old:                                       # pure insertion after line old_start
            if h["old_start"] is None:
                raise PatchError("hunk %d of %s has neither line numbers nor context" % (hi + 1, path))
            pos = max(min(h["old_start"] + offset, len(file_lines)), floor)
        else:
            hint = floor if h["old_start"] is None else h["old_start"] - 1 + offset
            pos = _find(file_lines, old, hint, floor)
            if pos is None:
                raise PatchError("hunk %d of %s does not apply (context not found)" % (hi + 1, path))
        file_lines[pos:pos + len(old)] = new
        floor = pos + len(new)
        offset += len(new) - len(old)
        if floor == len(file_lines):                      # the hunk reaches the end of the file
            eol = not h["no_eol_new"]
    return "\n".join(file_lines) + ("\n" if file_lines and eol else "")


# ---------------------------------------------------------------------------- applying
def plan_edits(answer):
    """Turn an answer into [(op, path, payload)] without touching disk. Returns (mode, edits, notes)."""
    notes = []
    blocks = parse_file_blocks(answer)
    if blocks:
        if diff_texts(answer):
            notes.append("diff blocks ignored because FILE:/DELETE: blocks are present")
        return "files", blocks, notes
    edits = []
    for txt in diff_texts(answer):
        for f in parse_unified(txt):
            if f["new"] == "/dev/null":
                edits.append(("delete", f["old"], None))
            elif f["old"] == "/dev/null":
                edits.append(("create_diff", f["new"], f["hunks"]))
            else:
                if f["old"] != f["new"]:
                    notes.append("rename %s -> %s treated as edit of %s" % (f["old"], f["new"], f["new"]))
                edits.append(("patch", f["new"], f["hunks"]))
    if not edits:
        raise PatchError("no FILE: block or unified diff found in the answer")
    return "diff", edits, notes


def apply_answer(answer, root, protected=DEFAULT_PROTECTED, allowed=None):
    """Apply the answer under `root`. Returns dict(ok, mode, edited, error, notes). Never raises PatchError."""
    res = {"ok": False, "mode": None, "edited": [], "error": None, "notes": []}
    try:
        mode, edits, notes = plan_edits(answer or "")
        res["mode"], res["notes"] = mode, notes
        # validate every path before writing anything
        for op, rel, _ in edits:
            norm = check_rel_path(rel)
            resolve_in(root, norm)
            if matches_any(norm, protected or ()):
                raise PatchError("answer modifies a protected path: %s" % norm)
            if allowed and not matches_any(norm, allowed):
                raise PatchError("answer edits a path outside the allowed set: %s" % norm)
        for op, rel, payload in edits:
            norm = check_rel_path(rel)
            target = resolve_in(root, norm)
            if op == "delete":
                if not os.path.isfile(target):
                    raise PatchError("cannot delete missing file %s" % norm)
                os.remove(target)
            elif op == "write":
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="") as f:
                    f.write(payload)
            else:
                if op == "patch":
                    if not os.path.isfile(target):
                        raise PatchError("diff patches missing file %s" % norm)
                    with open(target, encoding="utf-8", newline="") as f:
                        content = f.read()
                else:
                    if os.path.exists(target):
                        raise PatchError("diff creates %s but it already exists" % norm)
                    content = ""
                new = apply_hunks(content, payload, norm)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
            res["edited"].append({"op": op, "path": norm})
        res["ok"] = True
    except PatchError as e:
        res["error"] = str(e)
    except (OSError, UnicodeDecodeError) as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
    return res
