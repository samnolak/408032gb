"""Answer extraction and matching for long_doc_qa, reasoning, tool_use final answers and
structured_output (stdlib only). Only the assistant `content` is graded, never reasoning.

Rules (documented in tools/eval/README.md):
  strip_think   if content contains "</think>", only the text after the LAST "</think>" is graded;
                an unclosed "<think>" drops everything from it to the end.
  normalize     NFKC, casefold, "**"/"__"/backticks removed, whitespace collapsed, surrounding quotes
                and trailing . , ; : ! removed. Used by normalized / contains_* matchers.
  extract       whole       : the stripped content
                answer_line : text after the LAST line starting with "ANSWER:" / "Answer:" / "ОТВЕТ:" /
                              "Ответ:" (markdown bold allowed); None when there is no such line
                last_line   : the last non-empty line
                regex       : group 1 (or the whole match) of extract_regex, last match wins
"""
from __future__ import annotations

import json
import math
import re
import unicodedata

from ev_schema import json_equal

MATCHERS = ("exact", "normalized", "regex", "contains_all", "contains_any", "number")
EXTRACTORS = ("whole", "answer_line", "last_line", "regex")

ANSWER_LINE_RE = re.compile(r"^[ \t>*_#-]*(?:\*\*|__)?(?:final[ \t]+|итоговый[ \t]+)?(?:answer|ответ)(?:\*\*|__)?"
                            r"[ \t]*[:：](?:\*\*|__)?[ \t]*(.*?)[ \t]*$", re.I | re.M)
# digit-group separators: space, NBSP (U+00A0), narrow NBSP (U+202F), underscore, apostrophe
GROUP_SEPS = " " + chr(0xA0) + chr(0x202F) + "_'"
MINUS_SIGN = chr(0x2212)
# digits, optional groups of exactly 3 digits after a separator or ',', optional decimal part
NUMBER_RE = re.compile("[-+" + MINUS_SIGN + r"]?\d+(?:[" + re.escape(GROUP_SEPS) + r",]\d{3}(?!\d))*(?:[.,]\d+)?")
_QUOTES = "\"'`«»“”„‘’"


def strip_think(content):
    s = content or ""
    if "</think>" in s:
        s = s.rsplit("</think>", 1)[1]
    elif "<think>" in s:
        s = s.split("<think>", 1)[0]
    return s.strip()


def normalize(s):
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = s.strip(_QUOTES + " ").rstrip(".,;:!").strip()
    return s


def extract(content, mode="whole", regex=None):
    """Returns (answer or None, how)."""
    s = strip_think(content)
    if mode == "whole":
        return s, "whole"
    if mode == "answer_line":
        found = ANSWER_LINE_RE.findall(s)
        found = [f for f in found if f.strip()]
        return (found[-1].strip(), "answer_line") if found else (None, "no ANSWER: line")
    if mode == "last_line":
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        return (lines[-1], "last_line") if lines else (None, "empty content")
    if mode == "regex":
        ms = list(re.finditer(regex, s, re.M))
        if not ms:
            return None, "extract_regex did not match"
        m = ms[-1]
        return (m.group(1) if m.groups() else m.group(0)).strip(), "regex"
    raise ValueError("unknown extract mode %r" % mode)


def parse_number(s):
    """First number in s, or None. Digit groups of exactly 3 digits may be separated by a space, NBSP,
    narrow NBSP, '_', "'" or ',' ('1 234 567' -> 1234567). A decimal comma is accepted ('1,5' -> 1.5),
    except that a token of the form '1,234' / '12,345,678' is read as thousands (English); with both
    ',' and '.' the commas are thousands separators ('1,234.5' -> 1234.5)."""
    m = NUMBER_RE.search(s or "")
    if not m:
        return None
    tok = m.group(0).replace(MINUS_SIGN, "-")
    for sep in GROUP_SEPS:
        tok = tok.replace(sep, "")
    if "," in tok and "." in tok:
        tok = tok.replace(",", "")
    elif "," in tok:
        body = tok.lstrip("+-")
        tok = tok.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", body) else tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def _flags(spec):
    f = 0
    for ch in spec.get("flags", ""):
        f |= {"i": re.I, "m": re.M, "s": re.S}[ch]
    return f


def match(answer, spec):
    """Grade one extracted answer against a matcher spec. Returns (ok, detail)."""
    t = spec["type"]
    if answer is None:
        return False, "no answer extracted"
    answers = spec.get("answers") or []
    if t == "exact":
        ok = any(answer.strip() == a for a in answers)
        return ok, "exact" if ok else "no exact match among %d answers" % len(answers)
    if t == "normalized":
        na = normalize(answer)
        ok = any(na == normalize(a) for a in answers)
        return ok, "normalized" if ok else "normalized answer %r matches none of %d answers" % (na[:120], len(answers))
    if t == "regex":
        fn = re.fullmatch if spec.get("fullmatch") else re.search
        ok = fn(spec["pattern"], answer, _flags(spec)) is not None
        return ok, "regex matched" if ok else "regex %r did not match" % spec["pattern"]
    if t in ("contains_all", "contains_any"):
        na = normalize(answer)
        hits = [normalize(x) in na for x in spec["values"]]
        ok = all(hits) if t == "contains_all" else any(hits)
        missing = [v for v, h in zip(spec["values"], hits) if not h]
        return ok, t if ok else "missing %s" % missing
    if t == "number":
        x = parse_number(answer)
        if x is None:
            return False, "no number in answer %r" % answer[:80]
        target = float(spec["value"])
        tol = max(float(spec.get("abs_tol", 0.0)), float(spec.get("rel_tol", 0.0)) * abs(target))
        ok = math.isfinite(x) and abs(x - target) <= tol
        return ok, "number %s vs %s (tol %s)" % (x, target, tol)
    raise ValueError("unknown matcher %r" % t)


def grade_text(content, spec):
    """extract + match. Returns dict(ok, extracted, how, detail)."""
    ans, how = extract(content, spec.get("extract", "whole"), spec.get("extract_regex"))
    ok, detail = match(ans, spec)
    return {"ok": ok, "extracted": None if ans is None else ans[:2000], "how": how, "detail": detail}


# ---------------------------------------------------------------------------- JSON answers
JSON_PARSE_OK = ("strict", "fenced", "span")     # extract_json "how" values that mean success
FENCE_JSON_RE = re.compile(r"```(?:json|JSON)?[ \t]*\n(.*?)\n[ \t]*```", re.S)


def extract_json(content, mode="strict"):
    """strict  : the whole content (after strip_think) must be one JSON value.
    lenient : also accept the first ```json fenced block, then the span from the first '{' or '['
              to the last '}' or ']'. Returns (value, how) or (None, error)."""
    s = strip_think(content)
    try:
        return json.loads(s), "strict"
    except ValueError as e:
        if mode == "strict":
            return None, "content is not valid JSON: %s" % e
    for m in FENCE_JSON_RE.finditer(s):
        try:
            return json.loads(m.group(1)), "fenced"
        except ValueError:
            pass
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if starts:
        a = min(starts)
        b = max(s.rfind("}"), s.rfind("]"))
        if b > a:
            try:
                return json.loads(s[a:b + 1]), "span"
            except ValueError:
                pass
    return None, "no JSON value found"


def pointer_get(obj, pointer):
    """RFC 6901 JSON pointer. Returns (found, value)."""
    if pointer in ("", "/"):
        return True, obj
    node = obj
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if part not in node:
                return False, None
            node = node[part]
        elif isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                return False, None
            node = node[int(part)]
        else:
            return False, None
    return True, node


VALUE_OPS = ("equals", "approx", "regex", "one_of", "length", "exists")


def check_value(obj, spec):
    """One value check {"path": "/a/0", <op>: ...}. Returns (ok, detail)."""
    path = spec["path"]
    found, v = pointer_get(obj, path)
    if "exists" in spec:
        return found == bool(spec["exists"]), "%s exists=%s" % (path, found)
    if not found:
        return False, "%s: not found" % path
    if "equals" in spec:
        ok = json_equal(v, spec["equals"])
        return ok, "%s == %r: %s" % (path, spec["equals"], ok)
    if "approx" in spec:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, "%s: not a number" % path
        ok = abs(v - spec["approx"]) <= float(spec.get("tol", 1e-9))
        return ok, "%s ~= %s (tol %s): got %s" % (path, spec["approx"], spec.get("tol", 1e-9), v)
    if "regex" in spec:
        ok = isinstance(v, str) and re.search(spec["regex"], v) is not None
        return ok, "%s =~ %r: %s" % (path, spec["regex"], ok)
    if "one_of" in spec:
        ok = any(json_equal(v, x) for x in spec["one_of"])
        return ok, "%s in %r: %s" % (path, spec["one_of"], ok)
    if "length" in spec:
        ok = isinstance(v, (list, str, dict)) and len(v) == spec["length"]
        return ok, "%s length == %s: %s" % (path, spec["length"], ok)
    raise ValueError("value check needs one of %s" % (VALUE_OPS,))
