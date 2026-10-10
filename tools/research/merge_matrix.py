#!/usr/bin/env python3
"""Validate and merge the research technology-matrix rows (docs/reassessment/research/matrix/R*.json).

Every research report Rn writes a JSON list of rows with exactly the keys in KEYS (BRIEF rule 5). This tool

  1. validates every row (keys, types, URL / SHA / status format, evidence labels on claim fields);
  2. merges rows that describe the same project (normalised repository URL, plus the explicit ALIASES below),
     keeping every source row and, when sources disagree, keeping every value and marking the disagreement;
  3. writes matrix/merged.json and renders docs/reassessment/research/MATRIX.md (overview table, conflicts,
     validation findings, one section per project).

Output is deterministic: no timestamps, everything sorted, the inputs identified by their sha256.
The agents' JSON files are never modified; invalid rows are reported, not repaired.

Python standard library only.

    python3 tools/research/merge_matrix.py            # write merged.json and MATRIX.md
    python3 tools/research/merge_matrix.py --check    # exit 1 if the outputs are not up to date
    python3 tools/research/merge_matrix.py --strict   # exit 1 if any row has a validation error
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESEARCH_DIR = os.path.join(ROOT, "docs", "reassessment", "research")
MATRIX_DIR = os.path.join(RESEARCH_DIR, "matrix")

KEYS = (
    "name", "url", "sha", "last_activity", "license", "models_supported", "hardware", "offload_methods",
    "scheduler", "kernels", "multi_gpu", "long_context_262k", "overlap_gpu_cpu_ssd", "benchmarks",
    "independent_confirmation", "limitations", "key_code", "portability_to_us", "adaptation_cost",
    "testable_without_target_hw", "potential_impact", "confidence", "status",
)
LABELS = ("VERIFIED", "REPORTED", "HYPOTHESIS", "UNKNOWN", "DISPROVED", "NOT FOUND")
# Factual fields: each value is a claim and must carry a label (BRIEF rule 3).
CLAIM_FIELDS = (
    "models_supported", "hardware", "offload_methods", "scheduler", "kernels", "multi_gpu",
    "long_context_262k", "overlap_gpu_cpu_ssd", "benchmarks", "independent_confirmation", "limitations",
)
# Judgements of the report author. By the BRIEF's definition they are HYPOTHESIS; an unlabelled value is read
# as HYPOTHESIS and only counted (warning), not treated as an invalid row.
ASSESSMENT_FIELDS = ("portability_to_us", "adaptation_cost", "testable_without_target_hw", "potential_impact")
STATUS_VALUES = ("active", "stale", "abandoned", "not-found")
OVERVIEW_FIELDS = ("potential_impact", "portability_to_us", "adaptation_cost", "testable_without_target_hw")

LABEL_RE = re.compile(r"(?<![A-Za-z-])(VERIFIED|REPORTED|HYPOTHESIS|UNKNOWN|DISPROVED|NOT FOUND)(?![A-Za-z-])")
NONCANONICAL_RE = re.compile(r"(?<![A-Za-z-])(NOT-FOUND|CONFIRMED|PROVISIONAL|Verified|Reported|Hypothesis|"
                             r"verified:|reported:|hypothesis:)(?![A-Za-z])")
URL_RE = re.compile(r"https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?::\d+)?(/\S*)?")
SHA_RE = re.compile(r"[0-9a-f]{40}")
HEX40_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
DATE_RE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")
NA_RE = re.compile(r"^\s*n/a\b", re.IGNORECASE)
ABSENCE_RE = re.compile(r"^\s*(none|none found|no)\s*$", re.IGNORECASE)
REPO_HOSTS = ("github.com", "gitlab.com", "codeberg.org", "bitbucket.org")
LICENSE_RE = re.compile(
    r"\b(MIT|Apache-2\.0|Apache 2\.0|AGPL-3\.0(?:-only|-or-later)?|LGPL-[23]\.[01]|GPL-[23]\.0|GPLv[23]|"
    r"BSD-[23]-Clause|BSD|MPL-2\.0|CC[- ]BY[A-Z0-9 .-]*|Unlicense|ISC)\b")
NO_LICENSE_RE = re.compile(r"\b(no licen[cs]e|none found|no LICENSE)\b", re.IGNORECASE)
POSITIVE = ("VERIFIED", "REPORTED")
NEGATIVE = ("DISPROVED", "NOT FOUND")

# Rows that describe one project under different identities. Each entry is evidence-backed; the key on the left
# is rewritten to the key on the right before grouping.
ALIASES = {
    # renamed/transferred repository: same HEAD 6769b27 under both names (R5 §2.1, R10 §2.7 `git ls-remote`).
    "github.com/0xsero/glm53-flash-offload": "github.com/sybil-solutions/glm-flash-lite",
    # R5 recorded ProMoE code as NOT FOUND; R8 cloned promoe-opensource/promoe @022f8b2; R12 re-checked ls-remote.
    "name:promoe": "github.com/promoe-opensource/promoe",
    # R5 recorded SpecPrefetch code as NOT FOUND; R12 found wei390/SpecPrefetch (linked from the HF model card).
    "name:specprefetch": "github.com/wei390/specprefetch",
}


# ----------------------------------------------------------------------------------------------- loading

def report_id(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"(R\d+)", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def report_sort_key(rid: str):
    m = re.match(r"R(\d+)$", rid)
    return (0, int(m.group(1)), "") if m else (1, 0, rid)


def report_markdown(rid: str, research_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(research_dir, rid + "-*.md")))
    return os.path.basename(hits[0]) if hits else None


def load_inputs(matrix_dir: str):
    """Return (inputs, rows, file_issues). rows: dicts with report/file/index/row."""
    inputs, rows, issues = [], [], []
    paths = sorted(glob.glob(os.path.join(matrix_dir, "R*.json")),
                   key=lambda p: (report_sort_key(report_id(p)), os.path.basename(p)))
    for path in paths:
        fname = os.path.basename(path)
        rid = report_id(path)
        with open(path, "rb") as fh:
            raw = fh.read()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(_issue(fname, None, None, "error", None, "E_JSON", f"not valid UTF-8 JSON: {exc}"))
            inputs.append({"file": fname, "report": rid, "rows": 0, "sha256": digest})
            continue
        if not isinstance(data, list):
            issues.append(_issue(fname, None, None, "error", None, "E_FILE_NOT_LIST", "top level is not a list"))
            inputs.append({"file": fname, "report": rid, "rows": 0, "sha256": digest})
            continue
        inputs.append({"file": fname, "report": rid, "rows": len(data), "sha256": digest})
        for i, row in enumerate(data):
            rows.append({"report": rid, "file": fname, "index": i, "row": row})
    return inputs, rows, issues


# ----------------------------------------------------------------------------------------------- validation

def _issue(fname, index, name, severity, field, code, message):
    return {"file": fname, "index": index, "name": name, "severity": severity, "field": field,
            "code": code, "message": message}


def leading_label(text: str):
    m = LABEL_RE.search(text)
    return m.group(1) if m else None


def has_label(text: str) -> bool:
    return LABEL_RE.search(text) is not None


def sha_ok(value: str) -> bool:
    v = value.strip()
    if SHA_RE.fullmatch(v):
        return True
    for word in ("UNKNOWN", "NOT FOUND"):
        if v == word or v.startswith(word + " ") or v.startswith(word + ":"):
            return True
    return False


def url_ok(value: str) -> bool:
    v = value.strip()
    if v == "NOT FOUND" or v.startswith("NOT FOUND ") or v.startswith("NOT FOUND:"):
        return True
    return URL_RE.fullmatch(v) is not None


def canonical_status(value: str):
    v = value.strip().lower()
    for s in STATUS_VALUES:
        if v == s or v.startswith(s + " ") or v.startswith(s + "(") or v.startswith(s + ";"):
            return s
    return None


def validate_row(fname: str, index: int, row) -> list:
    """All issues of one row (empty list = valid and clean)."""
    out = []
    if not isinstance(row, dict):
        return [_issue(fname, index, None, "error", None, "E_NOT_OBJECT", "row is not a JSON object")]
    name = row.get("name") if isinstance(row.get("name"), str) else None

    def add(sev, field, code, msg):
        out.append(_issue(fname, index, name, sev, field, code, msg))

    for k in KEYS:
        if k not in row:
            add("error", k, "E_MISSING_KEY", f"missing key '{k}'")
    for k in sorted(set(row) - set(KEYS)):
        add("error", k, "E_EXTRA_KEY", f"unexpected key '{k}'")

    for k in KEYS:
        if k not in row:
            continue
        v = row[k]
        if k == "key_code":
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                add("error", k, "E_TYPE", "key_code must be a list of strings")
            continue
        if not isinstance(v, str):
            add("error", k, "E_TYPE", f"'{k}' must be a string, got {type(v).__name__}")
            continue
        m = NONCANONICAL_RE.search(v)
        if m:
            add("warning", k, "W_NONCANONICAL_LABEL", f"non-canonical label '{m.group(1)}'")

    if isinstance(row.get("name"), str) and not row["name"].strip():
        add("error", "name", "E_EMPTY_NAME", "empty name")
    if isinstance(row.get("url"), str) and not url_ok(row["url"]):
        add("error", "url", "E_URL", f"url is neither a URL nor 'NOT FOUND': {row['url'][:60]!r}")
    if isinstance(row.get("sha"), str) and not sha_ok(row["sha"]):
        add("error", "sha", "E_SHA", f"sha is neither 40 hex nor UNKNOWN/NOT FOUND: {row['sha'][:60]!r}")
    status = row.get("status")
    if isinstance(status, str):
        cs = canonical_status(status)
        if cs is None:
            add("error", "status", "E_STATUS", f"status not one of {'/'.join(STATUS_VALUES)}: {status!r}")
        elif status.strip().lower() != cs:
            add("warning", "status", "W_STATUS_NOTE", f"status carries extra text: {status!r}")

    for k in CLAIM_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and not has_label(v) and not NA_RE.match(v):
            hint = " (absence: write 'NOT FOUND: ...')" if ABSENCE_RE.match(v) else ""
            add("error", k, "E_UNLABELED", f"claim without an evidence label{hint}")
    for k in ASSESSMENT_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and not has_label(v) and not NA_RE.match(v):
            add("warning", k, "W_IMPLICIT_HYPOTHESIS", "unlabelled judgement, read as HYPOTHESIS")

    la = row.get("last_activity")
    if isinstance(la, str) and not DATE_RE.search(la) and not has_label(la):
        add("warning", "last_activity", "W_META", "last_activity has neither a date nor a label")
    lic = row.get("license")
    if isinstance(lic, str) and not LICENSE_RE.search(lic) and not has_label(lic) and not NO_LICENSE_RE.search(lic):
        add("warning", "license", "W_META", "license names no known licence and carries no label")

    # consistency between identity fields
    url, sha = row.get("url"), row.get("sha")
    if isinstance(url, str) and isinstance(sha, str) and isinstance(status, str):
        cs = canonical_status(status)
        repo = repo_key(url) is not None
        if cs == "not-found" and repo and SHA_RE.fullmatch(sha.strip()):
            add("warning", "status", "W_STATUS_URL", "status not-found but a repository URL and SHA are given")
        if cs in ("active", "stale", "abandoned") and url.strip().startswith("NOT FOUND"):
            add("warning", "status", "W_STATUS_URL", f"status {cs} but url is NOT FOUND")
    kc = row.get("key_code")
    if isinstance(kc, list) and not kc:
        verified = any(isinstance(row.get(k), str) and "VERIFIED" in LABEL_RE.findall(row[k]) for k in CLAIM_FIELDS)
        if verified:
            add("warning", "key_code", "W_VERIFIED_NO_KEYCODE", "VERIFIED claims but key_code is empty")
    return out


# ----------------------------------------------------------------------------------------------- merge keys

def repo_key(url: str):
    """'github.com/owner/repo' (lower case) for a code-hosting URL, else None."""
    m = re.match(r"https?://(?:www\.)?([^/\s]+)/([^/\s]+)/([^/\s#?]+)", url.strip())
    if not m:
        return None
    host = m.group(1).lower()
    if host not in REPO_HOSTS:
        return None
    repo = m.group(3)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"{host}/{m.group(2).lower()}/{repo.lower()}"


def short_name(name: str) -> str:
    s = name
    for sep in (" (", " - ", " — ", " / ", ": "):
        s = s.split(sep)[0]
    return re.sub(r"[^a-z0-9]", "", s.lower())


def project_key(row: dict, aliases: dict) -> str:
    url = row.get("url") if isinstance(row.get("url"), str) else ""
    key = repo_key(url)
    if key is None:
        key = "name:" + short_name(row.get("name") if isinstance(row.get("name"), str) else "")
    seen = set()
    while key in aliases and key not in seen:
        seen.add(key)
        key = aliases[key]
    return key


# ----------------------------------------------------------------------------------------------- merge

def norm_value(v):
    if isinstance(v, list):
        return tuple(" ".join(str(x).split()) for x in v)
    if isinstance(v, str):
        return " ".join(v.split())
    return json.dumps(v, sort_keys=True)


def licence_ids(text: str) -> frozenset:
    ids = {m.group(1).replace("Apache 2.0", "Apache-2.0").replace("GPLv3", "GPL-3.0").replace("GPLv2", "GPL-2.0")
           for m in LICENSE_RE.finditer(text)}
    if not ids and NO_LICENSE_RE.search(text):
        ids = {"none"}
    return frozenset(ids)


def classify(field: str, groups: list) -> tuple:
    """(agreement, reason) for the distinct values of one field across sources."""
    if len(groups) <= 1:
        return "single", ""
    texts = [g["value"] if isinstance(g["value"], str) else " ".join(g["value"]) for g in groups]
    if field == "sha":
        sets = [frozenset(HEX40_RE.findall(t)) for t in texts]
        if any(sets) and not all(sets):
            return "conflict", "one source gives a commit, another found none"
        nonempty = [s for s in sets if s]
        if any(not (a & b) for i, a in enumerate(nonempty) for b in nonempty[i + 1:]):
            return "conflict", "different commits read (claims may refer to different snapshots)"
        return "multi", "same or overlapping commits / same absence, different notes"
    if field == "url":
        missing = [t.strip().startswith("NOT FOUND") for t in texts]
        repos = [repo_key(t) is not None for t in texts]
        if any(missing) and any(repos):
            return "conflict", "one source found a code repository, another found none"
        return "multi", "different URLs of one project (alias, paper vs code)"
    if field == "status":
        st = {canonical_status(t) for t in texts}
        return ("conflict", "different status") if len(st) > 1 else ("multi", "same status, different notes")
    if field == "license":
        ids = {licence_ids(t) for t in texts if licence_ids(t)}
        return ("conflict", "different licences") if len(ids) > 1 else ("multi", "same licence, different notes")
    if field not in CLAIM_FIELDS:
        return "multi", "per-source values"
    leads = [leading_label(t) for t in texts]
    pos = any(l in POSITIVE for l in leads)
    if pos and any(l == "DISPROVED" for l in leads):
        return "conflict", "one source DISPROVED what another reports"
    if any(l == "VERIFIED" for l in leads) and any(l == "NOT FOUND" for l in leads):
        return "conflict", "one source VERIFIED what another did NOT FIND"
    return "multi", "per-source values (complementary or not comparable)"


def merge(rows: list, aliases: dict) -> list:
    groups = {}
    for r in rows:
        row = r["row"]
        if not isinstance(row, dict):
            continue
        key = project_key(row, aliases)
        groups.setdefault(key, []).append(r)
    projects = []
    for key, members in groups.items():
        members.sort(key=lambda r: (report_sort_key(r["report"]), r["file"], r["index"]))
        sources = [{"report": r["report"], "file": r["file"], "index": r["index"],
                    "name": r["row"].get("name") if isinstance(r["row"].get("name"), str) else None}
                   for r in members]
        tags = [f"{r['report']}#{r['index']}" for r in members]
        fields, conflicts = {}, []
        for f in KEYS:
            if f == "name":
                continue
            distinct = []
            for tag, r in zip(tags, members):
                if f not in r["row"]:
                    continue
                v = r["row"][f]
                nv = norm_value(v)
                for g in distinct:
                    if g["_n"] == nv:
                        g["sources"].append(tag)
                        break
                else:
                    distinct.append({"_n": nv, "value": v, "sources": [tag]})
            for g in distinct:
                del g["_n"]
            agreement, reason = classify(f, distinct)
            fields[f] = {"agreement": agreement, "values": distinct}
            if reason:
                fields[f]["reason"] = reason
            if agreement == "conflict":
                conflicts.append(f)
        names = []
        for s in sources:
            if s["name"] and s["name"] not in names:
                names.append(s["name"])
        projects.append({
            "key": key,
            "name": names[0] if names else key,
            "names": names,
            "sources": sources,
            "fields": fields,
            "conflicts": conflicts,
        })
    projects.sort(key=lambda p: (p["name"].casefold(), p["key"]))
    used = {}
    for p in projects:
        base = re.sub(r"[^a-z0-9]+", "-", p["key"].replace("name:", "").lower()).strip("-") or "project"
        n = used.get(base, 0)
        used[base] = n + 1
        p["anchor"] = "p-" + base + (f"-{n}" if n else "")
    return projects


def build(matrix_dir: str, aliases: dict | None = None):
    aliases = ALIASES if aliases is None else aliases
    inputs, rows, issues = load_inputs(matrix_dir)
    for r in rows:
        issues.extend(validate_row(r["file"], r["index"], r["row"]))
    projects = merge(rows, aliases)
    by_row = {}
    for i in issues:
        if i["index"] is not None:
            by_row.setdefault((i["file"], i["index"]), []).append(i)
    order = {inp["file"]: n for n, inp in enumerate(inputs)}
    for p in projects:
        found = [(f"{s['report']}#{s['index']}", i) for s in p["sources"] for i in by_row.get((s["file"], s["index"]), [])]
        p["validation_errors"] = [f"{tag} {i['code']}:{i['field'] or '-'}" for tag, i in found if i["severity"] == "error"]
        p["validation_warnings"] = sum(1 for _, i in found if i["severity"] == "warning")
    issues.sort(key=lambda i: (order.get(i["file"], 1 << 30), -1 if i["index"] is None else i["index"],
                               0 if i["severity"] == "error" else 1, i["code"], i["field"] or ""))
    invalid = sorted({(i["file"], i["index"]) for i in issues if i["severity"] == "error" and i["index"] is not None},
                     key=lambda t: (order.get(t[0], 1 << 30), t[1]))
    invalid_files = sorted({i["file"] for i in issues if i["severity"] == "error" and i["index"] is None},
                           key=lambda f: order.get(f, 1 << 30))
    merged = {
        "generator": "tools/research/merge_matrix.py",
        "schema_keys": list(KEYS),
        "labels": list(LABELS),
        "claim_fields": list(CLAIM_FIELDS),
        "assessment_fields": list(ASSESSMENT_FIELDS),
        "aliases": dict(sorted(aliases.items())),
        "inputs": inputs,
        "summary": {
            "rows": len(rows),
            "projects": len(projects),
            "projects_with_several_sources": sum(1 for p in projects if len(p["sources"]) > 1),
            "projects_with_conflicts": sum(1 for p in projects if p["conflicts"]),
            "errors": sum(1 for i in issues if i["severity"] == "error"),
            "warnings": sum(1 for i in issues if i["severity"] == "warning"),
            "invalid_rows": len(invalid),
            "invalid_files": len(invalid_files),
        },
        "invalid_files": invalid_files,
        "invalid_rows": [{"file": f, "index": i} for f, i in invalid],
        "issues": issues,
        "projects": projects,
    }
    return merged


# ----------------------------------------------------------------------------------------------- rendering

def md_escape(text: str) -> str:
    t = " ".join(str(text).split())
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|")


def clip(text: str, n: int) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3].rstrip() + "..."


def cell(field_entry, width: int | None = None) -> str:
    vals = field_entry["values"]
    if not vals:
        return ""
    if len(vals) == 1:
        v = vals[0]["value"]
        text = "; ".join(v) if isinstance(v, list) else v
        return md_escape(clip(text, width) if width else text)
    parts = []
    for g in vals:
        v = g["value"]
        text = "; ".join(v) if isinstance(v, list) else v
        parts.append(f"[{','.join(g['sources'])}] " + md_escape(clip(text, width) if width else text))
    prefix = "**CONFLICT** " if field_entry["agreement"] == "conflict" else ""
    return prefix + "<br>".join(parts)


def render_md(merged: dict, research_dir: str = RESEARCH_DIR) -> str:
    s = merged["summary"]
    out = []
    w = out.append
    w("# Engine technology matrix (merged from the research reports)")
    w("")
    w("Generated by `python3 tools/research/merge_matrix.py` from `docs/reassessment/research/matrix/R*.json`. "
      "Do not edit by hand: change the source rows (or the report that owns them) and re-run the tool. "
      "`--check` verifies that this file and `matrix/merged.json` are up to date.")
    w("")
    w("Evidence labels are those of the source rows: VERIFIED / REPORTED / HYPOTHESIS / UNKNOWN / DISPROVED / "
      "NOT FOUND (BRIEF rule 3). The merge adds no claims of its own. Where two reports disagree, both values "
      "are kept and the field is marked **CONFLICT**; complementary values from several reports are listed per "
      "source as `[R5#3]` (report R5, row 3 of `matrix/R5.json`, counting from 0).")
    w("")
    w("## Inputs")
    w("")
    w("| file | report | rows | sha256 |")
    w("|---|---|---:|---|")
    for inp in merged["inputs"]:
        md = report_markdown(inp["report"], research_dir)
        rep = f"[{inp['report']}]({md})" if md else inp["report"]
        w(f"| `matrix/{inp['file']}` | {rep} | {inp['rows']} | `{inp['sha256'][:16]}` |")
    w("")
    w(f"Totals: {s['rows']} rows, {s['projects']} projects after merging, {s['projects_with_several_sources']} "
      f"projects described by more than one row, {s['projects_with_conflicts']} with at least one conflict; "
      f"validation: {s['errors']} errors in {s['invalid_rows']} rows, {s['warnings']} warnings "
      "(section [Validation findings](#validation-findings)).")
    w("")
    w("Merge rule: rows with the same code-hosting repository (`host/owner/repo`, case, `.git`, `/tree/<branch>` "
      "ignored) are one project; rows without a repository URL are grouped by the first words of their name; "
      "the explicit aliases below join renamed repositories and earlier NOT FOUND rows to the repository found "
      "later.")
    w("")
    w("| alias key | merged into |")
    w("|---|---|")
    for a, b in merged["aliases"].items():
        w(f"| `{a}` | `{b}` |")
    w("")
    w("## Overview")
    w("")
    w("Status and licence as recorded; the relevance columns are the report authors' judgements (HYPOTHESIS "
      "unless labelled). Cells are shortened here; the project sections below have the full values.")
    w("")
    w("| project | status | licence | impact for us | portability | adaptation cost | testable without target HW | sources |")
    w("|---|---|---|---|---|---|---|---|")
    for p in merged["projects"]:
        f = p["fields"]
        src = ", ".join(sorted({x["report"] for x in p["sources"]}, key=report_sort_key))
        name = f"[{md_escape(clip(p['name'], 60))}](#{p['anchor']})"
        cols = [cell(f["status"], 30), cell(f["license"], 40)] + [cell(f[k], 60) for k in OVERVIEW_FIELDS]
        w(f"| {name} | " + " | ".join(cols) + f" | {src} |")
    w("")
    w("## Conflicts between reports")
    w("")
    conflicted = [p for p in merged["projects"] if p["conflicts"]]
    if not conflicted:
        w("None.")
    else:
        w("| project | field | why | values |")
        w("|---|---|---|---|")
        for p in conflicted:
            for fld in p["conflicts"]:
                e = p["fields"][fld]
                w(f"| [{md_escape(clip(p['name'], 50))}](#{p['anchor']}) | `{fld}` | {md_escape(e.get('reason', ''))} | "
                  f"{cell(e, 90)} |")
    w("")
    w("## Validation findings")
    w("")
    w("Rules: all 23 keys present and no others; `key_code` a list of strings, every other value a string; `url` "
      "a URL or `NOT FOUND`; `sha` 40 lower-case hex or starting with `UNKNOWN` / `NOT FOUND`; `status` one of "
      f"{', '.join(STATUS_VALUES)}; the factual fields ({', '.join(CLAIM_FIELDS)}) carry at least one evidence "
      "label (a value starting with `n/a` makes no claim). Errors make a row invalid. Warnings: non-canonical "
      "labels (`NOT-FOUND`, `CONFIRMED`, lower case), unlabelled judgements in "
      f"{', '.join(ASSESSMENT_FIELDS)} (read as HYPOTHESIS), `last_activity` without a date, `license` naming no "
      "known licence, status inconsistent with url/sha, VERIFIED claims with an empty `key_code`. "
      "The source JSON files are not modified by this tool.")
    w("")
    codes = {}
    for i in merged["issues"]:
        codes.setdefault((i["file"], i["code"], i["severity"]), 0)
        codes[(i["file"], i["code"], i["severity"])] += 1
    all_codes = sorted({(c, sev) for (_, c, sev) in codes}, key=lambda t: (0 if t[1] == "error" else 1, t[0]))
    if all_codes:
        w("| file | " + " | ".join(f"{c}" for c, _ in all_codes) + " |")
        w("|---|" + "---:|" * len(all_codes))
        for inp in merged["inputs"]:
            w(f"| `{inp['file']}` | " + " | ".join(str(codes.get((inp["file"], c, sev), 0)) for c, sev in all_codes)
              + " |")
        w("")
    if merged["invalid_files"]:
        w("### Invalid files")
        w("")
        for f in merged["invalid_files"]:
            msgs = "; ".join(i["message"] for i in merged["issues"] if i["file"] == f and i["index"] is None)
            w(f"- `{f}`: {md_escape(msgs)}")
        w("")
    w("### Invalid rows (at least one error)")
    w("")
    errs = {}
    for i in merged["issues"]:
        if i["severity"] == "error" and i["index"] is not None:
            errs.setdefault((i["file"], i["index"]), []).append(i)
    if not merged["invalid_rows"]:
        w("None.")
    else:
        w("| row | name | errors |")
        w("|---|---|---|")
        for inv in merged["invalid_rows"]:
            items = errs[(inv["file"], inv["index"])]
            name = items[0]["name"] or ""
            by_code = {}
            for it in items:
                by_code.setdefault(it["code"], []).append(it["field"] or "-")
            desc = "; ".join(f"{c}: {', '.join(fs)}" for c, fs in sorted(by_code.items()))
            extra = [it["message"] for it in items if it["code"] in ("E_SHA", "E_URL", "E_STATUS", "E_JSON",
                                                                       "E_FILE_NOT_LIST", "E_TYPE")]
            if extra:
                desc += " (" + "; ".join(extra) + ")"
            w(f"| `{inv['file']}` #{inv['index']} | {md_escape(clip(name, 60))} | {md_escape(desc)} |")
    w("")
    w("### Warnings by row")
    w("")
    warns = {}
    for i in merged["issues"]:
        if i["severity"] == "warning" and i["code"] != "W_IMPLICIT_HYPOTHESIS":
            warns.setdefault((i["file"], i["index"]), []).append(i)
    implicit = sum(1 for i in merged["issues"] if i["code"] == "W_IMPLICIT_HYPOTHESIS")
    w(f"`W_IMPLICIT_HYPOTHESIS` ({implicit} values) is counted in the table above and not listed per row.")
    w("")
    if warns:
        w("| row | name | warnings |")
        w("|---|---|---|")
        for (f, idx), items in sorted(warns.items(), key=lambda kv: (
                [x["file"] for x in merged["inputs"]].index(kv[0][0]), -1 if kv[0][1] is None else kv[0][1])):
            name = items[0]["name"] or ""
            desc = "; ".join(f"{it['code']} ({it['field']}): {it['message']}" for it in items)
            w(f"| `{f}` #{idx} | {md_escape(clip(name, 60))} | {md_escape(clip(desc, 300))} |")
        w("")
    w("## Projects")
    w("")
    for p in merged["projects"]:
        w(f'<a id="{p["anchor"]}"></a>')
        w("")
        w(f"### {md_escape(p['name'])}")
        w("")
        w(f"Merge key `{p['key']}`.")
        w("")
        w("Sources:")
        for s in p["sources"]:
            md = report_markdown(s["report"], research_dir)
            rep = f"[{s['report']}]({md})" if md else s["report"]
            w(f"- [{s['report']}#{s['index']}] {rep}, `matrix/{s['file']}` row {s['index']}: "
              f"{md_escape(s['name'] or '')}")
        if p["conflicts"]:
            w("")
            w("Conflicting fields: " + ", ".join(f"`{c}`" for c in p["conflicts"]) + ".")
        if p["validation_errors"]:
            w("")
            w("Validation errors in the source rows: " + ", ".join(f"`{e}`" for e in p["validation_errors"]) + ".")
        w("")
        w("| field | value |")
        w("|---|---|")
        for f in KEYS:
            if f == "name":
                continue
            e = p["fields"].get(f)
            if e is None or not e["values"]:
                w(f"| {f} | (missing in every source row) |")
                continue
            if f == "key_code":
                parts = []
                for g in e["values"]:
                    items = g["value"] if isinstance(g["value"], list) else [g["value"]]
                    tag = f"[{','.join(g['sources'])}] " if len(e["values"]) > 1 else ""
                    parts.append(tag + ("<br>".join(f"`{md_escape(x).replace('`', '')}`" for x in items)
                                        if items else "(none)"))
                w(f"| key_code | " + "<br>".join(parts) + " |")
                continue
            w(f"| {f} | {cell(e)} |")
        w("")
    return "\n".join(out).rstrip() + "\n"


def dump_json(merged: dict) -> str:
    """Indented top level; one compact line per element of the long lists (small, diff-friendly)."""
    lines = ["{"]
    items = list(merged.items())
    for n, (k, v) in enumerate(items):
        comma = "," if n < len(items) - 1 else ""
        if k in ("issues", "projects", "invalid_rows") and isinstance(v, list) and v:
            lines.append(f" {json.dumps(k)}: [")
            for i, e in enumerate(v):
                lines.append("  " + json.dumps(e, ensure_ascii=False, separators=(",", ":"))
                             + ("," if i < len(v) - 1 else ""))
            lines.append(" ]" + comma)
        else:
            lines.append(f" {json.dumps(k)}: " + json.dumps(v, ensure_ascii=False, indent=1).replace("\n", "\n ")
                         + comma)
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matrix-dir", default=MATRIX_DIR)
    ap.add_argument("--json", dest="out_json", default=None, help="default: <matrix-dir>/merged.json")
    ap.add_argument("--md", dest="out_md", default=None, help="default: <matrix-dir>/../MATRIX.md")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the outputs differ")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any row has a validation error")
    args = ap.parse_args(argv)
    matrix_dir = os.path.abspath(args.matrix_dir)
    research_dir = os.path.dirname(matrix_dir)
    out_json = args.out_json or os.path.join(matrix_dir, "merged.json")
    out_md = args.out_md or os.path.join(research_dir, "MATRIX.md")
    merged = build(matrix_dir)
    j, m = dump_json(merged), render_md(merged, research_dir)
    s = merged["summary"]
    print(f"rows {s['rows']} projects {s['projects']} multi-source {s['projects_with_several_sources']} "
          f"conflicts {s['projects_with_conflicts']} errors {s['errors']} invalid-rows {s['invalid_rows']} "
          f"warnings {s['warnings']}")
    if args.check:
        stale = []
        for path, text in ((out_json, j), (out_md, m)):
            try:
                with open(path, encoding="utf-8") as fh:
                    if fh.read() != text:
                        stale.append(path)
            except FileNotFoundError:
                stale.append(path)
        for path in stale:
            print(f"out of date: {os.path.relpath(path, ROOT)}")
        if stale:
            return 1
    else:
        for path, text in ((out_json, j), (out_md, m)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"wrote {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")
    if args.strict and s["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
