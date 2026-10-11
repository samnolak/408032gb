#!/usr/bin/env python3
"""Read only the GGUF headers of Hugging Face files, with HTTP Range requests (runs in CI: workflow gguf-headers).

The agent sandbox cannot reach huggingface.co; a GitHub-hosted runner can. Stdlib only. A request is written by an
agent under data/gguf-headers/<name>/REQUEST.json; its results go to third_party/gguf-headers/<name>/ (CI-fetched
upstream bytes, never typed by an agent: third_party/README.md; agents' Edit/Write tools are blocked there).

  data/gguf-headers/<name>/REQUEST.json, kind "headers" (the default):
    {"repo": "org/name", "revision": "main" | <40 hex>, "files": ["<fnmatch glob over the repo path>", ...],
     "expect_count": N, "max_header_bytes": <= 64 MiB (default 64 MiB), "initial_bytes": (default 1 MiB),
     "analysis": {...}  (read by tools/gguf_headers/analyze.py only)}
  kind "discover" (lists repos through the Hub search API and/or by name, reads no file):
    {"kind": "discover", "searches": [{"search": "GLM-5.3-Flash", "architectures": ["glm5-next", "glm5next"]}],
     "limit": 100, "list_files": true,
     "repos": [{"repo": "org/name", "revision": "main" | <40 hex>}, ...]}   (searches and/or repos, at least one)
    A named repo is pinned like a headers request (revision API; a 40-hex request must come back unchanged; 401/403/404
    recorded as "repo_unavailable") and its tree listed: every .gguf path with size and LFS sha256. Named repos are
    listed first and count against the same per-request listing cap as the search matches.

A request is pending while third_party/gguf-headers/<name>/ has no PIN.json (headers) / LISTING.json (discover).
For a headers request:
  1. the revision is resolved through /api/models/<repo>/revision/<rev> and pinned to the commit sha (a 40-hex request
     must come back unchanged); a 401/403/404 there is recorded as status "repo_unavailable" (PIN.json, not an error);
  2. the files are listed through /api/models/<repo>/tree/<sha>?recursive=true (paginated by the Link header) and
     matched against the globs; no match is recorded as status "no_match" with the repo's *.gguf listing; any other
     count than expect_count is an error;
  3. for each file, /<repo>/resolve/<sha>/<path> is read with Range requests starting at initial_bytes and growing
     (at least doubling) until tools/gguf_headers/gguf_header.py parses magic, version, KV pairs and tensor infos up
     to the aligned data offset. A request never goes past max_header_bytes or the file size: a header that needs
     more is an error ("cap exceeded"), never a full download. Every response must be 206 with a Content-Range for
     exactly the bytes asked for and the file's total size from the tree listing; a 200 is rejected before its body
     is read. At most the asked-for bytes are read from any response.
Outputs in third_party/gguf-headers/<name>/ (committed by the workflow; written only when every file succeeded):
  <file basename>.header.gz      the bytes [0, data_offset) of the file, gzip -9 with mtime 0 (deterministic);
  <file basename>.summary.json   gguf_header.summarize() of those bytes (all KV; arrays over 4096 elements as count
                                 and sha256; the tensor list: name, ggml type, shape, offset, bytes);
  PIN.json                       repo, requested and resolved revision, per file: path, size and LFS sha256 from the
                                 tree listing, header bytes and their sha256, bytes fetched, the Range requests, the
                                 hub's X-Linked-Etag / X-Repo-Commit / X-Linked-Size of the resolve redirect;
  ERROR.json                     instead of the above when a request fails (it stays pending; the job fails).
--plan checks every pending request against the caps and prints the worst-case download, fetching nothing; the
workflow runs it first, and the fetch runs it again. Annotations (::notice:: / ::warning:: / ::error::) are the
channel the sandbox reads through the check-runs API.

Usage: python3 scripts/gguf_headers_fetch.py [--plan] [--requests DIR] [--out DIR] [--endpoint URL] [REQUEST.json ...]
  (a REQUEST.json given on the command line writes to <--out>/<name of its directory>/)
"""
import argparse
import fnmatch
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "gguf_headers"))
import gguf_header  # noqa: E402

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HARD_CAP = 64 << 20           # per file: no request may set max_header_bytes above this
DEFAULT_INITIAL = 1 << 20
MIN_INITIAL = 4096
MAX_FILES = 16                # per request
RUN_CAP = 1 << 30             # worst case over all pending requests of one run
MAX_SEARCHES, MAX_LIMIT, MAX_LISTED_REPOS = 8, 200, 25
API_MAX_BYTES = 32 << 20      # any JSON answer of the hub
TIMEOUT = 120
RETRIES = 3
SLEEP = time.sleep
USER_AGENT = "408032gb-ci-gguf-headers"
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REV_RE = re.compile(r"^(main|[0-9a-f]{40})$")


class FetchError(Exception):
    pass


def annotate(level, msg):
    print(f"::{level}::" + msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A"), flush=True)


# ---------------------------------------------------------------------------------------------------------- HTTP

class _Recorder(urllib.request.HTTPRedirectHandler):
    """Follows redirects like urllib does (the Range header goes along) and keeps every hop's status and headers."""

    def __init__(self):
        self.hops = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.hops.append({"status": code, "url": req.full_url, "headers": dict(headers.items())})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url, headers):
    rec = _Recorder()
    opener = urllib.request.build_opener(rec)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    return opener.open(req, timeout=TIMEOUT), rec.hops


def _retrying(fn):
    for i in range(RETRIES):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == RETRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == RETRIES - 1:
                raise
        SLEEP(5 * (i + 1))


def api_json(url):
    """GET a JSON answer of the hub; returns (object, response headers)."""
    def go():
        r, _ = _open(url, {"Accept": "application/json"})
        with r:
            body = r.read(API_MAX_BYTES + 1)
            if len(body) > API_MAX_BYTES:
                raise FetchError(f"{url}: answer larger than {API_MAX_BYTES} bytes")
            return json.loads(body), dict(r.headers.items())
    return _retrying(go)


def _next_link(headers):
    for part in (headers.get("Link") or headers.get("link") or "").split(","):
        m = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part)
        if m:
            return m.group(1)
    return None


def range_get(url, start, end, total):
    """Bytes [start, end] of a file of `total` bytes, or FetchError. Reads at most end - start + 1 bytes."""
    n = end - start + 1

    def go():
        r, hops = _open(url, {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with r:
            if r.status != 206:
                raise FetchError(f"HTTP {r.status} instead of 206 for a Range request (body not read)")
            cr = r.headers.get("Content-Range", "")
            m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr.strip())
            if not m or int(m.group(1)) != start or int(m.group(2)) != end or m.group(3) != str(total):
                raise FetchError(f"Content-Range {cr!r} does not match bytes {start}-{end}/{total}")
            data = r.read(n + 1)
            if len(data) != n:
                raise FetchError(f"got {len(data)} bytes for a {n}-byte range")
            return data, hops
    return _retrying(go)


# ---------------------------------------------------------------------------------------------------------- requests

def load_request(path):
    with open(path) as f:
        req = json.load(f)
    kind = req.get("kind", "headers")
    problems = []
    if kind == "headers":
        if not REPO_RE.match(str(req.get("repo", ""))):
            problems.append(f"repo {req.get('repo')!r} is not org/name")
        if not REV_RE.match(str(req.get("revision", "main"))):
            problems.append(f"revision {req.get('revision')!r} is neither main nor a 40-hex sha")
        files = req.get("files")
        if not isinstance(files, list) or not files or not all(isinstance(g, str) and g for g in files):
            problems.append("files must be a non-empty list of globs")
        elif any(g.startswith("/") or ".." in g for g in files):
            problems.append("globs must be relative repo paths")
        n = req.get("expect_count")
        if not isinstance(n, int) or not 1 <= n <= MAX_FILES:
            problems.append(f"expect_count must be an integer in [1, {MAX_FILES}]")
        cap = req.get("max_header_bytes", HARD_CAP)
        if not isinstance(cap, int) or not MIN_INITIAL <= cap <= HARD_CAP:
            problems.append(f"max_header_bytes {cap} must be in [{MIN_INITIAL}, {HARD_CAP}] (hard cap 64 MiB)")
        init = req.get("initial_bytes", DEFAULT_INITIAL)
        if not isinstance(init, int) or not MIN_INITIAL <= init <= (cap if isinstance(cap, int) else HARD_CAP):
            problems.append(f"initial_bytes {init} must be in [{MIN_INITIAL}, max_header_bytes]")
    elif kind == "discover":
        s, named = req.get("searches"), req.get("repos")
        if s is None and named is None:
            problems.append("a discover request needs searches and/or repos")
        if s is not None and (not isinstance(s, list) or not 1 <= len(s) <= MAX_SEARCHES or
                              not all(isinstance(x, dict) and isinstance(x.get("search"), str) and x["search"]
                                      for x in s)):
            problems.append(f"searches must be 1..{MAX_SEARCHES} objects with a non-empty 'search'")
        if named is not None:
            if not isinstance(named, list) or not 1 <= len(named) <= MAX_LISTED_REPOS or \
                    not all(isinstance(x, dict) and REPO_RE.match(str(x.get("repo", ""))) and
                            REV_RE.match(str(x.get("revision", "main"))) for x in named):
                problems.append(f"repos must be 1..{MAX_LISTED_REPOS} objects with an org/name 'repo' and a "
                                "'revision' main or 40-hex")
            elif len({x["repo"] for x in named}) != len(named):
                problems.append("repos must not repeat a repo")
        lim = req.get("limit", 100)
        if not isinstance(lim, int) or not 1 <= lim <= MAX_LIMIT:
            problems.append(f"limit must be in [1, {MAX_LIMIT}]")
    else:
        problems.append(f"unknown kind {kind!r}")
    if problems:
        raise FetchError(f"{path}: " + "; ".join(problems))
    req["kind"] = kind
    return req


def result_name(req):
    return "LISTING.json" if req["kind"] == "discover" else "PIN.json"


def out_dir(request_path, out_root):
    """third_party/gguf-headers/<name> for data/gguf-headers/<name>/REQUEST.json."""
    return os.path.join(out_root, os.path.basename(os.path.dirname(os.path.abspath(request_path))))


def pending(paths, out_root):
    out = []
    for p in paths:
        req = load_request(p)
        if not os.path.exists(os.path.join(out_dir(p, out_root), result_name(req))):
            out.append((p, req))
    return out


def plan(todo):
    """Worst-case bytes per request and in total; FetchError when a cap is broken. Fetches nothing."""
    rows, total = [], 0
    for p, req in todo:
        if req["kind"] == "discover":
            rows.append(f"{p}: discover, {len(req.get('searches') or [])} searches, "
                        f"{len(req.get('repos') or [])} named repos, reads no file")
            continue
        cap = req.get("max_header_bytes", HARD_CAP)
        worst = cap * req["expect_count"]
        total += worst
        rows.append(f"{p}: {req['repo']} @ {req.get('revision', 'main')}, globs {req['files']}, "
                    f"{req['expect_count']} files x cap {cap} B = at most {worst} B")
    if total > RUN_CAP:
        raise FetchError(f"pending requests could fetch {total} B, over the per-run cap {RUN_CAP} B")
    return rows, total


# ---------------------------------------------------------------------------------------------------------- headers

def fetch_header(url, size, cap, initial):
    """Header bytes [0, data_offset) of the file at url, by growing Range reads. Returns (header bytes, Header, info)."""
    buf = bytearray()
    want = min(initial, cap, size)
    ranges, hops0 = [], None
    while True:
        if want > len(buf):
            data, hops = range_get(url, len(buf), want - 1, size)
            ranges.append([len(buf), want - 1])
            if hops0 is None:
                hops0 = hops
            buf += data
        try:
            h = gguf_header.parse(buf)
            break
        except gguf_header.NeedMore as e:
            if e.need > size:
                raise FetchError(f"the file ({size} B) ends inside its header (needs {e.need} B)") from None
            if e.need > cap:
                raise FetchError(f"cap exceeded: the header needs at least {e.need} B, the cap is {cap} B "
                                 f"({len(buf)} B fetched)") from None
            want = min(max(e.need, 2 * len(buf)), cap, size)
    info = {"bytes_fetched": len(buf), "range_requests": ranges, "redirect_hops": hops0 or []}
    return bytes(buf[:h.data_offset]), h, info


def _hop_header(hops, name):
    for hop in hops:
        for k, v in hop["headers"].items():
            if k.lower() == name.lower():
                return v
    return None


def list_tree(repo, sha):
    url = f"{ENDPOINT}/api/models/{repo}/tree/{sha}?recursive=true"
    entries, pages = [], 0
    while url:
        page, headers = api_json(url)
        entries += page
        pages += 1
        url = _next_link(headers)
        if url and not url.startswith(ENDPOINT + "/"):
            raise FetchError(f"tree pagination left the endpoint: {url}")
        if pages > 50:
            raise FetchError("tree listing over 50 pages")
    return [e for e in entries if e.get("type") == "file"]


def _lfs_sha(e):
    lfs = e.get("lfs") or {}
    return lfs.get("oid") or lfs.get("sha256")


def headers_request(path, req, out_dir):
    """Fetch one headers request into out_dir. Returns the PIN dict (status ok / repo_unavailable / no_match)."""
    repo, requested = req["repo"], req.get("revision", "main")
    cap, initial = req.get("max_header_bytes", HARD_CAP), req.get("initial_bytes", DEFAULT_INITIAL)
    pin = {"request": req, "repo": repo, "requested_revision": requested, "max_header_bytes": cap,
           "initial_bytes": initial, "endpoint": ENDPOINT, "tool": "scripts/gguf_headers_fetch.py",
           "workflow_sha": os.environ.get("GITHUB_SHA"), "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        info, _ = api_json(f"{ENDPOINT}/api/models/{repo}/revision/{requested}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            body = e.read(2048).decode("utf-8", errors="replace")
            pin.update(status="repo_unavailable", http_status=e.code, http_body=body, revision=None, files=[])
            annotate("warning", f"{repo}: HTTP {e.code} for the revision API ({body[:200]}); recorded, nothing fetched")
            return pin
        raise
    sha = info.get("sha")
    if not SHA_RE.match(str(sha)):
        raise FetchError(f"{repo}: revision API returned no commit sha")
    if SHA_RE.match(requested) and sha != requested:
        raise FetchError(f"{repo}: asked for {requested}, the hub answered {sha}")
    pin.update(revision=sha, gated=info.get("gated"),
               license_tags=[t for t in info.get("tags", []) if str(t).startswith("license:")])
    tree = list_tree(repo, sha)
    gguf_listing = sorted(({"path": e["path"], "size": e.get("size"), "lfs_sha256": _lfs_sha(e)}
                           for e in tree if e["path"].endswith(".gguf")), key=lambda x: x["path"])
    pin["gguf_listing"] = gguf_listing
    matched = sorted((e for e in tree if any(fnmatch.fnmatchcase(e["path"], g) for g in req["files"])),
                     key=lambda e: e["path"])
    if not matched:
        pin.update(status="no_match", files=[])
        annotate("warning", f"{repo} @ {sha}: no file matches {req['files']}; recorded with the repo's "
                            f"{len(gguf_listing)} .gguf paths, nothing fetched")
        return pin
    if len(matched) != req["expect_count"]:
        raise FetchError(f"{repo} @ {sha}: {len(matched)} files match {req['files']}, expected "
                         f"{req['expect_count']}: {[e['path'] for e in matched]}")
    if len({os.path.basename(e["path"]) for e in matched}) != len(matched):
        raise FetchError(f"{repo}: matched files share a basename")
    files = []
    for e in matched:
        size = e.get("size")
        if not isinstance(size, int) or size <= 0:
            raise FetchError(f"{repo}:{e['path']}: no size in the tree listing")
        if (e.get("lfs") or {}).get("size") not in (None, size):
            raise FetchError(f"{repo}:{e['path']}: tree size {size} != LFS size {e['lfs']['size']}")
        url = f"{ENDPOINT}/{repo}/resolve/{sha}/{urllib.parse.quote(e['path'], safe='/')}"
        hdr, h, got = fetch_header(url, size, cap, initial)
        base = os.path.basename(e["path"])
        summary = gguf_header.summarize(h, hdr)
        gz = gzip.compress(hdr, compresslevel=9, mtime=0)
        with open(os.path.join(out_dir, base + ".header.gz"), "wb") as f:
            f.write(gz)
        with open(os.path.join(out_dir, base + ".summary.json"), "w") as f:
            f.write(gguf_header.dumps(summary))
        lfs_sha = _lfs_sha(e)
        etag = _hop_header(got["redirect_hops"], "X-Linked-Etag")
        etag = etag.strip().strip('"') if etag else None
        commit = _hop_header(got["redirect_hops"], "X-Repo-Commit")
        rec = {"path": e["path"], "size": size, "lfs_sha256": lfs_sha, "lfs_size": (e.get("lfs") or {}).get("size"),
               "git_oid": e.get("oid"),
               "xet_hash": e.get("xetHash"), "header_bytes": len(hdr),
               "header_sha256": hashlib.sha256(hdr).hexdigest(), "header_gz": base + ".header.gz",
               "header_gz_bytes": len(gz), "header_gz_sha256": hashlib.sha256(gz).hexdigest(),
               "summary": base + ".summary.json", "bytes_fetched": got["bytes_fetched"],
               "range_requests": got["range_requests"],
               "resolve_redirect": {"x_linked_etag": etag, "x_repo_commit": commit,
                                    "x_linked_size": _hop_header(got["redirect_hops"], "X-Linked-Size"),
                                    "hops": len(got["redirect_hops"])},
               "etag_matches_lfs_sha256": (etag == lfs_sha) if (etag and lfs_sha) else None,
               "repo_commit_matches": (commit == sha) if commit else None,
               "tensor_offsets": summary["tensor_offsets"],
               "size_check": gguf_header.size_check(summary, size)}
        for flag, what in (("etag_matches_lfs_sha256", "X-Linked-Etag != LFS sha256"),
                           ("repo_commit_matches", "X-Repo-Commit != pinned revision")):
            if rec[flag] is False:
                annotate("warning", f"{repo}:{e['path']}: {what}")
        if not rec["size_check"].startswith("exact"):
            annotate("warning", f"{repo}:{e['path']}: {rec['size_check']} (tensor offsets {rec['tensor_offsets']})")
        files.append(rec)
        meta = {k["key"]: k.get("value") for k in summary["kv"]}
        tmpl = meta.get("tokenizer.chat_template")
        annotate("notice", f"{repo}:{e['path']}: GGUF v{h.version}, arch {meta.get('general.architecture')!r}, "
                           f"{h.n_kv} KV, {h.n_tensors} tensors, header {len(hdr)} B "
                           f"(sha256 {rec['header_sha256'][:16]}), {got['bytes_fetched']} B fetched in "
                           f"{len(got['range_requests'])} Range requests; size check {rec['size_check']}; "
                           f"chat_template {'absent' if tmpl is None else hashlib.sha256(tmpl.encode()).hexdigest()[:16]}")
    pin.update(status="ok", files=files, bytes_fetched=sum(f["bytes_fetched"] for f in files),
               committed_bytes=sum(f["header_gz_bytes"] for f in files))
    return pin


# ---------------------------------------------------------------------------------------------------------- discover

def list_named_repo(spec):
    """A named repo of a discover request: revision pinned like a headers request, .gguf paths with size and LFS sha256
    from the tree listing. 401/403/404 of the revision API is recorded, not raised. Reads no file."""
    repo, rev = spec["repo"], spec.get("revision", "main")
    row = {"repo": repo, "requested_revision": rev}
    try:
        info, _ = api_json(f"{ENDPOINT}/api/models/{repo}/revision/{rev}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            body = e.read(2048).decode("utf-8", errors="replace")
            row.update(status="repo_unavailable", http_status=e.code, http_body=body, revision=None)
            annotate("warning", f"discover {repo}: HTTP {e.code} for the revision API; recorded")
            return row
        raise
    sha = info.get("sha")
    if not SHA_RE.match(str(sha)):
        raise FetchError(f"{repo}: revision API returned no commit sha")
    if rev != "main" and sha != rev:
        raise FetchError(f"{repo}: asked for {rev}, the hub answered {sha}")
    g = info.get("gguf") or {}
    tmpl = g.get("chat_template")
    tree = list_tree(repo, sha)
    row.update(status="ok", revision=sha, lastModified=info.get("lastModified"), gated=info.get("gated"),
               gguf_architecture=g.get("architecture"), gguf_total_params=g.get("total"),
               gguf_chat_template_sha256=hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
               gguf_files=sorted(({"path": e["path"], "size": e.get("size"), "lfs_sha256": _lfs_sha(e)}
                                  for e in tree if e["path"].endswith(".gguf")), key=lambda x: x["path"]))
    return row


def discover_request(path, req):
    out = {"request": req, "endpoint": ENDPOINT, "tool": "scripts/gguf_headers_fetch.py",
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "searches": []}
    if req.get("repos"):
        out["repos"] = [list_named_repo(spec) for spec in req["repos"]]
        annotate("notice", "discover named repos: " + ", ".join(
            f"{r['repo']} @ {str(r['revision'])[:12]} ({len(r['gguf_files'])} .gguf, {r['gguf_architecture']})"
            if r["status"] == "ok" else f"{r['repo']} ({r['status']})" for r in out["repos"]))
    listed = len(out.get("repos", []))
    named = {r["repo"] for r in out.get("repos", [])}
    for s in req.get("searches") or []:
        q = {"search": s["search"], "filter": "gguf", "limit": str(req.get("limit", 100))}
        expand = ["gguf", "sha", "lastModified", "downloads", "gated", "tags"]
        url = f"{ENDPOINT}/api/models?{urllib.parse.urlencode(q)}&" + urllib.parse.urlencode([("expand", x) for x in expand])
        try:
            items, _ = api_json(url)
            expanded = True
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            url = f"{ENDPOINT}/api/models?{urllib.parse.urlencode(q)}"
            items, _ = api_json(url)
            expanded = False
        want = set(s.get("architectures", []))
        rows = []
        for it in items:
            g = it.get("gguf") or {}
            tmpl = g.get("chat_template")
            row = {"id": it.get("id") or it.get("modelId"), "sha": it.get("sha"), "lastModified": it.get("lastModified"),
                   "downloads": it.get("downloads"), "gated": it.get("gated"),
                   "gguf_architecture": g.get("architecture"), "gguf_total_params": g.get("total"),
                   "gguf_context_length": g.get("context_length"),
                   "gguf_chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
                   "arch_match": g.get("architecture") in want if want else None}
            if req.get("list_files") and row["arch_match"] and listed < MAX_LISTED_REPOS and row["sha"] and \
                    REPO_RE.match(str(row["id"])) and row["id"] not in named:
                tree = list_tree(row["id"], row["sha"])
                row["gguf_files"] = sorted(({"path": e["path"], "size": e.get("size")} for e in tree
                                            if e["path"].endswith(".gguf")), key=lambda x: x["path"])
                listed += 1
            rows.append(row)
        out["searches"].append({"search": s["search"], "architectures": sorted(want), "url": url,
                                "expanded": expanded, "n_results": len(rows), "results": rows})
        annotate("notice", f"discover {s['search']!r}: {len(rows)} GGUF repos; architecture in {sorted(want)}: " +
                 ", ".join(f"{r['id']} ({r['gguf_architecture']})" for r in rows if r["arch_match"]))
    return out


# ---------------------------------------------------------------------------------------------------------- main

def _request_ref(path):
    with open(path, "rb") as f:
        raw = f.read()
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    if rel.startswith(".."):
        rel = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
    return {"request_path": rel,
            "request_sha256": hashlib.sha256(raw).hexdigest()}


def run_one(path, req, dest):
    os.makedirs(dest, exist_ok=True)
    if os.path.exists(os.path.join(dest, "ERROR.json")):
        os.remove(os.path.join(dest, "ERROR.json"))
    if req["kind"] == "discover":
        res = {**_request_ref(path), **discover_request(path, req)}
        with open(os.path.join(dest, "LISTING.json"), "w") as f:
            f.write(gguf_header.dumps(res))
        return
    tmp = tempfile.mkdtemp(prefix="gguf-headers-")
    try:
        pin = {**_request_ref(path), **headers_request(path, req, tmp)}
        for name in sorted(os.listdir(tmp)):
            shutil.move(os.path.join(tmp, name), os.path.join(dest, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(dest, "PIN.json"), "w") as f:
        f.write(gguf_header.dumps(pin))
    annotate("notice", f"{path}: status {pin['status']}, revision {pin.get('revision')}, "
                       f"{len(pin.get('files', []))} headers, {pin.get('bytes_fetched', 0)} B fetched, "
                       f"{pin.get('committed_bytes', 0)} B committed (gzip)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requests", nargs="*")
    ap.add_argument("--requests", dest="req_root", default=os.path.join(ROOT, "data", "gguf-headers"))
    ap.add_argument("--out", dest="out_root", default=os.path.join(ROOT, "third_party", "gguf-headers"))
    ap.add_argument("--endpoint", default=None, help="hub base URL (default $HF_ENDPOINT or https://huggingface.co)")
    ap.add_argument("--plan", action="store_true", help="check the pending requests against the caps, fetch nothing")
    a = ap.parse_args(argv)
    global ENDPOINT
    if a.endpoint:
        ENDPOINT = a.endpoint.rstrip("/")
    paths = a.requests or sorted(glob.glob(os.path.join(a.req_root, "*", "REQUEST.json")))
    try:
        todo = pending(paths, a.out_root)
        rows, total = plan(todo)
    except (FetchError, ValueError, OSError) as e:
        annotate("error", f"{type(e).__name__}: {e}")
        return 1
    for r in rows:
        print(r)
    print(f"pending requests: {len(todo)}; worst-case download {total} B (per-file hard cap {HARD_CAP} B, "
          f"per-run cap {RUN_CAP} B)")
    if a.plan or not todo:
        if not todo:
            annotate("notice", "nothing pending: every REQUEST.json already has its result")
        return 0
    failed = 0
    for p, req in todo:
        dest = out_dir(p, a.out_root)
        try:
            run_one(p, req, dest)
        except Exception as e:  # noqa: BLE001 - recorded per request; the annotation is what the sandbox reads
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError):
                msg = f"HTTP {e.code} for {e.url}"
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "ERROR.json"), "w") as f:
                f.write(gguf_header.dumps({**_request_ref(p), "request": req, "error": msg, "endpoint": ENDPOINT,
                                           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            annotate("error", f"{p}: {msg[:800]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
