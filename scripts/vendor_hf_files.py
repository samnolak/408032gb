#!/usr/bin/env python3
"""Vendor a few small text files of a Hugging Face repo, byte-exact, at one revision (runs in CI).

The agent sandbox cannot reach huggingface.co; a GitHub-hosted runner can. One directory per request:
    third_party/hf/<name>/REQUEST.json   {"repo": "org/name", "revision": "main" | 40 hex, "files": ["config.json", ...],
                                          optional "lfs_files": ["tokenizer.json"], "lfs_max_bytes": N}
Output next to it: the files, PIN.json (resolved revision, sizes, sha256, git blob ids), SHA256SUMS.
Git blob ids are checked against the HF tree API. Only files under 1 MiB that are not LFS objects are taken: never weights.

lfs_files (N23): LFS objects that are not weights (tokenizer.json), each at most lfs_max_bytes and never more than
LFS_MAX_BYTES (64 MiB); refused unless the path is in LFS_ALLOWED. Each is downloaded from resolve/<revision>, its size
and sha256 checked against the tree API's LFS pointer (lfs.size, lfs.oid = sha256 of the content), and only recorded in
PIN.json "lfs_files" (path, size, sha256, pointer git blob, URL): the bytes are NOT written to the repository. Tests and
CI fetch them on demand by that pin (scripts/hf_lfs_fetch.py, sha256-checked, cached outside the tree). A request with
lfs_files also records the license fields of the model card (cardData) next to the license tags.
Results go to ::notice:: / ::error:: annotations (the sandbox reads those through the check-runs API).
"""
import glob
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

HF = "https://huggingface.co"
MAX_BYTES = 1 << 20
LFS_MAX_BYTES = 64 << 20
LFS_ALLOWED = ("tokenizer.json",)   # basenames: tokenizers only, never weights


def http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "408032gb-ci"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def notice(msg):
    print("::notice::" + msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A"))


def one(request_path):
    dest = os.path.dirname(request_path)
    with open(request_path) as f:
        req = json.load(f)
    repo = req["repo"]
    info = json.loads(http(f"{HF}/api/models/{repo}/revision/{req.get('revision', 'main')}"))
    rev = info["sha"]
    tree = {e["path"]: e for e in json.loads(http(f"{HF}/api/models/{repo}/tree/{rev}?recursive=true"))}
    files, skipped = [], []
    for path in req["files"]:
        e = tree.get(path)
        if e is None or e["type"] != "file":
            skipped.append((path, "absent upstream"))
            continue
        if "lfs" in e or e["size"] > MAX_BYTES:
            skipped.append((path, f"{e['size']} bytes or LFS: not a small text file"))
            continue
        data = http(f"{HF}/{repo}/resolve/{rev}/{path}")
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != e["oid"]:
            raise RuntimeError(f"{repo}:{path}: git blob id {blob} != upstream {e['oid']}")
        out = os.path.join(dest, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        files.append({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob})
    lfs_files = [lfs_one(repo, rev, tree, path, req) for path in req.get("lfs_files", [])]
    listing = sorted((p, e["size"], "lfs" in e) for p, e in tree.items() if e["type"] == "file")
    with open(os.path.join(dest, "SHA256SUMS"), "w") as f:
        for x in files:
            f.write(f"{x['sha256']}  {x['path']}\n")
    pin = {"repo": repo, "revision": rev, "requested": req.get("revision", "main"),
           "gated": info.get("gated"), "license_tags": [t for t in info.get("tags", []) if t.startswith("license:")],
           "files": files, "not_vendored": [{"path": p, "why": w} for p, w in skipped],
           "upstream_listing": [{"path": p, "size": s, "lfs": l} for p, s, l in listing]}
    if "lfs_files" in req:
        card = info.get("cardData") or {}
        pin["card_license"] = {k: card.get(k) for k in ("license", "license_name", "license_link")}
        pin["lfs_files"] = lfs_files
    with open(os.path.join(dest, "PIN.json"), "w") as f:
        json.dump(pin, f, indent=1)
        f.write("\n")
    notice(f"{repo} @ {rev}: {len(files)} files vendored, git blob ids verified; gated={info.get('gated')}; "
           f"{[t for t in info.get('tags', []) if t.startswith('license:')]}")
    notice(f"{repo} files upstream:\n" + "\n".join(f"{s:>12} {'lfs ' if l else '    '}{p}" for p, s, l in listing))
    if skipped:
        notice(f"{repo} not vendored:\n" + "\n".join(f"{p}: {w}" for p, w in skipped))
    for x in lfs_files:
        notice(f"{repo} @ {rev}: LFS {x['path']} {x['size']} B sha256 {x['sha256']} = LFS oid, pinned in PIN.json, "
               f"not committed (fetch: scripts/hf_lfs_fetch.py)")
    if "lfs_files" in req:
        notice(f"{repo} card license: {pin['card_license']}")


def lfs_one(repo, rev, tree, path, req):
    """Download one LFS object of the pinned revision, check its size and sha256 against the LFS pointer of the tree
    API, and return its PIN.json record. Raises on anything unexpected (absent, not LFS, not allowed, too big)."""
    if os.path.basename(path) not in LFS_ALLOWED:
        raise RuntimeError(f"{repo}:{path}: not an allowed LFS file {LFS_ALLOWED}")
    e = tree.get(path)
    if e is None or e["type"] != "file" or "lfs" not in e:
        raise RuntimeError(f"{repo}:{path}: not an LFS file at {rev}")
    size, oid = int(e["lfs"].get("size", e["size"])), e["lfs"]["oid"]
    cap = min(int(req.get("lfs_max_bytes", LFS_MAX_BYTES)), LFS_MAX_BYTES)
    if size > cap:
        raise RuntimeError(f"{repo}:{path}: {size} B over the cap {cap} B")
    url = f"{HF}/{repo}/resolve/{rev}/{path}"
    data = http(url)
    sha = hashlib.sha256(data).hexdigest()
    if len(data) != size or sha != oid:
        raise RuntimeError(f"{repo}:{path}: got {len(data)} B sha256 {sha}, LFS pointer says {size} B {oid}")
    return {"path": path, "size": size, "sha256": sha, "lfs_oid": oid, "pointer_git_blob": e["oid"], "url": url,
            "stored": "not committed: fetched on demand by scripts/hf_lfs_fetch.py"}


if __name__ == "__main__":
    try:
        for r in sorted(glob.glob("third_party/hf/*/REQUEST.json")):
            one(r)
    except urllib.error.HTTPError as e:
        print(f"::error::HTTP {e.code} for {e.url}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - the annotation is the only channel the sandbox can read
        print(f"::error::{type(e).__name__}: {str(e)[:800]}")
        sys.exit(1)
