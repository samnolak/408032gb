#!/usr/bin/env python3
"""Model weights logistics for an inference server that may have no internet (R1.3 of docs/reassessment/ROADMAP.md).

Three steps, stdlib only:
  manifest  (on a machine with internet) list the files of a Hugging Face repo at a PINNED revision with sizes and the LFS
            sha256 from the HF API, filtered by a glob -> JSON manifest committed to the repo / carried with the files
  download  (same machine, or the server if it has internet) stream the files with resume (HTTP Range), verify sha256
  verify    (on the server, offline) check every file of a manifest: size and sha256; exit 1 on any mismatch

Examples:
  python3 scripts/offline/weights.py manifest unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
      --glob 'UD-Q4_K_XL/*' -o weights/qwen38fn-ud-q4kxl.manifest.json
  python3 scripts/offline/weights.py download weights/qwen38fn-ud-q4kxl.manifest.json --dest /nvme0/models/qwen38fn
  python3 scripts/offline/weights.py verify weights/qwen38fn-ud-q4kxl.manifest.json --dest /nvme0/models/qwen38fn
A revision must be a 40-hex commit (never a branch name): the manifest must describe exactly one set of bytes.
HF_TOKEN in the environment is sent as a bearer token when set (gated repos).
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

API = os.environ.get("WEIGHTS_API_BASE", "https://huggingface.co")   # override only for tests
CHUNK = 8 << 20


def _req(url, headers=None):
    h = dict(headers or {})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return urllib.request.Request(url, headers=h)


def build_manifest(repo, revision, entries, pattern):
    """entries: the JSON list returned by GET /api/models/{repo}/tree/{revision}?recursive=1."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-hex commit id")
    files = []
    for e in entries:
        if e.get("type") != "file" or not fnmatch.fnmatch(e["path"], pattern):
            continue
        lfs = e.get("lfs") or {}
        files.append({"path": e["path"], "size": lfs.get("size", e.get("size")),
                      "sha256": lfs.get("oid"), "git_oid": e.get("oid")})
    if not files:
        raise ValueError(f"no files match {pattern!r}")
    files.sort(key=lambda f: f["path"])
    return {"repo": repo, "revision": revision, "glob": pattern, "files": files,
            "total_bytes": sum(f["size"] or 0 for f in files),
            "_label": "CONFIRMED from the Hugging Face API at the pinned revision; sha256 is the LFS oid "
                      "(git_oid only for small non-LFS files)"}


def fetch_tree(repo, revision):
    url = f"{API}/api/models/{repo}/tree/{revision}?recursive=1"
    with urllib.request.urlopen(_req(url), timeout=60) as r:
        return json.load(r)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(manifest, dest, log=print):
    bad = 0
    for f in manifest["files"]:
        p = os.path.join(dest, f["path"])
        if not os.path.exists(p):
            log(f"MISSING  {f['path']}")
            bad += 1
            continue
        size = os.path.getsize(p)
        if f["size"] is not None and size != f["size"]:
            log(f"SIZE     {f['path']}: {size} != {f['size']}")
            bad += 1
            continue
        if f["sha256"]:
            got = sha256_file(p)
            if got != f["sha256"]:
                log(f"SHA256   {f['path']}: {got} != {f['sha256']}")
                bad += 1
                continue
            log(f"OK       {f['path']} {size} sha256 {got}")
        else:
            log(f"OK-SIZE  {f['path']} {size} (no sha256 in manifest: not an LFS file)")
    log(f"verify: {len(manifest['files']) - bad}/{len(manifest['files'])} files OK")
    return bad == 0


def download(manifest, dest, log=print):
    for f in manifest["files"]:
        p = os.path.join(dest, f["path"])
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if f["size"] is not None and have == f["size"]:
            log(f"present  {f['path']}")
            continue
        url = f"{API}/{manifest['repo']}/resolve/{manifest['revision']}/{urllib.parse.quote(f['path'])}"
        headers = {"Range": f"bytes={have}-"} if have else {}
        t0 = time.time()
        with urllib.request.urlopen(_req(url, headers), timeout=120) as r, open(p, "ab" if have else "wb") as out:
            if have and r.status != 206:
                raise RuntimeError(f"{f['path']}: server ignored Range (status {r.status}); delete the partial file")
            n = 0
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                out.write(b)
                n += len(b)
        dt = max(time.time() - t0, 1e-9)
        log(f"fetched  {f['path']} +{n} bytes in {dt:.0f} s ({n / dt / 1e6:.0f} MB/s)")
    return verify(manifest, dest, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("repo")
    m.add_argument("--revision", required=True)
    m.add_argument("--glob", default="*")
    m.add_argument("-o", "--out", required=True)
    for name in ("download", "verify"):
        s = sub.add_parser(name)
        s.add_argument("manifest")
        s.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        man = build_manifest(a.repo, a.revision, fetch_tree(a.repo, a.revision), a.glob)
        with open(a.out, "w") as f:
            json.dump(man, f, indent=1)
        print(f"{len(man['files'])} files, {man['total_bytes'] / 1e9:.2f} GB -> {a.out}")
        return 0
    with open(a.manifest) as f:
        man = json.load(f)
    ok = download(man, a.dest) if a.cmd == "download" else verify(man, a.dest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
