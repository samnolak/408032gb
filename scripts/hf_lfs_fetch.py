#!/usr/bin/env python3
"""Fetch the Hugging Face LFS files that a PIN.json pins (its "lfs_files", written by scripts/vendor_hf_files.py for a
request with "lfs_files"), verify them and keep them in a cache outside the repository.

The files (tokenizer.json, 13-20 MB) are never committed: PIN.json holds the repository, the 40-hex revision, path, size
and sha256 (= the LFS oid of that revision), which is all a reproducible fetch needs. Cache layout:
    <cache>/<sha256>/<basename>
<cache> = --cache, else $HF_LFS_CACHE, else ~/.cache/408032gb/hf-lfs (the path actions/cache saves in CI). A cached file
is re-hashed on every lookup; a file with another size or sha256 is never returned and is replaced on fetch.
Download URL: <endpoint>/<repo>/resolve/<revision>/<path>, endpoint = $HF_ENDPOINT or https://huggingface.co (the tests
point it at a local server).

Usage: hf_lfs_fetch.py [--cache DIR] [--offline] PIN.json [PIN.json ...]
       prints "<sha256>  <cache file>  <repo>:<path>" per file; exit 1 on any failure (::error:: annotation)
Library: path_for(pin_path, rel_path, cache=None) -> verified cache path or None (never downloads).
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "408032gb", "hf-lfs")


def cache_dir(cache=None):
    return cache or os.environ.get("HF_LFS_CACHE") or DEFAULT_CACHE


def entries(pin_path):
    with open(pin_path) as f:
        pin = json.load(f)
    for e in pin.get("lfs_files", []):
        if len(e["sha256"]) != 64 or len(pin["revision"]) != 40:
            raise ValueError(f"{pin_path}: {e['path']}: needs a 40-hex revision and a sha256")
        yield pin, e


def target(e, cache=None):
    return os.path.join(cache_dir(cache), e["sha256"], os.path.basename(e["path"]))


def verified(path, e):
    if not os.path.isfile(path) or os.path.getsize(path) != e["size"]:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == e["sha256"]


def path_for(pin_path, rel_path, cache=None):
    """The verified cache file of rel_path pinned by pin_path, or None (missing, unpinned or corrupt)."""
    if not os.path.isfile(pin_path):
        return None
    for _pin, e in entries(pin_path):
        if e["path"] == rel_path:
            p = target(e, cache)
            return p if verified(p, e) else None
    return None


def fetch(pin, e, cache=None):
    p = target(e, cache)
    if verified(p, e):
        return p, False
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{pin['repo']}/resolve/{pin['revision']}/{e['path']}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    h = hashlib.sha256()
    n = 0
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), prefix=".part-")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "408032gb-ci"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=300) as r:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                n += len(chunk)
                if n > e["size"]:
                    raise RuntimeError(f"{url}: more than the pinned {e['size']} B")
                h.update(chunk)
                out.write(chunk)
        if n != e["size"] or h.hexdigest() != e["sha256"]:
            raise RuntimeError(f"{url}: got {n} B sha256 {h.hexdigest()}, pinned {e['size']} B {e['sha256']}")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pins", nargs="+")
    ap.add_argument("--cache")
    ap.add_argument("--offline", action="store_true", help="never download; fail if a file is not cached")
    a = ap.parse_args(argv)
    try:
        n = 0
        for pin_path in a.pins:
            for pin, e in entries(pin_path):
                if a.offline:
                    p = target(e, a.cache)
                    if not verified(p, e):
                        raise RuntimeError(f"{pin['repo']}:{e['path']}: not in the cache ({p})")
                    got = False
                else:
                    p, got = fetch(pin, e, a.cache)
                print(f"{e['sha256']}  {p}  {pin['repo']}:{e['path']}{' (downloaded)' if got else ' (cached)'}")
                n += 1
        if n == 0:
            raise RuntimeError("no lfs_files in " + " ".join(a.pins))
    except Exception as ex:  # noqa: BLE001 - the annotation is the only channel the sandbox can read
        print(f"::error::{type(ex).__name__}: {str(ex)[:800]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
