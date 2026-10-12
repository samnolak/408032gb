#!/usr/bin/env python3
"""Vendor only the Hugging Face requests that have not been vendored yet (no PIN.json in their destination).

For the vendor-hf-branch workflow, which commits back to the claude/* branch it ran on: unlike vendor-hf (pushes to
main), it must not re-vendor directories that are already pinned. The fetching itself is scripts/vendor_hf_files.py's
one(), unchanged. Usage: python3 scripts/vendor_hf_new.py [<REQUEST.json> ...]   (default: every pending request)

Where requests live: agents write data/hf-requests/<name>/REQUEST.json, because scripts/hooks/guard-upstream.sh keeps
third_party/ read-only for them; the request is copied to third_party/hf/<name>/REQUEST.json and the files are vendored
next to it there, by the workflow only. A request already under third_party/hf/<name>/ (the convention of vendor-hf.yml
on main) is vendored in place.
"""
import glob
import os
import shutil
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vendor_hf_files  # noqa: E402

REQUEST_GLOBS = ("third_party/hf/*/REQUEST.json", "data/hf-requests/*/REQUEST.json")


def dest_of(request_path):
    """Directory the request is vendored into: <root>/third_party/hf/<name> for <root>/data/hf-requests/<name>/
    REQUEST.json, the request's own directory otherwise."""
    name_dir = os.path.dirname(os.path.abspath(request_path))
    requests_dir = os.path.dirname(name_dir)
    if os.path.basename(requests_dir) == "hf-requests" and os.path.basename(os.path.dirname(requests_dir)) == "data":
        root = os.path.dirname(os.path.dirname(requests_dir))
        return os.path.join(root, "third_party", "hf", os.path.basename(name_dir))
    return name_dir


def pending(paths):
    return [p for p in paths if not os.path.exists(os.path.join(dest_of(p), "PIN.json"))]


def stage(request_path):
    """The REQUEST.json that vendor_hf_files.one() reads (it writes the files and PIN.json into that file's directory):
    a data/hf-requests/ request is copied into its destination first."""
    dest = dest_of(request_path)
    if dest == os.path.dirname(os.path.abspath(request_path)):
        return request_path
    os.makedirs(dest, exist_ok=True)
    staged = os.path.join(dest, "REQUEST.json")
    shutil.copyfile(request_path, staged)
    return staged


def main(argv):
    todo = pending(argv or sorted(p for g in REQUEST_GLOBS for p in glob.glob(g)))
    if not todo:
        vendor_hf_files.notice("nothing pending: every REQUEST.json already has a PIN.json")
        return 0
    try:
        for r in todo:
            vendor_hf_files.one(stage(r))
    except urllib.error.HTTPError as e:
        print(f"::error::HTTP {e.code} for {e.url}")
        return 1
    except Exception as e:  # noqa: BLE001 - the annotation is the only channel the sandbox can read
        print(f"::error::{type(e).__name__}: {str(e)[:800]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
