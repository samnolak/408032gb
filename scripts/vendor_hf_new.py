#!/usr/bin/env python3
"""Vendor only the Hugging Face requests that have not been vendored yet (no PIN.json next to their REQUEST.json).

For the vendor-hf-branch workflow, which commits back to the claude/* branch it ran on: unlike vendor-hf (pushes to
main), it must not re-vendor directories that are already pinned. The fetching itself is scripts/vendor_hf_files.py's
one(), unchanged. Usage: python3 scripts/vendor_hf_new.py [<REQUEST.json> ...]   (default: every pending request)
"""
import glob
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vendor_hf_files  # noqa: E402


def pending(paths):
    return [p for p in paths if not os.path.exists(os.path.join(os.path.dirname(p), "PIN.json"))]


def main(argv):
    todo = pending(argv or sorted(glob.glob("third_party/hf/*/REQUEST.json")))
    if not todo:
        vendor_hf_files.notice("nothing pending: every REQUEST.json already has a PIN.json")
        return 0
    try:
        for r in todo:
            vendor_hf_files.one(r)
    except urllib.error.HTTPError as e:
        print(f"::error::HTTP {e.code} for {e.url}")
        return 1
    except Exception as e:  # noqa: BLE001 - the annotation is the only channel the sandbox can read
        print(f"::error::{type(e).__name__}: {str(e)[:800]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
