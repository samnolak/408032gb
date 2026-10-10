#!/usr/bin/env python3
"""Download one GGUF of a Hugging Face repo at a pinned revision and print a JSON manifest line (repo, revision sha,
file, size, sha256). The file is chosen by a regex over the repo listing, so the exact name is recorded, not assumed.
Needs network access to huggingface.co (GitHub Actions; the agent sandbox has none) and `pip install huggingface_hub`.
Usage: python3 scripts/traces/fetch_hf_gguf.py --repo unsloth/X-GGUF --regex '^X-Q4_K_M\\.gguf$' --dir models/ [--revision sha]
"""
import argparse
import hashlib
import json
import os
import re
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--regex", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--revision", default=None, help="commit sha; default: the repo's current main, resolved and recorded")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(a.repo, revision=a.revision, files_metadata=True)
    rx = re.compile(a.regex)
    cands = [s for s in info.siblings if rx.search(s.rfilename)]
    if len(cands) != 1:
        names = [s.rfilename for s in info.siblings if s.rfilename.endswith(".gguf")]
        print(json.dumps({"error": f"{len(cands)} files match {a.regex!r}", "gguf_files": names}), file=sys.stderr)
        return 1
    s = cands[0]
    path = hf_hub_download(a.repo, s.rfilename, revision=info.sha, local_dir=a.dir)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    lfs = getattr(s, "lfs", None)
    lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    out = {"repo": a.repo, "revision": info.sha, "file": s.rfilename, "path": os.path.abspath(path),
           "size": os.path.getsize(path), "sha256": h.hexdigest(), "hf_lfs_sha256": lfs_sha,
           "sha256_matches_hf": (lfs_sha == h.hexdigest()) if lfs_sha else None}
    print(json.dumps(out))
    return 0 if out["sha256_matches_hf"] in (True, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
