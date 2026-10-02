#!/usr/bin/env python3
"""Vendor the DeepSeek-V4.1-Flash reference inference code byte-exact from Hugging Face.

Runs in GitHub Actions (job vendor-deepseek-ref): the agent sandbox cannot reach huggingface.co
(x-deny-reason: host_not_allowed), a GitHub-hosted runner can. Nothing here is typed by an agent:
every file is downloaded at one pinned revision and its git blob id is checked against the id the
Hugging Face tree API reports, so "byte-exact" is verified, not assumed.

Input:  third_party/deepseek-v41-flash/REQUEST  - one line: a 40-hex revision, or "main".
Output: third_party/deepseek-v41-flash/<same paths as upstream>, PIN.json, SHA256SUMS.
Results are reported as ::notice:: / ::error:: annotations (Actions logs are not readable from the
sandbox; annotations are, through the check-runs API).

Upstream is MIT (LICENSE is vendored next to the code). Vendored files are never edited in place:
changes live in patches/ or in overlay modules (AGENTS.md, Upstream pins).
"""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
DEST = "third_party/deepseek-v41-flash"
HF = "https://huggingface.co"
MAX_BYTES = 1 << 20                                   # source files only; weights are never vendored
TEXT_EXT = (".py", ".json", ".md", ".txt", ".sh", ".jinja")
TOP_LEVEL = ("config.json", "LICENSE", "README.md", "tokenizer_config.json", "generation_config.json")
REPORT_ONLY = ("tokenizer.json",)                     # size and hash reported, not vendored


def http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "408032gb-ci"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def git_blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def notice(msg):
    print("::notice::" + msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A"))


def main():
    want = open(os.path.join(DEST, "REQUEST")).read().split()[0]
    info = json.loads(http(f"{HF}/api/models/{REPO}/revision/{want}"))
    rev = info["sha"]
    if len(rev) != 40:
        raise RuntimeError(f"unexpected revision {rev!r}")

    top = {e["path"]: e for e in json.loads(http(f"{HF}/api/models/{REPO}/tree/{rev}"))}
    inf = json.loads(http(f"{HF}/api/models/{REPO}/tree/{rev}/inference?recursive=true"))

    wanted, skipped = [], []
    for name in TOP_LEVEL:
        e = top.get(name)
        if e is None or e["type"] != "file":
            skipped.append((name, "absent upstream"))
        elif "lfs" in e or e["size"] > MAX_BYTES:
            skipped.append((name, f"{e['size']} bytes, not a small text source"))
        else:
            wanted.append(e)
    for e in sorted(inf, key=lambda e: e["path"]):
        if e["type"] != "file":
            continue
        if "lfs" in e or e["size"] > MAX_BYTES or not e["path"].endswith(TEXT_EXT):
            skipped.append((e["path"], f"{e['size']} bytes, not a small text source"))
            continue
        wanted.append(e)

    files, total, oid_ok = [], 0, 0
    for e in wanted:
        path = e["path"]
        data = http(f"{HF}/{REPO}/resolve/{rev}/{path}")
        if len(data) != e["size"]:
            raise RuntimeError(f"{path}: {len(data)} bytes downloaded, tree API says {e['size']}")
        blob = git_blob_id(data)
        if "lfs" not in e:
            if blob != e["oid"]:
                raise RuntimeError(f"{path}: git blob id {blob} != upstream {e['oid']}")
            oid_ok += 1
        out = os.path.join(DEST, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        files.append({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob})
        total += len(data)

    report = []
    for name in REPORT_ONLY:
        e = top.get(name)
        if e:
            report.append({"path": name, "size": e["size"], "oid": e["oid"], "lfs_sha256": e.get("lfs", {}).get("oid")})

    with open(os.path.join(DEST, "SHA256SUMS"), "w") as f:
        for x in sorted(files, key=lambda x: x["path"]):
            f.write(f"{x['sha256']}  {x['path']}\n")
    with open(os.path.join(DEST, "PIN.json"), "w") as f:
        json.dump({"repo": REPO, "revision": rev, "requested": want, "last_modified": info.get("lastModified"),
                   "files": files, "not_vendored": [{"path": p, "why": w} for p, w in skipped],
                   "reported_only": report}, f, indent=1)
        f.write("\n")

    notice(f"vendored {REPO} @ {rev} (requested: {want}); {len(files)} files, {total} bytes; "
           f"git blob id verified against the HF tree API for {oid_ok}/{len(files)}")
    notice("files:\n" + "\n".join(f"{x['sha256'][:16]} {x['size']:>7} {x['path']}" for x in files))
    if skipped:
        notice("not vendored:\n" + "\n".join(f"{p}: {w}" for p, w in skipped))
    for r in report:
        notice(f"reported only: {r['path']} {r['size']} bytes, oid {r['oid']}, lfs sha256 {r['lfs_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        print(f"::error::HTTP {e.code} for {e.url}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - the annotation is the only channel the sandbox can read
        print(f"::error::{type(e).__name__}: {str(e)[:800]}")
        sys.exit(1)
