#!/usr/bin/env python3
"""Backup git bundles of this repository that restore in an empty directory (.claude/skills/bundle-sync).

  create  --out FILE [--ref REF ...]   refuse a shallow clone (its bundle has no parents below the shallow boundary,
                                       yet `git bundle verify` passes inside that clone); bundle the refs (default: main
                                       and the current branch); then run `check` on the result
  check   FILE [--expect REF=SHA ...]  restore into a new empty directory: `git clone --mirror` of the bundle,
                                       `git fsck --full`, every ref's commit equal to the bundle header (and to
                                       --expect), the whole history of every ref readable, and a normal `git clone
                                       --branch` with a checkout; exit 1 on any failure

`git bundle verify` alone is not a restore test: in a shallow repository it treats the shallow boundary as a root and
reports "The bundle records a complete history" for a bundle that a fresh clone rejects ("did not send all necessary
objects"). Exit codes: 0 ok, 1 a check failed, 2 usage or a shallow repository.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle_heads(path):
    p = git("bundle", "list-heads", path)
    heads = {}
    for line in p.stdout.splitlines():
        sha, _, ref = line.partition(" ")
        if ref:
            heads[ref] = sha
    return p.returncode, heads, p.stderr.strip()


def check(path, expect=()):
    """Restore the bundle in a fresh empty directory -> list of (ok, what, detail)."""
    res = []
    path = os.path.abspath(path)
    rc, heads, err = bundle_heads(path)
    res.append((rc == 0 and bool(heads), "bundle header", "%d refs" % len(heads) if rc == 0 else err))
    if rc or not heads:
        return res
    for spec in expect:
        ref, _, sha = spec.partition("=")
        ref = ref if ref.startswith("refs/") else "refs/heads/" + ref
        res.append((heads.get(ref) == sha, "expected %s" % ref, "%s (want %s)" % (heads.get(ref), sha)))
    with tempfile.TemporaryDirectory(prefix="bundle-restore-") as tmp:
        assert not os.listdir(tmp)                     # the restore really starts from an empty directory
        mirror = os.path.join(tmp, "mirror.git")
        p = git("clone", "-q", "--mirror", path, mirror)
        res.append((p.returncode == 0, "git clone --mirror", "ok" if p.returncode == 0 else p.stderr.strip()[-300:]))
        if p.returncode:
            return res
        p = git("-C", mirror, "fsck", "--full", "--no-dangling")
        res.append((p.returncode == 0, "git fsck --full", "ok" if p.returncode == 0 else p.stderr.strip()[-300:]))
        for ref, sha in sorted(heads.items()):
            got = git("-C", mirror, "rev-parse", "--verify", "-q", ref + "^{commit}").stdout.strip()
            n = git("-C", mirror, "rev-list", "--count", ref)
            ok = got == sha and n.returncode == 0
            res.append((ok, "ref %s" % ref, "%s, %s commits" % (got[:12], n.stdout.strip()) if ok else
                        "got %r, header %s; %s" % (got, sha, n.stderr.strip()[-200:])))
        branch = next((r[len("refs/heads/"):] for r in sorted(heads) if r.startswith("refs/heads/")), None)
        if branch:
            work = os.path.join(tmp, "work")
            p = git("clone", "-q", "--branch", branch, path, work)
            head = git("-C", work, "rev-parse", "HEAD").stdout.strip() if p.returncode == 0 else ""
            ok = p.returncode == 0 and head == heads["refs/heads/" + branch]
            res.append((ok, "git clone --branch %s" % branch, "HEAD %s, checkout ok" % head[:12] if ok else
                        p.stderr.strip()[-300:] or "HEAD %s" % head))
    return res


def report(res):
    for ok, what, detail in res:
        print("%-4s %-46s %s" % ("ok" if ok else "FAIL", what, detail))
    bad = sum(not ok for ok, _w, _d in res)
    print("restore check: %s" % ("ok" if not bad else "%d check(s) failed" % bad))
    return 1 if bad else 0


def cmd_create(a):
    shallow = git("rev-parse", "--is-shallow-repository").stdout.strip()
    if shallow != "false":
        print("refusing: this repository is shallow (git rev-parse --is-shallow-repository = %r); a bundle made here "
              "cannot be restored. Run: git fetch --unshallow origin" % shallow, file=sys.stderr)
        return 2
    refs = a.ref or ["main", git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()]
    refs = list(dict.fromkeys(r for r in refs if r and r != "HEAD"))
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        print("%s exists: write a new file" % out, file=sys.stderr)
        return 2
    p = git("bundle", "create", out, *refs)
    if p.returncode:
        print(p.stderr, file=sys.stderr)
        return 1
    expect = ["%s=%s" % (r, git("rev-parse", r).stdout.strip()) for r in refs]
    print("bundle %s (%d bytes, sha256 %s) refs: %s" % (out, os.path.getsize(out), sha256_file(out), " ".join(expect)))
    return report(check(out, expect))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", required=True)
    c.add_argument("--ref", action="append", help="branch or ref to include (repeatable); default: main + current branch")
    k = sub.add_parser("check")
    k.add_argument("bundle")
    k.add_argument("--expect", action="append", default=[], metavar="REF=SHA")
    a = ap.parse_args(argv)
    if a.cmd == "create":
        return cmd_create(a)
    if not os.path.isfile(a.bundle):
        print("no such file: %s" % a.bundle, file=sys.stderr)
        return 2
    print("bundle %s (%d bytes, sha256 %s)" % (a.bundle, os.path.getsize(a.bundle), sha256_file(a.bundle)))
    return report(check(a.bundle, a.expect))


if __name__ == "__main__":
    sys.exit(main())
