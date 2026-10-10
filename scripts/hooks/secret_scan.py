#!/usr/bin/env python3
"""Secret scanner for what an agent is about to publish (docs/reassessment/ADR-012.1-git-publication-policy.md).

Scans the lines ADDED by commits that are not yet on the remote (git log -p ... --not --remotes=origin) and the names of
files those commits add. Used by scripts/hooks/guard_bash.py before it allows a push; also runnable by hand:

  python3 scripts/hooks/secret_scan.py                 # commits of HEAD that are not on origin
  python3 scripts/hooks/secret_scan.py --ref claude/x  # commits of a branch that are not on origin
  python3 scripts/hooks/secret_scan.py --all           # every line of every tracked file at HEAD

Exit 0 = nothing found, 1 = findings (printed without the secret value), 2 = git error.
The patterns are deliberately narrow (known token formats, private keys, credential files): a false positive blocks a
push, a false negative publishes a secret, so a finding is reported with its file and line for a human to judge.
"""
import argparse
import os
import re
import subprocess
import sys

PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")),
    ("AWS access key id", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("Anthropic API key", re.compile(r"(?<![A-Za-z0-9_-])sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI-style API key", re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_-]{35}")),
    ("Slack token", re.compile(r"(?<![A-Za-z0-9_])xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("credentials in URL", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]{6,}@[A-Za-z0-9.-]+")),
]
BAD_FILES = re.compile(r"(^|/)(\.env(\.[^/]*)?|id_(rsa|dsa|ecdsa|ed25519)|[^/]*\.(pem|p12|pfx|key)|\.netrc|credentials\.json)$")
ALLOW_FILES = re.compile(r"(^|/)\.env\.(example|sample|template)$")


def scan_line(line):
    """Names of the patterns found in one line (values are never returned)."""
    return [name for name, rx in PATTERNS if rx.search(line)]


def scan_patch(text):
    """Findings in `git log -p --format=commit:%H` output: (commit, file, line number in new file, pattern)."""
    out, commit, path, lineno = [], None, None, 0
    for raw in text.splitlines():
        if raw.startswith("commit:"):
            commit, path = raw[7:].strip(), None
            continue
        if raw.startswith("+++ "):
            p = raw[4:]
            path = p[2:] if p.startswith("b/") else (None if p == "/dev/null" else p)
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            lineno = int(m.group(1)) if m else 0
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            for name in scan_line(raw[1:]):
                out.append((commit, path, lineno, name))
            lineno += 1
        elif raw.startswith(" "):
            lineno += 1
    return out


def bad_file_names(names):
    return [n for n in names if BAD_FILES.search(n) and not ALLOW_FILES.search(n)]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def scan_outgoing(repo, ref="HEAD", remote="origin"):
    """Findings for commits reachable from `ref` and not from any remote-tracking ref of `remote`."""
    rng = [ref, "--not", "--remotes=%s" % remote]
    patch = git(repo, "log", "-p", "--no-color", "--unified=0", "--no-ext-diff", "--format=commit:%H", *rng)
    added = git(repo, "log", "--diff-filter=A", "--name-only", "--format=", *rng).split()
    findings = [(c, p, n, name) for c, p, n, name in scan_patch(patch)]
    findings += [(None, f, 0, "credential-like file name") for f in bad_file_names(added)]
    return findings


def scan_tree(repo):
    findings = []
    for f in git(repo, "ls-files").splitlines():
        if bad_file_names([f]):
            findings.append((None, f, 0, "credential-like file name"))
        try:
            with open(os.path.join(repo, f), encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    for name in scan_line(line):
                        findings.append((None, f, i, name))
        except (IsADirectoryError, FileNotFoundError):
            pass
    return findings


def report(findings):
    for c, p, n, name in findings:
        print("secret-scan: %s in %s:%s%s" % (name, p, n, (" (commit %s)" % c[:12]) if c else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--all", action="store_true", help="scan every tracked file at HEAD instead of outgoing commits")
    a = ap.parse_args(argv)
    try:
        findings = scan_tree(a.repo) if a.all else scan_outgoing(a.repo, a.ref, a.remote)
    except subprocess.CalledProcessError as e:
        print("secret-scan: git error: %s" % (e.stderr or e).strip(), file=sys.stderr)
        return 2
    report(findings)
    if not findings:
        print("secret-scan: clean")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
