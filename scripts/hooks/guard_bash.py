#!/usr/bin/env python3
"""PreToolUse guard for Bash commands (called by scripts/hooks/guard-bash.sh; JSON hook input on stdin, exit 2 blocks).

Policy: docs/reassessment/ADR-012.1-git-publication-policy.md (customer decision 2026-10-10), summarised in AGENTS.md
"Git and sync":
  * an agent may push ONLY a plain fast-forward of a `claude/*` branch to `origin` = github.com/samnolak/408032gb:
    `git push [-u|--set-upstream] origin [HEAD:][refs/heads/]claude/<name>`, from the project repository itself;
  * before such a push the commits that are not on origin yet are scanned for secrets (scripts/hooks/secret_scan.py);
  * blocked: pushes to main or any other branch, other remotes or URLs, force / +refspec / --mirror / --delete /
    --tags / --all / any other push option, git options before `push` (-C, -c), pushes from another directory or
    through wrappers, nested shells or env-prefixed commands, git aliases, changing remotes or push-related config,
    history rewrites (reset --hard, rebase -i, filter-branch, filter-repo), merges, deletions and other writes through
    `gh` or the GitHub REST API;
  * the isolated inference VM has no remote: it still syncs by git bundle (skill bundle-sync).
The check is deliberately strict: anything it cannot classify as an allowed push is blocked.
"""
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import secret_scan  # noqa: E402

OWN_REPO = re.compile(r"github\.com[/:]samnolak/408032gb(\.git)?/?$")
BRANCH = re.compile(r"^(HEAD:)?(refs/heads/)?claude/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
SEP = {"&&", "||", ";", "|", "&"}
REWRITE = re.compile(r"(^|[^A-Za-z0-9_-])git(\s+-\S+(\s+[^-\s]\S*)?)*\s+"
                     r"(reset\s+(\S+\s+)*--hard|filter-branch|filter-repo|rebase\s+(\S+\s+)*(-i|--interactive)\b|"
                     r"push\s[^;&|]*(--force|\s-f\b|\s\+))")
CONFIG_KEYS = re.compile(r"^(remote|url|push|branch|alias)\.|^core\.(hookspath|sshcommand)$", re.I)
CONFIG_READ = {"--get", "--get-all", "--get-regexp", "--list", "-l", "--get-urlmatch", "--show-origin", "--show-scope"}
WRAPPERS = {"bash", "sh", "zsh", "dash", "ksh", "fish", "eval", "ssh", "su", "sudo", "env", "xargs", "nohup", "timeout",
            "nice", "exec", "command", "time", "script", "watch", "parallel", "stdbuf", "setsid", "flock", "chronic"}
NESTED_PUSH = re.compile(r"(^|[^A-Za-z0-9_./-])git\b[^;&|]*\bpush\b")
GIT_OPTS_WITH_VALUE = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env")


def project_root():
    return os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(HERE)))


REDIRECT = re.compile(r"^\d*(>>?|<<?|&>>?|>&|<&|>\|)(.*)$")


def segments(cmd):
    """Split a shell command into simple commands (token lists), also on separators glued to words.
    Redirections (`2>&1`, `> file`, `&>/dev/null`, ...) are dropped together with their target."""
    toks = shlex.split(cmd.replace("\n", " ; "), posix=True)
    kept, skip = [], False
    for t in toks:
        if skip:
            skip = False
            continue
        m = REDIRECT.match(t)
        if m:
            skip = m.group(2) == ""
            continue
        kept.append(t)
    segs, cur = [], []
    for t in kept:
        for p in re.split(r"(&&|\|\||;|\||&)", t):
            if p == "":
                continue
            if p in SEP:
                segs.append(cur)
                cur = []
            else:
                cur.append(p)
    segs.append(cur)
    return [s for s in segs if s]


def git_out(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def check_push_target(repo, ref):
    """Conditions on the repository a push runs in. Returns a block reason or None."""
    rc, top = git_out(repo, "rev-parse", "--show-toplevel")
    if rc != 0 or os.path.realpath(top) != project_root():
        return "pushes are allowed only from the project repository %s (got %s)" % (project_root(), top or repo)
    # get-url --push expands insteadOf/pushInsteadOf, so a rewrite to another host or repository fails this check
    rc, url = git_out(top, "remote", "get-url", "--push", "origin")
    if rc != 0 or not OWN_REPO.search(url):
        return "origin must be github.com/samnolak/408032gb (got %r)" % url
    for key in ("remote.origin.push", "remote.origin.mirror"):
        rc, val = git_out(top, "config", "--get-all", key)
        if val:
            return "%s is configured (%s); refspec rewrites are not allowed" % (key, val)
    src = "HEAD" if ref.startswith("HEAD:") else ref
    if src != "HEAD":
        name = src if src.startswith("refs/heads/") else "refs/heads/" + src
        rc, _ = git_out(top, "show-ref", "--verify", "--quiet", name)
        if rc != 0:
            return "%s is not a local branch" % name
    try:
        findings = secret_scan.scan_outgoing(top, src, "origin")
    except subprocess.CalledProcessError as e:
        return "secret scan failed (%s)" % (e.stderr or e).strip()
    if findings:
        return "secret scan: " + "; ".join("%s in %s:%s" % (n, p, l) for _, p, l, n in findings[:5])
    return None


def check_git_segment(s, cwd, run_checks):
    """Verdict for one simple command that starts with `git`: a block reason or None."""
    sub_i = 1
    while sub_i < len(s) and s[sub_i].startswith("-"):
        if s[sub_i] in ("-c",) and sub_i + 1 < len(s) and CONFIG_KEYS.search(s[sub_i + 1].split("=", 1)[0]):
            return "remote/url/push/branch/alias config overrides are not allowed"
        sub_i += 2 if s[sub_i] in GIT_OPTS_WITH_VALUE else 1
    sub = s[sub_i] if sub_i < len(s) else ""
    if sub == "push":
        if sub_i != 1:
            return "git options before push are not allowed"
        args = s[2:]
        flags = [a for a in args if a.startswith("-")]
        pos = [a for a in args if not a.startswith("-")]
        if any(f not in ("-u", "--set-upstream") for f in flags):
            return "push options other than -u/--set-upstream are not allowed"
        if len(pos) != 2 or pos[0] != "origin" or not BRANCH.match(pos[1]):
            return "only `git push [-u] origin claude/<branch>` is allowed"
        return check_push_target(cwd, pos[1]) if run_checks else None
    if run_checks and sub and not sub.startswith("-"):
        rc, alias = git_out(cwd, "config", "--get", "alias." + sub)
        if rc == 0 and alias:
            return "git aliases are not allowed (alias.%s)" % sub
    if sub == "remote" and len(s) > sub_i + 1 and s[sub_i + 1] in (
            "add", "set-url", "rename", "remove", "rm", "set-head", "set-branches", "prune"):
        return "changing remotes is not allowed"
    if sub == "config":
        rest = s[sub_i + 1:]
        if not any(a in CONFIG_READ for a in rest) and any(CONFIG_KEYS.search(a) for a in rest if not a.startswith("-")):
            return "changing remote/url/push/branch/alias config is not allowed"
    return None


def check_gh_segment(s):
    if len(s) >= 3 and s[1] == "pr" and s[2] == "merge":
        return "merging pull requests is the owner's decision"
    if len(s) >= 3 and s[1] == "repo" and s[2] in ("delete", "fork", "create", "rename", "archive", "edit", "sync"):
        return "gh repo %s is not allowed" % s[2]
    if len(s) >= 3 and s[1] == "release" and s[2] in ("create", "delete", "upload", "edit"):
        return "gh release changes are not allowed"
    if len(s) >= 2 and s[1] == "api":
        method = "GET"
        for i, a in enumerate(s):
            if a in ("-X", "--method") and i + 1 < len(s):
                method = s[i + 1].upper()
            elif a.startswith("--method="):
                method = a.split("=", 1)[1].upper()
            elif a.startswith("-X") and len(a) > 2:
                method = a[2:].upper()
            elif a in ("-f", "-F", "--field", "--raw-field", "--input") and method == "GET":
                method = "POST"
        if method != "GET":
            return "gh api %s is not allowed (read-only GET only)" % method
    return None


def check_http_segment(s):
    if not any("github.com" in t for t in s):
        return None
    if os.path.basename(s[0]) == "curl":
        for i, a in enumerate(s):
            if a in ("-X", "--request") and i + 1 < len(s) and s[i + 1].upper() != "GET":
                return "non-GET requests to GitHub are not allowed"
            if a.startswith("-X") and len(a) > 2 and a[2:].upper() != "GET":
                return "non-GET requests to GitHub are not allowed"
            if a in ("-d", "--data", "--data-raw", "--data-binary", "--json", "-F", "--form", "-T", "--upload-file"):
                return "requests with a body to GitHub are not allowed"
    if os.path.basename(s[0]) == "wget" and any(a.startswith(("--method", "--post")) for a in s):
        return "non-GET requests to GitHub are not allowed"
    return None


def verdict(cmd, cwd=None, run_checks=True):
    """None if the command may run, else the reason it is blocked."""
    if REWRITE.search(cmd):
        return "history rewrite is not allowed (AGENTS.md, Git and sync)"
    try:
        segs = segments(cmd)
    except ValueError:
        return "unparsable command mentions git push" if NESTED_PUSH.search(cmd) else None
    cwd = os.path.realpath(cwd or os.getcwd())
    for s in segs:
        names = [os.path.basename(t) for t in s]
        head = names[0]
        if head != "git" and "git" in names[1:]:
            i = names.index("git")
            if "push" in s[i + 1:]:
                return "git push is allowed only as a plain top-level command `git push [-u] origin claude/<branch>`"
        if head in WRAPPERS or "=" in s[0]:
            if any(NESTED_PUSH.search(t) for t in s[1:]):
                return "git push inside another shell, wrapper or env-prefixed command is not allowed"
        if head in ("cd", "pushd"):
            target = os.path.expanduser(s[1]) if len(s) > 1 and s[1] != "-" else os.path.expanduser("~")
            cwd = os.path.realpath(os.path.join(cwd, target))
            continue
        if head == "git":
            why = check_git_segment(s, cwd, run_checks)
        elif head == "gh":
            why = check_gh_segment(s)
        elif head in ("curl", "wget"):
            why = check_http_segment(s)
        else:
            why = None
        if why:
            return why
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    why = verdict(cmd, data.get("cwd"))
    if why:
        print("Blocked: %s. Policy: AGENTS.md 'Git and sync' (pushes only to claude/* on origin; the VM syncs by "
              "git bundle)." % why, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
