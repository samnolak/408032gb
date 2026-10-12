#!/usr/bin/env python3
"""PreToolUse guard for GitHub MCP tools (mcp__github__*), called by scripts/hooks/guard-github.sh; exit 2 blocks.

Same policy as scripts/hooks/guard_bash.py (AGENTS.md "Git and sync", docs/reassessment/ADR-012.1-git-publication-
policy.md), applied to the tools that can write to GitHub without git:
  * writes to files or branches only in samnolak/408032gb and only on `claude/*` branches;
  * no merges (merge_pull_request, enable_pr_auto_merge), no file deletion, no new repositories or forks;
  * pull requests may be created and edited; their base must be `main` or a `claude/*` branch, their head `claude/*`.
Read-only tools are not restricted.
"""
import json
import re
import sys

REPO = ("samnolak", "408032gb")
CLAUDE = re.compile(r"^(refs/heads/)?claude/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
BLOCKED = {
    "merge_pull_request": "merging pull requests is the owner's decision",
    "enable_pr_auto_merge": "auto-merge is the owner's decision",
    "delete_file": "deleting files through the API is not allowed; commit a deletion on a claude/* branch instead",
    "create_repository": "creating repositories is not allowed",
    "fork_repository": "forks are third-party remotes",
}
BRANCH_WRITES = {"push_files", "create_or_update_file", "create_branch"}
PR_WRITES = {"create_pull_request", "update_pull_request", "update_pull_request_branch"}


def verdict(tool, args):
    name = tool.split("__")[-1]
    if name in BLOCKED:
        return BLOCKED[name]
    if name in BRANCH_WRITES or name in PR_WRITES:
        if (args.get("owner"), args.get("repo")) != REPO:
            return "writes are allowed only to %s/%s" % REPO
    if name in BRANCH_WRITES:
        branch = args.get("branch") or ""
        if not CLAUDE.match(branch):
            return "writes are allowed only to claude/* branches (got %r)" % branch
    if name == "create_pull_request":
        head, base = args.get("head") or "", args.get("base") or ""
        if not CLAUDE.match(head):
            return "pull requests must come from a claude/* branch (got %r)" % head
        if base != "main" and not CLAUDE.match(base):
            return "pull requests may target main or a claude/* branch (got %r)" % base
    if name == "update_pull_request":
        base = args.get("base")
        if base is not None and base != "main" and not CLAUDE.match(base):
            return "pull requests may target main or a claude/* branch (got %r)" % base
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    why = verdict(data.get("tool_name", ""), data.get("tool_input") or {})
    if why:
        print("Blocked: %s. Policy: AGENTS.md 'Git and sync'." % why, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
