#!/usr/bin/env bash
# PreToolUse hook for Bash. Rules: scripts/hooks/guard_bash.py; policy: AGENTS.md "Git and sync" and
# docs/reassessment/ADR-012.1-git-publication-policy.md. Exit 2 blocks the call; a guard that cannot run blocks too.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/guard_bash.py" || exit 2
