#!/usr/bin/env bash
# PreToolUse hook for GitHub MCP tools. Rules: scripts/hooks/guard_github.py; policy: AGENTS.md "Git and sync".
# Exit 2 blocks the call; a guard that cannot run blocks too.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/guard_github.py" || exit 2
