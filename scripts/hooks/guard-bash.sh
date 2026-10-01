#!/usr/bin/env bash
# PreToolUse hook for Bash: blocks pushes (sync is by git bundle, AGENTS.md) and
# history rewrites. Reads the hook JSON on stdin; exit 2 blocks the call.
cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])git[[:space:]]+push([[:space:]]|$)'; then
  echo "Blocked: no git push from agents. Sync is by git bundle (skill bundle-sync)." >&2
  exit 2
fi
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+(reset[[:space:]]+--hard|push[[:space:]].*--force|filter-branch|rebase[[:space:]]+-i)'; then
  echo "Blocked: history rewrite is not allowed (AGENTS.md, Git and sync)." >&2
  exit 2
fi
exit 0
