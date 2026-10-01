#!/usr/bin/env bash
# PreToolUse hook for Edit/Write: pinned upstream trees are read-only (AGENTS.md).
path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$path" in
  */upstream/*|upstream/*|*/third_party/*|third_party/*)
    echo "Blocked: $path is a pinned upstream tree. Put changes in patches/ or our own sources." >&2
    exit 2;;
esac
exit 0
