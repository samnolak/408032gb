#!/usr/bin/env bash
# runjob.sh "<name>\t<command>": runs the command from the worktree, writes out/<name>.txt with the command line,
# the literal stdout/stderr and the exit code
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/traces-analysis
name=${1%%$'\t'*}; cmd=${1#*$'\t'}
mkdir -p "$S/out/$(dirname "$name")"
cd /home/user/wt-research || exit 1
s=$(date +%s)
{ echo "\$ $cmd"; eval "$cmd" 2>&1; echo "[exit=$? seconds=$(( $(date +%s) - s )) commit=$(git rev-parse --short HEAD)]"; } > "$S/out/$name.txt"
