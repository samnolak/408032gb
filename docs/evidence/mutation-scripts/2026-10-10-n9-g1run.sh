#!/usr/bin/env bash
# G1 op parity (tools/tinygen/run_parity.sh) at a pin; usage: g1run.sh <ec|r1>
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
REPO=/home/user/408032gb/.claude/worktrees/agent-ad00a79701ebcb394
tag=$1
if [ "$tag" = ec ]; then src=llama-ec7630a; else src=llama.cpp; fi
cd "$REPO" || exit 1
echo "llama.cpp: $(git -C "$S/$src" rev-parse HEAD)"
tools/tinygen/run_parity.sh "$S/$src" "$S/n9/build-$tag"
