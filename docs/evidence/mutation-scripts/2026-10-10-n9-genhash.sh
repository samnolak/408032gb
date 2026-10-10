#!/usr/bin/env bash
# sha256 of tools/tinygen/glm5next_tiny.py outputs for several argument sets; usage: genhash.sh <outdir>
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
REPO=/home/user/408032gb/.claude/worktrees/agent-ad00a79701ebcb394
out=$1; mkdir -p "$out"
export PYTHONPATH=$S/llama-ec7630a/gguf-py
python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$out/default.gguf" >/dev/null
python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$out/l12.gguf" --layers 12 >/dev/null
python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$out/l12s3.gguf" --layers 12 --seed 3 >/dev/null
python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$out/arch.gguf" --layers 8 --arch glm5next >/dev/null
cd "$out" && sha256sum default.gguf l12.gguf l12s3.gguf arch.gguf
