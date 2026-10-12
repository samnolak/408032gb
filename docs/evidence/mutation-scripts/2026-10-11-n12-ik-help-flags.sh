#!/usr/bin/env bash
# N12 probe (not a mutation check): every flag of the engine argv of scripts/r1/configs/ik-llamacpp-server.json (plus
# -ictk and --slot-save-path) looked up in ik_llama.cpp's `llama-server --help` (CPU build), and the exit codes of
# `--help` of ik's and of llama.cpp's (R1 pin) llama-server.
# usage: 2026-10-11-n12-ik-help-flags.sh <repo> <ik llama-server> <llama.cpp llama-server> <work dir>
set -uo pipefail
REPO=$(cd "$1" && pwd); IK=$2; ML=$3; W=$4
mkdir -p "$W"; cd "$W"                     # ik's logger writes llama.log into the current directory
"$IK" --help > "$W/ik-help.txt" 2> "$W/ik-help.err"; echo "ik llama-server --help: exit $?, stdout $(wc -l < "$W/ik-help.txt") lines, stderr $(wc -l < "$W/ik-help.err") lines"
"$ML" --help > "$W/ml-help.txt" 2>&1; echo "llama.cpp (R1 pin) llama-server --help: exit $?"
FLAGS=$(python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['engine']['argv'][1:]; print(' '.join(dict.fromkeys(x for x in a if x.startswith('-'))))" \
  "$REPO/scripts/r1/configs/ik-llamacpp-server.json")
for f in $FLAGS -ictk --slot-save-path; do
  hit=$(grep -E -- "(^|[ ,])$f([ ,=]|$)" "$W/ik-help.txt" | head -n 1 | sed 's/^ *//' | cut -c1-110)
  printf "%-16s %s\n" "$f" "${hit:-NOT IN --help}"
done
