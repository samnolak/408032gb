#!/usr/bin/env bash
# N14 probe: how llama.cpp at the R1 pin maps GLM-5.3-Flash's n_group / topk_group into the GGUF and when the loader
# and the graph use them (follow-up of N12's open point on group-limited routing). Prints the cited lines.
#   bash docs/evidence/mutation-scripts/2026-10-11-n14-converter-ngroup.sh <llama.cpp checkout at 79e2e74>
set -euo pipefail
L=$1
echo "checkout $(git -C "$L" rev-parse HEAD)"
show() { echo "--- $1:$2"; sed -n "$2p" "$L/$1"; }
show conversion/__init__.py 112
show conversion/glm.py 407,411
show conversion/glm.py 456,458
show conversion/base.py 1340,1355
show conversion/base.py 1435,1436
show conversion/base.py 1524,1529
show gguf-py/gguf/constants.py 125,126
show src/llama-model.cpp 1362,1363
show src/llama-model.cpp 1390,1398
show src/llama-graph.cpp 2143,2146
echo "--- Glm5NextModel overrides of n_group / topk_group / expert_group (none expected):"
sed -n '/^class Glm5NextModel/,/^class /p' "$L/conversion/glm.py" | grep -n "n_group\|topk_group\|expert_group" || echo "(none)"
