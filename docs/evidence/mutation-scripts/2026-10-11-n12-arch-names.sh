#!/usr/bin/env bash
# N12 probe (not a mutation check, after the audit of aaf7757): the same weights written three ways
# (glm5next_tiny.py --layers 12 with --dialect ik, --arch glm5next, and the default glm5-next) give bit-identical
# logits in ik_llama.cpp with the flags of the p64 run of run_parity_ik_glm5next.sh. Runs from <out dir>.
# usage: 2026-10-11-n12-arch-names.sh <repo> <ik-dump-tensors> <gguf-py> <tokens.txt> <out dir>
set -uo pipefail
REPO=$(cd "$1" && pwd); DUMP=$2; GGUF_PY=$3; TOKF=$4; OUT=$5
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); cd "$OUT" || exit 1
TOK=$(cat "$TOKF")
for v in "dialect-ik:--dialect ik" "arch-glm5next:--arch glm5next" "glm5-next:"; do
  n=${v%%:*}; a=${v#*:}
  PYTHONPATH="$GGUF_PY" python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$OUT/$n.gguf" --layers 12 $a > /dev/null
  rm -rf "$OUT/$n"
  DUMP_DIR="$OUT/$n" DUMP_TOKENS="$TOK" "$DUMP" -m "$OUT/$n.gguf" -c 512 -b 512 -ub 64 -t 1 -fa off -ctk f32 -ctv f16 \
    -ictk f32 --dsa > "$OUT/$n.log" 2>&1
  echo "$n: exit $?, $(grep -m1 'general.architecture' "$OUT/$n.log" | sed 's/.*= //'), sha256 $(sha256sum "$OUT/$n.gguf" | cut -c1-16)..."
done
python3 - "$OUT" <<'PY'
import sys, numpy as np
o = sys.argv[1]
a = {n: np.fromfile(f"{o}/{n}/logits.bin", dtype=np.float32) for n in ("dialect-ik", "arch-glm5next", "glm5-next")}
print("arch-glm5next == dialect-ik bitwise:", np.array_equal(a["arch-glm5next"], a["dialect-ik"]))
print("glm5-next == dialect-ik bitwise:", np.array_equal(a["glm5-next"], a["dialect-ik"]))
PY
