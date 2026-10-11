#!/usr/bin/env bash
# N12 probe (not a mutation check): ik_llama.cpp on the tiny model with an MTP block (glm5next_tiny.py --layers 12
# --dialect ik --mtp): loads without MTP (the MTP tensors are skipped) with trunk logits equal to the model without the
# block, and how MTP is requested (the legacy -mtp flag). Also prints llm_prepare_mla's log lines.
# usage: 2026-10-11-n12-ik-mtp-load.sh <repo> <ik-dump-tensors> <gguf-py> <work dir with ik.gguf and n12/p64> <out dir>
set -uo pipefail
REPO=$1; DUMP=$2; GGUF_PY=$3; WORK=$4; OUT=$5
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
PYTHONPATH="$GGUF_PY" python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$OUT/ikmtp.gguf" --layers 12 --dialect ik --mtp
sha256sum "$OUT/ikmtp.gguf" | cut -c1-64
TOK=$(cat "$WORK/n12/tokens.txt")
for flag in "" "-mtp"; do
  (cd "$OUT" && DUMP_DIR="$OUT/run$flag" DUMP_TOKENS="$TOK" "$DUMP" -m "$OUT/ikmtp.gguf" -c 512 -b 512 -ub 64 -t 1 \
    -fa off -ctk f32 -ctv f16 -ictk f32 --dsa $flag > "$OUT/run$flag.log" 2>&1)
  echo "flags '-fa off -ctk f32 -ctv f16 -ictk f32 --dsa $flag': exit $?"
  grep -E "nextn_predict_layers|skipped|legacy|Computed blk|need to compute|ik-dump:" "$OUT/run$flag.log" | grep -v "kv " | cut -c1-160
  [ -f "$OUT/run$flag/logits.bin" ] || { echo "  no logits (the run did not get that far)"; continue; }
  python3 -c "
import numpy as np, sys
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
print('  logits vs the model without MTP (p64 run): max |diff| =', float(abs(a - b).max()))" \
    "$OUT/run$flag/logits.bin" "$WORK/n12/p64/logits.bin"
done
