#!/usr/bin/env bash
# N12 probe (not a mutation check): is the greedy continuation ik_llama.cpp's llama-server produced with the R1
# template's flags (2026-10-11-n12-ik-server-probe.py -> server_tokens.txt) what the engine computes and what
# ref/glm5next_ik.py predicts? The same token sequence (48 prompt tokens + the server's 16) is run by
# tools/dump/ik_dump_tensors.cpp with the server's CPU flags (prefill 48, then one token per call, as the server
# decodes) and checked by tests.test_ik_glm5next_parity.IkFa16Decode (which replays exact DSA ties).
# usage: 2026-10-11-n12-server-greedy.sh <repo> <ik-dump-tensors> <ik.gguf> <server_tokens.txt> <out dir> <gguf-py>
set -euo pipefail
REPO=$1; DUMP=$2; MODEL=$3; SEQ=$4; OUT=$5; export LLAMA_CPP_GGUF_PY=$6
rm -rf "$OUT"; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); cp "$SEQ" "$OUT/tokens.txt"
TOK=$(cat "$SEQ"); ONES16=$(python3 -c "print(','.join(['1']*16))")
PFX="inp_embd,hc_init,hc_attn_pre-,attn_norm-,linear_attn_out-,new_state-,kv_compressed-,kv_cache-,Qcur-"
PFX="$PFX,dsa_indexer_k-,dsa_indexer_gate-,dsa_cached_all-,dsa_indexer_k_pooled-,dsa_indexer_q-,dsa_indexer_weights-"
PFX="$PFX,dsa_indexer_score-,dsa_top_k-,kqv_out-,attn_out-,ffn_norm-,ffn_out-,ffn_moe_out-,ffn_shexp-,l_out-,hc_collapse"
# the server's flags without -ngl/-sm/-ts, --host/--port/--parallel/--jinja (server-only)
F="-fa on -ctk f16 -ctv f16 --dsa -muge -ncmoe 17"
(cd "$OUT" && DUMP_DIR="$OUT/fa16dec" DUMP_TOKENS="$TOK" DUMP_PREFIXES="$PFX" DUMP_CALLS="48,$ONES16" \
  "$DUMP" -m "$MODEL" -c 262144 -b 2048 -ub 1024 -t 4 $F > "$OUT/fa16dec.log" 2>&1)   # ik writes llama.log to cwd
tail -n 1 "$OUT/fa16dec.log"; echo "$F" > "$OUT/fa16dec/flags.txt"
python3 - "$OUT" <<'PY'
import sys, numpy as np
out = sys.argv[1]
seq = [int(x) for x in open(f"{out}/tokens.txt").read().split(",")]
lg = np.fromfile(f"{out}/fa16dec/logits.bin", dtype=np.float32).reshape(len(seq), -1)
am = lg[47:63].argmax(1).tolist()
print("engine argmax at positions 47..62:", ",".join(map(str, am)))
print("server tokens 48..63:            ", ",".join(map(str, seq[48:])))
print("equal:", sum(a == b for a, b in zip(am, seq[48:])), "/ 16")
PY
cd "$REPO" && IK_GLM5NEXT_DUMP="$OUT" IK_GLM5NEXT_GGUF="$MODEL" python3 -m unittest tests.test_ik_glm5next_parity.IkFa16Decode
