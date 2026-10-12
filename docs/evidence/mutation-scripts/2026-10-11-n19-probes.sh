#!/usr/bin/env bash
# N19 probes (not mutations): what llama.cpp's qwen4exp context does with requests the parity tests do not model; prints
# the exit status, the llama_process return code and seq_rm result of every step, llama_memory_seq_pos_max per sequence,
# and the relevant log line. Results: docs/evidence/2026-10-11-n19-qwen4exp-multiseq.md ("Probes").
# usage: 2026-10-11-n19-probes.sh <dump-tensors built from this tree at a pin> <tiny qwen4exp GGUF> <scratch dir>
set -uo pipefail
BIN=$1; GGUF=$2; OUT=$3
mkdir -p "$OUT"
probe() {   # probe <name> <np> <kv_unified> <n_rs_seq> <DUMP_CALLS> <n tokens>
  local d="$OUT/$1"; rm -rf "$d"
  local tok; tok=$(python3 -c "import random; random.seed(5); print(','.join(str(random.randrange(3,96)) for _ in range($6)))")
  (cd "$OUT" && env DUMP_DIR="$d" DUMP_TOKENS="$tok" DUMP_PER_GRAPH=1 DUMP_PREFIXES=result_output DUMP_KV_UNIFIED="$3" \
    DUMP_N_RS_SEQ="$4" DUMP_CALLS="$5" "$BIN" -m "$GGUF" -c 512 -b 512 -ub 512 -np "$2" -t 1 -fa off -ctk f32 -ctv f32 \
    > "$d.log" 2>&1)
  echo "== $1 (np $2, kv_unified $3, n_rs_seq $4, DUMP_CALLS $5): exit status $?"
  if [ -f "$d/manifest.json" ]; then
    python3 -c "import json,sys; m=json.load(open(sys.argv[1]))
for c in m['calls']:
    what = 'ret %d' % c['ret'] if c['kind'] == 'decode' else 'ok %s' % c['ok']
    print('  %-7s %-26s %-8s pos_max %s' % (c['kind'], json.dumps(c.get('parts', [c.get('seq'), c.get('p0')]))[:26], what, c['pos_max']))" "$d/manifest.json"
  fi
  grep -h -o "GGML_ASSERT([^)]*) failed\|E split_equal: .*\|W decode: .*\|E init: .*\|E decode: .*\|the tokens of sequence [0-9]* in the input batch have inconsistent.*" "$d.log" | sort -u | sed 's/^/  log: /'
}
# coupled sequence ids (a common prompt for sequences 0 and 1): one KV stream, then one stream per sequence
probe coupled_uni 2 1 0 "12@0|1" 12
probe coupled_str 2 0 0 "12@0|1" 12
# n_rs_seq 0: every partial rollback is refused
probe rs0 1 1 0 "10@0,r1@0" 10
# a refused rollback (4 > n_rs_seq 3) and a decode at the position the rollback would have left
probe refused_then_decode 1 1 3 "20@0,4@0,r4@0,1@0:20" 25
# rollback of one of two sequences that share a recurrent cell, then sequence 1 decoded at the position after what
# llama_memory_seq_pos_max reports for it (its KV cells still hold 22 and 23 at ec7630a)
probe shared_rb_then_pos_max 2 1 3 "20@0,4@0,c0:1,r2@0,1@1:22" 25
