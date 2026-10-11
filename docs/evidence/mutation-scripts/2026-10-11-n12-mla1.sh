#!/usr/bin/env bash
# N12 probe (not a mutation check, after the audit of aaf7757): with -mla 1 -fa off ik_llama.cpp gives the MLA latent
# K cache the -ctk type (f32 here) and keeps a transposed V copy in the -ctv type (f16), src/llama.cpp:1496-1497,
# 1513-1516 @5194a9e. Numerics of -mla 1 are not compared. Runs from <out dir> (ik writes llama.log into the cwd).
# usage: 2026-10-11-n12-mla1.sh <ik-dump-tensors> <ik.gguf> <out dir>
set -uo pipefail
DUMP=$1; MODEL=$2; OUT=$3
mkdir -p "$OUT"; cd "$OUT" || exit 1; rm -rf d
echo '$ DUMP_DIR=d DUMP_PREFIXES=kv_cache- DUMP_TOKENS=5,9,11,20 ik-dump-tensors -m ik.gguf -c 512 -b 512 -ub 512 -t 1 -mla 1 -fa off -ctk f32 -ctv f16 -ictk f32 --dsa'
DUMP_DIR="$OUT/d" DUMP_PREFIXES=kv_cache- DUMP_TOKENS=5,9,11,20 "$DUMP" -m "$MODEL" -c 512 -b 512 -ub 512 -t 1 -mla 1 \
  -fa off -ctk f32 -ctv f16 -ictk f32 --dsa > run.log 2>&1
echo "rc=$?"
grep -E "mla_attn|KV self size|ik-dump:" run.log
echo '$ cat d/g000/kv_cache-3__0.json'; cat d/g000/kv_cache-3__0.json
