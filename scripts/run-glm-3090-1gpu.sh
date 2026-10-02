#!/usr/bin/env bash
# GLM-5.3-Flash UD-Q4_K_XL на ОДНОЙ RTX 3090 (sm_86, 24 GiB). Проверено 2026-10-02, см. PROBLEMS.md.
#
# Расклад по памяти (замер лога загрузки):
#   -ncmoe 40   40 MoE слоёв (165.2 GiB) остаются в RAM, blk.43/blk.44 (8.2 GiB) уезжают на GPU
#   --ctx 8192  KV 236 MiB (на 131k было бы 1598 MiB)
#   -ub 512     GPU compute-буферы 319 MiB (на -ub 2048 было 4353 MiB)
#   NO_PINNED   не пинить 165 GiB: pinned невытесняем, и загрузка умирает от oom-killer (PROBLEMS.md, §1)
#
# Нужно свободной RAM: ~166 GiB. Итого с VRAM ~184 GiB.
set -uo pipefail
GPU=${GPU:-1}
PORT=${PORT:-8090}
CTX=${CTX:-8192}
NCMOE=${NCMOE:-40}
UB=${UB:-512}
B=${B:-2048}
THREADS=${THREADS:-48}
TBATCH=${TBATCH:-90}
MODEL=${MODEL:-/mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
BIN=${BIN:-/home/grishberg/ik_llama.cpp/build-cuda/bin/llama-server}
LOG=${LOG:-./server-3090-$(date +%H%M%S).log}
export CUDA_VISIBLE_DEVICES=$GPU
export GGML_CUDA_NO_PINNED=${GGML_CUDA_NO_PINNED:-1}
echo "GPU=$GPU PORT=$PORT CTX=$CTX NCMOE=$NCMOE UB=$UB NO_PINNED=$GGML_CUDA_NO_PINNED LOG=$LOG"
exec "$BIN" \
  -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" --parallel 1 --jinja \
  -ngl 999 -sm none -ncmoe "$NCMOE" \
  -fa on --dsa -ctk q8_0 -ctv q8_0 \
  -t "$THREADS" --threads-batch "$TBATCH" -b "$B" -ub "$UB" -muge 2>&1 | tee "$LOG"
