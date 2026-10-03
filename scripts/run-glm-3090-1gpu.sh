#!/usr/bin/env bash
# GLM-5.3-Flash UD-Q4_K_XL на ОДНОЙ RTX 3090 (sm_86, 24 GiB). Проверено 2026-10-02, см. PROBLEMS.md.
#
# Расклад по памяти (замер лога загрузки, тюнинг 03.10, матрица ncmoe x ub x pinned в
# ../ik-llama-run/bench-0310/results.tsv — pp 166.8 / tg 15.0 tok/s против базы 115.5 / 8.9):
#   -ncmoe 39   39 MoE слоёв (161.1 GiB pinned) в RAM, blk.42..44 (12.6 GiB) на GPU
#   --ctx 8192  KV 236 MiB (q8_0); на 131k было бы 1598 MiB
#   -ub 1024    GPU compute-буферы 637 MiB (-ub 2048 при ncmoe 39 уже OOM: 21079+3620 > 24576)
#   pinned ON   161.13 GiB закрепились рядом с движком Strata (available падал до ~9 GB, oom-killer молчал);
#               если не влезет — GGML_CUDA_NO_PINNED=1 (tg просядет до ~10)
#
# Нужно свободной RAM: ~166 GiB. Итого с VRAM ~184 GiB. Итог по карте: 22.6 GiB из 24.
set -uo pipefail
GPU=${GPU:-2}
PORT=${PORT:-8090}
CTX=${CTX:-8192}
NCMOE=${NCMOE:-39}
UB=${UB:-1024}
B=${B:-2048}
THREADS=${THREADS:-96}
TBATCH=${TBATCH:-90}
MODEL=${MODEL:-/mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
BIN=${BIN:-/home/grishberg/ik_llama.cpp/build-cuda/bin/llama-server}
LOG=${LOG:-./server-3090-$(date +%H%M%S).log}
export CUDA_VISIBLE_DEVICES=$GPU
# pinned по умолчанию; любое значение GGML_CUDA_NO_PINNED (ggml проверяет getenv) выключает его
[ -z "${GGML_CUDA_NO_PINNED:-}" ] || export GGML_CUDA_NO_PINNED
echo "GPU=$GPU PORT=$PORT CTX=$CTX NCMOE=$NCMOE UB=$UB NO_PINNED=${GGML_CUDA_NO_PINNED:-off-not-set} LOG=$LOG"
exec "$BIN" \
  -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" --parallel 1 --jinja \
  -ngl 999 -sm none -ncmoe "$NCMOE" \
  -fa on --dsa -ctk q8_0 -ctv q8_0 \
  -t "$THREADS" --threads-batch "$TBATCH" -b "$B" -ub "$UB" -muge 2>&1 | tee "$LOG"
