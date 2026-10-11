#!/usr/bin/env bash
set -euo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
cd $S/n16
for p in ec r1; do
  L=$S/llama-ec7630a; [ $p = r1 ] && L=$S/llama.cpp
  t0=$(date +%s)
  cmake -S /home/user/wt-n16/tools/g0 -B $S/n16/build-$p -DLLAMA_CPP_DIR=$L -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DG0_QUANTIZE=ON > cmake-$p.log 2>&1
  t1=$(date +%s)
  cmake --build $S/n16/build-$p -j4 --target dump-tensors llama-quantize > build-$p.log 2>&1
  t2=$(date +%s)
  echo "$p cmake $((t1-t0)) s, build $((t2-t1)) s"
done
echo BUILDS-DONE
