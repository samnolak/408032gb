#!/usr/bin/env bash
# N17: print every source range docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md cites, from the two llama.cpp checkouts
# (leading blanks stripped, cut at 150 columns). usage: 2026-10-11-n17-source-lines.sh <llama.cpp @ec7630a> <llama.cpp @79e2e74>
set -euo pipefail
EC=$1; R1=$2
p() {  # p <checkout> <file> <a,b>
  echo "--- $(git -C "$1" rev-parse --short HEAD) $2:$3"
  sed -n "${3}p" "$1/$2" | sed 's/^[[:space:]]*//' | cut -c1-150
}
# the MTP graph: the head is layer n_layer + offset, asserted in range
p "$EC" src/models/mimo2.cpp 276,279
p "$R1" src/models/mimo2.cpp 267,270
p "$EC" src/models/mimo2.cpp 179,179
p "$R1" src/models/mimo2.cpp 171,171
p "$EC" src/llama-ext.h 101,101
p "$R1" src/llama-ext.h 101,101
p "$EC" src/llama-context.cpp 1248,1250
p "$R1" src/llama-context.cpp 1264,1266
p "$EC" src/llama-graph.h 873,875
p "$R1" src/llama-graph.h 884,886
p "$EC" src/llama-model.cpp 2807,2812
p "$R1" src/llama-model.cpp 2846,2851
# common/speculative.cpp: n heads, chain_heads, n_max, process (pairing, heads loop), draft (step 0, seq_rm + offset,
# chained rows), accept
p "$EC" common/speculative.cpp 1378,1378
p "$EC" common/speculative.cpp 1420,1425
p "$EC" common/speculative.cpp 1519,1529
p "$EC" common/speculative.cpp 1533,1557
p "$EC" common/speculative.cpp 1604,1611
p "$EC" common/speculative.cpp 1623,1631
p "$EC" common/speculative.cpp 1686,1696
p "$EC" common/speculative.cpp 1743,1745
p "$R1" common/speculative.cpp 1439,1439
p "$R1" common/speculative.cpp 1481,1486
p "$R1" common/speculative.cpp 1583,1593
p "$R1" common/speculative.cpp 1597,1621
p "$R1" common/speculative.cpp 1681,1688
p "$R1" common/speculative.cpp 1700,1708
p "$R1" common/speculative.cpp 1767,1777
p "$R1" common/speculative.cpp 1824,1826
p "$EC" common/common.h 332,332
p "$R1" common/common.h 333,333
p "$EC" tools/server/server-context.cpp 3044,3044
p "$EC" tools/server/server-context.cpp 3747,3747
p "$R1" tools/server/server-context.cpp 3404,3404
p "$R1" tools/server/server-context.cpp 4232,4232
# llama-server around the drafts: seq_rm of the draft context after drafting, accept, common_memory::seq_rm on both
p "$EC" tools/server/server-context.cpp 3063,3063
p "$EC" tools/server/server-context.cpp 3964,3964
p "$EC" tools/server/server-context.cpp 3998,3998
p "$R1" tools/server/server-context.cpp 3423,3423
p "$R1" tools/server/server-context.cpp 4471,4471
p "$R1" tools/server/server-context.cpp 4505,4505
p "$EC" common/common.cpp 1558,1563
p "$EC" common/common.cpp 1502,1527
# seq_rm moves the cache head back to the first freed cell
p "$EC" src/llama-kv-cache.cpp 410,419
p "$R1" src/llama-kv-cache.cpp 428,437
# the mask: one template for f16 (flash attention) and f32
p "$EC" src/llama-graph.cpp 38,43
p "$R1" src/llama-graph.cpp 39,44
p "$EC" src/llama-kv-cache.cpp 1581,1582
p "$EC" src/llama-kv-cache.cpp 1694,1711
p "$EC" src/llama-kv-cache.cpp 1789,1793
p "$R1" src/llama-kv-cache.cpp 1792,1796
# flash attention in the graph: f32 K/V cast to f16, the sinks, the precision
p "$EC" src/llama-graph.cpp 2651,2675
p "$EC" ggml/src/ggml.c 5579,5593
p "$EC" ggml/src/ggml.c 4218,4232
# the CPU flash attention (ops.cpp): one_chunk, the sinks, tiled, split-KV, dispatcher
p "$EC" ggml/src/ggml-cpu/ops.cpp 8917,8918
p "$EC" ggml/src/ggml-cpu/ops.cpp 8961,8961
p "$EC" ggml/src/ggml-cpu/ops.cpp 8976,8978
p "$EC" ggml/src/ggml-cpu/ops.cpp 8993,9007
p "$EC" ggml/src/ggml-cpu/ops.cpp 9031,9056
p "$EC" ggml/src/ggml-cpu/ops.cpp 9066,9068
p "$EC" ggml/src/ggml-cpu/ops.cpp 9332,9354
p "$EC" ggml/src/ggml-cpu/ops.cpp 9490,9520
p "$EC" ggml/src/ggml-cpu/ops.cpp 9420,9435
p "$EC" ggml/src/ggml-cpu/ops.cpp 9546,9551
p "$EC" ggml/src/ggml-cpu/ops.cpp 5896,5912
p "$R1" ggml/src/ggml-cpu/ops.cpp 9039,9055
# the f16 vector code on AVX2: dot (SIMD lanes, leftovers), mad, scale; the macros
p "$EC" ggml/src/ggml-cpu/vec.cpp 345,377
p "$EC" ggml/src/ggml-cpu/vec.h 556,581
p "$EC" ggml/src/ggml-cpu/vec.h 833,855
p "$EC" ggml/src/ggml-cpu/simd-mappings.h 595,620
p "$EC" ggml/src/ggml-cpu/simd-mappings.h 635,647
p "$EC" ggml/src/ggml-cpu/simd-mappings.h 670,680
p "$EC" ggml/src/ggml-cpu/simd-mappings.h 1333,1340
# the converter: three MiMo MTP layers, remapped to blocks n_layer + i
p "$R1" conversion/mimo.py 22,24
p "$R1" conversion/mimo.py 290,298
# the output crop of the MTP graph (rows before it do not depend on the output flags); the SWA cache size rule
p "$R1" src/models/mimo2.cpp 378,378
p "$EC" src/llama-kv-cache-iswa.cpp 73,73
p "$R1" src/llama-kv-cache-iswa.cpp 73,73
# the code the two pins share: the CPU flash attention, the mask template, and the MTP drafting (sampler changes only)
d() {  # d <file> <a,b @ec7630a> <a,b @79e2e74>
  echo "--- diff $1:$2 @$(git -C "$EC" rev-parse --short HEAD) $1:$3 @$(git -C "$R1" rev-parse --short HEAD)"
  diff <(sed -n "${2}p" "$EC/$1") <(sed -n "${3}p" "$R1/$1") && echo "identical ($(sed -n "${2}p" "$EC/$1" | wc -l) lines)" || true
}
d ggml/src/ggml-cpu/ops.cpp 8843,9592 8842,9591
d src/llama-kv-cache.cpp 1566,1716 1569,1719
d common/speculative.cpp 1340,1747 1401,1828
