#!/usr/bin/env bash
# N21: every source line cited by docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md, printed from the pinned checkouts
# (llama.cpp 79e2e74 = the R1 engine pin; ec7630a where a pin difference matters) and from this repository.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
R1=$S/llama.cpp; EC=$S/llama-ec7630a; WT=${1:-/home/user/wt-n21}
show() { # show <checkout> <file> <from> <to>
  echo "== $2:$3-$4 @$(git -C "$1" rev-parse --short HEAD 2>/dev/null || echo worktree)"
  sed -n "$3,$4p" "$1/$2" | nl -ba -v "$3" | sed 's/^ *//'
}
show $R1 common/speculative.cpp 1481 1486
show $R1 common/speculative.cpp 1583 1593
show $R1 common/speculative.cpp 1595 1621
show $R1 common/speculative.cpp 1627 1642
show $R1 common/speculative.cpp 1663 1669
show $R1 common/speculative.cpp 1696 1708
show $R1 common/speculative.cpp 1728 1752
show $R1 common/speculative.cpp 1814 1827
show $R1 common/speculative.cpp 2612 2619
show $R1 common/common.h 329 337
show $R1 common/common.h 376 376
show $R1 common/common.h 456 457
show $R1 common/speculative.cpp 76 76
show $R1 common/speculative.cpp 1439 1439
show $R1 common/speculative.cpp 1757 1759
show $R1 common/speculative.cpp 2527 2531
show $R1 common/speculative.cpp 2641 2646
show $R1 common/arg.cpp 357 359
show $R1 common/arg.cpp 546 550
show $R1 common/arg.cpp 566 573
show $R1 common/arg.cpp 1693 1699
show $R1 common/arg.cpp 4160 4167
show $R1 common/arg.cpp 4177 4177
show $R1 common/arg.cpp 4190 4190
show $R1 common/arg.cpp 4233 4241
show $R1 common/arg.cpp 4284 4292
show $R1 common/common.cpp 1725 1725
show $R1 common/sampling.cpp 689 721
show $R1 common/sampling.cpp 732 735
show $R1 common/sampling.cpp 790 803
show $R1 tools/server/server-context.cpp 504 506
show $R1 tools/server/server-context.cpp 713 716
show $R1 tools/server/server-common.cpp 99 102
show $R1 tools/server/server-context.cpp 3400 3405
show $R1 tools/server/server-context.cpp 3416 3426
show $R1 tools/server/server-context.cpp 4225 4233
show $R1 tools/server/server-context.cpp 4410 4427
show $R1 tools/server/server-context.cpp 4468 4472
show $R1 tools/server/server-context.cpp 4497 4505
show $R1 src/llama-kv-cache.cpp 428 437
show $R1 src/llama-kv-cache.cpp 1019 1025
show $R1 src/llama-kv-cache.cpp 1061 1078
show $R1 src/llama-kv-cache.cpp 1185 1205
show $R1 src/llama-kv-cache-iswa.cpp 70 81
show $R1 src/llama-context.cpp 147 166
show $R1 src/models/mimo2.cpp 267 270
show $R1 conversion/mimo.py 134 136
show $R1 conversion/mimo.py 153 160
show $EC common/speculative.cpp 1533 1557
show $WT third_party/hf/mimo-v2.6-flash-chat/config.json 162 162
show $WT third_party/hf/mimo-v2.6-flash-chat/config.json 303 303
show $WT scripts/r1/configs/llamacpp-server.json 10 10
show $WT scripts/r1/configs/llamacpp-server.json 36 39
show $WT scripts/r1/configs/llamacpp-server.json 42 42
show $WT scripts/r1/configs/llamacpp-server.json 51 51
show $WT scripts/r1/configs/llamacpp-server.json 66 66
show $WT scripts/r1/configs/llamacpp-server-moecache.json 52 52
show $WT scripts/r1/configs/llamacpp-server-moecache.json 64 64
show $WT scripts/r1/configs/ik-llamacpp-server.json 50 50
show $WT scripts/r1/configs/ik-llamacpp-server.json 62 62
show $WT third_party/hf/mimo-v2.6-flash-chat/config.json 47 47
show $WT docs/reassessment/ROADMAP.md 55 55
