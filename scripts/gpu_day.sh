#!/usr/bin/env bash
# The GPU day: everything that needs the server (or a cloud sm_89 GPU), in order, with literal logs.
#
#   scripts/gpu_day.sh <step|all>      steps: env build pack tokens profile check1 check2 check3 bench
#
# Inputs (environment):
#   CKPT     nvidia/GLM-5.3-Flash-NVFP4 directory (204 GB, safetensors + tokenizer.json)
#   WORK     a work directory on the NVMe drives (the pack is ~171 GB)
#   CORPUS   directory of *.txt real prompts/transcripts for the expert profile (one domain per file)
#   PROMPT   a prompt text file for the checks and benchmarks
#   GPUS     devices for the 4-way split (default 0,1,2,3)
# Every step writes docs/evidence/gpu-day/<step>.log (command lines + output) and a done-marker; rerunning
# skips finished steps (delete WORK/.done/<step> to redo one). It stops at the first failure.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${WORK:?set WORK}"
EV="$ROOT/docs/evidence/gpu-day"
mkdir -p "$EV" "$WORK/.done"
GPUS=${GPUS:-0,1,2,3}
SG="$WORK/strata-glm"
BIN="$SG/build/strata-glm"
PACK="$WORK/pack"
PIN=ed37419fccd0c52e07d26d526a29c2098f547843
SPLIT=14,25,35                     # docs/design/G4-glm-multigpu.md: 11/11/10/10 MoE layers per card

log() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
step() {   # step NAME FUNCTION
    local name=$1
    if [ -e "$WORK/.done/$name" ]; then echo "== $name: done before (rm $WORK/.done/$name to redo)"; return; fi
    LOG="$EV/$name.log"; : > "$LOG"
    { echo "# $name  $(date -u +%FT%TZ)"; echo "# repo $(git -C "$ROOT" log --oneline -n 1)"; } >> "$LOG"
    echo "== $name"
    "$2"
    touch "$WORK/.done/$name"
}
need() { [ -n "${!1:-}" ] || { echo "set $1"; exit 2; }; }

s_env() {
    log nvidia-smi
    log nvidia-smi topo -m
    log nvidia-smi -q -d PCIE
    log nvcc --version || true
    log free -g
    log lsblk -o NAME,SIZE,MODEL,MOUNTPOINT
    log sudo -n dmidecode -t memory || echo "dmidecode needs root: count the populated DIMM channels by hand" | tee -a "$LOG"
}
s_build() {
    [ -d "$SG/.git" ] || log git clone https://github.com/sergqwer/strata-glm.git "$SG"
    log git -C "$SG" checkout -f "$PIN"
    for p in "$ROOT"/patches/strata-glm/*.patch; do log git -C "$SG" apply "$p"; done
    log cmake -G Ninja -S "$SG" -B "$SG/build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
    log cmake --build "$SG/build" --target strata-glm
    log "$ROOT/tests/strata_glm/run_diskreader_test.sh" "$SG"
}
s_pack() {
    need CKPT
    log python3 "$SG/tools/glm_pack.py" --model "$CKPT" --out "$PACK" --verify 24
    log ls -la "$PACK"
}
s_tokens() {
    need CKPT; need PROMPT
    log python3 - "$CKPT/tokenizer.json" "$PROMPT" "$WORK/prompt_ids.txt" <<'PY'
import sys
from tokenizers import Tokenizer
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2], encoding="utf-8").read()).ids
open(sys.argv[3], "w").write(" ".join(map(str, ids)))
print(len(ids), "tokens")
PY
}
s_profile() {   # also G0: the measured share of routed reads that the top-N experts cover
    need CORPUS; need CKPT
    log python3 "$SG/tools/glm_profile.py" split --corpus "$CORPUS" --tokenizer "$CKPT/tokenizer.json" --out "$WORK/prof"
    log python3 "$SG/tools/glm_profile.py" run --out "$WORK/prof" --exe "$BIN" --pack "$PACK" --chunk 2048 --budget 7200
    # VRAM-tier sizes: one 4080 (~1,950 experts) and four (~8,000, tools/fitplan.py --quant NVFP4)
    for v in 1950 8000; do
        log python3 "$SG/tools/glm_profile.py" stats --out "$WORK/prof" --vram "$v" --ram 7000
    done
    log python3 "$SG/tools/glm_profile.py" stats --out "$WORK/prof" --vram 8000 --ram 7000 --profile "$PACK/expert-profile-boot.bin"
}
RUN=( "$BIN" --pack "$PACK" --profile "$PACK/expert-profile-boot.bin" --chunk 8192 --max-new 64 --skip-disk 0 )
s_check1() {   # one stage = the original engine (patch 0002 must not change it)
    log "${RUN[@]}" --tokens "$WORK/prompt_ids.txt" --dump-logits "$WORK/ref.f32"
    grep "output :" "$LOG" | tail -1 > "$WORK/ref.out"
}
s_check2() {   # 4 stages on GPU 0: identical tokens (and logits, UNKNOWN until run)
    log "${RUN[@]}" --tokens "$WORK/prompt_ids.txt" --layer-split "$SPLIT" --vram-experts 300 --dump-logits "$WORK/split1.f32"
    grep "output :" "$LOG" | tail -1 > "$WORK/split1.out"
    log diff "$WORK/ref.out" "$WORK/split1.out"
    log cmp "$WORK/ref.f32" "$WORK/split1.f32" || echo "logits differ (tokens are the gate; record the max difference)" | tee -a "$LOG"
}
s_check3() {   # 4 GPUs: identical tokens
    log "${RUN[@]}" --tokens "$WORK/prompt_ids.txt" --layer-split "$SPLIT" --gpus "$GPUS" --dump-logits "$WORK/split4.f32"
    grep "output :" "$LOG" | tail -1 > "$WORK/split4.out"
    log diff "$WORK/ref.out" "$WORK/split4.out"
}
s_bench() {   # default (lossy --skip-disk 0.1) and exact, one GPU vs four; 3 repeats each
    for _ in 1 2 3; do
        log "$BIN" --pack "$PACK" --profile "$PACK/expert-profile-boot.bin" --chunk 8192 --max-new 192 --tokens "$WORK/prompt_ids.txt"
        log "$BIN" --pack "$PACK" --profile "$PACK/expert-profile-boot.bin" --chunk 8192 --max-new 192 --tokens "$WORK/prompt_ids.txt" \
            --layer-split "$SPLIT" --gpus "$GPUS"
        log "$BIN" --pack "$PACK" --profile "$PACK/expert-profile-boot.bin" --chunk 8192 --max-new 192 --tokens "$WORK/prompt_ids.txt" \
            --layer-split "$SPLIT" --gpus "$GPUS" --skip-disk 0
    done
    grep -E "decode [0-9]+ steps|prompt [0-9]+ tokens|decode experts" "$LOG" | tee "$EV/bench-summary.txt"
}

case "${1:-}" in
    env) step env s_env ;; build) step build s_build ;; pack) step pack s_pack ;; tokens) step tokens s_tokens ;;
    profile) step profile s_profile ;; check1) step check1 s_check1 ;; check2) step check2 s_check2 ;;
    check3) step check3 s_check3 ;; bench) step bench s_bench ;;
    all) for s in env build pack tokens profile check1 check2 check3 bench; do "$0" "$s"; done ;;
    *) sed -n 2,16p "$0"; exit 2 ;;
esac
