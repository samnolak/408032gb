#!/usr/bin/env bash
# scripts/diag/bench_all.sh - run the opt-in benchmarks into an existing diag-out directory.
#
# Runs, in this order and only what is selected:
#   bench_mem.sh   (--mem)                         host memory bandwidth
#   bench_gpu.sh   (--gpu, + --vram-check)          GPU P2P/PCIe matrix (+ VRAM fill/verify)
#   bench_nvme.sh  (--nvme-dir / --nvme-all / --nvme-raw)   fio reads
# Default selection when none is given: --mem --gpu.
# Every sub-script log goes to DIAGDIR/bench/bench_all-<ts>.log; a failing benchmark does not
# stop the next one. At the end summarize.py rewrites DIAGDIR/HARDWARE_FACTS.draft.md and
# DIAGDIR.tar.gz is rebuilt so it includes the benchmark results. Exit code 1 if any benchmark failed.
# Refuses to run without --run; --plan prints the plans of the selected sub-scripts.
set -u
export LC_ALL=C
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/bench_common.sh
. "$KIT_DIR/bench_common.sh"

usage() {
  cat <<'EOF'
usage: bench_all.sh (--run | --plan) --out DIAGDIR [--mem] [--gpu] [--vram-check] [--force]
                    [--nvme-dir DIR | --nvme-all DIR1,DIR2,... | --nvme-raw /dev/nvmeXn1[,...]]
                    [--size-gib N] [--runtime S] [--mem-size-mib N]

  --out DIAGDIR   the directory written by collect.sh (results go to DIAGDIR/bench/)
  --mem           bench_mem.sh               --gpu         bench_gpu.sh
  --vram-check    bench_gpu.sh --vram-check  --force       pass --force to the VRAM check
  --nvme-dir/--nvme-all/--nvme-raw  bench_nvme.sh in file mode / multi-dir mode / raw read-only mode
  --size-gib N    test file size for the NVMe file mode (required there)
EOF
}

RUN=0; PLAN=0; OUT=""; MEM=0; GPU=0; VRAM=0; FORCE=0; NVME_DIR=""; NVME_ALL=""; NVME_RAW=""; SIZE_GIB=""; RT=""; MEM_MIB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift;;
    --plan) PLAN=1; shift;;
    --out) OUT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --mem) MEM=1; shift;;
    --gpu) GPU=1; shift;;
    --vram-check) VRAM=1; GPU=1; shift;;
    --force) FORCE=1; shift;;
    --nvme-dir) NVME_DIR=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --nvme-all) NVME_ALL=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --nvme-raw) NVME_RAW=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --size-gib) SIZE_GIB=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --runtime) RT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --mem-size-mib) MEM_MIB=${2:-}; shift 2 || { usage >&2; exit 2; };;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "--out DIAGDIR is required (the directory written by collect.sh)" >&2; usage >&2; exit 2; }
if [ -d "$OUT" ]; then OUT=$(cd "$OUT" && pwd)
elif [ $PLAN -ne 1 ]; then echo "refusing: $OUT is not a directory (run collect.sh first)" >&2; exit 2; fi
NVME=0; [ -n "$NVME_DIR$NVME_ALL$NVME_RAW" ] && NVME=1
[ $MEM -eq 0 ] && [ $GPU -eq 0 ] && [ $NVME -eq 0 ] && { MEM=1; GPU=1; }

MEM_CMD=(bash "$KIT_DIR/bench_mem.sh" --out "$OUT")
[ -n "$MEM_MIB" ] && MEM_CMD+=(--size-mib "$MEM_MIB")
GPU_CMD=(bash "$KIT_DIR/bench_gpu.sh" --out "$OUT")
[ $VRAM -eq 1 ] && GPU_CMD+=(--vram-check)
[ $FORCE -eq 1 ] && GPU_CMD+=(--force)
NVME_CMD=(bash "$KIT_DIR/bench_nvme.sh" --out "$OUT")
[ -n "$NVME_DIR" ] && NVME_CMD+=(--dir "$NVME_DIR")
[ -n "$NVME_ALL" ] && NVME_CMD+=(--all "$NVME_ALL")
[ -n "$NVME_RAW" ] && NVME_CMD+=(--raw-readonly "$NVME_RAW")
[ -n "$SIZE_GIB" ] && NVME_CMD+=(--size-gib "$SIZE_GIB")
[ -n "$RT" ] && NVME_CMD+=(--runtime "$RT")

echo "bench_all.sh plan: out=$OUT mem=$MEM gpu=$GPU vram_check=$VRAM nvme=$NVME"
rc_plan=0
[ $MEM -eq 1 ] && { "${MEM_CMD[@]}" --plan || rc_plan=1; }
[ $GPU -eq 1 ] && { "${GPU_CMD[@]}" --plan || rc_plan=1; }
[ $NVME -eq 1 ] && { "${NVME_CMD[@]}" --plan || rc_plan=1; }
[ $rc_plan -eq 0 ] || { echo "a sub-plan reported an error (see above); nothing was executed"; exit 2; }
if [ $PLAN -eq 1 ]; then exit 0; fi
if [ $RUN -ne 1 ]; then echo "Refusing to run without --run (nothing was executed)."; exit 2; fi

mkdir -p "$OUT/bench" || exit 2
LOG="$OUT/bench/bench_all-$(bench_ts).log"
status=()
run_one() { # run_one NAME CMD...
  local name=$1 rc
  shift
  echo "=== $(bench_utc) $name: $(printf '%q ' "$@")" | tee -a "$LOG"
  "$@" --run 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== $(bench_utc) $name exit $rc" | tee -a "$LOG"
  status+=("$name=$rc")
}
[ $MEM -eq 1 ] && run_one mem "${MEM_CMD[@]}"
[ $GPU -eq 1 ] && run_one gpu "${GPU_CMD[@]}"
[ $NVME -eq 1 ] && run_one nvme "${NVME_CMD[@]}"
echo "bench_all: ${status[*]}" | tee -a "$LOG"
rc_all=0
for st in "${status[@]}"; do [ "${st##*=}" = 0 ] || rc_all=1; done

if command -v python3 >/dev/null 2>&1; then
  python3 "$KIT_DIR/summarize.py" "$OUT" --output "$OUT/HARDWARE_FACTS.draft.md" >> "$LOG" 2>&1 \
    && echo "draft facts: $OUT/HARDWARE_FACTS.draft.md" | tee -a "$LOG"
fi
if command -v tar >/dev/null 2>&1 && tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz.part" "$(basename "$OUT")" \
   && mv -f "$OUT.tar.gz.part" "$OUT.tar.gz"; then
  echo "archive: $OUT.tar.gz  <- send this file"
else
  rm -f "$OUT.tar.gz.part"
  echo "tar failed or not available: send the directory $OUT"
fi
exit $rc_all
