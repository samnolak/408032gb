#!/usr/bin/env bash
# scripts/diag/bench_nvme.sh - NVMe read benchmark with fio (opt-in).
#
# File mode (default, recommended):
#   --dir DIR --size-gib N        one directory (one drive)
#   --all DIR1,DIR2,... --size-gib N
#                                 every directory alone, then all of them at the same time
#   For each directory: checks free space >= 2*N GiB, creates a NEW sub-directory
#   diag-fio-<host>-<ts>/ with ONE test file of N GiB (sequential write with fio, so the
#   reads hit real data), runs the read jobs with O_DIRECT and fio --readonly, and deletes
#   the test file and the sub-directory at the end (also on Ctrl-C). Nothing else in DIR
#   is touched.
# Raw mode (only when asked):
#   --raw-readonly /dev/nvmeXn1[,/dev/nvmeYn1]   reads the block device(s) directly with
#   fio --readonly (fio itself refuses any write). No file is created.
#
# Read jobs (all O_DIRECT, --readonly):
#   rr4k_qd1      randread 4k   iodepth 1   numjobs 1
#   rr4k_qd32_j4  randread 4k   iodepth 32  numjobs 4
#   rr128k_qd32   randread 128k iodepth 32  numjobs 1
#   rr1m_qd8      randread 1m   iodepth 8   numjobs 1
#   rr16m_qd4     randread 16m  iodepth 4   numjobs 1
#   sr1m_qd32     read (seq) 1m iodepth 32  numjobs 1
# Output: one fio JSON per run (--output-format=json), drive temperatures before/after
# each run (sysfs hwmon; plus nvme smart-log when root), summary.txt / summary.csv with
# bandwidth, IOPS and p50/p95/p99/p99.9 completion latency.
# Refuses to run without --run; --plan only prints what would run.
set -u
export LC_ALL=C
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/bench_common.sh
. "$KIT_DIR/bench_common.sh"
PATH="$PATH:/usr/local/sbin:/usr/sbin:/sbin"

usage() {
  cat <<'EOF'
usage: bench_nvme.sh (--run | --plan) (--dir DIR | --all DIR1,DIR2,...) --size-gib N [options]
       bench_nvme.sh (--run | --plan) --raw-readonly /dev/nvmeXn1[,/dev/nvmeYn1] [options]
options:
  --out DIAGDIR   results go to DIAGDIR/bench/nvme/<UTC timestamp>/
  --runtime S     seconds per job (default 30)     --ramp S   ramp time (default 5)
  --ioengine E    fio ioengine (default libaio)
  --jobs LIST     subset of: rr4k_qd1,rr4k_qd32_j4,rr128k_qd32,rr1m_qd8,rr16m_qd4,sr1m_qd32
EOF
}

RUN=0; PLAN=0; OUT=""; DIRS_ARG=""; SIZE_GIB=""; RAW_ARG=""; RT=30; RAMP=5; ENGINE=libaio; JOBSEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift;;
    --plan) PLAN=1; shift;;
    --out) OUT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --dir) DIRS_ARG=${2:-}; MODE_ALL=0; shift 2 || { usage >&2; exit 2; };;
    --all) DIRS_ARG=${2:-}; MODE_ALL=1; shift 2 || { usage >&2; exit 2; };;
    --size-gib) SIZE_GIB=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --raw-readonly) RAW_ARG=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --runtime) RT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --ramp) RAMP=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --ioengine) ENGINE=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --jobs) JOBSEL=${2:-}; shift 2 || { usage >&2; exit 2; };;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
MODE_ALL=${MODE_ALL:-0}
case "$RT$RAMP" in *[!0-9]*|'') echo "--runtime/--ramp must be integers" >&2; exit 2;; esac
case "$ENGINE" in *[!a-z0-9_]*|'') echo "bad --ioengine" >&2; exit 2;; esac

# name|rw|bs|iodepth|numjobs
ALLJOBS=("rr4k_qd1|randread|4k|1|1" "rr4k_qd32_j4|randread|4k|32|4" "rr128k_qd32|randread|128k|32|1"
         "rr1m_qd8|randread|1m|8|1" "rr16m_qd4|randread|16m|4|1" "sr1m_qd32|read|1m|32|1")
JOBS=()
for j in "${ALLJOBS[@]}"; do
  if [ -z "$JOBSEL" ] || [[ ",$JOBSEL," == *",${j%%|*},"* ]]; then JOBS+=("$j"); fi
done
[ ${#JOBS[@]} -gt 0 ] || { echo "no job selected by --jobs $JOBSEL" >&2; exit 2; }

TARGETS=(); MODE=""
if [ -n "$RAW_ARG" ] && [ -n "$DIRS_ARG" ]; then echo "use either --dir/--all or --raw-readonly, not both" >&2; exit 2; fi
if [ -n "$RAW_ARG" ]; then
  MODE="raw"
  IFS=',' read -r -a TARGETS <<<"$RAW_ARG"
  for d in "${TARGETS[@]}"; do
    case "$d" in /dev/*) ;; *) echo "refusing: $d is not a /dev path" >&2; exit 2;; esac
    [ -b "$d" ] || { echo "refusing: $d is not a block device" >&2; exit 2; }
  done
elif [ -n "$DIRS_ARG" ]; then
  MODE="file"
  IFS=',' read -r -a DIRS <<<"$DIRS_ARG"
  case "$SIZE_GIB" in ''|*[!0-9]*|0) echo "--size-gib N (positive integer) is required in file mode" >&2; exit 2;; esac
  for d in "${DIRS[@]}"; do
    case "$d" in /dev|/dev/*) echo "refusing: $d is under /dev; file mode needs a directory on a mounted file system" >&2; exit 2;; esac
    [ -d "$d" ] || { echo "refusing: $d is not a directory" >&2; exit 2; }
    [ -w "$d" ] || { echo "refusing: $d is not writable" >&2; exit 2; }
  done
else
  usage >&2; exit 2
fi

HOST=$(bench_host); TS=$(bench_ts)
[ -n "$OUT" ] || OUT="diag-out/$HOST-$TS-bench"
OUTD="$OUT/bench/nvme/$TS"
if [ "$MODE" = file ]; then
  for d in "${DIRS[@]}"; do TARGETS+=("$(cd "$d" && pwd)/diag-fio-$HOST-$TS/testfile.bin"); done
fi

fio_common() { # fio_common JSON RW BS QD NJ -> prints argv (one per line) of the common part
  printf '%s\n' fio --output-format=json "--output=$1" --direct=1 "--ioengine=$ENGINE" "--rw=$2" "--bs=$3" \
    "--iodepth=$4" "--numjobs=$5" --time_based "--runtime=$RT" "--ramp_time=$RAMP" \
    --percentile_list=50:95:99:99.9 --group_reporting --norandommap --randrepeat=0 --readonly
  [ "$MODE" = file ] && printf '%s\n' "--size=${SIZE_GIB}G"
}

plan() {
  local n_solo=${#TARGETS[@]} n_conc=0 per j
  [ "$n_solo" -gt 1 ] && n_conc=1
  per=$(( (RT + RAMP) * ${#JOBS[@]} ))
  echo "bench_nvme.sh plan ($MODE mode)"
  echo "  output: $OUTD"
  if [ "$MODE" = file ]; then
    local i=0 d
    for d in "${DIRS[@]}"; do
      echo "  dir $d: needs >= $((2 * SIZE_GIB)) GiB free; creates ${TARGETS[$i]} ($SIZE_GIB GiB, sequential write once), deletes it at the end"
      i=$((i + 1))
    done
    echo "  prep:   fio --name=prep --filename=<test file> --rw=write --bs=1m --iodepth=16 --direct=1 --size=${SIZE_GIB}G --end_fsync=1 (writes ONLY the new test file)"
  else
    for d in "${TARGETS[@]}"; do echo "  device $d: read only (fio --readonly), no file is created"; done
    echo "  note:   never-written LBAs can read back faster than real data; prefer file mode"
  fi
  echo "  jobs (each: --direct=1 --readonly --time_based --runtime=$RT --ramp_time=$RAMP --ioengine=$ENGINE):"
  for j in "${JOBS[@]}"; do IFS='|' read -r n rw bs qd nj <<<"$j"; printf '    %-13s rw=%-8s bs=%-5s iodepth=%-3s numjobs=%s\n' "$n" "$rw" "$bs" "$qd" "$nj"; done
  echo "  runs:   each target alone$([ $n_conc -eq 1 ] && echo ', then all targets concurrently (one fio, one job group per target)')"
  echo "  expected duration: about $(( per * (n_solo + n_conc) / 60 + 1 )) min plus the prep write"
  echo "  temperatures: before/after every run from /sys/class/nvme/*/hwmon* (and nvme smart-log when root)"
}

plan
if [ $PLAN -eq 1 ]; then exit 0; fi
if [ $RUN -ne 1 ]; then echo "Refusing to run without --run (nothing was executed)."; exit 2; fi
command -v fio >/dev/null 2>&1 || { echo "NOT AVAILABLE: fio not found"; exit 3; }

if [ "$MODE" = file ]; then
  for d in "${DIRS[@]}"; do
    avail=$(df -P -B1 "$d" 2>/dev/null | awk 'NR == 2 {print $4}')
    case "$avail" in ''|*[!0-9]*) echo "refusing: cannot read free space of $d" >&2; exit 2;; esac
    if [ "$avail" -lt $((2 * SIZE_GIB * 1073741824)) ]; then
      echo "refusing: $d has $((avail / 1073741824)) GiB free, need >= $((2 * SIZE_GIB)) GiB (2 x --size-gib)" >&2; exit 2
    fi
    fst=$(stat -f -c %T "$d" 2>/dev/null)
    [ "$fst" = tmpfs ] && echo "warning: $d is on tmpfs (no O_DIRECT, not a drive); results will be meaningless"
  done
fi
mkdir -p "$OUTD" || exit 2

CREATED_FILES=(); CREATED_DIRS=()
cleanup() {
  local f d
  for f in ${CREATED_FILES[@]+"${CREATED_FILES[@]}"}; do [ -f "$f" ] && rm -f -- "$f" && echo "deleted test file $f"; done
  for d in ${CREATED_DIRS[@]+"${CREATED_DIRS[@]}"}; do [ -d "$d" ] && rmdir -- "$d" 2>/dev/null; done
  CREATED_FILES=(); CREATED_DIRS=()
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

IS_ROOT=0; [ "$(id -u)" = 0 ] && IS_ROOT=1
TEMPS="$OUTD/temps.tsv"
printf 'utc\tlabel\tcontroller\tsource\ttemp_c\n' > "$TEMPS"
record_temps() { # record_temps LABEL
  local c hw t now name
  now=$(bench_utc)
  for c in /sys/class/nvme/nvme*; do
    [ -e "$c" ] || continue
    name=${c##*/}
    for hw in "$c"/hwmon* "$c"/device/hwmon/hwmon*; do
      [ -r "$hw/temp1_input" ] || continue
      t=$(cat "$hw/temp1_input" 2>/dev/null)
      case "$t" in ''|*[!0-9-]*) continue;; esac
      printf '%s\t%s\t%s\tsysfs_hwmon\t%s\n' "$now" "$1" "$name" "$(awk -v t="$t" 'BEGIN {printf "%.1f", t / 1000}')" >> "$TEMPS"
      break
    done
    if [ $IS_ROOT -eq 1 ] && command -v nvme >/dev/null 2>&1 && [ -e "/dev/$name" ]; then
      t=$(nvme smart-log "/dev/$name" 2>/dev/null | awk -F: 'tolower($1) ~ /^temperature[ \t]*$/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
      [ -n "$t" ] && printf '%s\t%s\t%s\tnvme_smart_log\t%s\n' "$now" "$1" "$name" "$t" >> "$TEMPS"
    fi
  done
}

# run_fio NAME RW BS QD NJ TARGET...   (always --readonly)
run_fio() {
  local name=$1 rw=$2 bs=$3 qd=$4 nj=$5 i=0 t
  shift 5
  local -a args
  mapfile -t args < <(fio_common "$OUTD/$name.json" "$rw" "$bs" "$qd" "$nj")
  for t in "$@"; do args+=("--name=d$i" "--filename=$t" --new_group); i=$((i + 1)); done
  record_temps "$name before"
  run_logged "$OUTD/$name.txt" "${args[@]}"
  local rc=$?
  record_temps "$name after"
  return $rc
}

# prep_file FILE (file mode only; writes only FILE, which this script just created the directory for)
prep_file() {
  case "$1" in /dev/*) echo "internal error: prep target under /dev" >&2; exit 2;; esac
  record_temps "prep $(basename "$(dirname "$(dirname "$1")")") before"
  run_logged "$OUTD/prep-$2.txt" fio --output-format=json "--output=$OUTD/prep-$2.json" --name=prep "--filename=$1" \
    --rw=write --bs=1m --iodepth=16 --direct=1 "--ioengine=$ENGINE" "--size=${SIZE_GIB}G" --end_fsync=1
  local rc=$?
  record_temps "prep $(basename "$(dirname "$(dirname "$1")")") after"
  return $rc
}

rc_all=0
if [ "$MODE" = file ]; then
  i=0
  for t in "${TARGETS[@]}"; do
    wd=$(dirname "$t")
    mkdir -- "$wd" || { echo "cannot create $wd (exists?)"; exit 2; }
    CREATED_DIRS+=("$wd"); CREATED_FILES+=("$t")
    prep_file "$t" "d$i" || { echo "prep failed for $t"; exit 3; }
    sz=$(stat -c %s "$t" 2>/dev/null || echo 0)
    [ "$sz" -ge $((SIZE_GIB * 1073741824)) ] || { echo "test file $t has $sz bytes, expected $((SIZE_GIB * 1073741824))"; exit 3; }
    i=$((i + 1))
  done
fi

{
  echo "mode=$MODE runtime=$RT ramp=$RAMP ioengine=$ENGINE size_gib=${SIZE_GIB:-n/a}"
  i=0; for t in "${TARGETS[@]}"; do echo "d$i=$t"; i=$((i + 1)); done
} > "$OUTD/targets.txt"

for j in "${JOBS[@]}"; do
  IFS='|' read -r n rw bs qd nj <<<"$j"
  i=0
  for t in "${TARGETS[@]}"; do run_fio "solo-d$i-$n" "$rw" "$bs" "$qd" "$nj" "$t" || rc_all=1; i=$((i + 1)); done
  if [ ${#TARGETS[@]} -gt 1 ]; then run_fio "concurrent-$n" "$rw" "$bs" "$qd" "$nj" "${TARGETS[@]}" || rc_all=1; fi
done

cleanup
if command -v python3 >/dev/null 2>&1; then
  python3 "$KIT_DIR/summarize.py" --fio-summary --csv "$OUTD/summary.csv" "$OUTD"/*.json > "$OUTD/summary.txt" 2>&1
  cat "$OUTD/summary.txt"
else
  echo "python3 not available: no summary.txt; the fio JSON files are complete"
fi
echo "results: $OUTD"
exit $rc_all
