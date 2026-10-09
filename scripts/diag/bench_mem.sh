#!/usr/bin/env bash
# scripts/diag/bench_mem.sh - host memory bandwidth with our membw.c (opt-in benchmark).
#
# Compiles membw.c into the output directory and runs it with:
#   1 thread; all physical cores with OMP_PROC_BIND=spread and =close (OMP_PLACES=cores);
#   all hardware threads (spread, OMP_PLACES=threads);
#   per NUMA node via numactl --cpunodebind=N --membind=N (if numactl and >1 node);
#   all cores with numactl --interleave=all (if numactl and >1 node).
# Only allocates and touches ordinary process memory; changes no system setting.
# Refuses to run without --run; --plan only prints what would run.
set -u
export LC_ALL=C
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/bench_common.sh
. "$KIT_DIR/bench_common.sh"

usage() {
  cat <<'EOF'
usage: bench_mem.sh (--run | --plan) [--out DIAGDIR] [--size-mib N] [--ntimes N] [--cc CC] [--no-numa]

  --run          execute (without it the plan is printed and the script exits with code 2)
  --plan         print the plan and exit 0
  --out DIAGDIR  results go to DIAGDIR/bench/mem/<UTC timestamp>/ (default ./diag-out/<host>-<ts>-bench)
  --size-mib N   MiB per array, 3 arrays are used (default 2048 -> 6 GiB in total)
  --ntimes N     runs per variant including 1 warm-up (default 11)
  --cc CC        C compiler with OpenMP (default gcc)
  --no-numa      skip the numactl variants
EOF
}

RUN=0; PLAN=0; OUT=""; SIZE_MIB=2048; NTIMES=11; CC_BIN=${CC:-gcc}; USE_NUMA=1
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift;;
    --plan) PLAN=1; shift;;
    --out) OUT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --size-mib) SIZE_MIB=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --ntimes) NTIMES=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --cc) CC_BIN=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --no-numa) USE_NUMA=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$SIZE_MIB$NTIMES" in *[!0-9]*|'') echo "--size-mib/--ntimes must be integers" >&2; exit 2;; esac

LOGICAL=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
PHYS=""
if command -v lscpu >/dev/null 2>&1; then
  PHYS=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)
fi
case "$PHYS" in ''|0) PHYS=$LOGICAL;; esac

# variants: name|threads|OMP_PROC_BIND|OMP_PLACES|numactl args (or -)
VARIANTS=("t1|1|close|cores|-" "cores_spread|$PHYS|spread|cores|-" "cores_close|$PHYS|close|cores|-"
          "threads_spread|$LOGICAL|spread|threads|-")
NODES=()
if [ $USE_NUMA -eq 1 ] && command -v numactl >/dev/null 2>&1; then
  for n in /sys/devices/system/node/node[0-9]*; do
    [ -e "$n" ] || continue
    id=${n##*node}
    cl=$(cat "$n/cpulist" 2>/dev/null)
    [ -n "$cl" ] || continue
    NODES+=("$id")
  done
  if [ ${#NODES[@]} -gt 1 ]; then
    for id in "${NODES[@]}"; do
      nc=$(lscpu -p=CORE,SOCKET,NODE 2>/dev/null | grep -v '^#' | awk -F, -v n="$id" '$3 == n {print $1","$2}' | sort -u | wc -l)
      [ "${nc:-0}" -gt 0 ] || nc=1
      VARIANTS+=("node${id}|$nc|spread|cores|--cpunodebind=$id --membind=$id")
    done
    VARIANTS+=("interleave_all|$PHYS|spread|cores|--interleave=all")
  fi
fi

[ -n "$OUT" ] || OUT="diag-out/$(bench_host)-$(bench_ts)-bench"
OUTD="$OUT/bench/mem/$(bench_ts)"
NEED_MIB=$((SIZE_MIB * 3))

plan() {
  echo "bench_mem.sh plan"
  echo "  output:   $OUTD"
  echo "  compile:  $CC_BIN -O3 -march=native -fopenmp $KIT_DIR/membw.c -o $OUTD/membw -lm"
  echo "  memory:   3 arrays x $SIZE_MIB MiB = $NEED_MIB MiB per run (process memory only)"
  echo "  cpus:     logical=$LOGICAL physical_cores=$PHYS"
  echo "  numa:     $(if [ $USE_NUMA -eq 0 ]; then echo 'skipped (--no-numa)'; elif ! command -v numactl >/dev/null 2>&1; then echo 'numactl not found: no per-node runs'; else echo "${#NODES[@]} node(s) with CPUs$([ ${#NODES[@]} -gt 1 ] || echo ': no per-node runs')"; fi)"
  echo "  runs ($NTIMES iterations each, first is warm-up):"
  local v name thr bind places na
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r name thr bind places na <<<"$v"
    printf '    %-16s OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s %s\n' "$name" "$thr" "$bind" "$places" \
      "$([ "$na" = - ] || echo "numactl $na")"
  done
  echo "  expected duration: about 1-5 minutes"
  echo "  safety: only allocates process memory; no system setting is changed"
}

plan
if [ $PLAN -eq 1 ]; then exit 0; fi
if [ $RUN -ne 1 ]; then echo "Refusing to run without --run (nothing was executed)."; exit 2; fi

command -v "$CC_BIN" >/dev/null 2>&1 || { echo "NOT AVAILABLE: compiler $CC_BIN not found"; exit 3; }
avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
if [ -n "$avail_kib" ] && [ $((avail_kib / 1024)) -lt $((NEED_MIB + NEED_MIB / 5 + 1024)) ]; then
  echo "refusing: MemAvailable $((avail_kib / 1024)) MiB < needed $((NEED_MIB + NEED_MIB / 5 + 1024)) MiB; use a smaller --size-mib"
  exit 2
fi
mkdir -p "$OUTD" || exit 2

if ! run_logged "$OUTD/compile.txt" "$CC_BIN" -O3 -march=native -fopenmp "$KIT_DIR/membw.c" -o "$OUTD/membw" -lm; then
  run_logged "$OUTD/compile_fallback.txt" "$CC_BIN" -O3 -fopenmp "$KIT_DIR/membw.c" -o "$OUTD/membw" -lm \
    || { echo "compile failed; see $OUTD/compile*.txt"; exit 3; }
fi

CSV="$OUTD/membw.csv"
echo "variant,threads,omp_proc_bind,omp_places,numactl,kernel,best_GBps,median_GBps,min_s,median_s,max_s,array_bytes,timed_runs,log" > "$CSV"
rc_all=0
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r name thr bind places na <<<"$v"
  log="$OUTD/run-$name.txt"
  cmd=(env "OMP_NUM_THREADS=$thr" "OMP_PROC_BIND=$bind" "OMP_PLACES=$places")
  if [ "$na" != - ]; then
    # shellcheck disable=SC2206
    cmd+=(numactl $na)
  fi
  cmd+=("$OUTD/membw" -s "$SIZE_MIB" -n "$NTIMES" -t "$thr")
  run_logged "$log" "${cmd[@]}" || rc_all=1
  na_csv=$na; [ "$na" = - ] && na_csv=""
  sed -n '/^===== STDOUT =====$/,/^===== STDERR =====$/p' "$log" | grep '^CSV,' | while IFS=, read -r _ k t b r best med mn md mx; do
    echo "$name,$t,$bind,$places,$na_csv,$k,$best,$med,$mn,$md,$mx,$b,$r,$(basename "$log")"
  done >> "$CSV"
done

echo
echo "summary (GB/s, best of runs; see $CSV):"
awk -F, 'NR > 1 && ($6 == "triad" || $6 == "read" || $6 == "copy") {printf "  %-16s threads=%-4s %-6s best=%9s median=%9s\n", $1, $2, $6, $7, $8}' "$CSV"
exit $rc_all
