#!/usr/bin/env bash
# scripts/diag/bench_gpu.sh - GPU P2P / PCIe bandwidth matrix and optional VRAM check (opt-in).
#
# Compiles gpu_p2p_matrix.cu (and gpu_vram_check.cu with --vram-check) with nvcc into the
# output directory, then runs them with CUDA_DEVICE_ORDER=PCI_BUS_ID (CUDA numbering =
# nvidia-smi numbering). While the bandwidth test runs, nvidia-smi samples the PCIe link
# generation/width every 500 ms (read-only query) to see the link speed under load.
# No GPU setting is changed. Peer access is enabled only inside the test process.
# Refuses to run without --run; --plan only prints what would run.
set -u
export LC_ALL=C
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=SCRIPTDIR/bench_common.sh
. "$KIT_DIR/bench_common.sh"
PATH="$PATH:/usr/local/cuda/bin"

usage() {
  cat <<'EOF'
usage: bench_gpu.sh (--run | --plan) [--out DIAGDIR] [--arch sm_89] [--nvcc PATH]
                    [--size-mib 256] [--reps 5] [--numa-node N]
                    [--vram-check [--margin-mib 512] [--force]]

  --run          execute (without it the plan is printed and the script exits with code 2)
  --plan         print the plan and exit 0
  --out DIAGDIR  results go to DIAGDIR/bench/gpu/<UTC timestamp>/
  --arch ARCH    nvcc -arch value (default sm_89)
  --numa-node N  run the bandwidth test under numactl --cpunodebind=N --membind=N
  --vram-check   also run gpu_vram_check (fills almost all free VRAM of every GPU with
                 test patterns and verifies them; refuses if compute processes run on a GPU)
  --force        pass --force to gpu_vram_check (run even if GPU processes are listed)
EOF
}

RUN=0; PLAN=0; OUT=""; ARCH=sm_89; NVCC_ARG=""; SIZE_MIB=256; REPS=5; NUMA_NODE=""; VRAM=0; FORCE=0; MARGIN=512
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift;;
    --plan) PLAN=1; shift;;
    --out) OUT=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --arch) ARCH=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --nvcc) NVCC_ARG=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --size-mib) SIZE_MIB=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --reps) REPS=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --numa-node) NUMA_NODE=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --vram-check) VRAM=1; shift;;
    --margin-mib) MARGIN=${2:-}; shift 2 || { usage >&2; exit 2; };;
    --force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$SIZE_MIB$REPS$MARGIN$NUMA_NODE" in *[!0-9]*|'') echo "numeric options must be integers" >&2; exit 2;; esac

NVCC=$(find_nvcc "$NVCC_ARG")
[ -n "$OUT" ] || OUT="diag-out/$(bench_host)-$(bench_ts)-bench"
OUTD="$OUT/bench/gpu/$(bench_ts)"
P2P_CMD=(env CUDA_DEVICE_ORDER=PCI_BUS_ID)
[ -n "$NUMA_NODE" ] && P2P_CMD+=(numactl "--cpunodebind=$NUMA_NODE" "--membind=$NUMA_NODE")
P2P_CMD+=("$OUTD/gpu_p2p_matrix" --size-mib "$SIZE_MIB" --reps "$REPS" --csv "$OUTD/p2p_matrix.csv")
VRAM_CMD=(env CUDA_DEVICE_ORDER=PCI_BUS_ID "$OUTD/gpu_vram_check" --margin-mib "$MARGIN")
[ $FORCE -eq 1 ] && VRAM_CMD+=(--force)

plan() {
  echo "bench_gpu.sh plan"
  echo "  output:  $OUTD"
  echo "  nvcc:    ${NVCC:-NOT FOUND (install the CUDA toolkit or pass --nvcc PATH)}"
  echo "  compile: nvcc -O2 -arch=$ARCH -o $OUTD/gpu_p2p_matrix $KIT_DIR/gpu_p2p_matrix.cu"
  [ $VRAM -eq 1 ] && echo "  compile: nvcc -O2 -arch=$ARCH -o $OUTD/gpu_vram_check $KIT_DIR/gpu_vram_check.cu"
  echo "  run:     $(printf '%q ' "${P2P_CMD[@]}")"
  echo "           (2 x $SIZE_MIB MiB per GPU + $SIZE_MIB MiB pinned host memory per GPU; about 1-3 minutes for 4 GPUs)"
  echo "  sample:  nvidia-smi --query-gpu=...pcie.link.gen.current,pcie.link.width.current... --format=csv -lms 500 (read-only)"
  if [ $VRAM -eq 1 ]; then
    echo "  run:     $(printf '%q ' "${VRAM_CMD[@]}")"
    echo "           (fills all free VRAM minus $MARGIN MiB on every GPU, 6 patterns; about 1-2 minutes per GPU;"
    echo "            refuses if nvidia-smi lists compute processes on a GPU$([ $FORCE -eq 1 ] && echo ', overridden by --force'))"
  fi
  echo "  safety:  no GPU setting is changed; peer access is enabled only inside the test process"
}

plan
if [ $PLAN -eq 1 ]; then exit 0; fi
if [ $RUN -ne 1 ]; then echo "Refusing to run without --run (nothing was executed)."; exit 2; fi
[ -n "$NVCC" ] || { echo "NOT AVAILABLE: nvcc not found"; exit 3; }
[ -z "$NUMA_NODE" ] || command -v numactl >/dev/null 2>&1 || { echo "NOT AVAILABLE: numactl not found"; exit 3; }
mkdir -p "$OUTD" || exit 2

run_logged "$OUTD/compile_p2p.txt" "$NVCC" -O2 "-arch=$ARCH" -o "$OUTD/gpu_p2p_matrix" "$KIT_DIR/gpu_p2p_matrix.cu" \
  || { echo "compile failed"; exit 3; }
if [ $VRAM -eq 1 ]; then
  run_logged "$OUTD/compile_vram.txt" "$NVCC" -O2 "-arch=$ARCH" -o "$OUTD/gpu_vram_check" "$KIT_DIR/gpu_vram_check.cu" \
    || { echo "compile failed"; exit 3; }
fi

SAMPLER=""
cleanup() { [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM
if command -v nvidia-smi >/dev/null 2>&1; then
  run_logged "$OUTD/compute_apps_before.txt" nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,power.draw,temperature.gpu \
    --format=csv -lms 500 > "$OUTD/link_samples.csv" 2> "$OUTD/link_samples.err" &
  SAMPLER=$!
else
  echo "nvidia-smi not found: no link samples, no compute-process check before the run"
fi

rc_all=0
run_logged "$OUTD/p2p_matrix.txt" "${P2P_CMD[@]}" || rc_all=1
cleanup; SAMPLER=""
if [ $VRAM -eq 1 ]; then
  run_logged "$OUTD/vram_check.txt" "${VRAM_CMD[@]}" || rc_all=1
fi
echo "results: $OUTD"
exit $rc_all
