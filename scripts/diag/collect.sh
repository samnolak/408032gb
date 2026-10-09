#!/usr/bin/env bash
# scripts/diag/collect.sh - read-only hardware/software inventory of an inference server.
#
# One entry point. Every probe is a literal command; its command line, UTC start time,
# exit code, duration and full stdout/stderr go to probes/NNN_<name>.txt, and a summary
# record goes to manifest.json. A failing probe never stops the run. At the end the
# output directory is packed into <dir>.tar.gz next to it.
#
# Safety: probes only READ. No settings are changed, no module is loaded, no service is
# touched, nothing is written to block devices, nothing is installed or downloaded.
# The only network access is the optional --net-check (one HEAD request, 5 s timeout).
#
# Root: probes that need root are recorded as "needs-root" when run as a normal user.
# With --sudo they are run through "sudo -n" (non-interactive; never asks a password).
#
# Statuses in manifest.json: ok | failed | not-available | needs-root | skipped
#
# Test hook (used by tests/test_diag_kit.py only): DIAG_ASSUME_NONROOT=1 makes the kit
# behave as if it were not root.
set -u
export LC_ALL=C
umask 022

KIT_VERSION=1
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SYSFS_HELPER="$KIT_DIR/probes_sysfs.sh"
ARGV_STR="$0 $*"

usage() {
  cat <<'EOF'
usage: collect.sh [--out DIR] [--sudo] [--net-check] [--timeout SEC] [--no-tar] [--no-summary]

  --out DIR      output directory (default: ./diag-out/<hostname>-<UTC timestamp>);
                 must not exist or be empty
  --sudo         run root-only probes via "sudo -n" (never prompts; if sudo needs a
                 password those probes are recorded as needs-root)
  --net-check    also test internet reachability: ONE HEAD request to https://github.com
                 with a 5 s timeout (default: off, no network access at all)
  --timeout SEC  per-probe timeout in seconds (default 120)
  --no-tar       do not create <DIR>.tar.gz
  --no-summary   do not run summarize.py (HARDWARE_FACTS.draft.md)
  -h, --help     this text

Read-only: nothing is changed on the machine; see docs/reassessment/HW_DIAG_RUNBOOK.md.
EOF
}

OUT=""; USE_SUDO=0; NET_CHECK=0; TMO=120; MAKE_TAR=1; RUN_SUMMARY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || { echo "--out needs a value" >&2; exit 2; }; OUT=$2; shift 2;;
    --out=*) OUT=${1#--out=}; shift;;
    --sudo) USE_SUDO=1; shift;;
    --net-check) NET_CHECK=1; shift;;
    --timeout) [ $# -ge 2 ] || { echo "--timeout needs a value" >&2; exit 2; }; TMO=$2; shift 2;;
    --timeout=*) TMO=${1#--timeout=}; shift;;
    --no-tar) MAKE_TAR=0; shift;;
    --no-summary) RUN_SUMMARY=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$TMO" in ''|*[!0-9]*) echo "--timeout must be a positive integer" >&2; exit 2;; esac
[ "$TMO" -gt 0 ] || { echo "--timeout must be a positive integer" >&2; exit 2; }

# Admin tools live in sbin on Debian/Proxmox; nvcc usually in /usr/local/cuda/bin.
# Appended, so the caller's own PATH order wins.
PATH="$PATH:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/cuda/bin"
export PATH

IS_ROOT=0
[ "$(id -u 2>/dev/null || echo 1)" = 0 ] && IS_ROOT=1
ASSUME_NONROOT=0
if [ "${DIAG_ASSUME_NONROOT:-0}" = 1 ]; then IS_ROOT=0; ASSUME_NONROOT=1; fi
SUDO_OK=0; SUDO_STATE=not-requested
if [ $USE_SUDO -eq 1 ]; then
  if [ $IS_ROOT -eq 1 ]; then SUDO_STATE=not-needed-already-root
  elif command -v sudo >/dev/null 2>&1 && sudo -n true </dev/null >/dev/null 2>&1; then SUDO_OK=1; SUDO_STATE=ok
  else SUDO_STATE=unavailable-password-required-or-no-sudo
  fi
fi
HAVE_TIMEOUT=0; command -v timeout >/dev/null 2>&1 && HAVE_TIMEOUT=1

HOST=$(uname -n 2>/dev/null || echo unknown-host)
HOST_SAFE=$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_')
TS=$(date -u +%Y%m%dT%H%M%SZ)
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
[ -n "$OUT" ] || OUT="diag-out/${HOST_SAFE}-${TS}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "refusing: output directory $OUT exists and is not empty" >&2; exit 2
fi
mkdir -p "$OUT/probes" "$OUT/.tmp" || { echo "cannot create $OUT" >&2; exit 2; }
OUT=$(cd "$OUT" && pwd)
TMPD="$OUT/.tmp"
LOG="$OUT/collect.log"
PARTS="$TMPD/manifest.parts"
: > "$PARTS"

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
q() { printf '%q' "$1"; }
now_us() {
  if [ -n "${EPOCHREALTIME:-}" ]; then local t=${EPOCHREALTIME/./}; echo "${t/,/}"
  else echo $(( $(date +%s) * 1000000 )); fi
}
fmt_dur() { printf '%d.%03d' $(( $1 / 1000000 )) $(( ($1 % 1000000) / 1000 )); }
json_s() { # JSON string literal
  local s=$1
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/\\r}
  printf '"%s"' "$s"
}
json_list() { # json_list "a b c" -> ["a","b","c"]
  local out="" x
  for x in $1; do [ "$x" = "-" ] && continue; out="$out,$(json_s "$x")"; done
  printf '[%s]' "${out#,}"
}

SEQ=0
declare -A COUNT=([ok]=0 [failed]=0 [not-available]=0 [needs-root]=0 [skipped]=0)

record() { # record NAME CMD REQUIRES ROOTMODE RUN_AS START_UTC RC DUR FILE STATUS NOTE
  local rc_json=null
  [ -n "$7" ] && rc_json=$7
  printf '{"seq":%d,"name":%s,"category":%s,"command":%s,"requires":%s,"root_mode":%s,"run_as":%s,"started_utc":%s,"exit_code":%s,"duration_s":%s,"output_file":%s,"status":%s,"note":%s}\n' \
    "$SEQ" "$(json_s "$1")" "$(json_s "${1%%.*}")" "$(json_s "$2")" "$(json_list "$3")" \
    "$(json_s "$4")" "$(json_s "$5")" "$(json_s "$6")" "$rc_json" "$8" "$(json_s "$9")" \
    "$(json_s "${10}")" "$(json_s "${11}")" >> "$PARTS"
  COUNT[${10}]=$(( ${COUNT[${10}]:-0} + 1 ))
  printf '[%3d] %-44s %-13s %ss%s\n' "$SEQ" "$1" "${10}" "$8" "${11:+  (${11})}" | tee -a "$LOG"
}

write_probe_file() { # FILE NAME CMD REQUIRES ROOTMODE RUN_AS START_UTC TMO RC DUR STATUS NOTE
  {
    echo "===== PROBE ====="
    echo "name: $2"
    echo "started_utc: $7"
    echo "root_mode: $5"
    echo "run_as: $6"
    echo "requires: $4"
    echo "timeout_s: $8"
    if [ "$6" = not-run ]; then echo "runner: (not run)"
    elif [ $HAVE_TIMEOUT -eq 1 ]; then echo "runner: timeout -k 5 $8 bash -o pipefail -c <command>"
    else echo "runner: bash -o pipefail -c <command> (no timeout command available)"; fi
    echo "===== COMMAND ====="
    printf '%s\n' "$3"
    echo "===== STDOUT ====="
    if [ -s "$TMPD/out" ]; then cat "$TMPD/out"; [ -n "$(tail -c 1 "$TMPD/out")" ] && echo; fi
    echo "===== STDERR ====="
    if [ -s "$TMPD/err" ]; then cat "$TMPD/err"; [ -n "$(tail -c 1 "$TMPD/err")" ] && echo; fi
    echo "===== RESULT ====="
    echo "exit_code: ${9:-not-run}"
    echo "duration_s: ${10}"
    echo "status: ${11}"
    echo "note: ${12}"
  } > "$OUT/$1"
}

# probe NAME ROOTMODE REQUIRES CMD [TIMEOUT]
#   ROOTMODE: none    - never needs root
#             root    - needs root; as a normal user it is NOT run (needs-root) unless --sudo works
#             partial - runs as the current user; a failure without root is recorded as needs-root,
#                       a success without root is "ok" with a note that output may be incomplete
#   REQUIRES: space-separated tools (looked up in PATH) or absolute paths; "a|b" = any of; "-" = none
# probe_h NAME ROOTMODE REQUIRES [TIMEOUT] <<'EOF' ... EOF   (command read from stdin)
probe() {
  local name=$1 rootmode=$2 requires=$3 cmd=$4 tmo=${5:-$TMO}
  local file start_utc start_us rc="" status note="" run_as missing="" r alt any dur
  local -a runner
  SEQ=$((SEQ + 1))
  file=$(printf 'probes/%03d_%s.txt' "$SEQ" "$name")
  start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_us=$(now_us)
  : > "$TMPD/out"; : > "$TMPD/err"
  for r in $requires; do
    [ "$r" = "-" ] && continue
    any=0
    for alt in ${r//|/ }; do
      case "$alt" in
        /*) [ -e "$alt" ] && any=1;;
        *) command -v "$alt" >/dev/null 2>&1 && any=1;;
      esac
    done
    [ $any -eq 1 ] || missing="$missing $r"
  done
  if [ -n "$missing" ]; then
    status=not-available; run_as=not-run; note="not found:$missing"
  elif [ "$rootmode" = root ] && [ $IS_ROOT -eq 0 ] && [ $SUDO_OK -eq 0 ]; then
    status=needs-root; run_as=not-run; note="requires root; not run (rerun as root or with --sudo)"
  else
    if [ $IS_ROOT -eq 1 ]; then run_as=root; runner=(bash -o pipefail -c "$cmd")
    elif [ "$rootmode" != none ] && [ $SUDO_OK -eq 1 ]; then
      run_as=sudo; runner=(sudo -n env LC_ALL=C "PATH=$PATH" bash -o pipefail -c "$cmd")
    else run_as=user; runner=(bash -o pipefail -c "$cmd"); fi
    [ $HAVE_TIMEOUT -eq 1 ] && runner=(timeout -k 5 "$tmo" "${runner[@]}")
    "${runner[@]}" </dev/null >"$TMPD/out" 2>"$TMPD/err"
    rc=$?
    if [ $rc -eq 0 ]; then
      status=ok
      [ "$rootmode" = partial ] && [ "$run_as" = user ] && note="ran without root: output may be incomplete"
    elif [ $HAVE_TIMEOUT -eq 1 ] && { [ $rc -eq 124 ] || [ $rc -eq 137 ]; }; then
      status=failed; note="timed out after ${tmo}s"
    elif [ "$rootmode" != none ] && [ "$run_as" = user ]; then
      status=needs-root; note="failed without root (exit $rc); rerun as root or with --sudo"
    else
      status=failed
      [ $rc -eq 127 ] && note="command not found inside the probe (exit 127)"
    fi
  fi
  dur=$(fmt_dur $(( $(now_us) - start_us )))
  write_probe_file "$file" "$name" "$cmd" "$requires" "$rootmode" "$run_as" "$start_utc" "$tmo" "$rc" "$dur" "$status" "$note"
  record "$name" "$cmd" "$requires" "$rootmode" "$run_as" "$start_utc" "$rc" "$dur" "$file" "$status" "$note"
}
probe_h() { local c; c=$(cat); probe "$1" "$2" "$3" "$c" "${4:-$TMO}"; }

skip_probe() { # skip_probe NAME CMD REASON
  local file start_utc
  SEQ=$((SEQ + 1))
  file=$(printf 'probes/%03d_%s.txt' "$SEQ" "$1")
  start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  : > "$TMPD/out"; : > "$TMPD/err"
  write_probe_file "$file" "$1" "$2" "-" none not-run "$start_utc" "$TMO" "" "0.000" skipped "$3"
  record "$1" "$2" "-" none not-run "$start_utc" "" "0.000" "$file" skipped "$3"
}

finalize() {
  local finished rc_note=${1:-completed} first=1 line
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  {
    echo "{"
    echo "  \"kit\": \"408032gb scripts/diag\","
    echo "  \"kit_version\": $KIT_VERSION,"
    echo "  \"schema\": 1,"
    echo "  \"synthetic\": false,"
    echo "  \"run_state\": $(json_s "$rc_note"),"
    echo "  \"hostname\": $(json_s "$HOST"),"
    echo "  \"started_utc\": $(json_s "$STARTED_UTC"),"
    echo "  \"finished_utc\": $(json_s "$finished"),"
    echo "  \"argv\": $(json_s "$ARGV_STR"),"
    echo "  \"uid\": $(id -u 2>/dev/null || echo -1),"
    echo "  \"user\": $(json_s "$(id -un 2>/dev/null || echo unknown)"),"
    echo "  \"is_root\": $([ $IS_ROOT -eq 1 ] && echo true || echo false),"
    echo "  \"assume_nonroot_test_hook\": $([ $ASSUME_NONROOT -eq 1 ] && echo true || echo false),"
    echo "  \"sudo\": $(json_s "$SUDO_STATE"),"
    echo "  \"net_check\": $([ $NET_CHECK -eq 1 ] && echo true || echo false),"
    echo "  \"timeout_s\": $TMO,"
    echo "  \"have_timeout_cmd\": $([ $HAVE_TIMEOUT -eq 1 ] && echo true || echo false),"
    echo "  \"path\": $(json_s "$PATH"),"
    echo "  \"kit_dir\": $(json_s "$KIT_DIR"),"
    echo "  \"counts\": {\"ok\": ${COUNT[ok]}, \"failed\": ${COUNT[failed]}, \"not-available\": ${COUNT[not-available]}, \"needs-root\": ${COUNT[needs-root]}, \"skipped\": ${COUNT[skipped]}},"
    echo "  \"probes\": ["
    while IFS= read -r line; do
      if [ $first -eq 1 ]; then first=0; else echo ","; fi
      printf '    %s' "$line"
    done < "$PARTS"
    [ $first -eq 1 ] || echo
    echo "  ]"
    echo "}"
  } > "$OUT/manifest.json"
}
# shellcheck disable=SC2329  # invoked through trap
on_signal() {
  say "interrupted: writing a partial manifest"
  finalize interrupted
  rm -rf "$TMPD"
  exit 130
}
trap on_signal INT TERM

say "diag kit v$KIT_VERSION: $STARTED_UTC host=$HOST out=$OUT"
say "root=$IS_ROOT sudo=$SUDO_STATE net_check=$NET_CHECK timeout=${TMO}s"
[ $IS_ROOT -eq 1 ] || say "note: not root: root-only probes are recorded as needs-root (use --sudo or run as root)"

H=$(q "$SYSFS_HELPER")
K=$(q "$KIT_DIR")

# ---------------------------------------------------------------- meta
probe meta.id none "id" 'id'
probe meta.kit_sha256 none "sha256sum" "cd $K && sha256sum -- *.sh *.py *.c *.cu"
probe meta.kit_git none "git" "git -C $K log -1 --format='%H %cI %s' && git -C $K status --short -- ."

# ---------------------------------------------------------------- system
probe system.uname none "uname" 'uname -a'
probe system.os_release none "/etc/os-release" 'cat /etc/os-release'
probe system.cmdline none "/proc/cmdline" 'cat /proc/cmdline'
probe system.virt none "systemd-detect-virt" 'systemd-detect-virt || [ $? -eq 1 ]'
probe system.cpuinfo_hypervisor_flag none "/proc/cpuinfo" 'grep -m1 -c -w hypervisor /proc/cpuinfo || [ $? -eq 1 ]'
probe system.uptime none "uptime" 'uptime'
probe_h system.dmesg_filtered partial "dmesg" <<'EOF'
out=$(dmesg -T) || exit $?
printf '%s\n' "$out" | grep -iE 'nvidia|nvrm|xid|pcie|aer|nvme|mce|machine check|hardware error|edac|iommu|amd-vi|dmar|vfio|bar [0-9]|resizable|fallen off|link is down|limited by|downgrad' || [ $? -eq 1 ]
EOF
probe system.dmesg_full partial "dmesg" 'dmesg -T'
probe system.journal_kernel_warnings partial "journalctl" "journalctl -k -b 0 -p 0..4 --no-pager | sed -n '1,5000p'"
probe system.lsmod none "lsmod" 'lsmod'
probe system.modinfo_nvidia none "modinfo" 'modinfo nvidia'
probe system.proxmox_pveversion none "pveversion" 'pveversion -v'
probe system.proxmox_qm_list root "qm" 'qm list'
probe_h system.proxmox_qm_config root "qm" <<'EOF'
for id in $(qm list | awk 'NR > 1 {print $1}'); do echo "== VM $id"; qm config "$id"; done
EOF

# ---------------------------------------------------------------- cpu
probe cpu.lscpu none "lscpu" 'lscpu'
probe cpu.lscpu_extended none "lscpu" 'lscpu -e'
probe cpu.lscpu_parse none "lscpu" 'lscpu -p=CPU,CORE,SOCKET,NODE'
probe_h cpu.flags none "/proc/cpuinfo" <<'EOF'
flags=$(grep -m1 -E '^flags' /proc/cpuinfo)
flags=" ${flags#*:} "
for f in sse4_2 avx avx2 fma f16c bmi2 avx_vnni avx512f avx512cd avx512bw avx512dq avx512vl avx512_vnni avx512_bf16 avx512_fp16 amx_tile amx_bf16 amx_int8 sha_ni vaes; do
  case "$flags" in *" $f "*) echo "$f=yes";; *) echo "$f=no";; esac
done
echo "avx512_any=$(case "$flags" in *" avx512"*) echo yes;; *) echo no;; esac)"
EOF
probe_h cpu.cpuinfo_model none "/proc/cpuinfo" <<'EOF'
grep -m1 -E '^model name' /proc/cpuinfo
echo "processors: $(grep -c '^processor' /proc/cpuinfo)"
grep -m1 -E '^microcode' /proc/cpuinfo || true
grep -E '^cpu MHz' /proc/cpuinfo | sort | uniq -c | sort -rn | sed -n '1,10p'
EOF
probe cpu.cpupower_frequency_info none "cpupower" 'cpupower frequency-info'
probe cpu.cpufreq_sysfs none "/sys/devices/system/cpu" "bash $H cpufreq"
probe cpu.numactl_hardware none "numactl" 'numactl -H'
probe cpu.lstopo none "lstopo-no-graphics" 'lstopo-no-graphics --of console'
probe cpu.numa_sysfs none "/sys/devices/system/node" "bash $H numa"

# ---------------------------------------------------------------- memory
probe memory.meminfo none "/proc/meminfo" 'cat /proc/meminfo'
probe memory.hugepages none "/sys/kernel/mm" "bash $H hugepages"
probe memory.ulimit_l none "-" 'ulimit -l'
probe memory.ulimit_a none "-" 'ulimit -a'
probe_h memory.limits_conf_memlock none "/etc/security" <<'EOF'
for f in /etc/security/limits.conf /etc/security/limits.d/*; do [ -f "$f" ] && grep -Hn memlock "$f"; done
true
EOF
probe memory.systemd_default_memlock none "systemctl" 'systemctl show --property=DefaultLimitMEMLOCK --property=DefaultLimitMEMLOCKSoft'
probe memory.dmidecode_16_17 root "dmidecode" 'dmidecode -t 16,17'
probe memory.edac_sysfs none "/sys/devices/system/edac/mc" "bash $H edac"
probe memory.edac_util none "edac-util" 'edac-util -v'
probe memory.ras_mc_ctl_status partial "ras-mc-ctl" 'ras-mc-ctl --status'
probe memory.ras_mc_ctl_summary partial "ras-mc-ctl" 'ras-mc-ctl --summary'
probe memory.ras_mc_ctl_errors partial "ras-mc-ctl" 'ras-mc-ctl --errors'

# ---------------------------------------------------------------- board / BMC
probe board.dmidecode_0_1_2_3 root "dmidecode" 'dmidecode -t 0,1,2,3'
probe board.dmidecode_38_ipmi root "dmidecode" 'dmidecode -t 38'
probe board.dmi_sysfs none "/sys/class/dmi/id" "bash $H dmi"
IPMI_DEV="/dev/ipmi0|/dev/ipmi/0|/dev/ipmidev/0"
probe bmc.ipmitool_mc_info root "ipmitool $IPMI_DEV" 'ipmitool -I open mc info'
probe bmc.ipmitool_fru root "ipmitool $IPMI_DEV" 'ipmitool -I open fru print' 180
probe bmc.ipmitool_sdr root "ipmitool $IPMI_DEV" 'ipmitool -I open sdr list' 240
probe bmc.ipmitool_sel root "ipmitool $IPMI_DEV" 'ipmitool -I open sel list' 240

# ---------------------------------------------------------------- pcie
probe pcie.lspci_tree none "lspci" 'lspci -tv'
probe pcie.lspci_nn none "lspci" 'lspci -nn'
probe pcie.lspci_vvv_nvidia partial "lspci" 'lspci -vvv -nn -d 10de:'
probe_h pcie.lspci_vvv_nvme partial "lspci" <<'EOF'
for d in $(lspci -Dn | awk '$2 ~ /^0108/ {print $1}'); do lspci -vvv -nn -s "$d"; echo; done
EOF
probe pcie.lspci_vvv_all partial "lspci" 'lspci -vvv -nn' 240
probe pcie.sysfs_links none "/sys/bus/pci/devices" "bash $H pci-links"
probe pcie.sysfs_links_all none "/sys/bus/pci/devices" "bash $H pci-links-all"
probe pcie.iommu_groups none "/sys/kernel/iommu_groups" "bash $H iommu"
probe_h pcie.iommu_cmdline none "/proc/cmdline" <<'EOF'
grep -oE '(amd_|intel_)?iommu[^ ]*|vfio[^ ]*|pcie_[^ ]*|pci=[^ ]*|default_hugepagesz[^ ]*|hugepage[^ ]*|isolcpus[^ ]*|nohz[^ ]*|mitigations[^ ]*' /proc/cmdline || [ $? -eq 1 ]
EOF
probe_h pcie.iommu_dmesg partial "dmesg" <<'EOF'
out=$(dmesg) || exit $?
printf '%s\n' "$out" | grep -iE 'iommu|amd-vi|dmar' || [ $? -eq 1 ]
EOF

# ---------------------------------------------------------------- gpu
probe gpu.nvidia_smi none "nvidia-smi" 'nvidia-smi'
probe gpu.nvidia_smi_list none "nvidia-smi" 'nvidia-smi -L'
probe gpu.nvidia_smi_q none "nvidia-smi" 'nvidia-smi -q' 240
probe gpu.topo_matrix none "nvidia-smi" 'nvidia-smi topo -m'
for cap in r w n a p; do
  probe "gpu.topo_p2p_$cap" none "nvidia-smi" "nvidia-smi topo -p2p $cap"
done
probe gpu.query_csv none "nvidia-smi" 'nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.memory,power.limit,temperature.gpu,vbios_version --format=csv'
probe gpu.query_csv_ids none "nvidia-smi" 'nvidia-smi --query-gpu=index,uuid,serial,pci.bus_id,pci.device_id,pci.sub_device_id,driver_version,persistence_mode,compute_mode,pstate,memory.used,memory.free --format=csv'
probe gpu.query_csv_link_power none "nvidia-smi" 'nvidia-smi --query-gpu=index,pcie.link.gen.gpumax,pcie.link.gen.hostmax,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.graphics,fan.speed --format=csv'
for d in PERFORMANCE CLOCK POWER TEMPERATURE ECC PAGE_RETIREMENT ROW_REMAPPER MEMORY; do
  probe "gpu.q_${d,,}" none "nvidia-smi" "nvidia-smi -q -d $d"
done
probe gpu.compute_apps none "nvidia-smi" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv'
probe gpu.proc_driver_nvidia none "/proc/driver/nvidia" "bash $H nvidia-proc"

# ---------------------------------------------------------------- storage
probe storage.lsblk_json none "lsblk" 'lsblk -O -J'
probe storage.lsblk none "lsblk" 'lsblk -o NAME,MODEL,SERIAL,SIZE,ROTA,TRAN,TYPE,FSTYPE,MOUNTPOINTS,PHY-SEC,LOG-SEC,SCHED,RQ-SIZE || lsblk -o NAME,MODEL,SERIAL,SIZE,ROTA,TRAN,TYPE,FSTYPE,MOUNTPOINT,PHY-SEC,LOG-SEC,SCHED,RQ-SIZE'
probe storage.findmnt none "findmnt" 'findmnt'
probe storage.df none "df" 'df -hT'
probe storage.nvme_list partial "nvme" 'nvme list'
probe storage.nvme_sysfs none "/sys/class/nvme" "bash $H nvme"
for c in /sys/class/nvme/nvme*; do
  [ -e "$c" ] || continue
  cn=${c##*/}
  case "$cn" in nvme[0-9]*) ;; *) continue;; esac
  probe "storage.nvme_id_ctrl.$cn" root "nvme /dev/$cn" "nvme id-ctrl /dev/$cn"
  probe "storage.nvme_smart_log.$cn" root "nvme /dev/$cn" "nvme smart-log /dev/$cn"
  probe "storage.nvme_error_log.$cn" root "nvme /dev/$cn" "nvme error-log /dev/$cn"
  for ns in "$c"/"$cn"n*; do
    [ -e "$ns" ] || continue
    nsn=${ns##*/}
    probe "storage.nvme_id_ns.$nsn" root "nvme /dev/$nsn" "nvme id-ns -H /dev/$nsn"
  done
  probe "storage.smartctl.$cn" root "smartctl /dev/$cn" "smartctl -a /dev/$cn; rc=\$?; echo \"smartctl_exit_status=\$rc\"; [ \$((rc & 3)) -eq 0 ]"
done
probe storage.block_queues none "/sys/block" "bash $H block"
probe storage.mdstat none "/proc/mdstat" 'cat /proc/mdstat'
probe storage.mdadm_scan root "mdadm" 'mdadm --detail --scan'
probe storage.zpool_status partial "zpool" 'zpool status'
probe storage.zpool_list partial "zpool" 'zpool list -v'
probe storage.lvm root "pvs vgs lvs" 'pvs; vgs; lvs -a -o +devices'

# ---------------------------------------------------------------- software
probe software.nvcc none "nvcc" 'nvcc --version'
probe_h software.cuda_dirs none "-" <<'EOF'
ls -ld /usr/local/cuda* /opt/cuda* 2>&1
for f in /usr/local/cuda*/version.json /usr/local/cuda*/version.txt; do [ -f "$f" ] && { echo "== $f"; sed -n '1,60p' "$f"; }; done
true
EOF
probe software.gcc none "gcc" 'gcc --version'
probe software.gxx none "g++" 'g++ --version'
probe software.clang none "clang" 'clang --version'
probe software.cmake none "cmake" 'cmake --version'
probe software.make none "make" 'make --version'
probe software.ninja none "ninja" 'ninja --version'
probe software.git none "git" 'git --version'
probe software.python3 none "python3" 'python3 --version'
probe software.pip_list none "python3" "python3 -m pip list --disable-pip-version-check | sed -n '1,300p'"
probe_h software.python_ml_packages none "python3" <<'EOF'
python3 - <<'PY'
import importlib.util as u
for m in ("numpy", "torch", "transformers", "vllm", "gguf", "safetensors"):
    s = u.find_spec(m)
    print(m + ": " + ("not installed" if s is None else "installed at " + str(s.origin)))
PY
EOF
probe software.docker none "docker" 'docker --version'
probe software.podman none "podman" 'podman --version'
probe software.nvidia_container_toolkit none "nvidia-ctk|nvidia-container-cli" 'if command -v nvidia-ctk >/dev/null; then nvidia-ctk --version; fi; if command -v nvidia-container-cli >/dev/null; then nvidia-container-cli --version; fi'
probe software.cuda_libraries none "ldconfig" "ldconfig -p | grep -E 'libcuda|libcudart|cublas|nccl|cudnn|libnvidia-ml' || [ \$? -eq 1 ]"
probe_h software.llm_tools none "-" <<'EOF'
for b in ollama llama-server llama-cli llama-bench vllm fio nvbandwidth p2pBandwidthLatencyTest; do
  printf '%s\t%s\n' "$b" "$(command -v "$b" || echo 'not found')"
done
EOF

# ---------------------------------------------------------------- network
probe network.ip_br_addr none "ip" 'ip -br a'
probe network.ip_br_link none "ip" 'ip -br link'
probe network.sysfs none "/sys/class/net" "bash $H net"
NETCMD="if command -v curl >/dev/null 2>&1; then curl -sS -I --max-time 5 -o /dev/null -w 'http_code=%{http_code}\n' https://github.com; else wget --spider -T 5 -t 1 https://github.com; fi"
if [ $NET_CHECK -eq 1 ]; then
  probe network.internet_check none "curl|wget" "$NETCMD" 15
else
  skip_probe network.internet_check "$NETCMD" "off by default; enable with --net-check"
fi

# ---------------------------------------------------------------- finish
finalize completed
mkdir -p "$OUT/kit"
for f in "$KIT_DIR"/*.sh "$KIT_DIR"/*.py "$KIT_DIR"/*.c "$KIT_DIR"/*.cu; do
  [ -f "$f" ] && cp -p "$f" "$OUT/kit/"
done
rm -rf "$TMPD"
say "probes: ok=${COUNT[ok]} failed=${COUNT[failed]} not-available=${COUNT[not-available]} needs-root=${COUNT[needs-root]} skipped=${COUNT[skipped]}"
if [ $RUN_SUMMARY -eq 1 ]; then
  if command -v python3 >/dev/null 2>&1; then
    if python3 "$KIT_DIR/summarize.py" "$OUT" --output "$OUT/HARDWARE_FACTS.draft.md" >>"$LOG" 2>&1; then
      say "draft facts: $OUT/HARDWARE_FACTS.draft.md"
    else
      say "summarize.py failed (see collect.log); raw outputs are complete"
    fi
  else
    say "python3 not available: HARDWARE_FACTS.draft.md NOT generated (raw outputs are complete)"
  fi
fi
if [ $MAKE_TAR -eq 1 ]; then
  if command -v tar >/dev/null 2>&1 && tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz.part" "$(basename "$OUT")" \
     && mv -f "$OUT.tar.gz.part" "$OUT.tar.gz"; then
    say "archive: $OUT.tar.gz  <- send this file"
  else
    rm -f "$OUT.tar.gz.part"
    say "tar failed or not available: send the directory $OUT instead"
  fi
fi
exit 0
