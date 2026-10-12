#!/usr/bin/env bash
# Read-only helpers for scripts/diag/collect.sh. Every subcommand only READS files
# under /sys and /proc and prints them as text (mostly TSV with a header line).
# Nothing here writes anywhere, loads modules or changes settings.
#
# usage: probes_sysfs.sh <subcommand>
#   pci-links     GPUs (PCI class 03xx, vendor 10de) and NVMe (class 0108) with link
#                 speed/width (current and max), BAR sizes, NUMA node, IOMMU group,
#                 AER counters and the upstream bridge chain (bridges listed too)
#   pci-links-all the same columns for every PCI device (no bridge-chain expansion)
#   nvme          NVMe controllers: model, serial, firmware, PCIe link, temperature
#   block         /sys/block/*/queue settings and sizes
#   cpufreq       cpufreq policies (governor, driver, min/max), boost, amd_pstate
#   hugepages     hugepage pools per size and per node, THP mode, vm knobs (read only)
#   numa          /sys/devices/system/node: cpulists, per-node meminfo, distances
#   edac          /sys/devices/system/edac/mc: memory controllers, DIMMs, CE/UE counters
#   dmi           /sys/class/dmi/id: board/BIOS/system strings (serials need root)
#   iommu         /sys/kernel/iommu_groups: group -> devices
#   net           /sys/class/net: operstate, speed, mtu, driver
#   nvidia-proc   /proc/driver/nvidia: version, params, gpus/*/information
set -u
export LC_ALL=C

rd() { # rd FILE -> first line of FILE, or "-" if missing/unreadable
  local v
  if [ -r "$1" ] && v=$(head -n 1 -- "$1" 2>/dev/null); then
    printf '%s' "${v:--}"
  else
    printf '%s' "-"
  fi
}

human_bytes() { # human_bytes N -> 256M / 32G / 16K
  local n=$1
  if [ "$n" -ge $((1 << 30)) ] && [ $((n % (1 << 30))) -eq 0 ]; then echo "$((n >> 30))G"
  elif [ "$n" -ge $((1 << 20)) ] && [ $((n % (1 << 20))) -eq 0 ]; then echo "$((n >> 20))M"
  elif [ "$n" -ge $((1 << 10)) ] && [ $((n % (1 << 10))) -eq 0 ]; then echo "$((n >> 10))K"
  else echo "${n}B"
  fi
}

bars_of() { # BAR sizes from the sysfs "resource" file (lines 0..5 are BAR0..BAR5)
  local f=$1/resource i=0 out="" start end size
  [ -r "$f" ] || { printf '%s' "-"; return; }
  while read -r start end _ && [ $i -le 5 ]; do
    if [ "$end" != "0x0000000000000000" ] && [ -n "$end" ]; then
      size=$(( end - start + 1 ))
      [ "$size" -gt 0 ] && out="$out BAR$i=$(human_bytes "$size")"
    fi
    i=$((i + 1))
  done < "$f"
  out=${out# }
  printf '%s' "${out:--}"
}

sectors_bytes() { # sectors_bytes FILE -> 512 * value of a sysfs "size" file (0 if unreadable)
  local v
  v=$(rd "$1" | tr -cd 0-9)
  echo $(( ${v:-0} * 512 ))
}

aer_total() { # aer_total FILE KEY -> value of KEY line or "-"
  local v
  if [ -r "$1" ]; then
    v=$(awk -v k="$2" '$1==k {print $2}' "$1" 2>/dev/null)
    printf '%s' "${v:--}"
  else
    printf '%s' "-"
  fi
}

upstream_of() { # BDFs of the bridges above a device, nearest first
  local p bdf chain=""
  p=$(readlink -f "/sys/bus/pci/devices/$1" 2>/dev/null) || { printf '%s' "-"; return; }
  p=${p%/*}
  while :; do
    bdf=${p##*/}
    case "$bdf" in
      [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) chain="$chain $bdf";;
      *) break;;
    esac
    p=${p%/*}
  done
  chain=${chain# }
  printf '%s' "${chain:--}"
}

pci_row() { # pci_row ROLE BDF
  local d=/sys/bus/pci/devices/$2 drv grp
  drv=$(basename "$(readlink -f "$d/driver" 2>/dev/null)" 2>/dev/null)
  [ -e "$d/driver" ] || drv="-"
  grp=$(basename "$(readlink -f "$d/iommu_group" 2>/dev/null)" 2>/dev/null)
  [ -e "$d/iommu_group" ] || grp="-"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$(rd "$d/vendor")" "$(rd "$d/device")" "$(rd "$d/class")" "$drv" \
    "$(rd "$d/numa_node")" "$(rd "$d/current_link_speed")" "$(rd "$d/current_link_width")" \
    "$(rd "$d/max_link_speed")" "$(rd "$d/max_link_width")" "$grp" \
    "$(aer_total "$d/aer_dev_correctable" TOTAL_ERR_COR)" \
    "$(aer_total "$d/aer_dev_nonfatal" TOTAL_ERR_NONFATAL)" \
    "$(aer_total "$d/aer_dev_fatal" TOTAL_ERR_FATAL)" \
    "$(bars_of "$d")" "$(upstream_of "$2")"
}

cmd_pci_links() { # cmd_pci_links [all]
  local d bdf cls ven role up b n=0 all=${1:-}
  local -A bridges=()
  local -a order=()
  [ -d /sys/bus/pci/devices ] || { echo "no /sys/bus/pci/devices"; return 3; }
  printf 'role\tbdf\tvendor\tdevice\tclass\tdriver\tnuma_node\tcur_speed\tcur_width\tmax_speed\tmax_width\tiommu_group\taer_cor\taer_nonfatal\taer_fatal\tbars\tupstream\n'
  for d in /sys/bus/pci/devices/*; do
    [ -e "$d" ] || continue
    bdf=${d##*/}
    cls=$(rd "$d/class"); ven=$(rd "$d/vendor")
    role=""
    case "$cls" in
      0x03*) [ "$ven" = "0x10de" ] && role=gpu || role=display;;
      0x0108*) role=nvme;;
      0x0604*) [ "$all" = all ] && role=bridge-any;;
      *) [ "$all" = all ] && role=other;;
    esac
    [ -n "$role" ] || continue
    pci_row "$role" "$bdf"
    n=$((n + 1))
    [ "$all" = all ] && continue
    up=$(upstream_of "$bdf")
    [ "$up" = "-" ] && continue
    for b in $up; do
      if [ -z "${bridges[$b]+x}" ]; then bridges[$b]=1; order+=("$b"); fi
    done
  done
  for b in ${order[@]+"${order[@]}"}; do pci_row bridge "$b"; done
  echo "# devices listed: $n (gpu = class 03xx vendor 0x10de, nvme = class 0108); bridges: ${#order[@]}"
  echo "# note: GPUs lower the current link speed at idle (power saving); compare max_* and the speed under load."
}

cmd_nvme() {
  local c name d ns hw t
  [ -d /sys/class/nvme ] || { echo "no /sys/class/nvme"; return 3; }
  printf 'controller\tkey\tvalue\n'
  for c in /sys/class/nvme/nvme*; do
    [ -e "$c" ] || continue
    name=${c##*/}
    d=$c/device
    printf '%s\tmodel\t%s\n' "$name" "$(rd "$c/model" | sed 's/[[:space:]]*$//')"
    printf '%s\tserial\t%s\n' "$name" "$(rd "$c/serial" | sed 's/[[:space:]]*$//')"
    printf '%s\tfirmware_rev\t%s\n' "$name" "$(rd "$c/firmware_rev" | sed 's/[[:space:]]*$//')"
    printf '%s\ttransport\t%s\n' "$name" "$(rd "$c/transport")"
    printf '%s\taddress\t%s\n' "$name" "$(rd "$c/address")"
    printf '%s\tstate\t%s\n' "$name" "$(rd "$c/state")"
    printf '%s\tnuma_node\t%s\n' "$name" "$(rd "$d/numa_node")"
    printf '%s\tcur_link_speed\t%s\n' "$name" "$(rd "$d/current_link_speed")"
    printf '%s\tcur_link_width\t%s\n' "$name" "$(rd "$d/current_link_width")"
    printf '%s\tmax_link_speed\t%s\n' "$name" "$(rd "$d/max_link_speed")"
    printf '%s\tmax_link_width\t%s\n' "$name" "$(rd "$d/max_link_width")"
    for hw in "$c"/hwmon* "$d"/hwmon/hwmon*; do
      [ -r "$hw/temp1_input" ] || continue
      t=$(rd "$hw/temp1_input")
      case "$t" in ''|-|*[!0-9-]*) ;; *) printf '%s\ttemp_c\t%s.%03d\n' "$name" "$((t / 1000))" "$(( (t < 0 ? -t : t) % 1000 ))";; esac
      break
    done
    for ns in /sys/block/"$name"n*; do
      [ -e "$ns" ] || continue
      printf '%s\tnamespace\t%s size_bytes=%s lbs=%s\n' "$name" "${ns##*/}" \
        "$(sectors_bytes "$ns/size")" "$(rd "$ns/queue/logical_block_size")"
    done
  done
}

cmd_block() {
  local b q
  printf 'dev\tsize_bytes\trotational\tscheduler\tnr_requests\tmax_sectors_kb\tmax_hw_sectors_kb\tread_ahead_kb\tlogical_block\tphysical_block\tmodel\n'
  for b in /sys/block/*; do
    [ -e "$b" ] || continue
    q=$b/queue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${b##*/}" \
      "$(sectors_bytes "$b/size")" "$(rd "$q/rotational")" "$(rd "$q/scheduler")" \
      "$(rd "$q/nr_requests")" "$(rd "$q/max_sectors_kb")" "$(rd "$q/max_hw_sectors_kb")" \
      "$(rd "$q/read_ahead_kb")" "$(rd "$q/logical_block_size")" "$(rd "$q/physical_block_size")" \
      "$(rd "$b/device/model" | sed 's/[[:space:]]*$//')"
  done
}

cmd_cpufreq() {
  local p found=0
  printf 'policy\taffected_cpus\tdriver\tgovernor\tcpuinfo_min_khz\tcpuinfo_max_khz\tscaling_min_khz\tscaling_max_khz\tcur_khz\tepp\n'
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -e "$p" ] || continue
    found=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${p##*/}" "$(rd "$p/affected_cpus")" \
      "$(rd "$p/scaling_driver")" "$(rd "$p/scaling_governor")" "$(rd "$p/cpuinfo_min_freq")" \
      "$(rd "$p/cpuinfo_max_freq")" "$(rd "$p/scaling_min_freq")" "$(rd "$p/scaling_max_freq")" \
      "$(rd "$p/scaling_cur_freq")" "$(rd "$p/energy_performance_preference")"
  done
  echo "# boost=$(rd /sys/devices/system/cpu/cpufreq/boost) amd_pstate_status=$(rd /sys/devices/system/cpu/amd_pstate/status) smt_active=$(rd /sys/devices/system/cpu/smt/active) smt_control=$(rd /sys/devices/system/cpu/smt/control)"
  [ $found -eq 1 ] || { echo "# no cpufreq policies exposed (typical inside a VM)"; }
}

cmd_hugepages() {
  local h n
  printf 'scope\tsize\tnr_hugepages\tfree_hugepages\tresv_hugepages\tsurplus_hugepages\n'
  for h in /sys/kernel/mm/hugepages/hugepages-*; do
    [ -e "$h" ] || continue
    printf 'global\t%s\t%s\t%s\t%s\t%s\n' "${h##*/hugepages-}" "$(rd "$h/nr_hugepages")" \
      "$(rd "$h/free_hugepages")" "$(rd "$h/resv_hugepages")" "$(rd "$h/surplus_hugepages")"
  done
  for n in /sys/devices/system/node/node*; do
    for h in "$n"/hugepages/hugepages-*; do
      [ -e "$h" ] || continue
      printf '%s\t%s\t%s\t%s\t-\t%s\n' "${n##*/}" "${h##*/hugepages-}" "$(rd "$h/nr_hugepages")" \
        "$(rd "$h/free_hugepages")" "$(rd "$h/surplus_hugepages")"
    done
  done
  echo "thp_enabled: $(rd /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "thp_defrag: $(rd /sys/kernel/mm/transparent_hugepage/defrag)"
  for k in vm/nr_hugepages vm/nr_overcommit_hugepages vm/overcommit_memory vm/swappiness \
           vm/zone_reclaim_mode kernel/numa_balancing vm/max_map_count; do
    echo "proc_sys_$k: $(rd "/proc/sys/$k")"
  done
}

cmd_numa() {
  local n
  [ -d /sys/devices/system/node ] || { echo "no /sys/devices/system/node"; return 3; }
  echo "online: $(rd /sys/devices/system/node/online) possible: $(rd /sys/devices/system/node/possible) has_memory: $(rd /sys/devices/system/node/has_memory)"
  for n in /sys/devices/system/node/node*; do
    [ -e "$n" ] || continue
    echo "== ${n##*/} cpulist=$(rd "$n/cpulist") distance=$(rd "$n/distance")"
    grep -E 'MemTotal|MemFree|HugePages_(Total|Free)' "$n/meminfo" 2>/dev/null
  done
}

cmd_edac() {
  local mc dm
  [ -d /sys/devices/system/edac/mc ] || { echo "no /sys/devices/system/edac/mc (EDAC driver not loaded)"; return 3; }
  printf 'kind\tname\tlabel_or_mc\tsize\tlocation_or_type\tce\tue\n'
  for mc in /sys/devices/system/edac/mc/mc*; do
    [ -e "$mc" ] || continue
    printf 'mc\t%s\t%s\t%s_MB\t-\t%s\t%s\n' "${mc##*/}" "$(rd "$mc/mc_name")" "$(rd "$mc/size_mb")" \
      "$(rd "$mc/ce_count")" "$(rd "$mc/ue_count")"
    for dm in "$mc"/dimm* "$mc"/rank*; do
      [ -e "$dm" ] || continue
      printf 'dimm\t%s/%s\t%s\t%s_MB\t%s %s\t%s\t%s\n' "${mc##*/}" "${dm##*/}" "$(rd "$dm/dimm_label")" \
        "$(rd "$dm/size")" "$(rd "$dm/dimm_location")" "$(rd "$dm/dimm_mem_type")" \
        "$(rd "$dm/dimm_ce_count")" "$(rd "$dm/dimm_ue_count")"
    done
  done
}

cmd_dmi() {
  local f n
  [ -d /sys/class/dmi/id ] || { echo "no /sys/class/dmi/id"; return 3; }
  for f in /sys/class/dmi/id/*; do
    [ -f "$f" ] || continue
    n=${f##*/}
    case "$n" in uevent|modalias) continue;; esac
    if [ -r "$f" ] && head -c 0 "$f" 2>/dev/null; then
      printf '%s\t%s\n' "$n" "$(head -n 1 "$f" 2>/dev/null)"
    else
      printf '%s\t%s\n' "$n" "<not readable without root>"
    fi
  done
}

cmd_iommu() {
  local g d n=0
  [ -d /sys/kernel/iommu_groups ] || { echo "no /sys/kernel/iommu_groups"; return 3; }
  for g in /sys/kernel/iommu_groups/*; do
    [ -e "$g" ] || continue
    n=$((n + 1))
    for d in "$g"/devices/*; do
      [ -e "$d" ] || continue
      printf 'group %s\t%s\tclass=%s\tid=%s:%s\tdriver=%s\n' "${g##*/}" "${d##*/}" "$(rd "$d/class")" \
        "$(rd "$d/vendor")" "$(rd "$d/device")" \
        "$( [ -e "$d/driver" ] && basename "$(readlink -f "$d/driver")" || echo -)"
    done
  done
  echo "# iommu groups: $n (0 groups usually means the IOMMU is off or not exposed)"
  for g in /sys/class/iommu/*; do [ -e "$g" ] && echo "# iommu unit: ${g##*/}"; done
  return 0
}

cmd_net() {
  local i
  printf 'iface\toperstate\tspeed_mbps\tmtu\tdriver\n'
  for i in /sys/class/net/*; do
    [ -e "$i" ] || continue
    printf '%s\t%s\t%s\t%s\t%s\n' "${i##*/}" "$(rd "$i/operstate")" "$(rd "$i/speed")" "$(rd "$i/mtu")" \
      "$( [ -e "$i/device/driver" ] && basename "$(readlink -f "$i/device/driver")" || echo -)"
  done
}

cmd_nvidia_proc() {
  local f
  [ -d /proc/driver/nvidia ] || { echo "no /proc/driver/nvidia (nvidia kernel module not loaded)"; return 3; }
  for f in /proc/driver/nvidia/version /proc/driver/nvidia/params /proc/driver/nvidia/gpus/*/information; do
    [ -r "$f" ] || continue
    echo "== $f"
    cat "$f"
  done
}

case "${1:-}" in
  pci-links) cmd_pci_links;;
  pci-links-all) cmd_pci_links all;;
  nvme) cmd_nvme;;
  block) cmd_block;;
  cpufreq) cmd_cpufreq;;
  hugepages) cmd_hugepages;;
  numa) cmd_numa;;
  edac) cmd_edac;;
  dmi) cmd_dmi;;
  iommu) cmd_iommu;;
  net) cmd_net;;
  nvidia-proc) cmd_nvidia_proc;;
  *) echo "usage: $0 {pci-links|pci-links-all|nvme|block|cpufreq|hugepages|numa|edac|dmi|iommu|net|nvidia-proc}" >&2; exit 2;;
esac
