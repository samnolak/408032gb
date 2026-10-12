#!/usr/bin/env python3
"""Draft HARDWARE_FACTS from a diag-out directory written by scripts/diag/collect.sh.

Standard library only. The rule is the repo's evidence rule (AGENTS.md):
  CONFIRMED   value parsed from a literal command output inside the directory; the label
              names the file it came from
  PROVISIONAL value derived from confirmed outputs; the label says how
  UNKNOWN     not present / not parseable; the label says why. Nothing is filled in.

usage:
  summarize.py DIAGDIR [--output FILE] [--json FILE]
  summarize.py --fio-summary [--csv FILE] FIO_JSON_OR_DIR...
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import sys

SECTION_RE = re.compile(r"^===== (PROBE|COMMAND|STDOUT|STDERR|RESULT) =====$")


# --------------------------------------------------------------------------- probe files
def parse_probe_text(text):
    """Split a probe/bench log into sections. STDOUT runs to the LAST '===== STDERR ====='."""
    lines = text.split("\n")
    idx = {}
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line)
        if m:
            idx.setdefault(m.group(1), []).append(i)
    out = {"meta": {}, "command": "", "stdout": "", "stderr": "", "result": {}}
    if "STDOUT" not in idx:
        out["stdout"] = text
        return out
    so = idx["STDOUT"][0]
    se = idx["STDERR"][-1] if "STDERR" in idx else len(lines)
    rs = idx["RESULT"][-1] if "RESULT" in idx else len(lines)
    cm = idx["COMMAND"][0] if "COMMAND" in idx else so
    pr = idx["PROBE"][0] if "PROBE" in idx else 0
    for line in lines[pr + 1:cm]:
        if ": " in line:
            k, v = line.split(": ", 1)
            out["meta"][k.strip()] = v
    out["command"] = "\n".join(lines[cm + 1:so]).strip("\n")
    out["stdout"] = "\n".join(lines[so + 1:se])
    out["stderr"] = "\n".join(lines[se + 1:rs]) if se < rs else ""
    for line in lines[rs + 1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            out["result"][k.strip()] = v.strip()
    return out


class Run:
    def __init__(self, d):
        self.dir = d
        self.manifest = {}
        self.probes = {}
        mpath = os.path.join(d, "manifest.json")
        if os.path.isfile(mpath):
            with open(mpath, encoding="utf-8") as f:
                self.manifest = json.load(f)
            for p in self.manifest.get("probes", []):
                self.probes[p["name"]] = dict(p)
        else:  # fall back to the probe files themselves
            for path in sorted(glob.glob(os.path.join(d, "probes", "*.txt"))):
                sec = parse_probe_text(open(path, encoding="utf-8", errors="replace").read())
                name = sec["meta"].get("name") or os.path.basename(path)
                self.probes[name] = {"name": name, "status": sec["result"].get("status", "unknown"),
                                     "note": sec["result"].get("note", ""),
                                     "output_file": os.path.relpath(path, d)}
        self._cache = {}

    def get(self, name):
        """-> (stdout, source_file) when the probe ran with status ok, else (None, reason)."""
        p = self.probes.get(name)
        if p is None:
            return None, "probe %s not in this run" % name
        if p.get("status") != "ok":
            note = p.get("note") or ""
            return None, "probe %s: %s%s" % (name, p.get("status"), (" - " + note) if note else "")
        rel = p.get("output_file", "")
        if rel not in self._cache:
            path = os.path.join(self.dir, rel)
            if not os.path.isfile(path):
                return None, "probe %s: output file %s missing" % (name, rel)
            with open(path, encoding="utf-8", errors="replace") as f:
                self._cache[rel] = parse_probe_text(f.read())
        return self._cache[rel]["stdout"], rel

    def names(self, prefix):
        return sorted(n for n in self.probes if n.startswith(prefix))


# --------------------------------------------------------------------------- facts
class Facts:
    def __init__(self):
        self.rows = []
        self.flags = []
        self.raw = []

    def confirmed(self, section, fact, value, src):
        self.rows.append({"section": section, "fact": fact, "value": str(value),
                          "label": "CONFIRMED", "source": src})

    def provisional(self, section, fact, value, how):
        self.rows.append({"section": section, "fact": fact, "value": str(value),
                          "label": "PROVISIONAL", "source": how})

    def unknown(self, section, fact, why):
        self.rows.append({"section": section, "fact": fact, "value": "UNKNOWN",
                          "label": "UNKNOWN", "source": why})

    def flag(self, text, src):
        self.flags.append({"flag": text, "source": src})

    def find(self, fact):
        for r in self.rows:
            if r["fact"] == fact:
                return r
        return None


def kv_lines(text, sep=":"):
    d = {}
    for line in text.splitlines():
        if sep in line:
            k, v = line.split(sep, 1)
            k = k.strip()
            if k and k not in d:
                d[k] = v.strip()
    return d


def tsv_table(text):
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    if not rows:
        return []
    head = rows[0].split("\t")
    return [dict(zip(head, r.split("\t"))) for r in rows[1:]]


def first_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


# --------------------------------------------------------------------------- extractors
def ex_run(run, F):
    m = run.manifest
    S = "run"
    if not m:
        F.unknown(S, "manifest", "manifest.json missing")
        return
    for key, fact in (("hostname", "hostname"), ("started_utc", "collected (UTC)"),
                      ("kit_version", "kit version"), ("is_root", "collected as root"),
                      ("sudo", "sudo mode"), ("run_state", "run state")):
        if key in m:
            F.confirmed(S, fact, m[key], "manifest.json")
    c = m.get("counts")
    if c:
        F.confirmed(S, "probe status counts", ", ".join("%s=%s" % kv for kv in c.items()), "manifest.json")
    out, src = run.get("meta.kit_git")
    if out is not None and first_line(out):
        dirty = [l for l in out.splitlines()[1:] if l.strip()]
        F.confirmed(S, "kit git commit (repo HEAD)", first_line(out).split()[0] +
                    (" + %d uncommitted/untracked entries under scripts/diag" % len(dirty) if dirty else " (kit files clean)"), src)
    else:
        F.unknown(S, "kit git commit (repo HEAD)", src)


def ex_system(run, F):
    S = "system"
    out, src = run.get("system.os_release")
    if out is not None and "PRETTY_NAME" in out:
        F.confirmed(S, "OS", kv_lines(out, "=")["PRETTY_NAME"].strip('"'), src)
    else:
        F.unknown(S, "OS", src if out is None else "PRETTY_NAME not found")
    out, src = run.get("system.uname")
    if out and first_line(out):
        F.confirmed(S, "kernel (uname -a)", first_line(out), src)
    else:
        F.unknown(S, "kernel (uname -a)", src)
    out, src = run.get("system.virt")
    virt = first_line(out) if out else None
    if virt:
        F.confirmed(S, "virtualization (systemd-detect-virt)", virt, src)
        if virt != "none":
            F.flag("collected inside a virtual machine/container (%s): DIMM, BIOS, PCIe topology and BMC data "
                   "describe the virtual hardware; run collect.sh on the Proxmox host too" % virt, src)
    else:
        F.unknown(S, "virtualization (systemd-detect-virt)", src)
    out, src = run.get("system.cpuinfo_hypervisor_flag")
    if out is not None and first_line(out) is not None:
        F.confirmed(S, "CPU 'hypervisor' flag present", "yes" if first_line(out) != "0" else "no", src)
    out, src = run.get("system.proxmox_pveversion")
    if out and first_line(out):
        F.confirmed(S, "Proxmox (pveversion -v)", first_line(out), src)
    else:
        F.unknown(S, "Proxmox (pveversion -v)", src)
    out, src = run.get("pcie.iommu_cmdline")
    if out is not None:
        v = " ".join(l.strip() for l in out.splitlines() if l.strip())
        F.confirmed(S, "kernel cmdline: iommu/vfio/pcie/hugepage params", v or "(none)", src)
    else:
        F.unknown(S, "kernel cmdline: iommu/vfio/pcie/hugepage params", src)
    out, src = run.get("system.proxmox_qm_config")
    if out:
        vm = None
        keep = ("memory", "balloon", "numa", "cpu", "cores", "sockets", "hugepages", "machine", "bios")
        cur = {}
        for line in out.splitlines() + ["== END"]:
            if line.startswith("== "):
                if vm is not None:
                    F.confirmed(S, "Proxmox VM %s config" % vm, "; ".join(cur.get(k) for k in sorted(cur)), src)
                vm = line[3:].replace("VM", "").strip()
                cur = {}
                continue
            k = line.split(":", 1)[0].strip()
            if k in keep or k.startswith("hostpci"):
                cur[k] = line.strip()


def ex_cpu(run, F):
    S = "cpu"
    out, src = run.get("cpu.lscpu")
    want = ["Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "CPU(s)",
            "NUMA node(s)", "L3 cache", "Hypervisor vendor", "Virtualization"]
    if out is not None:
        kv = kv_lines(out)
        for k in want:
            if k in kv:
                F.confirmed(S, "lscpu " + k, kv[k], src)
            elif k not in ("Hypervisor vendor", "Virtualization"):
                F.unknown(S, "lscpu " + k, "field not in lscpu output (%s)" % src)
        for k, v in kv.items():
            if re.match(r"NUMA node\d+ CPU\(s\)", k):
                F.confirmed(S, "lscpu " + k, v, src)
    else:
        for k in want[:7]:
            F.unknown(S, "lscpu " + k, src)
    out, src = run.get("cpu.lscpu_parse")
    if out is not None:
        cores = set()
        for line in out.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(",")
            if len(p) >= 3:
                cores.add((p[1], p[2]))
        if cores:
            F.provisional(S, "physical cores (unique core,socket pairs)", len(cores),
                          "counted from lscpu -p=CPU,CORE,SOCKET,NODE in " + src)
    out, src = run.get("cpu.flags")
    if out is not None:
        kv = kv_lines(out, "=")
        for k in ("avx2", "fma", "f16c", "avx_vnni"):
            if k in kv:
                F.confirmed(S, "flag " + k, kv[k], src)
        if "avx512_any" in kv:
            av = [k for k, v in kv.items() if k.startswith("avx512") and k != "avx512_any" and v == "yes"]
            F.confirmed(S, "AVX-512 (any avx512* flag)", kv["avx512_any"] + (" (" + " ".join(av) + ")" if av else ""), src)
            if kv["avx512_any"] == "yes":
                F.flag("CPU reports AVX-512; the stated CPU (EPYC 7H43, Zen 3) has none: different CPU or flags passed by a hypervisor", src)
        amx = [k for k in ("amx_tile", "amx_bf16", "amx_int8") if kv.get(k) == "yes"]
        if "amx_tile" in kv:
            F.confirmed(S, "AMX", "yes (" + " ".join(amx) + ")" if amx else "no", src)
        if kv.get("avx2") == "no":
            F.flag("CPU does not report AVX2 (CPU expert kernels need it)", src)
    else:
        F.unknown(S, "flag avx2", src)
        F.unknown(S, "AVX-512 (any avx512* flag)", src)
    out, src = run.get("cpu.numactl_hardware")
    m = re.search(r"available:\s*(\d+)\s+nodes", out or "")
    if m:
        F.confirmed(S, "NUMA nodes (numactl -H)", m.group(1), src)
    else:
        F.unknown(S, "NUMA nodes (numactl -H)", src if out is None else "pattern 'available: N nodes' not found")
    out, src = run.get("cpu.cpufreq_sysfs")
    if out is not None:
        rows = tsv_table(out)
        gov = sorted(set(r.get("governor", "-") for r in rows))
        drv = sorted(set(r.get("driver", "-") for r in rows))
        if rows:
            F.confirmed(S, "cpufreq governors / driver", "%s / %s (%d policies)" % (",".join(gov), ",".join(drv), len(rows)), src)
        else:
            F.confirmed(S, "cpufreq governors / driver", "no cpufreq policies exposed", src)
        m = re.search(r"boost=(\S+)", out)
        if m and m.group(1) != "-":
            F.confirmed(S, "cpufreq boost", m.group(1), src)
        m = re.search(r"smt_active=(\S+)", out)
        if m and m.group(1) != "-":
            F.confirmed(S, "SMT active", m.group(1), src)


def _dmi_blocks(text, type_no):
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith("Handle "):
            if cur is not None:
                blocks.append(cur)
            m = re.search(r"DMI type (\d+)", line)
            cur = {"_type": int(m.group(1)) if m else -1}
            continue
        if cur is None:
            continue
        if line.startswith("\t") and ":" in line and not line.startswith("\t\t"):
            k, v = line.strip().split(":", 1)
            cur.setdefault(k.strip(), v.strip())
        elif line.strip() and not line.startswith("\t") and "_title" not in cur:
            cur["_title"] = line.strip()
    if cur is not None:
        blocks.append(cur)
    return [b for b in blocks if b["_type"] == type_no]


def _size_bytes(s):
    m = re.match(r"(\d+)\s*(KB|MB|GB|TB|kB)", s or "")
    if not m:
        return None
    return int(m.group(1)) * {"KB": 1 << 10, "kB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}[m.group(2)]


def infer_channels(dimms):
    """-> (count, how) or (None, why). Uses Bank Locator / Locator naming only."""
    ids = []
    for d in dimms:
        loc, bank = d.get("Locator", ""), d.get("Bank Locator", "")
        both = bank + " " + loc
        s = re.search(r"(?:^|[^A-Z0-9])(?:P|CPU|SOCKET)[ _-]?(\d)(?![0-9])", both, re.I)
        sock = s.group(1) if s else "0"
        m = (re.search(r"CHANNEL[ _-]?([A-Z0-9])(?![A-Z0-9])", bank, re.I)
             or re.search(r"CHANNEL[ _-]?([A-Z0-9])(?![A-Z0-9])", loc, re.I)
             or re.search(r"(?:^|[^A-Z0-9])DIMM[ _-]?(?:P\d[ _-]?)?([A-P])[ _-]?\d(?![0-9])", loc, re.I)
             or re.search(r"(?:^|[_ -])([A-P])[ _-]?[0-2]$", loc, re.I))
        if not m:
            return None, 'locator naming not recognised (Locator="%s", Bank Locator="%s")' % (loc, bank)
        ids.append((sock, m.group(1).upper()))
    if not ids:
        return None, "no populated DIMMs"
    return len(set(ids)), "distinct (socket, channel) ids parsed from the Bank Locator/Locator strings of %d populated DIMMs: %s" % (
        len(dimms), " ".join("P%s:%s" % i for i in sorted(set(ids))))


def ex_memory(run, F):
    S = "memory"
    out, src = run.get("memory.meminfo")
    kv = kv_lines(out) if out is not None else {}
    if "MemTotal" in kv:
        kb = int(kv["MemTotal"].split()[0])
        F.confirmed(S, "MemTotal", "%s (%.2f GiB)" % (kv["MemTotal"], kb / 1048576.0), src)
    else:
        F.unknown(S, "MemTotal", src)
    for k in ("HugePages_Total", "HugePages_Free", "Hugepagesize", "Hugetlb"):
        if k in kv:
            F.confirmed(S, "meminfo " + k, kv[k], src)
    out, src = run.get("memory.hugepages")
    if out is not None:
        for r in tsv_table("\n".join(l for l in out.splitlines() if "\t" in l)):
            if r.get("scope") == "global":
                F.confirmed(S, "hugepages %s pool (nr/free)" % r.get("size"), "%s / %s" % (r.get("nr_hugepages"), r.get("free_hugepages")), src)
        m = re.search(r"thp_enabled: (.*)", out)
        if m:
            sel = re.search(r"\[(\w+)\]", m.group(1))
            F.confirmed(S, "transparent hugepages", sel.group(1) if sel else m.group(1), src)
    out, src = run.get("memory.ulimit_l")
    v = first_line(out) if out is not None else None
    if v:
        F.confirmed(S, "memlock limit (ulimit -l, KiB, for the collecting user/session)", v, src)
        if v != "unlimited" and v.isdigit() and int(v) < 100 * 1024 * 1024:
            F.flag("memlock limit is %s KiB (< 100 GiB): pinned host buffers of the engine will fail or be capped; "
                   "check limits.conf / systemd LimitMEMLOCK for the service user" % v, src)
    else:
        F.unknown(S, "memlock limit (ulimit -l)", src)

    out, src = run.get("memory.dmidecode_16_17")
    if out is None:
        reason = src
        F.unknown(S, "DIMM slots (SMBIOS type 16)", reason)
        F.unknown(S, "populated DIMMs (SMBIOS type 17)", reason)
        F.unknown(S, "memory channels populated", reason + "; run collect.sh as root (dmidecode) on the host")
        F.unknown(S, "configured memory speed", reason)
    else:
        arrays = _dmi_blocks(out, 16)
        devs = _dmi_blocks(out, 17)
        slots = 0
        for a in arrays:
            if a.get("Number Of Devices", "").isdigit():
                slots += int(a["Number Of Devices"])
        if arrays:
            F.confirmed(S, "DIMM slots (SMBIOS type 16)", "%d (Number Of Devices)" % slots, src)
        pop = [d for d in devs if _size_bytes(d.get("Size")) is not None]
        if devs:
            total = sum(_size_bytes(d["Size"]) for d in pop)
            F.confirmed(S, "populated DIMMs (SMBIOS type 17)", "%d of %d devices, total %.0f GiB" % (len(pop), len(devs), total / (1 << 30)), src)
            sizes = sorted(set(d["Size"] for d in pop))
            F.confirmed(S, "DIMM sizes", ", ".join(sizes) or "-", src)
            for key, fact in (("Type", "DIMM type"), ("Speed", "DIMM rated speed"),
                              ("Manufacturer", "DIMM manufacturers"), ("Part Number", "DIMM part numbers"),
                              ("Rank", "DIMM ranks")):
                vals = sorted(set(d.get(key, "?") for d in pop))
                if pop:
                    F.confirmed(S, fact, ", ".join(vals), src)
            conf = sorted(set(d.get("Configured Memory Speed") or d.get("Configured Clock Speed") or "?" for d in pop))
            if pop and conf != ["?"]:
                F.confirmed(S, "configured memory speed", ", ".join(conf), src)
            else:
                F.unknown(S, "configured memory speed", "no 'Configured Memory Speed' field (%s)" % src)
            F.confirmed(S, "populated DIMM locators", "; ".join("%s / %s" % (d.get("Locator", "?"), d.get("Bank Locator", "?")) for d in pop) or "-", src)
            n, how = infer_channels(pop)
            if n is not None:
                F.provisional(S, "memory channels populated", n, how + " (" + src + ")")
                if n < 8:
                    F.flag("only %d memory channels populated (EPYC SP3 has 8): memory bandwidth is capped at about %d/8 of the platform maximum" % (n, n), src)
            else:
                F.unknown(S, "memory channels populated", how + "; populated DIMMs = %d (%s)" % (len(pop), src))
            if len(pop) < 8:
                F.flag("%d DIMMs populated (fewer than 8): at most %d of 8 channels can be in use" % (len(pop), len(pop)), src)
            speeds = [int(m.group(1)) for c in conf for m in [re.match(r"(\d+)", c)] if m]
            if speeds and min(speeds) < 3200:
                F.flag("configured memory speed %s is below DDR4-3200" % ", ".join(conf), src)
        else:
            F.unknown(S, "populated DIMMs (SMBIOS type 17)", "no type 17 records in " + src)
    out, src = run.get("memory.edac_sysfs")
    if out is not None:
        rows = tsv_table(out)
        dimms = [r for r in rows if r.get("kind") == "dimm" and r.get("size", "0_MB").split("_")[0] not in ("0", "-")]
        ce = sum(int(r["ce"]) for r in rows if r.get("kind") == "mc" and r.get("ce", "").isdigit())
        ue = sum(int(r["ue"]) for r in rows if r.get("kind") == "mc" and r.get("ue", "").isdigit())
        F.confirmed(S, "EDAC: DIMMs with size > 0 / corrected errors / uncorrected errors", "%d / %d / %d" % (len(dimms), ce, ue), src)
        if ue:
            F.flag("EDAC reports %d uncorrected memory errors" % ue, src)
        elif ce:
            F.flag("EDAC reports %d corrected memory errors" % ce, src)


def ex_board(run, F):
    S = "board"
    out, src = run.get("board.dmidecode_0_1_2_3")
    dmi_src = None
    if out is not None:
        for t, title, keys in ((0, "BIOS", ("Vendor", "Version", "Release Date")),
                               (1, "system", ("Manufacturer", "Product Name", "Version")),
                               (2, "board", ("Manufacturer", "Product Name", "Version"))):
            bl = _dmi_blocks(out, t)
            if bl:
                for k in keys:
                    if k in bl[0]:
                        F.confirmed(S, "%s %s" % (title, k), bl[0][k], src)
                dmi_src = src
    if dmi_src is None:
        out2, src2 = run.get("board.dmi_sysfs")
        if out2 is not None:
            kv = dict(l.split("\t", 1) for l in out2.splitlines() if "\t" in l)
            for f, k in (("BIOS Vendor", "bios_vendor"), ("BIOS Version", "bios_version"), ("BIOS Release Date", "bios_date"),
                         ("system Manufacturer", "sys_vendor"), ("system Product Name", "product_name"),
                         ("board Manufacturer", "board_vendor"), ("board Product Name", "board_name"),
                         ("board Version", "board_version")):
                if k in kv:
                    F.confirmed(S, f, kv[k], src2)
                else:
                    F.unknown(S, f, "%s not in %s" % (k, src2))
        else:
            F.unknown(S, "board Product Name", "%s; %s" % (src, src2))
            F.unknown(S, "BIOS Version", "%s; %s" % (src, src2))
    out, src = run.get("board.dmidecode_38_ipmi")
    if out is not None:
        bl = _dmi_blocks(out, 38)
        F.confirmed(S, "SMBIOS IPMI device (type 38)", ("present, interface " + bl[0].get("Interface Type", "?")) if bl else "no type 38 record", src)
    out, src = run.get("bmc.ipmitool_mc_info")
    if out is not None:
        kv = kv_lines(out)
        for k in ("Manufacturer Name", "Product Name", "Firmware Revision", "IPMI Version"):
            if k in kv:
                F.confirmed(S, "BMC " + k, kv[k], src)
    else:
        F.unknown(S, "BMC (ipmitool mc info)", src)
    out, src = run.get("bmc.ipmitool_sel")
    if out is not None:
        n = len([l for l in out.splitlines() if "|" in l])
        F.confirmed(S, "BMC SEL entries", n, src)
    out, src = run.get("bmc.ipmitool_sdr")
    if out is not None:
        bad = [l.strip() for l in out.splitlines() if l.count("|") >= 2 and l.rsplit("|", 1)[1].strip() not in ("ok", "ns")]
        F.confirmed(S, "BMC sensors not 'ok'/'ns'", "; ".join(bad) if bad else "none", src)
        if bad:
            F.flag("BMC sensors in a non-ok state: " + "; ".join(bad[:5]), src)


def _csv_rows(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    rdr = csv.reader(io.StringIO("\n".join(lines)), skipinitialspace=True)
    rows = list(rdr)
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, [c.strip() for c in r])) for r in rows[1:]]


def _p2p_matrix(text):
    lines = [l for l in text.splitlines() if l.strip()]
    head = None
    mat = {}
    for line in lines:
        toks = line.split()
        if head is None and toks and all(re.match(r"GPU\d+$", t) for t in toks):
            head = toks
            continue
        if head and toks and re.match(r"GPU\d+$", toks[0]) and len(toks) == len(head) + 1:
            for j, v in enumerate(toks[1:]):
                mat[(toks[0], head[j])] = v
    return head, mat


def _topo_m(text):
    """GPU-to-GPU cells of `nvidia-smi topo -m` (header: GPU0..GPUn, then affinity columns)."""
    head, mat = None, {}
    for line in text.splitlines():
        toks = line.split()
        if head is None:
            lead = []
            for t in toks:
                if re.match(r"(GPU|NIC)\d+$", t):
                    lead.append(t)
                else:
                    break
            if lead and lead[0] == "GPU0":
                head = lead
            continue
        if toks and toks[0] in head and len(toks) > len(head):
            for j, v in enumerate(toks[1:1 + len(head)]):
                mat[(toks[0], head[j])] = v
        elif not toks or toks[0] == "Legend:":
            if mat:
                break
    return mat


def ex_gpu(run, F):
    S = "gpu"
    out, src = run.get("gpu.nvidia_smi")
    m = re.search(r"Driver Version:\s*([\d.]+)", out or "")
    if m:
        F.confirmed(S, "NVIDIA driver version", m.group(1), src)
        c = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if c:
            F.confirmed(S, "CUDA version supported by the driver", c.group(1), src)
    else:
        F.unknown(S, "NVIDIA driver version", src if out is None else "not in nvidia-smi output")
    out, src = run.get("gpu.proc_driver_nvidia")
    m = re.search(r"NVRM version:\s*(.*)", out or "")
    if m:
        F.confirmed(S, "kernel module (NVRM version line)", m.group(1).strip(), src)
        F.confirmed(S, "kernel module flavour", "open" if "Open Kernel Module" in m.group(1) else "proprietary", src)
    out, src = run.get("gpu.query_csv")
    rows = _csv_rows(out) if out is not None else []
    if rows:
        F.confirmed(S, "GPU count (nvidia-smi)", len(rows), src)
        for r in rows:
            i = r.get("index", "?")
            def g(k):
                for kk, vv in r.items():
                    if kk.split(" [")[0] == k:
                        return vv
                return None
            F.confirmed(S, "GPU%s name" % i, g("name"), src)
            F.confirmed(S, "GPU%s PCI bus id" % i, g("pci.bus_id"), src)
            F.confirmed(S, "GPU%s memory.total" % i, g("memory.total"), src)
            F.confirmed(S, "GPU%s PCIe gen current/max" % i, "%s / %s" % (g("pcie.link.gen.current"), g("pcie.link.gen.max")), src)
            F.confirmed(S, "GPU%s PCIe width current/max" % i, "x%s / x%s" % (g("pcie.link.width.current"), g("pcie.link.width.max")), src)
            F.confirmed(S, "GPU%s power limit / max memory clock / vbios" % i, "%s / %s / %s" % (g("power.limit"), g("clocks.max.memory"), g("vbios_version")), src)
            mt = re.match(r"(\d+)\s*MiB", g("memory.total") or "")
            if mt and int(mt.group(1)) < 30 * 1024:
                F.flag("GPU%s reports %s total memory (expected about 32 GB on the stated cards)" % (i, g("memory.total")), src)
            gm, wm, wc = g("pcie.link.gen.max"), g("pcie.link.width.max"), g("pcie.link.width.current")
            if gm and gm.isdigit() and int(gm) < 4:
                F.flag("GPU%s PCIe max generation is Gen%s (card supports Gen4): slot, riser or BIOS setting limits it" % (i, gm), src)
            if wm and wm.isdigit() and int(wm) < 16:
                F.flag("GPU%s PCIe max width is x%s (not x16)" % (i, wm), src)
            if wc and wm and wc.isdigit() and wm.isdigit() and int(wc) < int(wm):
                F.flag("GPU%s PCIe link runs at x%s although x%s is possible" % (i, wc, wm), src)
    else:
        F.unknown(S, "GPU count (nvidia-smi)", src if out is None else "no rows in query output")
    out, src = run.get("gpu.q_memory")
    if out is not None:
        gpu, sect = None, None
        for line in out.splitlines():
            m = re.match(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", line.strip())
            if m:
                gpu = m.group(1)
                continue
            if "BAR1 Memory Usage" in line:
                sect = "bar1"
                continue
            if "Memory Usage" in line:
                sect = "other"
                continue
            m = re.match(r"\s*Total\s*:\s*(.*)", line)
            if m and sect == "bar1" and gpu:
                F.confirmed(S, "BAR1 total %s" % gpu, m.group(1).strip(), src)
                mm = re.match(r"(\d+)\s*MiB", m.group(1).strip())
                if mm and int(mm.group(1)) <= 256:
                    F.flag("GPU %s BAR1 is %s (Resizable BAR off): P2P over BAR1 and large mappings are limited" % (gpu, m.group(1).strip()), src)
                sect = None
    for cap in ("r", "w"):
        out, src = run.get("gpu.topo_p2p_" + cap)
        if out is None:
            F.unknown(S, "P2P %s matrix (nvidia-smi topo -p2p %s)" % ({"r": "read", "w": "write"}[cap], cap), src)
            continue
        head, mat = _p2p_matrix(out)
        if not mat:
            F.unknown(S, "P2P %s matrix (nvidia-smi topo -p2p %s)" % ({"r": "read", "w": "write"}[cap], cap), "matrix not parseable in " + src)
            continue
        pairs = ["%s>%s:%s" % (a[3:], b[3:], v) for (a, b), v in sorted(mat.items()) if a != b]
        F.confirmed(S, "P2P %s matrix (nvidia-smi topo -p2p %s)" % ({"r": "read", "w": "write"}[cap], cap), " ".join(pairs), src)
        bad = [p for p in pairs if not p.endswith(":OK")]
        if bad:
            F.flag("nvidia-smi topo -p2p %s: pairs without P2P support: %s" % (cap, " ".join(bad)), src)
        F.raw.append(("nvidia-smi topo -p2p " + cap, src, out))
    out, src = run.get("gpu.topo_matrix")
    if out is not None:
        F.raw.append(("nvidia-smi topo -m", src, out))
        mat = _topo_m(out)
        pairs = ["%s-%s:%s" % (a[3:], b[3:], v) for (a, b), v in sorted(mat.items())
                 if a.startswith("GPU") and b.startswith("GPU") and a < b]
        if pairs:
            F.confirmed(S, "GPU-GPU topology (topo -m)", " ".join(pairs), src)
        else:
            F.unknown(S, "GPU-GPU topology (topo -m)", "matrix not parseable in " + src)
    out, src = run.get("gpu.compute_apps")
    if out is not None:
        rows = _csv_rows(out)
        F.confirmed(S, "GPU compute processes at collection time", len(rows), src)


def ex_pcie(run, F):
    S = "pcie"
    out, src = run.get("pcie.sysfs_links")
    if out is None:
        F.unknown(S, "PCIe links of GPUs/NVMe (sysfs)", src)
        return
    rows = tsv_table(out)
    gpus = [r for r in rows if r.get("role") == "gpu"]
    nv = [r for r in rows if r.get("role") == "nvme"]
    F.confirmed(S, "NVIDIA display-class PCI functions (sysfs)", len(gpus), src)
    F.confirmed(S, "NVMe controllers on PCI (sysfs)", len(nv), src)
    for r in gpus + nv:
        F.confirmed(S, "%s %s link current / max" % (r["role"], r["bdf"]),
                    "%s x%s / %s x%s" % (r.get("cur_speed"), r.get("cur_width"), r.get("max_speed"), r.get("max_width")), src)
        F.confirmed(S, "%s %s numa_node / BARs / upstream" % (r["role"], r["bdf"]),
                    "%s / %s / %s" % (r.get("numa_node"), r.get("bars"), r.get("upstream")), src)
        for k in ("aer_cor", "aer_nonfatal", "aer_fatal"):
            v = r.get(k, "-")
            if v.isdigit() and int(v) > 0:
                F.flag("%s %s has %s=%s (PCIe link errors; check riser/cable/slot)" % (r["role"], r["bdf"], k, v), src)
        if r["role"] == "gpu":
            mw, cw, ms = r.get("max_width", "-"), r.get("cur_width", "-"), r.get("max_speed", "")
            if mw.isdigit() and int(mw) < 16:
                F.flag("GPU %s max link width x%s in sysfs (not x16)" % (r["bdf"], mw), src)
            if ms.startswith(("2.5", "5.0", "8.0")):
                F.flag("GPU %s max link speed %s in sysfs (Ada cards are Gen4, 16.0 GT/s)" % (r["bdf"], ms), src)
            if cw.isdigit() and mw.isdigit() and int(cw) < int(mw):
                F.flag("GPU %s link runs at x%s of x%s (sysfs)" % (r["bdf"], cw, mw), src)
        if r["role"] == "nvme":
            mw = r.get("max_width", "-")
            if mw.isdigit() and int(mw) < 4:
                F.flag("NVMe %s max link width x%s (expected x4)" % (r["bdf"], mw), src)
            ms = r.get("max_speed", "")
            if ms.startswith(("2.5", "5.0", "8.0")):
                F.flag("NVMe %s max link speed %s (990 Pro is Gen4, 16.0 GT/s)" % (r["bdf"], ms), src)
            cw = r.get("cur_width", "-")
            if cw.isdigit() and mw.isdigit() and int(cw) < int(mw):
                F.flag("NVMe %s link runs at x%s of x%s" % (r["bdf"], cw, mw), src)
    for r in rows:
        if r.get("role") == "bridge":
            for k in ("aer_cor", "aer_nonfatal", "aer_fatal"):
                v = r.get(k, "-")
                if v.isdigit() and int(v) > 0:
                    F.flag("bridge %s (upstream of a GPU/NVMe) has %s=%s" % (r["bdf"], k, v), src)
    out, src = run.get("pcie.iommu_groups")
    m = re.search(r"# iommu groups: (\d+)", out or "")
    if m:
        F.confirmed(S, "IOMMU groups", m.group(1), src)
    out, src = run.get("pcie.lspci_vvv_all")
    if out is not None:
        acs = len(re.findall(r"ACSCtl:.*ReqRedir\+", out))
        if "ACSCtl" in out:
            F.confirmed(S, "bridges with ACS ReqRedir+ enabled (lspci -vvv)", acs, src)
            if acs:
                F.flag("%d PCIe functions have ACS request redirect enabled: GPU P2P traffic is forced through the root complex" % acs, src)


def ex_storage(run, F):
    S = "storage"
    out, src = run.get("storage.nvme_sysfs")
    if out is not None:
        ctrls = {}
        for line in out.splitlines()[1:]:
            p = line.split("\t", 2)
            if len(p) == 3:
                ctrls.setdefault(p[0], {}).setdefault(p[1], []).append(p[2])
        F.confirmed(S, "NVMe controllers (sysfs)", len(ctrls), src)
        for c, kv in sorted(ctrls.items()):
            g = lambda k: kv.get(k, ["?"])[0]
            F.confirmed(S, "%s model / firmware" % c, "%s / %s" % (g("model"), g("firmware_rev")), src)
            F.confirmed(S, "%s link current / max" % c, "%s x%s / %s x%s" % (g("cur_link_speed"), g("cur_link_width"), g("max_link_speed"), g("max_link_width")), src)
            if "temp_c" in kv:
                F.confirmed(S, "%s temperature (C, hwmon at collection)" % c, g("temp_c"), src)
                try:
                    if float(g("temp_c")) > 70:
                        F.flag("%s is at %s C at collection time (> 70 C)" % (c, g("temp_c")), src)
                except ValueError:
                    pass
            if "namespace" in kv:
                F.confirmed(S, "%s namespaces" % c, "; ".join(kv["namespace"]), src)
    else:
        F.unknown(S, "NVMe controllers (sysfs)", src)
    for name in run.names("storage.nvme_smart_log."):
        out, src = run.get(name)
        c = name.rsplit(".", 1)[1]
        if out is None:
            F.unknown(S, "%s SMART (nvme smart-log)" % c, src)
            continue
        kv = {k.lower().strip(): v for k, v in kv_lines(out).items()}
        parts = []
        for k in ("critical_warning", "temperature", "percentage_used", "media_errors", "num_err_log_entries",
                  "warning temperature time", "critical composite temperature time"):
            if k in kv:
                parts.append("%s=%s" % (k, kv[k]))
        F.confirmed(S, "%s SMART (nvme smart-log)" % c, ", ".join(parts) or "fields not found", src)
        if kv.get("critical_warning", "0").split()[0] not in ("0", "0x0", "0x00"):
            F.flag("%s critical_warning=%s" % (c, kv["critical_warning"]), src)
        me = kv.get("media_errors", "0").replace(",", "").split()[0]
        if me.isdigit() and int(me) > 0:
            F.flag("%s media_errors=%s" % (c, me), src)
        t = re.match(r"(\d+)", kv.get("temperature", ""))
        if t and int(t.group(1)) > 70:
            F.flag("%s SMART temperature %s (> 70 C)" % (c, kv["temperature"]), src)
    out, src = run.get("storage.block_queues")
    if out is not None:
        nv = [r for r in tsv_table(out) if r.get("dev", "").startswith("nvme")]
        if nv:
            F.confirmed(S, "NVMe block schedulers", "; ".join("%s:%s" % (r["dev"], re.sub(r".*\[(\S+)\].*", r"\1", r.get("scheduler", "?"))) for r in nv), src)
    for probe, fact in (("storage.mdstat", "md RAID (/proc/mdstat)"), ("storage.zpool_status", "ZFS pools (zpool status)")):
        out, src = run.get(probe)
        if out is not None:
            F.confirmed(S, fact, (first_line(out) or "(empty)") if probe != "storage.mdstat" else
                        (" ".join(l.split(":")[0] for l in out.splitlines() if re.match(r"md\d+\s*:", l)) or "no md arrays"), src)


def ex_software(run, F):
    S = "software"
    out, src = run.get("software.nvcc")
    m = re.search(r"release ([\d.]+), (V[\d.]+)", out or "")
    if m:
        F.confirmed(S, "nvcc (CUDA toolkit)", "%s %s" % m.groups(), src)
    else:
        F.unknown(S, "nvcc (CUDA toolkit)", src if out is None else "release line not found")
    for probe, fact in (("software.gcc", "gcc"), ("software.cmake", "cmake"), ("software.python3", "python3"),
                        ("software.docker", "docker"), ("software.git", "git")):
        out, src = run.get(probe)
        if out is not None and first_line(out):
            F.confirmed(S, fact, first_line(out), src)
        else:
            F.unknown(S, fact, src)
    out, src = run.get("software.cuda_libraries")
    if out is not None:
        libs = sorted(set(re.findall(r"^\s*(\S+)", out, re.M)))
        F.confirmed(S, "CUDA libraries in ldconfig cache", ", ".join(libs) if libs else "none", src)
    out, src = run.get("software.llm_tools")
    if out is not None:
        found = [l.split("\t")[0] for l in out.splitlines() if "\t" in l and not l.endswith("not found")]
        F.confirmed(S, "tools found in PATH (ollama/llama.cpp/vllm/fio/...)", ", ".join(found) if found else "none", src)


def ex_dmesg(run, F):
    S = "kernel log"
    out, src = run.get("system.dmesg_filtered")
    if out is None:
        F.unknown(S, "dmesg error lines", src)
        return
    lines = out.splitlines()
    checks = (("NVIDIA Xid lines", r"\bXid\b"), ("AER lines", r"\bAER\b"),
              ("machine check / hardware error lines", r"mce:|machine check|Hardware Error"),
              ("EDAC error lines", r"EDAC.*(CE|UE|error)"),
              ("NVMe timeout/reset/error lines", r"nvme.*(timeout|reset|error|I/O)"),
              ("'GPU has fallen off the bus' lines", r"fallen off"),
              ("PCIe bandwidth 'limited by' lines", r"limited by"))
    for fact, pat in checks:
        hit = [l for l in lines if re.search(pat, l, re.I)]
        F.confirmed(S, fact, len(hit), src)
        if not hit:
            continue
        if fact == "AER lines" and not any(re.search(r"error", l, re.I) for l in hit):
            continue  # AER capability announcements only
        F.flag("%s in dmesg: %d (first: %s)" % (fact, len(hit), hit[0].strip()[:160]), src)


# --------------------------------------------------------------------------- benchmarks
def ex_bench(run, F):
    S = "benchmarks"
    d = run.dir
    for path in sorted(glob.glob(os.path.join(d, "bench", "mem", "*", "membw.csv"))):
        rel = os.path.relpath(path, d)
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("kernel") in ("triad", "read", "copy"):
                    F.confirmed(S, "membw %s %s (threads=%s) best / median GB/s" % (r["variant"], r["kernel"], r["threads"]),
                                "%s / %s" % (r["best_GBps"], r["median_GBps"]), rel)
    for path in sorted(glob.glob(os.path.join(d, "bench", "gpu", "*", "p2p_matrix.csv"))):
        rel = os.path.relpath(path, d)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        devs = {}
        for r in rows:
            if r["kind"] == "device":
                devs.setdefault(r["src"], {})[r["metric"]] = r["value"]
        for i, kv in sorted(devs.items()):
            tg = kv.get("total_global_mem", "")
            F.confirmed(S, "CUDA device %s name / bus / totalGlobalMem" % i,
                        "%s / %s / %s bytes (%.2f GiB)" % (kv.get("name"), kv.get("pci_bus_id"), tg, int(tg) / 2**30 if tg.isdigit() else 0), rel)
        cap = ["%s>%s:%s" % (r["src"], r["dst"], r["value"]) for r in rows if r["kind"] == "can_access_peer"]
        if cap:
            F.confirmed(S, "cudaDeviceCanAccessPeer", " ".join(cap), rel)
            if any(c.endswith(":0") for c in cap):
                F.flag("cudaDeviceCanAccessPeer = 0 for: " + " ".join(c for c in cap if c.endswith(":0")), rel)
        en = ["%s>%s:%s" % (r["src"], r["dst"], r["value"]) for r in rows if r["kind"] == "enable_peer"]
        if en:
            F.confirmed(S, "cudaDeviceEnablePeerAccess", " ".join(en), rel)
        for kind in ("d2d_peer", "d2d_nopeer", "h2d_pinned", "d2h_pinned"):
            v = ["%s>%s:%s" % (r["src"], r["dst"], r["value"]) for r in rows if r["kind"] == kind and r["metric"] == "best_GBps"]
            if v:
                F.confirmed(S, "%s best GB/s" % kind, " ".join(v), rel)
        for kind in ("h2d_all_simultaneous", "d2h_all_simultaneous"):
            for r in rows:
                if r["kind"] == kind and r["metric"] in ("aggregate_GBps", "aggregate_over_sum_of_solo"):
                    F.confirmed(S, "%s %s" % (kind, r["metric"]), r["value"], rel)
                    if r["metric"] == "aggregate_over_sum_of_solo":
                        try:
                            if float(r["value"]) < 0.75:
                                F.flag("%s: aggregate is %s of the sum of solo bandwidths (shared PCIe upstream?)" % (kind, r["value"]), rel)
                        except ValueError:
                            pass
    for path in sorted(glob.glob(os.path.join(d, "bench", "gpu", "*", "link_samples.csv"))):
        rel = os.path.relpath(path, d)
        with open(path, encoding="utf-8") as f:
            rows = _csv_rows(f.read())
        best = {}
        for r in rows:
            def g(k):
                for kk, vv in r.items():
                    if kk.split(" [")[0] == k:
                        return vv
                return None
            i, gen, w = g("index"), g("pcie.link.gen.current"), g("pcie.link.width.current")
            if i is None or not (gen or "").isdigit() or not (w or "").isdigit():
                continue
            b = best.setdefault(i, [0, 0])
            b[0], b[1] = max(b[0], int(gen)), max(b[1], int(w))
        for i, (gen, w) in sorted(best.items()):
            F.confirmed(S, "GPU%s highest PCIe gen / width seen during the bandwidth test" % i, "Gen%d x%d" % (gen, w), rel)
            if gen < 4 or w < 16:
                F.flag("GPU%s reached only Gen%d x%d under load" % (i, gen, w), rel)
    for path in sorted(glob.glob(os.path.join(d, "bench", "gpu", "*", "vram_check.txt"))):
        rel = os.path.relpath(path, d)
        with open(path, encoding="utf-8", errors="replace") as f:
            sec = parse_probe_text(f.read())
        for line in sec["stdout"].splitlines():
            if line.startswith("RESULT\t"):
                kv = dict(p.split("=", 1) for p in line.split("\t")[1:] if "=" in p)
                F.confirmed(S, "VRAM check GPU%s" % kv.get("dev"),
                            "total_global_mem=%s allocated=%s verified_bytes=%s mismatches=%s bad_bits=%s" % (
                                kv.get("total_global_mem"), kv.get("allocated"), kv.get("verified_bytes"),
                                kv.get("mismatches"), kv.get("bad_bits")), rel)
                if kv.get("mismatches", "0") != "0":
                    F.flag("VRAM check GPU%s: %s mismatches (bad bits %s)" % (kv.get("dev"), kv.get("mismatches"), kv.get("bad_bits")), rel)
    for path in sorted(glob.glob(os.path.join(d, "bench", "nvme", "*", "*.json"))):
        rel = os.path.relpath(path, d)
        for r in fio_rows(path):
            F.confirmed(S, "fio %s %s" % (os.path.basename(path)[:-5], r["group"]),
                        "%s MB/s, %s IOPS, clat p50/p99/p99.9 %s/%s/%s us" % (r["MBps"], r["iops"], r["p50_us"], r["p99_us"], r["p99.9_us"]), rel)
    for path in sorted(glob.glob(os.path.join(d, "bench", "nvme", "*", "temps.tsv"))):
        rel = os.path.relpath(path, d)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        mx = {}
        for r in rows:
            try:
                t = float(re.match(r"[\d.]+", r["temp_c"]).group(0))
            except (AttributeError, ValueError, TypeError):
                continue
            mx[r["controller"]] = max(mx.get(r["controller"], 0), t)
        for c, t in sorted(mx.items()):
            F.confirmed(S, "%s max temperature during fio (C)" % c, "%.1f" % t, rel)
            if t > 70:
                F.flag("%s reached %.1f C during fio (> 70 C: thermal throttling likely)" % (c, t), rel)


def fio_rows(path):
    """One row per fio job group: bandwidth, IOPS and completion-latency percentiles."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        j = json.loads(text[text.find("{"):])
    except (ValueError, OSError):
        return []
    rows = []
    gopt = j.get("global options", {})
    for job in j.get("jobs", []):
        for ddir in ("read", "write"):
            r = job.get(ddir, {})
            if not r or not r.get("io_bytes"):
                continue
            clat = r.get("clat_ns") or {}
            pct = clat.get("percentile") or {}

            def p(k):
                v = pct.get(k)
                return "%.1f" % (v / 1000.0) if isinstance(v, (int, float)) else "n/a"
            bw = r.get("bw_bytes", r.get("bw", 0) * 1024)
            opt = dict(gopt)
            opt.update(job.get("job options", {}))
            rows.append({"file": os.path.basename(path), "group": job.get("jobname", "?"), "dir": ddir,
                         "rw": opt.get("rw", "?"), "bs": opt.get("bs", "?"), "iodepth": opt.get("iodepth", "?"),
                         "numjobs": opt.get("numjobs", "1"), "filename": opt.get("filename", "?"),
                         "MBps": "%.1f" % (bw / 1e6), "iops": "%.0f" % r.get("iops", 0),
                         "p50_us": p("50.000000"), "p95_us": p("95.000000"), "p99_us": p("99.000000"),
                         "p99.9_us": p("99.900000")})
    return rows


def fio_summary(paths, csv_path=None):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        rows += fio_rows(f)
    cols = ["file", "group", "dir", "rw", "bs", "iodepth", "numjobs", "MBps", "iops", "p50_us", "p95_us", "p99_us", "p99.9_us", "filename"]
    out = io.StringIO()
    out.write("fio summary (MB/s = 1e6 bytes/s; latency = completion latency percentiles in microseconds)\n")
    w = [max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c) for c in cols]
    out.write("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + "\n")
    for r in rows:
        out.write("  ".join(str(r[c]).ljust(w[i]) for i, c in enumerate(cols)) + "\n")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    return out.getvalue(), rows


# --------------------------------------------------------------------------- report
EXTRACTORS = (ex_run, ex_system, ex_cpu, ex_memory, ex_board, ex_gpu, ex_pcie, ex_storage, ex_software, ex_dmesg, ex_bench)


def summarize(d):
    run = Run(d)
    F = Facts()
    for ex in EXTRACTORS:
        try:
            ex(run, F)
        except Exception as e:  # a parser bug must not turn into an invented value
            F.unknown(ex.__name__[3:], "extractor %s" % ex.__name__, "parser error: %s: %s" % (type(e).__name__, e))
    return run, F


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def render(run, F):
    m = run.manifest
    out = io.StringIO()
    w = out.write
    w("# HARDWARE_FACTS (draft): %s, collected %s\n\n" % (m.get("hostname", "?"), m.get("started_utc", "?")))
    if m.get("synthetic"):
        w("> **SYNTHETIC TEST FIXTURE - NOT EVIDENCE.** The input directory is marked synthetic; "
          "every value below is invented test data.\n\n")
    w("> Draft generated by `scripts/diag/summarize.py` from `%s`. CONFIRMED = parsed from the literal\n"
      "> output in the named file of this directory/archive; PROVISIONAL = derived, the derivation is stated;\n"
      "> UNKNOWN = missing or not parseable, nothing was filled in. Review before copying into docs/evidence/.\n\n"
      % os.path.basename(os.path.abspath(run.dir)))
    w("## Red flags (automatic checks; verify each against its source)\n\n")
    if F.flags:
        for fl in F.flags:
            w("- %s (source: `%s`)\n" % (fl["flag"], fl["source"]))
    else:
        w("- none detected by the automatic checks (this is not a proof of health)\n")
    w("\n")
    sections = []
    for r in F.rows:
        if r["section"] not in sections:
            sections.append(r["section"])
    for s in sections:
        w("## %s\n\n| fact | value | label |\n|---|---|---|\n" % s)
        for r in F.rows:
            if r["section"] != s:
                continue
            lab = "%s (%s)" % (r["label"], r["source"]) if r["source"] else r["label"]
            w("| %s | %s | %s |\n" % (md_escape(r["fact"]), md_escape(r["value"]), md_escape(lab)))
        w("\n")
    if F.raw:
        w("## Literal excerpts\n\n")
        for title, src, text in F.raw:
            w("### %s (`%s`)\n\n```\n%s\n```\n\n" % (title, src, text.rstrip("\n")))
    w("## Probes that did not return ok\n\n")
    bad = [p for p in run.probes.values() if p.get("status") != "ok"]
    if bad:
        w("| probe | status | note |\n|---|---|---|\n")
        for p in sorted(bad, key=lambda p: p.get("seq", 0)):
            w("| %s | %s | %s |\n" % (md_escape(p["name"]), p.get("status"), md_escape(p.get("note", ""))))
    else:
        w("all probes ok\n")
    return out.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="diag-out directory, or fio JSON files/dirs with --fio-summary")
    ap.add_argument("--output", help="markdown output (default DIAGDIR/HARDWARE_FACTS.draft.md)")
    ap.add_argument("--json", help="also write the facts as JSON")
    ap.add_argument("--fio-summary", action="store_true", help="print a summary table of fio JSON outputs")
    ap.add_argument("--csv", help="with --fio-summary: also write CSV")
    a = ap.parse_args(argv)
    if a.fio_summary:
        text, rows = fio_summary(a.paths, a.csv)
        sys.stdout.write(text)
        return 0 if rows else 1
    if len(a.paths) != 1 or not os.path.isdir(a.paths[0]):
        ap.error("expected one diag-out directory")
    d = a.paths[0]
    run, F = summarize(d)
    md = render(run, F)
    outp = a.output or os.path.join(d, "HARDWARE_FACTS.draft.md")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"synthetic": bool(run.manifest.get("synthetic")), "facts": F.rows, "red_flags": F.flags}, f, indent=1)
    print("wrote %s (%d facts, %d red flags)" % (outp, len(F.rows), len(F.flags)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
