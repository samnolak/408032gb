"""scripts/diag: read-only hardware diagnostic kit.

Runs without GPU and without root requirements:
  * collect.sh end to end (as "non-root" via the DIAG_ASSUME_NONROOT test hook, as real root
    when the sandbox is root, and with --sudo against a stub sudo), with stub tools that
    record their argv so the test sees exactly what the kit executed;
  * manifest.json validity and status consistency (not-available / needs-root / skipped),
    probe files, tarball, per-probe timeout, failing probes do not stop the run;
  * static scan of scripts/diag for forbidden (state-changing) commands;
  * opt-in benchmarks refuse to run without --run; bench_nvme.sh with a stub fio
    (only the prep job writes, only into the new test file; every read job is --readonly;
    the test file is deleted; raw mode stays read-only);
  * membw.c compiled with gcc -fopenmp and run on a tiny array; bench_mem.sh on a tiny array;
  * summarize.py on the SYNTHETIC fixtures in tests/fixtures/diag (never invents values);
  * nvcc compile of the .cu files for sm_89 when nvcc is available (env NVCC or PATH),
    shellcheck when available (env SHELLCHECK or PATH). Otherwise these two are skipped.
"""
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAG = ROOT / "scripts" / "diag"
FIX = ROOT / "tests" / "fixtures" / "diag"
sys.path.insert(0, str(DIAG))
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep scripts/diag free of __pycache__
import summarize  # noqa: E402
sys.dont_write_bytecode = _dwb

STUB_TOOLS = ["dmidecode", "nvidia-smi", "nvme", "smartctl", "ipmitool", "lspci", "cpupower", "numactl",
              "mdadm", "pvs", "vgs", "lvs", "zpool", "edac-util", "ras-mc-ctl", "lstopo-no-graphics",
              "pveversion", "qm", "curl", "wget", "sudo", "fio", "modprobe", "sysctl", "setpci", "nvcc"]

STUB = r"""#!/bin/sh
me=$(basename "$0")
{ printf '%s' "$me"; for a in "$@"; do printf '\t%s' "$a"; done; printf '\n'; } >> "$DIAG_STUB_LOG"
case "$me" in
  dmidecode)
    printf '# dmidecode stub\n\nHandle 0x0010, DMI type 16, 23 bytes\nPhysical Memory Array\n\tNumber Of Devices: 2\n\n'
    printf 'Handle 0x0011, DMI type 17, 92 bytes\nMemory Device\n\tSize: 16 GB\n\tLocator: DIMM_A1\n\tBank Locator: P0 CHANNEL A\n\tType: DDR4\n\tSpeed: 3200 MT/s\n\tConfigured Memory Speed: 3200 MT/s\n\n'
    printf 'Handle 0x0012, DMI type 17, 92 bytes\nMemory Device\n\tSize: 16 GB\n\tLocator: DIMM_B1\n\tBank Locator: P0 CHANNEL B\n\tType: DDR4\n\tSpeed: 3200 MT/s\n\tConfigured Memory Speed: 3200 MT/s\n\n'
    ;;
  cpupower) sleep 30 ;;
  numactl) echo "stub numactl failure" >&2; exit 3 ;;
  nvidia-smi)
    case "$*" in
      --query-gpu=*) echo "index, name"; echo "0, STUB GPU" ;;
      *) echo "stub nvidia-smi" ;;
    esac ;;
  sudo)
    [ "$1" = "-n" ] || { echo "stub sudo: interactive use" >&2; exit 1; }
    exit 0 ;;
  fio)
    out=""; file=""; prep=0
    for a in "$@"; do
      case "$a" in
        --output=*) out=${a#--output=} ;;
        --filename=*) file=${a#--filename=} ;;
        --name=prep) prep=1 ;;
      esac
    done
    if [ $prep -eq 1 ]; then truncate -s "${DIAG_STUB_FIO_SIZE:-1G}" "$file"; fi
    if [ -n "$out" ]; then
      printf '{"fio version": "stub", "global options": {"rw": "randread", "bs": "4k"}, "jobs": [{"jobname": "d0", "job options": {"name": "d0"}, "read": {"io_bytes": 4096, "bw_bytes": 1000000, "iops": 244.0, "clat_ns": {"percentile": {"50.000000": 1000, "95.000000": 2000, "99.000000": 3000, "99.900000": 4000}}}, "write": {"io_bytes": 0}}]}\n' > "$out"
    fi ;;
  *) echo "stub $me" ;;
esac
exit 0
"""

NVSMI_ALLOWED = [r"", r"-L", r"-q", r"--version", r"topo -m", r"topo -p2p [rwnap]", r"-q -d [A-Z_]+",
                 r"--query-gpu=[a-z0-9_.,]+ --format=csv(,noheader)?( -lms \d+)?",
                 r"--query-compute-apps=[a-z0-9_.,]+ --format=csv(,noheader)?"]
NVME_ALLOWED = {"list", "id-ctrl", "id-ns", "smart-log", "error-log"}

# state-changing commands that must never appear anywhere in scripts/diag
FORBIDDEN = [
    (r"nvidia-smi[^|;&\n]*\s(-pl|-pm|-ac|-rac|-lgc|-rgc|-lmc|-rmc|-r|-e|-c|-am|-caa|-mig|-acp|-cc)(\s|$)", "nvidia-smi setting"),
    (r"nvidia-smi[^|;&\n]*--(power-limit|persistence-mode|applications-clocks|reset-applications-clocks|lock-gpu-clocks|lock-memory-clocks|gpu-reset|ecc-config|compute-mode|auto-boost)", "nvidia-smi setting"),
    (r"\bnvidia-smi\s+(drain|mig|conf-compute|vgpu)\b", "nvidia-smi admin subcommand"),
    (r"\bsysctl\b", "sysctl"),
    (r"\bmkfs", "mkfs"),
    (r"\bdd\b[^\n]*\bof=", "dd of="),
    (r"\bwipefs\b", "wipefs"),
    (r"\bnvme\s+(format|sanitize|write|write-zeroes|write-uncor|dsm|fw-download|fw-commit|fw-activate|set-feature|ns-delete|delete-ns|create-ns|detach-ns|attach-ns|reset|subsystem-reset|security-send|admin-passthru|io-passthru)\b", "nvme write/admin"),
    (r"\b(blkdiscard|fdisk|sfdisk|sgdisk|gdisk|parted|hdparm|shred|setpci)\b", "disk/PCI writer"),
    (r"\bsmartctl\b[^\n]*\s-(t|s|o|S|X)(\s|$)", "smartctl self-test/setting"),
    (r"\bipmitool\b[^\n]*\b(power|chassis|raw|reset|sol|shell|exec|set|clear|write|fill|delete)\b", "ipmitool action"),
    (r"\b(modprobe|insmod|rmmod)\b", "kernel module change"),
    (r"\bcpupower\s+(frequency-set|set|idle-set)\b", "cpupower set"),
    (r"\bsystemctl\s+(stop|start|restart|reload|disable|enable|mask|kill|reboot|poweroff|isolate|set-property)\b", "systemctl action"),
    (r"(^|[;&|]\s*)(reboot|shutdown|poweroff|halt)\b", "power action"),
    (r"\b(apt|apt-get|yum|dnf|zypper)\s+(install|remove|upgrade)\b|\bpip3?\s+install\b", "package install"),
    (r"\bqm\s+(set|start|stop|shutdown|reset|destroy|create|migrate|resize|move-disk|unlink|reboot|suspend|resume)\b", "qm action"),
    (r"\bmdadm\b[^\n]*--(create|grow|remove|fail|zero-superblock|assemble|stop|add)\b", "mdadm change"),
    (r"\bzpool\s+(create|destroy|import|export|clear|scrub|offline|online|replace|attach|detach|add|remove|set|labelclear|upgrade)\b", "zpool change"),
    (r"\b(pvcreate|vgcreate|lvcreate|pvremove|lvremove|vgremove|lvchange|vgchange|pvchange|lvextend|lvresize)\b", "lvm change"),
    (r"(echo|printf|cat)\b[^\n]*>\s*/(sys|proc)/", "write to sysfs/procfs"),
    (r"\btee\s+(-a\s+)?/(sys|proc)/", "write to sysfs/procfs"),
    (r">\s*/dev/(nvme|sd|vd|hd|md|dm-|mapper|nbd|loop|zd)", "redirect into a block device"),
    (r"\brm\s+-rf?\s+/(\s|$)", "rm -rf /"),
]


def which_nvcc():
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and os.access(c, os.X_OK):
            return c
    return None


def which_shellcheck():
    for c in (os.environ.get("SHELLCHECK"), shutil.which("shellcheck")):
        if c and os.access(c, os.X_OK):
            return c
    return None


def gcc_openmp():
    if not shutil.which("gcc"):
        return False
    r = subprocess.run(["gcc", "-fopenmp", "-x", "c", "-", "-o", os.devnull], input="#include <omp.h>\nint main(void){return omp_get_max_threads()<0;}\n",
                       capture_output=True, text=True)
    return r.returncode == 0


class StubEnv:
    """Temp dir with argv-recording stub tools first in PATH."""

    def __init__(self, tools=STUB_TOOLS):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.bin = self.dir / "stubbin"
        self.bin.mkdir()
        self.log = self.dir / "stub.log"
        self.log.write_text("")
        for t in tools:
            p = self.bin / t
            p.write_text(STUB)
            p.chmod(0o755)

    def env(self, **extra):
        e = dict(os.environ)
        e["PATH"] = "%s:%s" % (self.bin, e.get("PATH", "/usr/bin:/bin"))
        e["DIAG_STUB_LOG"] = str(self.log)
        e.pop("DIAG_ASSUME_NONROOT", None)
        e.update(extra)
        return e

    def calls(self, tool=None):
        out = []
        for line in self.log.read_text().splitlines():
            parts = line.split("\t")
            if tool is None or parts[0] == tool:
                out.append(parts)
        return out

    def cleanup(self):
        self.tmp.cleanup()


def run_collect(stub, out, *args, **env):
    return subprocess.run(["bash", str(DIAG / "collect.sh"), "--out", str(out), "--timeout", "3", *args],
                          capture_output=True, text=True, env=stub.env(**env), timeout=600)


def tool_available(req, path):
    for alt in req.split("|"):
        if alt.startswith("/"):
            if os.path.exists(alt):
                return True
        elif shutil.which(alt, path=path):
            return True
    return False


class Collect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = StubEnv()
        cls.out = cls.stub.dir / "run-nonroot"
        cls.res = run_collect(cls.stub, cls.out, DIAG_ASSUME_NONROOT="1")
        cls.manifest = json.loads((cls.out / "manifest.json").read_text()) if (cls.out / "manifest.json").exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.stub.cleanup()

    def probe(self, name):
        for p in self.manifest["probes"]:
            if p["name"] == name:
                return p
        self.fail("probe %s missing" % name)

    def test_exit_zero_and_manifest(self):
        self.assertEqual(self.res.returncode, 0, self.res.stdout[-2000:] + self.res.stderr[-2000:])
        m = self.manifest
        self.assertIsNotNone(m)
        self.assertEqual(m["run_state"], "completed")
        self.assertFalse(m["is_root"])
        self.assertTrue(m["assume_nonroot_test_hook"])
        self.assertFalse(m["synthetic"])
        self.assertGreater(len(m["probes"]), 80)
        names = [p["name"] for p in m["probes"]]
        self.assertEqual(len(names), len(set(names)))
        for cat in ("system", "cpu", "memory", "board", "bmc", "pcie", "gpu", "storage", "software", "network"):
            self.assertIn(cat, {p["category"] for p in m["probes"]}, cat)
        counts = {}
        for p in m["probes"]:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        for k, v in m["counts"].items():
            self.assertEqual(counts.get(k, 0), v, k)

    def test_probe_files_have_literal_command_and_result(self):
        for p in self.manifest["probes"]:
            text = (self.out / p["output_file"]).read_text(errors="replace")
            sec = summarize.parse_probe_text(text)
            self.assertEqual(sec["command"], p["command"], p["name"])
            self.assertEqual(sec["result"]["status"], p["status"], p["name"])
            self.assertIn("started_utc", sec["meta"], p["name"])
            self.assertRegex(p["started_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            if p["status"] in ("ok", "failed") or (p["status"] == "needs-root" and p["run_as"] == "user"):
                self.assertIsInstance(p["exit_code"], int, p["name"])
                self.assertEqual(sec["result"]["exit_code"], str(p["exit_code"]), p["name"])
            else:
                self.assertIsNone(p["exit_code"], p["name"])

    def test_status_consistency(self):
        path = self.manifest["path"]
        root_only = re.compile(r"^(dmidecode|ipmitool|smartctl|mdadm|qm|pvs|nvme (id-ctrl|id-ns|smart-log|error-log))\b")
        for p in self.manifest["probes"]:
            if root_only.match(p["command"]):
                self.assertEqual(p["root_mode"], "root", p["name"])
            if p["status"] == "skipped":
                self.assertEqual(p["name"], "network.internet_check")
                continue
            missing = [r for r in p["requires"] if not tool_available(r, path)]
            if missing:
                self.assertEqual(p["status"], "not-available", (p["name"], missing))
            elif p["root_mode"] == "root":
                self.assertEqual(p["status"], "needs-root", p["name"])
                self.assertEqual(p["run_as"], "not-run", p["name"])
            else:
                self.assertIn(p["status"], ("ok", "failed", "needs-root"), p["name"])
                self.assertEqual(p["run_as"], "user", p["name"])
                if p["root_mode"] == "none":
                    self.assertNotEqual(p["status"], "needs-root", p["name"])

    def test_root_only_tools_not_executed_as_nonroot(self):
        for tool in ("dmidecode", "smartctl", "ipmitool", "mdadm", "pvs", "vgs", "lvs", "qm", "sudo",
                     "curl", "wget", "fio", "modprobe", "sysctl", "setpci"):
            self.assertEqual(self.stub.calls(tool), [], tool)
        self.assertEqual([c[1:] for c in self.stub.calls("nvcc")], [["--version"]])
        self.assertEqual(self.probe("memory.dmidecode_16_17")["status"], "needs-root")
        self.assertEqual(self.probe("board.dmidecode_0_1_2_3")["status"], "needs-root")

    def test_read_only_invocations(self):
        nv = self.stub.calls("nvidia-smi")
        self.assertGreater(len(nv), 10)
        for c in nv:
            argv = " ".join(c[1:])
            self.assertTrue(any(re.fullmatch(a, argv) for a in NVSMI_ALLOWED), argv)
        for c in self.stub.calls("nvme"):
            self.assertIn(c[1], NVME_ALLOWED, c)
        for c in self.stub.calls("cpupower"):
            self.assertEqual(c[1:], ["frequency-info"])
        for c in self.stub.calls("lspci"):
            self.assertTrue(all(a in ("-tv", "-nn", "-vvv", "-d", "10de:", "-Dn", "-s") or re.match(r"^[0-9a-f:.]+$", a) for a in c[1:]), c)

    def test_failing_and_hanging_probes_do_not_stop_the_run(self):
        cp = self.probe("cpu.cpupower_frequency_info")
        self.assertEqual(cp["status"], "failed")
        self.assertIn("timed out", cp["note"])
        nm = self.probe("cpu.numactl_hardware")
        self.assertEqual(nm["status"], "failed")
        self.assertEqual(nm["exit_code"], 3)
        self.assertEqual(self.manifest["probes"][-1]["name"], "network.internet_check")

    def test_net_check_off_by_default(self):
        p = self.probe("network.internet_check")
        self.assertEqual(p["status"], "skipped")
        self.assertIn("--max-time 5", p["command"])
        self.assertIn("-I", p["command"].split())

    def test_tarball_and_draft(self):
        tgz = pathlib.Path(str(self.out) + ".tar.gz")
        self.assertTrue(tgz.is_file())
        with tarfile.open(tgz) as t:
            names = t.getnames()
        base = self.out.name
        self.assertIn(base + "/manifest.json", names)
        self.assertIn(base + "/HARDWARE_FACTS.draft.md", names)
        self.assertIn(base + "/kit/collect.sh", names)
        self.assertTrue(any(n.startswith(base + "/probes/") for n in names))
        self.assertFalse(any("/.tmp" in n for n in names))
        self.assertFalse((self.out / ".tmp").exists())

    def test_summary_of_nonroot_run_does_not_invent(self):
        run, F = summarize.summarize(str(self.out))
        r = F.find("populated DIMMs (SMBIOS type 17)")
        self.assertEqual(r["label"], "UNKNOWN")
        self.assertIn("needs-root", r["source"])
        self.assertEqual(F.find("memory channels populated")["label"], "UNKNOWN")


class CollectArgs(unittest.TestCase):
    def test_unknown_argument_and_nonempty_out(self):
        r = subprocess.run(["bash", str(DIAG / "collect.sh"), "--bogus"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "keep.txt").write_text("x")
            r = subprocess.run(["bash", str(DIAG / "collect.sh"), "--out", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(os.listdir(d), ["keep.txt"])


@unittest.skipUnless(hasattr(os, "geteuid") and os.geteuid() == 0, "needs a root sandbox")
class CollectAsRoot(unittest.TestCase):
    def test_root_probes_run_and_parse(self):
        stub = StubEnv()
        try:
            out = stub.dir / "run-root"
            r = run_collect(stub, out)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = json.loads((out / "manifest.json").read_text())
            self.assertTrue(m["is_root"])
            dm = stub.calls("dmidecode")
            self.assertEqual(sorted(" ".join(c[1:]) for c in dm), ["-t 0,1,2,3", "-t 16,17", "-t 38"])
            self.assertEqual(stub.calls("sudo"), [])
            run, F = summarize.summarize(str(out))
            self.assertEqual(F.find("populated DIMMs (SMBIOS type 17)")["label"], "CONFIRMED")
            self.assertTrue(F.find("populated DIMMs (SMBIOS type 17)")["value"].startswith("2 of 2"))
            ch = F.find("memory channels populated")
            self.assertEqual((ch["label"], ch["value"]), ("PROVISIONAL", "2"))
        finally:
            stub.cleanup()


class CollectSudo(unittest.TestCase):
    def test_sudo_is_non_interactive_only(self):
        stub = StubEnv()
        try:
            out = stub.dir / "run-sudo"
            r = run_collect(stub, out, "--sudo", DIAG_ASSUME_NONROOT="1")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = json.loads((out / "manifest.json").read_text())
            self.assertEqual(m["sudo"], "ok")
            calls = stub.calls("sudo")
            self.assertGreater(len(calls), 3)
            for c in calls:
                self.assertEqual(c[1], "-n", c)
            dm = [p for p in m["probes"] if p["name"] == "memory.dmidecode_16_17"][0]
            self.assertEqual(dm["run_as"], "sudo")
        finally:
            stub.cleanup()


class StaticSafety(unittest.TestCase):
    def files(self):
        fs = sorted(p for p in DIAG.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        self.assertGreaterEqual(len(fs), 10)
        return fs

    def test_no_forbidden_commands(self):
        for f in self.files():
            text = f.read_text(errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                for pat, what in FORBIDDEN:
                    self.assertIsNone(re.search(pat, line), "%s:%d %s: %s" % (f.relative_to(ROOT), i, what, line.strip()))

    def test_fio_against_dev_is_readonly(self):
        for f in self.files():
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if re.search(r"\bfio\b", line) and re.search(r"/dev/(?!null\b)", line) and not line.lstrip().startswith("#"):
                    self.assertIn("--readonly", line, "%s:%d" % (f.name, i))
        text = (DIAG / "bench_nvme.sh").read_text()
        common = text[text.index("fio_common() {"):text.index("\n}\n", text.index("fio_common() {"))]
        self.assertIn("--readonly", common)
        self.assertEqual(len(re.findall(r"run_logged [^\n]*\bfio\b", text)), 1, "only the prep job may call fio outside run_fio")

    def test_shell_options(self):
        for f in DIAG.glob("*.sh"):
            text = f.read_text()
            if f.name != "bench_common.sh":
                self.assertIn("\nset -u\n", text, f.name)
            self.assertNotRegex(text, r"\nset -[a-z]*e", f.name)
        c = (DIAG / "collect.sh").read_text()
        self.assertIn("runner=(sudo -n env", c)
        self.assertNotRegex(c, r"\bsudo\s+-(?!n\b)")

    @unittest.skipUnless(which_shellcheck(), "shellcheck not available (env SHELLCHECK or PATH)")
    def test_shellcheck(self):
        files = sorted(str(p) for p in DIAG.glob("*.sh"))
        r = subprocess.run([which_shellcheck(), "-x", *files], capture_output=True, text=True, cwd=str(DIAG))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class BenchRefusal(unittest.TestCase):
    def test_refuse_without_run_and_plan_only(self):
        with tempfile.TemporaryDirectory() as d:
            stub = StubEnv()
            try:
                cases = [["bench_mem.sh", "--out", d], ["bench_gpu.sh", "--out", d, "--vram-check"],
                         ["bench_nvme.sh", "--out", d, "--dir", d, "--size-gib", "1"],
                         ["bench_all.sh", "--out", d, "--nvme-dir", d, "--size-gib", "1"]]
                for c in cases:
                    r = subprocess.run(["bash", str(DIAG / c[0]), *c[1:]], capture_output=True, text=True, env=stub.env())
                    self.assertEqual(r.returncode, 2, c[0] + r.stdout + r.stderr)
                    self.assertIn("Refusing to run without --run", r.stdout, c[0])
                    self.assertIn("plan", r.stdout, c[0])
                    r = subprocess.run(["bash", str(DIAG / c[0]), *c[1:], "--plan"], capture_output=True, text=True, env=stub.env())
                    self.assertEqual(r.returncode, 0, c[0] + r.stdout + r.stderr)
                self.assertEqual(stub.calls(), [], "a plan must not execute fio/nvcc/nvidia-smi")
                self.assertEqual(os.listdir(d), [], "a plan must not write anything")
            finally:
                stub.cleanup()


class BenchNvme(unittest.TestCase):
    def setUp(self):
        self.stub = StubEnv(["fio", "nvme"])
        self.data = self.stub.dir / "data"
        self.data.mkdir()
        self.out = self.stub.dir / "out"

    def tearDown(self):
        self.stub.cleanup()

    def run_nvme(self, *args):
        return subprocess.run(["bash", str(DIAG / "bench_nvme.sh"), "--out", str(self.out), *args],
                              capture_output=True, text=True, env=self.stub.env(), timeout=300)

    def test_file_mode_prep_writes_only_test_file_reads_are_readonly(self):
        r = self.run_nvme("--run", "--dir", str(self.data), "--size-gib", "1", "--runtime", "1", "--ramp", "0",
                          "--jobs", "rr4k_qd1,sr1m_qd32")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        calls = self.stub.calls("fio")
        prep = [c for c in calls if "--name=prep" in c]
        reads = [c for c in calls if "--name=prep" not in c]
        self.assertEqual(len(prep), 1)
        self.assertEqual(len(reads), 2)
        self.assertNotIn("--readonly", prep[0])
        fn = [a for a in prep[0] if a.startswith("--filename=")][0][len("--filename="):]
        self.assertTrue(fn.startswith(str(self.data) + "/diag-fio-"), fn)
        self.assertTrue(fn.endswith("/testfile.bin"))
        for c in reads:
            self.assertIn("--readonly", c)
            self.assertIn("--direct=1", c)
            self.assertIn("--output-format=json", c)
            self.assertIn("--filename=" + fn, c)
        self.assertEqual(os.listdir(self.data), [], "test file and its directory must be deleted")
        res = list(self.out.glob("bench/nvme/*"))
        self.assertEqual(len(res), 1)
        self.assertTrue((res[0] / "solo-d0-rr4k_qd1.json").is_file())
        self.assertTrue((res[0] / "temps.tsv").is_file())
        self.assertIn("244", (res[0] / "summary.txt").read_text())
        sec = summarize.parse_probe_text((res[0] / "solo-d0-rr4k_qd1.txt").read_text())
        self.assertIn("--readonly", sec["command"])
        self.assertEqual(sec["result"]["exit_code"], "0")

    def test_multi_dir_runs_solo_and_concurrent(self):
        d2 = self.stub.dir / "data2"
        d2.mkdir()
        r = self.run_nvme("--run", "--all", "%s,%s" % (self.data, d2), "--size-gib", "1", "--runtime", "1",
                          "--ramp", "0", "--jobs", "rr4k_qd32_j4")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        reads = [c for c in self.stub.calls("fio") if "--name=prep" not in c]
        self.assertEqual(len(reads), 3)  # d0 alone, d1 alone, both together
        conc = [c for c in reads if sum(a.startswith("--filename=") for a in c) == 2]
        self.assertEqual(len(conc), 1)
        self.assertIn("--numjobs=4", conc[0])
        self.assertEqual(conc[0].count("--new_group"), 2)
        self.assertEqual(os.listdir(self.data), [])
        self.assertEqual(os.listdir(d2), [])

    def test_refusals(self):
        r = self.run_nvme("--run", "--dir", str(self.data), "--size-gib", "100000000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("need >=", r.stderr)
        r = self.run_nvme("--run", "--dir", str(self.data))
        self.assertEqual(r.returncode, 2)
        r = self.run_nvme("--run", "--raw-readonly", "/etc/hostname")
        self.assertEqual(r.returncode, 2)
        r = self.run_nvme("--run", "--raw-readonly", "/dev/null")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not a block device", r.stderr)
        r = self.run_nvme("--run", "--dir", "/dev/shm", "--size-gib", "1")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.stub.calls("fio"), [])
        self.assertEqual(os.listdir(self.data), [])

    def test_raw_mode_is_readonly(self):
        dev = None
        for b in sorted(os.listdir("/sys/block")) if os.path.isdir("/sys/block") else []:
            p = "/dev/" + b
            if os.path.exists(p) and pathlib.Path(p).is_block_device() and not b.startswith(("loop", "zram", "ram")):
                dev = p
                break
        if dev is None:
            self.skipTest("no block device visible")
        r = self.run_nvme("--run", "--raw-readonly", dev, "--runtime", "1", "--ramp", "0", "--jobs", "rr4k_qd1,rr16m_qd4")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        calls = self.stub.calls("fio")
        self.assertEqual(len(calls), 2)
        for c in calls:
            self.assertIn("--readonly", c)
            self.assertIn("--filename=" + dev, c)
            self.assertFalse(any(a.startswith("--size=") or a == "--name=prep" or a.startswith("--rw=write") for a in c), c)


@unittest.skipUnless(gcc_openmp(), "gcc with OpenMP not available")
class Membw(unittest.TestCase):
    def test_compile_and_run_tiny(self):
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "membw")
            r = subprocess.run(["gcc", "-O2", "-fopenmp", "-Wall", "-Wextra", "-Werror", str(DIAG / "membw.c"), "-o", exe, "-lm"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run([exe, "-s", "8", "-n", "3", "-t", "2"], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rows = [l.split(",") for l in r.stdout.splitlines() if l.startswith("CSV,")]
            self.assertEqual([x[1] for x in rows], ["copy", "scale", "add", "triad", "read"])
            for x in rows:
                self.assertEqual(x[2], "2")
                self.assertEqual(x[3], str(8 << 20))
                self.assertGreater(float(x[5]), 0.0)
                self.assertGreaterEqual(float(x[5]), float(x[6]))  # best >= median
            self.assertIn("validation: ok", r.stdout)
            self.assertEqual(subprocess.run([exe, "-n", "1"], capture_output=True).returncode, 2)

    def test_bench_mem_tiny(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["bash", str(DIAG / "bench_mem.sh"), "--run", "--out", d, "--size-mib", "8",
                                "--ntimes", "3", "--no-numa"], capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            csvs = list(pathlib.Path(d).glob("bench/mem/*/membw.csv"))
            self.assertEqual(len(csvs), 1)
            rows = csvs[0].read_text().splitlines()
            variants = {l.split(",")[0] for l in rows[1:]}
            self.assertEqual(variants, {"t1", "cores_spread", "cores_close", "threads_spread"})
            self.assertEqual(len(rows) - 1, 4 * 5)
            log = csvs[0].parent / "run-t1.txt"
            sec = summarize.parse_probe_text(log.read_text())
            self.assertIn("OMP_NUM_THREADS=1", sec["command"])
            self.assertEqual(sec["result"]["exit_code"], "0")


@unittest.skipUnless(which_nvcc(), "nvcc not available (set NVCC=/path/to/nvcc)")
class CudaCompile(unittest.TestCase):
    def test_compile_sm89(self):
        nvcc = which_nvcc()
        with tempfile.TemporaryDirectory() as d:
            for src in ("gpu_p2p_matrix.cu", "gpu_vram_check.cu"):
                exe = os.path.join(d, src[:-3])
                r = subprocess.run([nvcc, "-O2", "-arch=sm_89", "-Xcompiler", "-Wall", "-o", exe, str(DIAG / src)],
                                   capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, src + r.stdout + r.stderr)
                self.assertTrue(os.path.isfile(exe))


class Summarize(unittest.TestCase):
    def facts(self, name):
        with tempfile.TemporaryDirectory() as d:
            md, js = os.path.join(d, "f.md"), os.path.join(d, "f.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summarize.main([str(FIX / name), "--output", md, "--json", js]), 0)
            with open(js) as f:
                data = json.load(f)
            with open(md) as f:
                text = f.read()
        return data, {r["fact"]: r for r in data["facts"]}, text

    def test_fixtures_are_marked_synthetic(self):
        self.assertIn("SYNTHETIC TEST FIXTURE", (FIX / "README.md").read_text())
        for name in ("synthetic-full", "synthetic-nonroot"):
            m = json.loads((FIX / name / "manifest.json").read_text())
            self.assertIs(m["synthetic"], True)
            data, _, text = self.facts(name)
            self.assertTrue(data["synthetic"])
            self.assertIn("SYNTHETIC TEST FIXTURE - NOT EVIDENCE", text)

    def test_full_fixture_values(self):
        data, f, _ = self.facts("synthetic-full")
        exp = {
            "lscpu Model name": "SYNTHETIC EPYC 48-Core Processor",
            "flag avx2": "yes",
            "AVX-512 (any avx512* flag)": "no",
            "NUMA nodes (numactl -H)": "1",
            "DIMM slots (SMBIOS type 16)": "8 (Number Of Devices)",
            "populated DIMMs (SMBIOS type 17)": "4 of 8 devices, total 128 GiB",
            "configured memory speed": "2933 MT/s",
            "board Product Name": "SYNTHETIC-BOARD",
            "BIOS Version": "SYNTH-0.1",
            "NVIDIA driver version": "999.99.99",
            "kernel module flavour": "open",
            "GPU count (nvidia-smi)": "4",
            "GPU1 PCIe width current/max": "x8 / x8",
            "GPU3 PCIe gen current/max": "1 / 3",
            "GPU2 memory.total": "16376 MiB",
            "BAR1 total 00000000:21:00.0": "256 MiB",
            "GPU-GPU topology (topo -m)": "0-1:SYS 0-2:SYS 0-3:SYS 1-2:SYS 1-3:SYS 2-3:PXB",
            "nvcc (CUDA toolkit)": "99.9 V99.9.1",
            "memlock limit (ulimit -l, KiB, for the collecting user/session)": "8192",
            "NVIDIA Xid lines": "1",
            "nvme1 temperature (C, hwmon at collection)": "72.850",
            "IOMMU groups": "42",
        }
        for k, v in exp.items():
            self.assertIn(k, f)
            self.assertEqual((f[k]["value"], f[k]["label"]), (v, "CONFIRMED"), k)
        self.assertIn("2>3:NS", f["P2P read matrix (nvidia-smi topo -p2p r)"]["value"])
        ch = f["memory channels populated"]
        self.assertEqual((ch["value"], ch["label"]), ("4", "PROVISIONAL"))
        self.assertIn("P0:A P0:C P0:E P0:G", ch["source"])
        self.assertIn("mismatches=3", f["VRAM check GPU2"]["value"])
        self.assertIn("16276 IOPS", f["fio solo-d0-rr4k_qd1 d0"]["value"])

    def test_full_fixture_red_flags(self):
        data, _, _ = self.facts("synthetic-full")
        flags = " || ".join(x["flag"] for x in data["red_flags"])
        for needle in ("only 4 memory channels", "fewer than 8", "below DDR4-3200", "GPU1 PCIe max width is x8",
                       "GPU3 PCIe max generation is Gen3", "GPU2 reports 16376 MiB", "BAR1 is 256 MiB", "2>3:NS",
                       "aer_cor=12", "x2 of x4", "8.0 GT/s PCIe (990 Pro", "ACS request redirect", "nvme1 is at 72.850 C",
                       "NVIDIA Xid lines", "fallen off", "limited by", "memlock limit is 8192", "FAN1",
                       "cudaDeviceCanAccessPeer = 0", "0.590 of the sum", "VRAM check GPU2: 3 mismatches",
                       "nvme0 reached 74.0 C", "GPU1 reached only Gen4 x8"):
            self.assertIn(needle, flags)
        self.assertNotIn("virtual machine", flags)  # systemd-detect-virt says none
        self.assertNotIn("AVX-512", flags)

    def test_confirmed_sources_exist(self):
        data, _, _ = self.facts("synthetic-full")
        for r in data["facts"]:
            if r["label"] == "CONFIRMED":
                self.assertTrue((FIX / "synthetic-full" / r["source"]).is_file(), r)
        for x in data["red_flags"]:
            self.assertTrue((FIX / "synthetic-full" / x["source"]).is_file(), x)

    def test_nonroot_fixture_unknowns(self):
        data, f, _ = self.facts("synthetic-nonroot")
        for k in ("populated DIMMs (SMBIOS type 17)", "memory channels populated", "configured memory speed",
                  "DIMM slots (SMBIOS type 16)"):
            self.assertEqual(f[k]["label"], "UNKNOWN", k)
            self.assertIn("needs-root", f[k]["source"], k)
        for k in ("GPU count (nvidia-smi)", "NVIDIA driver version", "NVMe controllers (sysfs)", "nvcc (CUDA toolkit)",
                  "NUMA nodes (numactl -H)", "dmesg error lines"):
            self.assertEqual(f[k]["label"], "UNKNOWN", k)
        self.assertEqual((f["board Product Name"]["value"], f["board Product Name"]["source"]),
                         ("SYNTHETIC-BOARD", "probes/011_board.dmi_sysfs.txt"))
        self.assertEqual(f["memlock limit (ulimit -l, KiB, for the collecting user/session)"]["value"], "unlimited")
        self.assertEqual([x["flag"].split(" (")[0] for x in data["red_flags"]],
                         ["collected inside a virtual machine/container"])
        self.assertFalse(any(r["label"] != "UNKNOWN" and r["fact"].startswith(("GPU0", "DIMM", "BAR1", "populated"))
                             for r in data["facts"]))

    def test_mutation_probe_lost_becomes_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            dst = pathlib.Path(d) / "fx"
            shutil.copytree(FIX / "synthetic-full", dst)
            m = json.loads((dst / "manifest.json").read_text())
            for p in m["probes"]:
                if p["name"] in ("memory.dmidecode_16_17", "gpu.query_csv"):
                    p["status"] = "needs-root" if p["name"].startswith("memory") else "failed"
            (dst / "manifest.json").write_text(json.dumps(m))
            run, F = summarize.summarize(str(dst))
            self.assertEqual(F.find("populated DIMMs (SMBIOS type 17)")["label"], "UNKNOWN")
            self.assertEqual(F.find("GPU count (nvidia-smi)")["label"], "UNKNOWN")
            self.assertIsNone(F.find("GPU0 name"))

    def test_parse_probe_text_keeps_marker_like_stdout(self):
        t = ("===== PROBE =====\nname: x\n===== COMMAND =====\necho\n===== STDOUT =====\na\n===== STDERR =====\nb\n"
             "===== STDERR =====\nreal err\n===== RESULT =====\nexit_code: 0\nstatus: ok\n")
        s = summarize.parse_probe_text(t)
        self.assertEqual(s["stdout"], "a\n===== STDERR =====\nb")
        self.assertEqual(s["stderr"], "real err")
        self.assertEqual(s["result"]["status"], "ok")

    def test_infer_channels_locator_styles(self):
        f = summarize.infer_channels
        self.assertEqual(f([{"Locator": "DIMM 0", "Bank Locator": "P0 CHANNEL A"},
                            {"Locator": "DIMM 1", "Bank Locator": "P0 CHANNEL A"},
                            {"Locator": "DIMM 0", "Bank Locator": "P0 CHANNEL B"}])[0], 2)
        self.assertEqual(f([{"Locator": "DIMMA1", "Bank Locator": "P0_Node0_Channel0_Dimm0"},
                            {"Locator": "DIMMB1", "Bank Locator": "P0_Node0_Channel1_Dimm0"}])[0], 2)
        self.assertEqual(f([{"Locator": "DIMM_P0_A0", "Bank Locator": "BANK 0"},
                            {"Locator": "DIMM_P0_B0", "Bank Locator": "BANK 1"},
                            {"Locator": "DIMM_P0_C0", "Bank Locator": "BANK 2"}])[0], 3)
        self.assertEqual(f([{"Locator": "CPU1_DIMM_A1", "Bank Locator": "Not Specified"}])[0], 1)
        n, why = f([{"Locator": "DIMM 0", "Bank Locator": "Not Specified"}])
        self.assertIsNone(n)
        self.assertIn("not recognised", why)

    def test_fio_summary_cli(self):
        with tempfile.TemporaryDirectory() as d:
            csvp = os.path.join(d, "s.csv")
            js = FIX / "synthetic-full" / "bench" / "nvme" / "20260101T000000Z" / "solo-d0-rr4k_qd1.json"
            r = subprocess.run([sys.executable, str(DIAG / "summarize.py"), "--fio-summary", "--csv", csvp, str(js)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("16276", r.stdout)
            self.assertIn("58.0", r.stdout)
            with open(csvp) as f:
                self.assertIn("p99.9_us", f.readline())


if __name__ == "__main__":
    unittest.main()
