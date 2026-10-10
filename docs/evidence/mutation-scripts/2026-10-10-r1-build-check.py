"""Mutation check for engine_build_check (scripts/r1/run_r1.py) and files_sha256 (scripts/engines/build_engine.py)."""
import os, shutil, subprocess, sys, tempfile
ROOT, SP = sys.argv[1], sys.argv[2]
R = "scripts/r1/run_r1.py"
B = "scripts/engines/build_engine.py"
MUT = [
 ("B1 engine_start not gated", R, 'DEPENDS = dict({"engine_start": ("engine_build_check",), "health_wait"', 'DEPENDS = dict({"health_wait"'),
 ("B2 --only does not pull the check", R, '        sel.add("engine_build_check")', '        pass'),
 ("B3 sha unchecked", R, '            chk(k, info.get(k) == want, info.get(k), want)', '            chk(k, k == "sha" or info.get(k) == want, info.get(k), want)'),
 ("B4 checked_out unchecked", R, 'chk("checked_out", info.get("checked_out") == pin["sha"]', 'chk("checked_out", True'),
 ("B5 deps unchecked", R, 'chk("deps", got_deps == want_deps', 'chk("deps", True'),
 ("B6 status unchecked", R, 'chk("status", isinstance(st, str) and st.startswith("ok")', 'chk("status", True'),
 ("B7 arch unchecked", R, 'chk("arch", info.get("arch") == e["arch"]', 'chk("arch", True'),
 ("B8 any cubin accepted", R, 'if isinstance(v, list) and sm in v)', 'if isinstance(v, list) and v)'),
 ("B9 file hashes unchecked", R, '                if got != h:\n', '                if False:\n'),
 ("B10 binary location unchecked", R, 'chk("binary", rel in pin["bins"] and isinstance(files, dict) and rel in files', 'chk("binary", True'),
 ("B11 source HEAD unchecked", R, 'chk("source_head", head == pin["sha"]', 'chk("source_head", True'),
 ("B12 commit contradiction accepted", R, '        elif not (re.fullmatch(r"[0-9a-f]{7,40}", e["commit"]) and sha.startswith(e["commit"])):', '        elif False:'),
 ("B13 unknown pin accepted", R, '        if e["pin"] not in pins:', '        if False:'),
 ("B14 build_info without pin accepted", R, '            if e[k] and not e["pin"]:', '            if False:'),
 ("B15 binaries flags unchecked", R, 'and sorted(bins) == sorted(pin["bins"]) and all(bins.values())', ''),
 ("B16 build_engine records no hashes", B, '            files[os.path.relpath(f, build)] = sha256_file(f)', '            pass'),
]
def tree():
    d = tempfile.mkdtemp(dir=SP)
    files = subprocess.run(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    for f in files:
        src = os.path.join(ROOT, f)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
            shutil.copy2(src, os.path.join(d, f))
    return d
def run(d, mods):
    p = subprocess.run([sys.executable, "-m", "unittest"] + mods, cwd=d, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]
MODS = ["tests.test_r1_orchestration", "tests.test_engines_build"]
d = tree(); rc, last = run(d, MODS); shutil.rmtree(d)
print("%-36s rc=%d %-8s %s" % ("baseline", rc, "ok" if rc == 0 else "BROKEN", last), flush=True)
caught = 0
for name, f, old, new in MUT:
    d = tree()
    p = os.path.join(d, f)
    s = open(p).read()
    assert s.count(old) == 1, (name, s.count(old))
    open(p, "w").write(s.replace(old, new))
    rc, last = run(d, MODS)
    caught += rc != 0
    print("%-36s rc=%d %-8s %s" % (name, rc, "caught" if rc else "SURVIVED", last), flush=True)
    shutil.rmtree(d)
print("caught %d/%d" % (caught, len(MUT)))
