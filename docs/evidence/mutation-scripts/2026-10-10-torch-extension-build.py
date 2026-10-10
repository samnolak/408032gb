"""Mutation check for the torch-extension path of scripts/engines/build_engine.py."""
import os, shutil, subprocess, sys, tempfile
ROOT, SP = sys.argv[1], sys.argv[2]
B = "scripts/engines/build_engine.py"
MUT = [
 ("T1 torch check step dropped", 'return [("check torch %s with CUDA %s" % (py["torch"], py["cuda"]), [python, "-c", TORCH_CHECK, py["torch"], py["cuda"]]),\n            ("build wheel"',
  'return [("build wheel"'),
 ("T2 arch list fixed to 8.6", '"TORCH_CUDA_ARCH_LIST=%d.%d" % divmod(arch, 10)', '"TORCH_CUDA_ARCH_LIST=8.6"'),
 ("T3 PYTORCH_NVCC not set", '"PYTORCH_NVCC=" + nvcc,', ''),
 ("T4 MAX_JOBS ignored", '"MAX_JOBS=%d" % jobs', '"MAX_JOBS=1"'),
 ("T5 unpack takes any .so", "x.startswith(sys.argv[3] + '.') and x.endswith('.so')", "x.endswith('.so')"),
 ("T6 wheel not hashed", 'f.endswith(".whl") or ', ''),
 ("T7 --extra-cmake accepted", '            raise ValueError("--extra-cmake does not apply to a torch-extension build")', '            pass'),
 ("T8 torch builds take the cmake path", '    if pin.get("kind") == "torch-extension":\n        if extra:', '    if False:\n        if extra:'),
 ("T9 build isolation on", '"--no-deps", "--no-build-isolation",', '"--no-deps",'),
]
def tree():
    d = tempfile.mkdtemp(dir=SP)
    files = subprocess.run(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    for f in files:
        src = os.path.join(ROOT, f)
        if os.path.isfile(src) and (f.startswith(("scripts/", "tools/engines/", "tests/test_engines_build.py"))):
            os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
            shutil.copy2(src, os.path.join(d, f))
    return d
def run(d):
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_engines_build"], cwd=d, capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]
d = tree(); rc, last = run(d); shutil.rmtree(d)
print("%-36s rc=%d %-8s %s" % ("baseline", rc, "ok" if rc == 0 else "BROKEN", last), flush=True)
caught = 0
for name, old, new in MUT:
    d = tree(); p = os.path.join(d, B); s = open(p).read()
    assert s.count(old) == 1, (name, s.count(old))
    open(p, "w").write(s.replace(old, new))
    rc, last = run(d); caught += rc != 0
    print("%-36s rc=%d %-8s %s" % (name, rc, "caught" if rc else "SURVIVED", last), flush=True)
    shutil.rmtree(d)
print("caught %d/%d" % (caught, len(MUT)))
