"""Mutation check for the offline engine kit: scripts/offline/engines.py and the --offline-kit path of
scripts/engines/build_engine.py, against tests/test_offline_engines.py and tests/test_engines_build.py.
Usage: python3 <this> <repository> <temporary directory>"""
import os, shutil, subprocess, sys, tempfile
ROOT, SP = sys.argv[1], sys.argv[2]
E = "scripts/offline/engines.py"
B = "scripts/engines/build_engine.py"
MUT = [
 ("K1 verify: commit check always ok", E, '        if got != sha:\n            return False, "the bundle gives %s, pinned %s" % (got, sha)',
  '        if False:\n            return False, "the bundle gives %s, pinned %s" % (got, sha)'),
 ("K2 verify: sha256 ignored", E, 'same = os.path.getsize(path) == s["size"] and sha256_file(path) == s["sha256"]', 'same = True'),
 ("K3 verify: pins.json not compared", E, 'line(want == got, "pins %s" % name,', 'line(True, "pins %s" % name,'),
 ("K4 build: kit for other commits accepted", B, '        if got != want:\n            print("offline kit %s was packed for other commits',
  '        if False:\n            print("offline kit %s was packed for other commits'),
 ("K5 build: dependency checkout unchecked", B, '                    for dep, h in dep_heads.items() if h != pin["deps"][dep]["sha"]]',
  '                    for dep, h in dep_heads.items() if False]'),
 ("K6 build: no shallow boundary", B, '                [sys.executable, "-c", SHALLOW, dest, sha],\n', ''),
 ("K7 pack: bundle under a branch ref", B, 'KIT_REF = "refs/pins/%s"', 'KIT_REF = "refs/heads/%s"'),
 ("K8 build: kit ignored (network fetch)", B, '    if kit:\n        return [["git", "init", "-q", dest],', '    if False:\n        return [["git", "init", "-q", dest],'),
]
def tree():
    d = tempfile.mkdtemp(dir=SP)
    for f in subprocess.run(["git", "-C", ROOT, "ls-files", "-co", "--exclude-standard"], capture_output=True,
                            text=True).stdout.split():
        if f.startswith(("scripts/", "tools/engines/", "tests/test_offline_engines.py", "tests/test_engines_build.py")):
            os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, f), os.path.join(d, f))
    return d
def run(d):
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_offline_engines", "tests.test_engines_build"],
                       cwd=d, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]
d = tree(); rc, last = run(d); shutil.rmtree(d)
print("%-42s rc=%d %-8s %s" % ("baseline", rc, "ok" if rc == 0 else "BROKEN", last), flush=True)
caught = 0
for name, f, old, new in MUT:
    d = tree(); p = os.path.join(d, f); s = open(p).read()
    assert s.count(old) == 1, (name, s.count(old))
    open(p, "w").write(s.replace(old, new))
    rc, last = run(d); caught += rc != 0
    print("%-42s rc=%d %-8s %s" % (name, rc, "caught" if rc else "SURVIVED", last), flush=True)
    shutil.rmtree(d)
print("caught %d/%d" % (caught, len(MUT)))
