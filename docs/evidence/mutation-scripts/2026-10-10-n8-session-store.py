"""Mutation check for tests/test_session_store.py: each mutant edits a copy of tools/kv/session_store.py.

Usage: python3 docs/evidence/mutation-scripts/2026-10-10-n8-session-store.py <repository> <scratch dir>
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT, SP = sys.argv[1], sys.argv[2]
MUT = {
    "S1 no sha256 check": ('elif _sha256(path) != m["sha256"]:', 'elif False:'),
    "S2 no size check": ('elif os.path.getsize(path) != m["size"]:', 'elif False:'),
    "S3 no n_restored check": ('if resp is not None and resp.get("n_restored") != m["n_tokens"]:', 'if False:'),
    "S4 old generation deleted before the manifest commit": (
        '            _fsync_dir(self.dir)\n            self._hook("after_rename")',
        '            _fsync_dir(self.dir)\n            if old and os.path.exists(os.path.join(self.dir, old["file"])):\n'
        '                os.remove(os.path.join(self.dir, old["file"]))\n            self._hook("after_rename")'),
    "S5 recover keeps temporaries": ('if name.startswith("tmp-") and name.endswith(".slot") or', 'if False and'),
    "S6 transient error judges the file": ('except urllib.error.HTTPError as e:', 'except (urllib.error.URLError, OSError) as e:'),
    "S7 identity not checked": ('if m.get("identity") != self.identity:', 'if False:'),
    "S8 eviction may evict a session being saved": ('busy = set(self._saving) | {keep}', 'busy = set()'),
    "S9 slot not erased after a refused restore": (
        '                    _post(url, f"/slots/{slot}?action=erase", {}, self.timeout)   # never leave a half-loaded slot',
        '                    pass'),
    "S10 save size check skipped": ('if size <= 0 or size != resp.get("n_written"):', 'if size <= 0:'),
    "S11 quarantine names not unique": ('f"{time.time_ns()}-{uuid.uuid4().hex[:8]}-{fname}"', 'f"{int(time.time())}-{fname}"'),
}


def run(tree):
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_session_store"], cwd=tree, capture_output=True,
                           text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT after 300 s (a hang counts as caught)"
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]


for name, edit in [("baseline", None)] + list(MUT.items()):
    tree = tempfile.mkdtemp(dir=SP)
    for d in ("tools", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tree, d), ignore=shutil.ignore_patterns("__pycache__"))
    f = os.path.join(tree, "tools", "kv", "session_store.py")
    s = open(f).read()
    if edit:
        old, new = edit
        assert s.count(old) == 1, name
        open(f, "w").write(s.replace(old, new))
    rc, last = run(tree)
    verdict = ("ok" if rc == 0 else "BROKEN") if edit is None else ("caught" if rc != 0 else "SURVIVED")
    print("%-56s rc=%d %-8s %s" % (name, rc, verdict, last), flush=True)
    shutil.rmtree(tree)
