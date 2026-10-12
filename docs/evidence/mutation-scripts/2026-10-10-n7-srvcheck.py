"""Mutation check for tests/test_srvcheck.py: each mutant edits a copy of tools/srvcheck/hybrid_server_checks.py.

Usage: python3 docs/evidence/mutation-scripts/2026-10-10-n7-srvcheck.py <repository> <scratch dir>
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT, SP = sys.argv[1], sys.argv[2]
MUT = {
    "M1 compare ignores tokens": (
        '    out["pass"] = bool(out["same_tokens"] and out["max_abs_dlogprob"] is not None',
        '    out["pass"] = bool(True and out["max_abs_dlogprob"] is not None'),
    "M2 compare ignores the tolerance": (
        'out["max_abs_dlogprob"] <= tol)', 'out["max_abs_dlogprob"] <= 1e9)'),
    "M3 nothing to compare counts as equal": (
        '        out["max_abs_dlogprob"] = None   # nothing to compare is not a pass',
        '        out["max_abs_dlogprob"] = 0.0'),
    "M4 determinism compares a run with itself": (
        '    b = complete(url, prompt, gen)\n    return {"cold_1": a',
        '    b = a\n    return {"cold_1": a'),
    "M5 prefix_extend erases before the warm request": (
        '    nxt = prompt + first["tokens"] + tail          # previous prompt + previous answer + new turn\n',
        '    nxt = prompt + first["tokens"] + tail\n    erase(url)\n'),
    "M6 cache_reuse erases before the warm request": (
        '        warm = complete(srv.url, nxt, gen, n_cache_reuse=8)',
        '        erase(srv.url)\n        warm = complete(srv.url, nxt, gen, n_cache_reuse=8)'),
    "M7 restart run never restores": (
        '        restored = post(s2.url, "/slots/0?action=restore", {"filename": "n7-restart.bin"})',
        '        restored = None'),
    "M8 isolation runs the two requests one after the other": (
        '        for t in ts:\n            t.start()\n        for t in ts:\n            t.join()\n        done.set()',
        '        for t in ts:\n            t.start()\n            t.join()\n        done.set()'),
    "M9 isolation does not require both slots busy at once": (
        ' and out["max_busy_slots"] >= 2 and', ' and'),
    "M15 isolation trusts client-side overlap instead of GET /slots": (
        ' and out["max_busy_slots"] >= 2 and', ' and out["overlap_s"] > 0 and'),
    "M10 MTP pass without any draft": (
        '            and bool(out["on"].get("draft_n"))       # MTP actually drafted: otherwise nothing was tested',
        '            and True'),
    "M11 MTP failure to answer counts as pass": (
        '    else:\n        out["pass"] = False\n    return out', '    else:\n        out["pass"] = True\n    return out'),
    "M12 a crashed server reports a clean exit": (
        '            rc = 0                       # stopped by us, not a crash\n        self.log.close()\n        return rc',
        '            rc = 0                       # stopped by us, not a crash\n        self.log.close()\n        return 0'),
    "M13 verdict drops the restart check": (
        '        "slot_persist_after_restart": res["slot_persist"]["cmp_restart"]["pass"],\n', ''),
    "M14 verdict drops the cache_reuse check": (
        '        "cache_reuse_correct": res["cache_reuse"]["cmp"]["pass"] and res["cache_reuse"]["exit"] == 0,\n', ''),
}


def run(tree):
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_srvcheck"], cwd=tree, capture_output=True,
                       text=True, timeout=1800)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]


for name, edit in [("baseline", None)] + list(MUT.items()):
    tree = tempfile.mkdtemp(dir=SP)
    for d in ("tools", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tree, d), ignore=shutil.ignore_patterns("__pycache__"))
    f = os.path.join(tree, "tools", "srvcheck", "hybrid_server_checks.py")
    s = open(f).read()
    if edit:
        old, new = edit
        assert s.count(old) == 1, name
        open(f, "w").write(s.replace(old, new))
    rc, last = run(tree)
    verdict = ("ok" if rc == 0 else "BROKEN") if edit is None else ("caught" if rc != 0 else "SURVIVED")
    print("%-56s rc=%d %-8s %s" % (name, rc, verdict, last), flush=True)
    shutil.rmtree(tree)
