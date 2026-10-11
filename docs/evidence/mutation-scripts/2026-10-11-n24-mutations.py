#!/usr/bin/env python3
"""Mutation check of the N24 changes: the declared-synthetic rule and the V6 near-tie tolerance of tools/verify/pack.py
(P1-P9), the gap fields of tools/quality/logprob_compare.py (L1-L2), the read-only GGUF host check
tools/hostcheck/glm_gguf_check.py (H1-H5) and tools/quality/swiglu_gate_stats.py (G1-G2). Each mutant must make
tests.test_verify_pack_n24, tests.test_hostcheck_glm_gguf or tests.test_swiglu_clamp_tools fail. The repository is never
edited: the files of `git ls-files -co --exclude-standard` (minus third_party/) are copied into <scratch>/<mutant>.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n24-mutations.py <repo root> <scratch>
Results: docs/evidence/2026-10-11-n24-owner-decisions.md (section 6)."""
import os
import shutil
import subprocess
import sys

wt, scratch = sys.argv[1:3]
PK, LC = "tools/verify/pack.py", "tools/quality/logprob_compare.py"
HC, GS = "tools/hostcheck/glm_gguf_check.py", "tools/quality/swiglu_gate_stats.py"
T_PACK, T_HOST, T_GATE = "tests.test_verify_pack_n24", "tests.test_hostcheck_glm_gguf", "tests.test_swiglu_clamp_tools"
MUTANTS = [   # (label, tests, [(path, old, new), ...])
    ("P1 declaration applied even when a run's model description does not say SYNTHETIC", T_PACK,
     [(PK, "    if unmarked:\n        return {\"declared\": True, \"applied\": False,",
       "    if False:\n        return {\"declared\": True, \"applied\": False,")]),
    ("P2 probe request errors turned into NOT RUN too", T_PACK,
     [(PK, "(r[\"status\"] == FAIL and (r.get(\"details\") or {}).get(\"request_errors\"))", "False")]),
    ("P3 a declared synthetic run may be PASS overall", T_PACK,
     [(PK, "    if decl and overall == PASS:", "    if False:")]),
    ("P4 any criterion may be declared not applicable", T_PACK,
     [(PK, "any(c not in BEHAVIOUR_CRITERIA for c in na)", "False")]),
    ("P5 the not-applicable list is ignored (every criterion converted)", T_PACK,
     [(PK, " or r[\"id\"] not in crit[\"synthetic_not_applicable\"]:", ":")]),
    ("P6 near_tie ignores the tolerance", T_PACK,
     [(PK, "    return g >= 0 and h >= 0 and g + h <= tol", "    return g >= 0 and h >= 0")]),
    ("P7 near_tie accepts negative gaps", T_PACK,
     [(PK, "    return g >= 0 and h >= 0 and g + h <= tol", "    return g + h <= tol")]),
    ("P8 near_tie treats a missing gap as 0", T_PACK,
     [(PK, "    if not (is_num(g) and is_num(h)) or isinstance(g, bool) or isinstance(h, bool):\n        return False",
       "    g, h = g or 0.0, h or 0.0")]),
    ("P9 the threshold judged on strict top-1 (near-tie flips not counted)", T_PACK,
     [(PK, "        if eff < thr:", "        if top1 < thr:")]),
    ("L1 cand_gap with the wrong sign", T_PACK,
     [(LC, "\"cand_gap\": (cand[c1] - cand[r1]) if r1 in cand else None,",
       "\"cand_gap\": (cand[r1] - cand[c1]) if r1 in cand else None,")]),
    ("L2 max_abs_dlogprob over one common token only", T_PACK,
     [(LC, "max(abs(ref[t] - cand[t]) for t in common)", "max(abs(ref[t] - cand[t]) for t in sorted(common)[:1])")]),
    ("H1 header check ignores the header sha256", T_HOST,
     [(HC, "row[\"header_sha256\"] == p[\"header_sha256\"])", "True)")]),
    ("H2 --out allowed inside --dir", T_HOST,
     [(HC, "    if out_r == dir_r or out_r.startswith(dir_r.rstrip(os.sep) + os.sep):", "    if False:")]),
    ("H3 MATCH without comparing the sha256", T_HOST,
     [(HC, "(r[\"size_ok\"] and r[\"sha256\"] == r[\"pin_sha256\"])", "r[\"size_ok\"]")]),
    ("H4 --no-hash reported as a pass", T_HOST,
     [(HC, "sha256 not computed (--no-hash)\", 3", "sha256 not computed (--no-hash)\", 0")]),
    ("H5 architecture-prefixed keys looked up under a fixed 'glm5-next.'", T_HOST,
     [(HC, "    a = (arch + \".\") if isinstance(arch, str) else \"?.\"", "    a = \"glm5-next.\"")]),
    ("G1 output difference computed with the same semantics on both sides", T_GATE,
     [(GS, "d = abs((min(silu(x), limit) - silu(min(x, limit))) * uc)", "d = abs((silu(min(x, limit)) - silu(min(x, limit))) * uc)")]),
    ("G2 gate values above the limit not counted", T_GATE,
     [(GS, "            if x > limit:\n                above += 1", "            if x > limit:\n                above += 0")]),
]


def tree_files():
    out = subprocess.run(["git", "-C", wt, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return [p for p in out if p and not p.startswith("third_party/") and os.path.isfile(os.path.join(wt, p))]


def main():
    files = tree_files()
    caught = 0
    for i, (label, tests, edits) in enumerate(MUTANTS):
        d = os.path.join(scratch, "m%02d" % i)
        shutil.rmtree(d, ignore_errors=True)
        for p in files:
            os.makedirs(os.path.dirname(os.path.join(d, p)), exist_ok=True)
            shutil.copy2(os.path.join(wt, p), os.path.join(d, p))
        for path, old, new in edits:
            fp = os.path.join(d, path)
            with open(fp) as f:
                s = f.read()
            assert s.count(old) == 1, (label, path, s.count(old))
            with open(fp, "w") as f:
                f.write(s.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "unittest", tests], cwd=d, capture_output=True, text=True)
        last = [l for l in r.stderr.splitlines() if l.startswith(("FAILED", "OK"))]
        ok = r.returncode != 0
        caught += ok
        print("%-8s %s  [%s: %s]" % ("CAUGHT" if ok else "SURVIVED", label, tests, last[-1] if last else r.returncode),
              flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print("%d/%d caught" % (caught, len(MUTANTS)))
    return 0 if caught == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
