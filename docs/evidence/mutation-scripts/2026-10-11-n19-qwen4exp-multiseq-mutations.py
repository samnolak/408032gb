#!/usr/bin/env python3
"""Mutation check of the N19 several-sequence oracle and tests (gap 10 of docs/reassessment/CORRECTNESS.md).

Oracle mutants (O*): single-point edits of ref/qwen4exp.py's several-sequence section (the state of the wrong sequence,
a seq_cp that does not share the recurrent state, off-by-one rollbacks, snapshot planes, the refusal rules, the
extra-range copy, the cell / stream model, the k-pool sharing and dedupe, the split) and of the test's plan
expectations; dump mutants (D*): a llama.cpp regression simulated on a copy of the artifacts (probability mass on another
sequence's cell, the state of another sequence read, a rollback > n_rs_seq accepted, a stale snapshot "fixed").
Every mutant must make tests.test_qwen4exp_multiseq_parity and/or tests.test_qwen4exp_multiseq_ref fail. The repository
is never edited: ref/, tests/ and tools/tinygen/ are copied into <scratch>/<mutant> and mutated there; the artifacts
are copied for the dump mutants only.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n19-qwen4exp-multiseq-mutations.py <repo root> <artifacts>
       <llama.cpp> <scratch>
(<artifacts>: QWEN4EXP_MULTISEQ_WORK of tools/tinygen/run_parity_qwen4exp_multiseq.sh, holding n19/; <llama.cpp>: the
checkout the dumps came from, for its gguf-py and commit). Results: docs/evidence/2026-10-11-n19-qwen4exp-multiseq.md.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

wt, art, llama, scratch = sys.argv[1:5]
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
R = "ref/qwen4exp.py"
P = "tools/tinygen/qwen4exp_multiseq_plans.py"
MUTANTS = [
    ("O1 every sequence of a ubatch reads the first sequence's recurrent row", R,
     "                    st_, pre = rr[k]", "                    st_, pre = rr[0]"),
    ("O2 seq_cp does not share the recurrent state (the copy starts from zeros)", R,
     "        if self.tail[src] >= 0:\n            self.seqs[self.tail[src]].add(dst)",
     "        if False:\n            self.seqs[self.tail[src]].add(dst)"),
    ("O3 rollback reads the snapshot one token too recent", R,
     "                    self.rs_idx[seq] = rollback\n", "                    self.rs_idx[seq] = rollback - 1\n"),
    ("O4 rollback keeps one position too many in the KV history", R,
     "        keep = max(p0, 0)", "        keep = max(p0 + 1, 0)"),
    ("O5 every delta-rule snapshot plane written (as if a short ubatch kept them all)", R,
     "                    for j in range(min(nst, n_planes)):", "                    for j in range(n_planes):"),
    ("O6 conv / PLE snapshot planes all hold the newest history", R,
     "    return [(full[max(0, n - j):max(0, n - j) + kc], max(0, n - j)) for j in range(n_planes)]",
     "    return [(full[n:n + kc], n) for j in range(n_planes)]"),
    ("O7 s_copy ignores a pending rollback (plane 0 always)", R,
     "                plane = self.rs_idx[seq]\n", "                plane = 0\n"),
    ("O8 a pending rollback is not single-use", R,
     "if self.rs_idx[seq] == 0 and 1 <= rollback <= self.n_rs_seq:", "if 1 <= rollback <= self.n_rs_seq:"),
    ("O9 no refusal of a rollback of a shared recurrent cell (R1 rule dropped)", R,
     "                if self.refuse_shared and len(self.seqs[t]) > 1:", "                if False:"),
    ("O10 the extra recurrent rows are not copied (a pending rollback lost)", R,
     "                    planes[0][head + i] = reads[key][i]", "                    pass"),
    ("O11 one KV stream for all sequences even without kv_unified", R,
     "        return 0 if self.n_stream == 1 else seq", "        return 0"),
    ("O12 ec7630a: shared cells never detected (cache_safe always true)", R,
     'if self.pin == "ec" and unified and not sq["shared"]:', 'if False:'),
    ("O13 79e2e74: a pool marked once per pool, not once per rep cell", R,
     "                if rep in reps:\n                    return", "                if False:\n                    return"),
    ("O14 ec7630a: dropping a sharing sequence does not stale every sequence", R,
     "        if shared:                                                        # HI:196-200 (absent @79e2e74)",
     "        if False:"),
    ("O15 split: the sets of a ubatch need not be consecutive with one stream per sequence", R,
     "                add = add and (not cur or s == last + 1)", "                add = add"),
    ("O16 split: the last n_rs_seq + 1 tokens of a sequence may straddle ubatches", R,
     "        if n_keep_tail > 0:                                           # :637-701",
     "        if False:                                                     # :637-701"),
    ("O17 ec7630a padding cell: always the first token's cell", R,
     "            if new and pools:", "            if False:"),
    ("O18 seq_cp: the destination's pools are not marked stale", R,
     "        self.rc.seq_cp(src, dst)\n        self.stale[dst] = 0", "        self.rc.seq_cp(src, dst)"),
    ("T1 plans: the rollback of a shared cell expected to be accepted at both pins", P,
     '("r", 0, 2, True, False),                         # one recurrent cell for both: ec accepts, R1 refuses',
     '("r", 0, 2, True, True),'),
    ("T2 plans: the stale rollback expected not to diverge", P,
     'diverge={"ec": {0: 4, 1: 10}, "r1": {0: 4, 1: 10}}', 'diverge={"ec": {1: 10}, "r1": {1: 10}}'),
]


def load(path):
    j = json.loads(open(path + ".json").read())
    return j, np.fromfile(path + ".bin", dtype=np.float32 if j["type"] == "f32" else np.int32)


def save(path, a):
    a.tofile(path + ".bin")


def d1_leak(root):
    """kq_soft_max of layer 7, multi_uni graph 3 (3 rows, one per sequence): row 0 gets 0.01 on a cell of sequence 1."""
    d = os.path.join(root, "multi_uni", "g003", "kq_soft_max-7__0")
    j, a = load(d)
    a = a.reshape(j["ne"][::-1])
    man = json.load(open(os.path.join(root, "multi_uni", "manifest.json")))
    assert man["calls"][1]["graphs"] == [3, 4]
    # cells 12..23 hold sequence 1 after the first call (ubatch 0 = 12 + 12 + 12 rows in cells 0..35)
    a[0, 0, 0, 12] = 0.01
    save(d, a)


def d2_wrong_state(root):
    """state_predelta of layer 0, multi_uni graph 5: the rows of sequences 0 and 1 swapped."""
    d = os.path.join(root, "multi_uni", "g005", "state_predelta-0__0")
    j, a = load(d)
    a = a.reshape(j["ne"][::-1]).copy()
    a[[0, 1]] = a[[1, 0]]
    save(d, a)


def d3_accept(root):
    """rb_uni: llama.cpp reports the rollback by 4 > n_rs_seq as accepted."""
    m = os.path.join(root, "rb_uni", "manifest.json")
    man = json.load(open(m))
    i = [k for k, c in enumerate(man["calls"]) if c["kind"] == "seq_rm" and c["n"] == 4][0]
    man["calls"][i]["ok"] = True
    json.dump(man, open(m, "w"))


def d4_fixed_snapshot(root):
    """stale: the state read after the rollback across calls replaced by the kept prefix's (as if llama.cpp kept the
    right snapshot): the oracle (B) predicts the stale one, the test expects divergence."""
    man = json.load(open(os.path.join(root, "stale", "manifest.json")))
    g = man["calls"][4]["graphs"][0]
    for name in os.listdir(os.path.join(root, "stale", f"g{g:03d}")):
        if name.startswith("state_predelta-") and name.endswith(".json"):
            p = os.path.join(root, "stale", f"g{g:03d}", name[:-5])
            j, a = load(p)
            save(p, (a * 0.5).astype(a.dtype))


DUMP_MUTANTS = [
    ("D1 llama.cpp puts attention probability on another sequence's cell", d1_leak),
    ("D2 llama.cpp reads another sequence's delta-rule state", d2_wrong_state),
    ("D3 llama.cpp accepts a rollback longer than n_rs_seq", d3_accept),
    ("D4 llama.cpp reads another state than the stale snapshot after a rollback across calls", d4_fixed_snapshot),
]


def run_tests(tree, dump):
    env = dict(os.environ, QWEN4EXP_MULTISEQ_DUMP=dump, QWEN4EXP_TINY_GGUF=os.path.join(art, "n19", "qwen4exp.gguf"),
               LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"), PARITY_LLAMA_COMMIT=commit)
    out = subprocess.run([sys.executable, "-m", "unittest", "tests.test_qwen4exp_multiseq_parity",
                          "tests.test_qwen4exp_multiseq_ref"], cwd=tree, env=env, capture_output=True, text=True)
    text = out.stdout + out.stderr
    ran = re.search(r"Ran \d+ tests? in [0-9.]+s", text)
    status = re.search(r"^(OK.*|FAILED.*)$", text, re.M)
    failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \((?:[\w.]+\.)?(\w+)\.\w+\)", text, re.M)))
    return out.returncode, (ran[0] if ran else "?") + " | " + (status[0] if status else "?"), failing


def tree_copy(dst):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for d in ("ref", "tests", "tools/tinygen"):
        shutil.copytree(os.path.join(wt, d), os.path.join(dst, d), ignore=shutil.ignore_patterns("__pycache__"))


os.makedirs(scratch, exist_ok=True)
base = os.path.join(scratch, "baseline")
tree_copy(base)
rc, summary, _ = run_tests(base, os.path.join(art, "n19"))
print(f"llama.cpp {commit[:7]}; baseline (no mutation): rc={rc} {summary}")
caught = 0
for name, path, old, new in MUTANTS:
    tree = os.path.join(scratch, "m")
    tree_copy(tree)
    f = os.path.join(tree, path)
    s = open(f).read()
    assert s.count(old) == 1, (name, s.count(old))
    open(f, "w").write(s.replace(old, new))
    rc, summary, failing = run_tests(tree, os.path.join(art, "n19"))
    caught += rc != 0
    print(f"{'caught' if rc else 'SURVIVED'}: {name}\n    rc={rc} {summary}\n    failing: "
          f"{', '.join(f'{c}.{t}' for t, c in failing) or 'none'}")
for name, fn in DUMP_MUTANTS:
    dump = os.path.join(scratch, "dumps")
    shutil.rmtree(dump, ignore_errors=True)
    shutil.copytree(os.path.join(art, "n19"), dump)
    fn(dump)
    rc, summary, failing = run_tests(base, dump)
    caught += rc != 0
    print(f"{'caught' if rc else 'SURVIVED'}: {name}\n    rc={rc} {summary}\n    failing: "
          f"{', '.join(f'{c}.{t}' for t, c in failing) or 'none'}")
    shutil.rmtree(dump)
print(f"caught {caught}/{len(MUTANTS) + len(DUMP_MUTANTS)}")
