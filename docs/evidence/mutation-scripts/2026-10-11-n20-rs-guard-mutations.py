#!/usr/bin/env python3
"""Mutation check of the N20 recurrent rollback guard (patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch) and
of its tests.

C* mutants are real llama.cpp builds: the patched copy's src/llama-memory-recurrent.{h,cpp} is reset to the pin plus
the patch, one hunk of the patch is disabled (C0: the whole patch reverted), dump-tensors is rebuilt, the 13 plans (9 N19 + 4 N20)
(tools/tinygen/qwen4exp_multiseq_plans.py and qwen4exp_rs_guard_plans.py) are dumped again exactly as
tools/tinygen/run_parity_qwen4exp_rs_guard.sh does, and the dump tests run with QWEN4EXP_RS_GUARD=1. At the end the
copy is reset to the patch and rebuilt, so the build dir is the patched build again.
P* / T* mutants edit a copy of ref/, tests/ and tools/tinygen/ (the oracle's guard model, the test's guard switch, the
plan expectations) and run against the patched build's dumps (baseline first). Every mutant must make a test fail.
The repository is never edited.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n20-rs-guard-mutations.py <repo root> <llama.cpp checkout at
       the pin> <patched copy> <build dir of the patched copy> <scratch> [C|P|CP]
Results: docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md.
"""
import os
import re
import shutil
import subprocess
import sys

wt, llama, copy, build, scratch = sys.argv[1:6]
which = sys.argv[6] if len(sys.argv) > 6 else "CP"
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
PATCH = os.path.join(wt, "patches", "llama.cpp", commit[:7], "0001-recurrent-rollback-guard.patch")
RC = "src/llama-memory-recurrent.cpp"
C_MUTANTS = [
    ("C0 the patch reverted (llama.cpp at the pin)", None, None),
    ("C1 seq_rm: the plane rule dropped (rollback onto an unwritten plane accepted)",
     "&& rollback >= (llama_pos) cell.rs_n) {", "&& false) {"),
    ("C2 seq_cp: the pending rollback not handed to the copy",
     "            rs_idx[seq_id_dst] = tail_src.tail >= 0 ? rs_idx[seq_id_src] : 0;", "            (void) 0;"),
    ("C3 find_slot: an extra cell keeps its planes after the extra-row copy",
     "                if (cells[i].src0 != i || idx != 0) {", "                if (false) {"),
    ("C4 find_slot: a ubatch cell gets 1 + n_rs_seq planes whatever its tokens",
     "                cells[i].rs_n = std::min(n_seq_tokens, 1 + n_rs_seq);", "                cells[i].rs_n = 1 + n_rs_seq;"),
    ("C5 state_read: a restored cell keeps the planes it had",
     "            cell.rs_n = 1;", "            (void) cell;"),
    ("C6 seq_rm: a rollback of a shared cell accepted (the ec7630a hunk; upstream code at 79e2e74)",
     "                if (cell.seq_id.size() > 1) {\n                    return false;",
     "                if (false) {\n                    return false;"),
    ("C7 s_copy: only the cell's first sequence's pending rollback reset (N20 audit FAIL 1)",
     "            for (const llama_seq_id s : mem->cells[cell_idx].seq_id) {\n                if (s >= 0 && (size_t) s < mem->rs_idx.size()) {\n                    mem->rs_idx[s] = 0;",
     "            for (const llama_seq_id s : {seq}) {\n                if (s >= 0 && (size_t) s < mem->rs_idx.size()) {\n                    mem->rs_idx[s] = 0;"),
]
R = "ref/qwen4exp.py"
T = "tests/test_qwen4exp_multiseq_parity.py"
P19 = "tools/tinygen/qwen4exp_multiseq_plans.py"
P20 = "tools/tinygen/qwen4exp_rs_guard_plans.py"
P_MUTANTS = [
    ("P1 oracle: the plane rule dropped", R,
     "                if self.guard and rollback >= self.rs_n[t]:", "                if False:"),
    ("P2 oracle: seq_cp does not hand the pending rollback over", R,
     "        if self.guard:                                                    # N20 patch: dst inherits src's state",
     "        if False:"),
    ("P3 oracle: an extra cell keeps its planes", R,
     "                if self.src0[i] != i or pend != 0:", "                if False:"),
    ("P4 oracle: a ubatch cell gets 1 + n_rs_seq planes", R,
     "                self.rs_n[i] = min(n_seq_tokens, 1 + self.n_rs_seq)",
     "                self.rs_n[i] = 1 + self.n_rs_seq"),
    ("P6 oracle: s_copy resets only the first sharer's pending rollback", R,
     "                for s in (self.seqs[c] if self.guard else (seq,)):", "                for s in (seq,):"),
    ("P5 tests: QWEN4EXP_RS_GUARD ignored (the pin's own behaviour expected)", T,
     'GUARD = os.environ.get("QWEN4EXP_RS_GUARD") == "1"', "GUARD = False"),
    ("T1 plans: stale expected to accept both rollbacks with the guard", P19,
     "guard_ok=[False, False], guard_log=2)", "guard_ok=[True, True], guard_log=2)"),
    ("T2 plans: stale expected to log no guard refusal", P19,
     "guard_ok=[False, False], guard_log=2)", "guard_ok=[False, False], guard_log=0)"),
    ("T3 plans: cp_pending expected to diverge with the guard", P19,
     'diverge={"ec": {1: 4}, "r1": {1: 4}, "guard": {}})', 'diverge={"ec": {1: 4}, "r1": {1: 4}, "guard": {1: 4}})'),
    ("T4 plans: extra_cell's second rollback expected accepted with the guard", P20,
     "guard_ok=[True, False], guard_log=1)", "guard_ok=[True, True], guard_log=1)"),
]
PFX = ("hc_init,hc_combine,ple_,linear_attn_out,conv_output,attn_,final_output,indexer_,kq_mask_qsa,kq_soft_max,Qcur,"
       "Kcur,Vcur,ffn_moe_topk,ffn_out,l_last,result_,conv_state_at,state_predelta,new_state")
GGUF_PY = os.path.join(llama, "gguf-py")


def sh(*args, **kw):
    return subprocess.run(list(args), check=True, capture_output=True, text=True, **kw).stdout


def reset_copy(patched=True):
    sh("git", "-C", copy, "checkout", "--", "src/llama-memory-recurrent.cpp", "src/llama-memory-recurrent.h")
    if patched:
        sh("git", "-C", copy, "apply", PATCH)


def rebuild():
    out = subprocess.run(["cmake", "--build", build, "-j4", "--target", "dump-tensors"], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
    return len(re.findall(r"Building CXX", out.stdout))


def dump_all(root, gguf):
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for plans in (P19, P20):
        path = os.path.join(wt, plans)
        for run in sh(sys.executable, path, "--list").split():
            lines = sh(sys.executable, path, run).splitlines()
            np_, kvu, rs, ub = lines[2].split()
            d = os.path.join(root, run)
            env = dict(os.environ, DUMP_DIR=d, DUMP_TOKENS=lines[1], DUMP_PER_GRAPH="1", DUMP_PREFIXES=PFX,
                       DUMP_CALLS=lines[0], DUMP_KV_UNIFIED=kvu, DUMP_N_RS_SEQ=rs)
            with open(os.path.join(root, f"{run}.log"), "w") as log:
                subprocess.run([os.path.join(build, "dump-tensors"), "-m", gguf, "-c", "512", "-b", "512", "-ub", ub,
                                "-np", np_, "-t", "1", "-fa", "off", "-ctk", "f32", "-ctv", "f32"], env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
            with open(os.path.join(d, "tokens.txt"), "w") as f:
                f.write(lines[1] + "\n")


def run_tests(tree, dump, gguf, modules):
    env = dict(os.environ, QWEN4EXP_MULTISEQ_DUMP=dump, QWEN4EXP_TINY_GGUF=gguf, LLAMA_CPP_GGUF_PY=GGUF_PY,
               PARITY_LLAMA_COMMIT=commit, QWEN4EXP_RS_GUARD="1")
    out = subprocess.run([sys.executable, "-m", "unittest", *modules], cwd=tree, env=env, capture_output=True, text=True)
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


def report(name, rc, summary, failing):
    """A run without a unittest summary (the mutant broke the import) is BROKEN, not caught."""
    verdict = "BROKEN" if summary.startswith("?") else "caught" if rc else "SURVIVED"
    print(f"{verdict}: {name}\n    rc={rc} {summary}\n    failing: "
          f"{', '.join(f'{c}.{t}' for t, c in failing) or 'none'}", flush=True)
    return verdict == "caught"


os.makedirs(scratch, exist_ok=True)
gguf = os.path.join(scratch, "qwen4exp.gguf")
sh(sys.executable, os.path.join(wt, "tools/tinygen/qwen4exp_tiny.py"), gguf, env=dict(os.environ, PYTHONPATH=GGUF_PY))
DUMP_MODULES = ["tests.test_qwen4exp_multiseq_parity", "tests.test_qwen4exp_rs_guard_parity"]
ALL_MODULES = DUMP_MODULES + ["tests.test_qwen4exp_multiseq_ref", "tests.test_qwen4exp_rs_guard_ref"]
base_tree = os.path.join(scratch, "tree-base")
tree_copy(base_tree)
reset_copy()
print(f"llama.cpp {commit[:7]}; patched build: {rebuild()} objects compiled", flush=True)
base_dump = os.path.join(scratch, "dumps-base")
dump_all(base_dump, gguf)
rc, summary, failing = run_tests(base_tree, base_dump, gguf, ALL_MODULES)
print(f"baseline (patched build, no mutation): rc={rc} {summary}", flush=True)
caught = total = 0
try:
    if "C" in which:
        for name, old, new in C_MUTANTS:
            reset_copy(patched=old is not None)
            if old is not None:
                f = os.path.join(copy, RC)
                s = open(f).read()
                if s.count(old) != 1:
                    print(f"n/a: {name} (not in this pin's patched source)")
                    continue
                open(f, "w").write(s.replace(old, new))
            n_obj = rebuild()
            dump = os.path.join(scratch, "dumps-mut")
            dump_all(dump, gguf)
            rc, summary, failing = run_tests(base_tree, dump, gguf, DUMP_MODULES)
            total += 1
            caught += report(f"{name} ({n_obj} objects rebuilt)", rc, summary, failing)
finally:
    reset_copy()
    print(f"restored the patched copy: {rebuild()} objects rebuilt; diff vs patch: "
          f"{'same' if sh('git', '-C', copy, 'apply', '--reverse', '--check', PATCH) == '' else '?'}", flush=True)
if "P" in which:
    for name, path, old, new in P_MUTANTS:
        tree = os.path.join(scratch, "tree-m")
        tree_copy(tree)
        f = os.path.join(tree, path)
        s = open(f).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(f, "w").write(s.replace(old, new))
        rc, summary, failing = run_tests(tree, base_dump, gguf, ALL_MODULES)
        total += 1
        caught += report(name, rc, summary, failing)
print(f"caught {caught}/{total}")
