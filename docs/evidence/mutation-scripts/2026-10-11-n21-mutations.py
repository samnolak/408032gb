#!/usr/bin/env python3
"""Mutation check of N21 (docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md): single-point mutants of
  * the llama.cpp patch (patches/llama.cpp/0001-mtp-heads-slot-replay.patch), applied to <patched llama.cpp>, a scratch
    git repo holding 79e2e74 plus the patch as its HEAD commit: the source is mutated there, <build> rebuilt, the N21
    driver run, and the file restored with `git checkout` of that scratch repo (the pinned checkouts are never touched);
  * the dump tool's N21 modes (tools/dump/dump_tensors.cpp): mutated in the worktree, rebuilt, the driver run, and the
    file restored from `git show HEAD:` (checked with git diff --quiet at the end);
  * the oracles and tests (ref/mimo2.py, ref/mtp_cells.py, tests/test_mimo2_mtp_*.py): on copies of ref/ and tests/ in
    <scratch>/<mutant>, run against the dumps of one patched driver run (as N17's mutation script).
Each mutant must make the N21 step fail: tests.test_mimo2_mtp_fix_parity and/or tests.test_mimo2_mtp_fix_ref (for the
C++ mutants: the driver, which runs both after its dumps, exits non-zero).

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n21-mutations.py <repo root> <patched llama.cpp> <build> <scratch>
"""
import glob
import os
import re
import shutil
import subprocess
import sys

wt, llama, build, scratch = sys.argv[1:5]
os.makedirs(scratch, exist_ok=True)
WORK = os.path.join(scratch, "work")
DRIVER = os.path.join(wt, "tools/tinygen/run_parity_mimo2_n21.sh")
TESTS = ["tests.test_mimo2_mtp_fix_parity", "tests.test_mimo2_mtp_fix_ref"]
SPEC, KV, TOOL = "common/speculative.cpp", "src/llama-kv-cache.cpp", "tools/dump/dump_tensors.cpp"
R, CELLS, TH, TF = "ref/mimo2.py", "ref/mtp_cells.py", "tests/test_mimo2_mtp_heads_parity.py", \
    "tests/test_mimo2_mtp_fix_parity.py"

CPP = [  # (label, where: "llama" | "wt", path, old, new)
    ("P1 llama_kv_cache::prepare ignores the slot replay (always searches)", "llama", KV,
     "    if (slot_replay) {\n        if (slot_replay_fits(ubatches)) {",
     "    if (false) {\n        if (slot_replay_fits(ubatches)) {"),
    ("P2 process_heads_once without replay (heads 1..H-1 search their own cells)", "llama", SPEC,
     "                llama_memory_set_slot_replay(mem_dft, head > 0);",
     "                llama_memory_set_slot_replay(mem_dft, false);"),
    ("P3 process_heads_once without chunks (every head on the whole batch, with replay)", "llama", SPEC,
     "        const int32_t n_ub = std::max<int32_t>(1, (int32_t) llama_n_ubatch(ctx_dft));",
     "        const int32_t n_ub = std::max<int32_t>(1, n_tokens);"),
    ("P4 process_heads_once: seq_rm from the batch's first position instead of the chunk's", "llama", SPEC,
     "                            llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[k].pos[0], -1);",
     "                            llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[0].pos[0], -1);"),
    ("P5 slot_replay_fits does not check that a replayed cell is free", "llama", KV,
     "                if (idx >= cells.size() || (!cells.is_empty(idx) && taken[strm].count(idx) == 0)) {",
     "                if (idx >= cells.size()) {"),
    ("D1 DUMP_MTP_FIX replays without slot replay", "wt", TOOL,
     "                    if (!llama_memory_set_slot_replay(mem, head > 0)) {",
     "                    if (!llama_memory_set_slot_replay(mem, false)) {"),
    ("D2 DUMP_MTP_FIX without chunks (every head on the whole call, with replay)", "wt", TOOL,
     "            const int n_ub = (int) llama_n_ubatch(mtp_ctx.get());",
     "            const int n_ub = (int) ids.size();"),
    ("D3 DUMP_MTP_SPEC: accept() told one draft more than the rejection kept", "wt", TOOL,
     "                        common_speculative_accept(spec.get(), 0, (uint16_t) (last_call_n - n - 1));",
     "                        common_speculative_accept(spec.get(), 0, (uint16_t) (last_call_n - n));"),
]
PY = [  # (label, path, old, new)
    ("O1 oracle: fixed process without replay (heads 1..H-1 search their own cells)", R,
     "                                      prec=prec, replay=rep)",
     "                                      prec=prec, replay=None)"),
    ("O2 oracle: fixed process seq_rm from the call's first position, not the chunk's", R,
     "                cache_truncate(cache, p0 + j)\n", "                cache_truncate(cache, p0)\n"),
    ("O3 oracle: fixed process without chunks (one ubatch per head for the whole call)", R,
     "        for j in range(0, len(tokens), n_ubatch):\n            rep = None\n",
     "        n_ubatch = len(tokens)\n        for j in range(0, len(tokens), n_ubatch):\n            rep = None\n"),
    ("O4 oracle: incomplete windows never reported (attn[\"missing\"] all False)", R,
     "        missing = held[pos + 1] - held[lo] < pos - lo + 1\n",
     "        missing = np.zeros(len(pos), dtype=bool)\n"),
    ("O5 oracle: replay accepted onto a taken cell", R,
     "            if len(idxs) != n or any(cache[which].pos[i] >= 0 for i in idxs):",
     "            if len(idxs) != n:"),
    ("C1 cell model: the fixed order behaves as upstream", CELLS,
     "        if not self.fixed:\n            for head in range(self.H):",
     "        if True:\n            for head in range(self.H):"),
    ("C2 cell model: the window one position wider (p - pos_cell <= n_swa)", CELLS,
     "            vis = (cp >= 0) & (cp <= p) & (p - cp < self.c.n_swa)",
     "            vis = (cp >= 0) & (cp <= p) & (p - cp <= self.c.n_swa)"),
    ("C3 cell model: no seq_rm before each draft step", CELLS,
     "            for k in range(min(n_steps, self.H)):\n                self.seq_rm(pos0)\n",
     "            for k in range(min(n_steps, self.H)):\n                pass\n"),
    ("T1 test: fixed records paired head-major instead of chunk-major", TH,
     "                    c, head = divmod(k, cls.H)\n", "                    head, c = divmod(k, n_chunks)\n"),
    ("T2 test: the fixed run's h rows of chunk c taken from the call's start", TH,
     "                want = full[off:off + rec[\"n\"]]\n", "                want = full[:rec[\"n\"]]\n"),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def driver():
    env = dict(os.environ, MIMO2_N21_WORK=WORK)
    p = sh([DRIVER, llama, build], env=env, timeout=7200)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.(\w+)\.(\w+)[.)]", out, re.M)))
    if not summary:
        summary = ["no unittest summary: the driver stopped before the tests: " + " / ".join(out.strip().splitlines()[-3:])]
    return p.returncode, summary, failed


def tests(tree):
    env = dict(os.environ, MIMO2_N21_DUMP=os.path.join(WORK, "n21"),
               LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"))
    p = sh([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, timeout=3600)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.(\w+)\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def report(label, path, old, new, rc, summary, failed):
    by_mod = {}
    for t, mod, cls in failed:
        by_mod.setdefault(mod, set()).add(f"{cls}.{t}" if not t.startswith("setUpClass") else f"{cls}.setUpClass")
    print(f"{label}\n    {path}: {old.strip()[:90]!r} -> {new.strip()[:90]!r}\n    rc={rc} {' | '.join(summary)}")
    for mod in sorted(by_mod):
        names = sorted(by_mod[mod])
        print(f"    {mod}: {len(names)} failing: " + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
    sys.stdout.flush()


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"), ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tree, "tests"))
    for f in glob.glob(os.path.join(wt, "tests", "*.py")):
        shutil.copy(f, os.path.join(tree, "tests"))
    return tree


head = sh(["git", "-C", llama, "log", "--oneline", "-1"]).stdout.strip()
print(f"patched llama.cpp {llama}: {head}; build {build}; worktree {sh(['git', '-C', wt, 'rev-parse', 'HEAD']).stdout.strip()}")
rc, summary, failed = driver()
print(f"baseline (no mutation), driver: rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = total = 0
# Python mutants first, on the baseline dumps
for label, path, old, new in PY:
    tree = make_tree(label.split()[0])
    fp = os.path.join(tree, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    open(fp, "w").write(src.replace(old, new))
    rc, summary, failed = tests(tree)
    total += 1
    caught += rc != 0
    report(label, path, old, new, rc, summary, failed)
    shutil.rmtree(tree, ignore_errors=True)
# C++ mutants: patch and dump tool
for label, where, path, old, new in CPP:
    root = llama if where == "llama" else wt
    fp = os.path.join(root, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    try:
        open(fp, "w").write(src.replace(old, new))
        rc, summary, failed = driver()
    finally:
        if where == "llama":
            sh(["git", "-C", llama, "checkout", "--", path], check=True)
        else:
            open(fp, "w").write(sh(["git", "-C", wt, "show", f"HEAD:{path}"], check=True).stdout)
    total += 1
    caught += rc != 0
    report(label, path, old, new, rc, summary, failed)
clean_llama = sh(["git", "-C", llama, "status", "--short"]).stdout.strip()
clean_wt = sh(["git", "-C", wt, "diff", "--quiet", "--", TOOL]).returncode == 0
rc, summary, failed = driver()
print(f"after the C++ mutants, restored sources (patched llama.cpp status: {clean_llama or 'clean'}; worktree "
      f"{TOOL}: {'unchanged' if clean_wt else 'CHANGED'}), driver: rc={rc} {' | '.join(summary)}")
print(f"caught {caught}/{total}")
