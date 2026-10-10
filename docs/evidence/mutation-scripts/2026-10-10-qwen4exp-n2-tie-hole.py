"""Lead check of the N2 tie-replay hole reported by parity-qa (N6): a mutant confined to ref/qwen4exp.py qsa_select
(one pool fewer) against tests.test_qwen4exp_tiny_parity, without and with the one-line fix in _tie_variants."""
import os, shutil, subprocess, sys, tempfile
ROOT, ART, GGUF_PY = sys.argv[1], sys.argv[2], sys.argv[3]
MUT = ("        top = vis[np.argsort(-scores[t, vis], kind=\"stable\")[:n_top]]\n        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]\n        out.append(sorted(cells + qsa_tail(t, kpool)))",
       "        top = vis[np.argsort(-scores[t, vis], kind=\"stable\")[:max(n_top - 1, 0)]]\n        cells = [c for b in sorted(top) for c in range(b * kpool, (b + 1) * kpool)]\n        out.append(sorted(cells + qsa_tail(t, kpool)))")
FIX = ("            ok, why = Q.qsa_tie_variant(scores[t], s, t, self.hp[\"idx_top_k\"], self.hp[\"kpool\"])\n",
       "            ok, why = Q.qsa_tie_variant(scores[t], s, t, self.hp[\"idx_top_k\"], self.hp[\"kpool\"])\n"
       "            own, why_own = Q.qsa_tie_variant(scores[t], mine[t], t, self.hp[\"idx_top_k\"], self.hp[\"kpool\"])\n"
       "            ok, why = ok and own, why or (\"oracle selection: \" + str(why_own))\n")
env = dict(os.environ, QWEN4EXP_TINY_DUMP=os.path.join(ART, "dump"), QWEN4EXP_TINY_GGUF=os.path.join(ART, "qwen4exp.gguf"),
           LLAMA_CPP_GGUF_PY=GGUF_PY)
def run(mutate, fix):
    d = tempfile.mkdtemp()
    for sub in ("ref", "tests", "tools"):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub), ignore=shutil.ignore_patterns("__pycache__"))
    for path, (old, new), on in (("ref/qwen4exp.py", MUT, mutate), ("tests/test_qwen4exp_tiny_parity.py", FIX, fix)):
        if on:
            p = os.path.join(d, path); s = open(p).read(); assert s.count(old) == 1, path
            open(p, "w").write(s.replace(old, new))
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_qwen4exp_tiny_parity"], cwd=d, env=env,
                       capture_output=True, text=True, timeout=600)
    shutil.rmtree(d)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1]
for name, m, f in (("baseline, no fix", 0, 0), ("qsa_select one pool fewer, no fix", 1, 0),
                   ("baseline, with fix", 0, 1), ("qsa_select one pool fewer, with fix", 1, 1)):
    rc, last = run(m, f)
    print("%-40s rc=%d %s" % (name, rc, last), flush=True)
