#!/usr/bin/env python3
"""Mutation check of the qwen4exp oracle: single-point mutations of ref/qwen4exp.py (and ref/kda.py); each mutant
must make tests.test_qwen4exp_tiny_parity / tests.test_qwen4exp_ref fail. The repo is never edited: ref/ and the two
test files are copied into <scratch dir>/<mutant> and mutated there.

usage: python3 tools/tinygen/mutate_qwen4exp.py <repo root> <dump dir> <gguf> <llama.cpp gguf-py> <scratch dir>
(dump dir and gguf: the artifacts printed by tools/tinygen/run_parity_qwen4exp.sh). Results:
docs/evidence/2026-10-10-qwen4exp-tiny-parity.md."""
import os
import re
import shutil
import subprocess
import sys

wt, dump, gguf, gguf_py, scratch = sys.argv[1:6]
Q, K = "ref/qwen4exp.py", "ref/kda.py"
MUTANTS = [
    ("M1 hc_combine: drop the 1/hc inside the scatter sigmoid", Q,
     "w = 2.0 * sigmoid(np.asarray(inject) / hc)", "w = 2.0 * sigmoid(np.asarray(inject))"),
    ("M2 pooled indexer keys rotated with text positions (4th = 0, not p0)", Q,
     "return np.stack([p0, p0, p0, p0], axis=1)", "return np.stack([p0, p0, p0, np.zeros_like(p0)], axis=1)"),
    ("M3 PLE hash: an EOS predecessor no longer resets the window", Q,
     "cut = cut or prev is None or prev == eos", "cut = cut or prev is None"),
    ("M4 GDN output gate silu(z) (Qwen3.5) instead of sigmoid(z)", Q,
     "return rms_norm(scan, eps) * np.asarray(w_norm) * sigmoid(z)",
     "return rms_norm(scan, eps) * np.asarray(w_norm) * silu(z)"),
    ("M5 MoE: routing weights not renormalised over the top-k", Q,
     "wts = wts / np.maximum(wts.sum(-1, keepdims=True), 6.103515625e-5)", "wts = wts / 1.0"),
    ("M6 HC mixer: sum over streams instead of mean", Q,
     "mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1) / hc",
     "mixed = (xn * sigmoid(gate)).reshape(T, hc, n).sum(axis=1)"),
    ("M7 QSA score: no relu on the head scores", Q,
     "s = np.maximum(np.einsum(\"thd,bd->thb\", iq, pooled), 0.0).sum(axis=1) / np.sqrt(d)",
     "s = np.einsum(\"thd,bd->thb\", iq, pooled).sum(axis=1) / np.sqrt(d)"),
    ("M8 IMRoPE: off-by-one in the t-section bound (pair 3 leaves the 4th section)", Q,
     "elif sector % 3 == 0 and sector < 3 * s[0]:", "elif sector % 3 == 0 and sector <= 3 * s[0]:"),
    ("M9 PLE conv: dilation 1 instead of the n-gram size", Q,
     "w_conv, dtype=np.float64), ngram)", "w_conv, dtype=np.float64), 1)"),
    ("M10 QSA tail: tail starts at kpool*floor(t/kpool) instead of kpool*floor((t+1)/kpool)", Q,
     "tail0 = ((t + 1) // kpool) * kpool", "tail0 = (t // kpool) * kpool"),
    ("M11 shared expert: gate without sigmoid", Q,
     "sh_gate = sigmoid(x @ w[\"ffn_gate_inp_shexp.weight\"])", "sh_gate = x @ w[\"ffn_gate_inp_shexp.weight\"]"),
    ("M12 text M-RoPE positions (p, p, p, p) instead of (p, p, p, 0)", Q,
     "return np.stack([p, p, p, np.zeros_like(p)], axis=1)", "return np.stack([p, p, p, p], axis=1)"),
    ("M13 GDN head map: value head h reads key head h // (Hv/Hk) instead of h % Hk", K,
     "hk = h % H_k", "hk = h // (H_v // H_k)"),
]


def run(tree):
    env = dict(os.environ, QWEN4EXP_TINY_DUMP=dump, QWEN4EXP_TINY_GGUF=gguf, LLAMA_CPP_GGUF_PY=gguf_py)
    p = subprocess.run([sys.executable, "-m", "unittest", "tests.test_qwen4exp_tiny_parity", "tests.test_qwen4exp_ref"],
                       cwd=tree, env=env, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (test_\w+) \(tests\.(\w+)", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"))
    os.makedirs(os.path.join(tree, "tests"))
    for f in ("test_qwen4exp_tiny_parity.py", "test_qwen4exp_ref.py"):
        shutil.copy(os.path.join(wt, "tests", f), os.path.join(tree, "tests", f))
    return tree


rc, summary, failed = run(make_tree("baseline"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
caught = 0
for label, path, old, new in MUTANTS:
    tree = make_tree(label.split()[0])
    fp = os.path.join(tree, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    open(fp, "w").write(src.replace(old, new))
    rc, summary, failed = run(tree)
    caught += rc != 0
    print(f"{label}\n    {path}: '{old}' -> '{new}'\n    rc={rc} {' | '.join(summary)}")
    print("    failing: " + (", ".join(f"{m}.{t}" for t, m in failed) or "none"))
print(f"caught {caught}/{len(MUTANTS)}")
