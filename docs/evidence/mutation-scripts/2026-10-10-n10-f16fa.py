#!/usr/bin/env python3
"""Mutation check of the N10 f16 KV / flash-attention parity (gap 3): single-point mutations of the precision model
(ref/fattn.py, ref/qwen4exp.py: O1-O7), of the tests' tolerance and replay settings (T1-T3) and of the llama.cpp side,
simulated by editing a copy of one dumped run (D1-D4). Each mutant must make tests.test_qwen4exp_f16fa_parity,
tests.test_glm5next_f16fa_parity and/or tests.test_fattn_ref fail. The repository is never edited: ref/ and tests/
are copied into <scratch>/<mutant>; a dump mutant gets a copy of the one run directory it edits (the other runs are
symlinked).

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-f16fa.py <repo root> <artifacts> <llama.cpp checkout>
       <scratch>
(<artifacts>: the directory tools/tinygen/run_parity_f16fa.sh printed / F16FA_WORK, holding q/, g/, qwen4exp.gguf and
glm5next.gguf, made with the default -ctk f16 -ctv f16 -fa on; <llama.cpp checkout>: the one the dumps came from).
Results: docs/evidence/2026-10-10-quant-fa-parity.md."""
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
FA_ = "ref/fattn.py"
QR = "ref/qwen4exp.py"
TQ = "tests/test_qwen4exp_f16fa_parity.py"
TG = "tests/test_glm5next_f16fa_parity.py"
MIX = "tests/f16fa.py"
MUTANTS = [   # (label, [(path, old, new), ...])
    ("O1 store(): no f16 rounding of what the caches hold",
     [(FA_, "return f16(x) if kv_f16(prec) else np.asarray(x, dtype=np.float64)",
       "return np.asarray(x, dtype=np.float64)")]),
    ("O2 one_chunk: Q kept in f32 (no Q_q = f16(q))",
     [(FA_, "    qq = f16(qh)  ", "    qq = np.asarray(qh, dtype=np.float64)  ")]),
    ("O3 one_chunk: accumulator kept in f32 in the mad step (VKQ16 not rounded)",
     [(FA_, "acc = f16(acc + vc[i][None, :] * vs[:, None])", "acc = f32(acc + vc[i][None, :] * vs[:, None])")]),
    ("O4 tiled path from 16 query rows instead of 64", [(FA_, "\nFA_TILE_Q = 64 ", "\nFA_TILE_Q = 16 ")]),
    ("O5 f16 rounding toward zero instead of to nearest even",
     [(FA_, "    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float16).astype(np.float64)",
       "    x32 = np.asarray(x, dtype=np.float64).astype(np.float32); r = x32.astype(np.float16); "
       "return np.where(np.abs(r.astype(np.float32)) > np.abs(x32), np.nextafter(r, np.float16(0)), r)"
       ".astype(np.float64)")]),
    ("O6 qwen4exp: raw indexer keys cached unrounded (incremental path only)",
     [(QR, "idxc[\"k_raw\"] = np.concatenate([idxc[\"k_raw\"], store(k_raw, prec)])",
       "idxc[\"k_raw\"] = np.concatenate([idxc[\"k_raw\"], k_raw])")]),
    ("O7 one_chunk: the score not rounded to f32 (float64 q.k * scale)",
     [(FA_, "s_all = (f32(qq @ kc.T).astype(np.float32) * sc32).astype(np.float64)", "s_all = (qq @ kc.T) * scale")]),
    ("T1 tests: the f32 tolerance 1e-4 downstream of the attention too",
     [(TQ, "    TIE_RTOL = MOE_TIE_RTOL = TOL16", "    TIE_RTOL = MOE_TIE_RTOL = TOL16\n    TOL = 1e-4"),
      (TG, "    TIE_RTOL = MOE_TIE_RTOL = TOL16", "    TIE_RTOL = MOE_TIE_RTOL = TOL16\n    TOL = 1e-4")]),
    ("T2 tests: no near-tie replay in the f16 runs (the N6/N9 settings)",
     [(TQ, "    TIE_RTOL = MOE_TIE_RTOL = TOL16", "    TIE_RTOL, MOE_TIE_RTOL = 1e-6, None"),
      (TG, "    TIE_RTOL = MOE_TIE_RTOL = TOL16", "    TIE_RTOL, MOE_TIE_RTOL = 1e-6, None")]),
    ("T3 tests: the plain f32 oracle used as the precision-aware one", [(MIX, "    PREC = PREC\n", "    PREC = None\n")]),
]


def load(path):
    m = json.load(open(path + ".json"))
    return m, np.fromfile(path + ".bin", dtype=np.float32)


def d1(root):            # one cached K element of qwen4exp layer 3, graph 1 of ub16, one binary16 spacing up
    p = os.path.join(root, "q", "ub16", "g001", "cache_k_l3 (view)__0")
    m, a = load(p)
    a = a.reshape(m["ne"][1], m["ne"][0])
    a[20, 5] = np.nextafter(np.float16(a[20, 5]), np.float16(np.inf))
    a.tofile(p + ".bin")


def d2(root):            # the flash-attention output of qwen4exp layer 3, graph 2 of ub16, scaled by 1 + 2^-8
    p = os.path.join(root, "q", "ub16", "g002", "attn_pregate-3__0")
    m, a = load(p)
    (a * np.float32(1 + 2.0 ** -8)).astype(np.float32).tofile(p + ".bin")


def d3(root):            # two cells of the glm5-next MLA latent cache (layer 7, graph 3 of decode) swapped
    p = os.path.join(root, "g", "decode", "g003", "cache_k_l7 (view)__0")
    m, a = load(p)
    a = a.reshape(m["ne"][1], m["ne"][0])
    a[[10, 11]] = a[[11, 10]]
    a.tofile(p + ".bin")


def d4(root):            # glm5-next pf64: the latent of cell 60 as the flash attention read it, scaled by 1 + 2^-9
    p = os.path.join(root, "g", "pf64", "g000")
    for name in ("cache_k_l3 (view)__1", "cache_k_l3 (view) (view)__0"):   # the get_k view and its V view
        m, a = load(os.path.join(p, name))
        a = a.reshape(-1, m["ne"][0])
        a[60] *= np.float32(1 + 2.0 ** -9)                                 # in the tail of tokens 60-62
        a.tofile(os.path.join(p, name + ".bin"))


DUMPS = [("D1 llama side: one cached K element one binary16 spacing off (qwen4exp L3, ub16 graph 1)", "q", "ub16", d1),
         ("D2 llama side: flash-attention output 2^-8 too large (qwen4exp L3, ub16 graph 2)", "q", "ub16", d2),
         ("D3 llama side: two latent-cache cells swapped (glm5-next L7, decode graph 3)", "g", "decode", d3),
         ("D4 llama side: one latent the attention read 2^-9 too large (glm5-next L3, pf64)", "g", "pf64", d4)]
TESTS = ["tests.test_qwen4exp_f16fa_parity", "tests.test_glm5next_f16fa_parity", "tests.test_fattn_ref"]


def run(tree, dumps=art):
    env = dict(os.environ, QWEN4EXP_F16FA_DUMP=os.path.join(dumps, "q"), GLM5NEXT_F16FA_DUMP=os.path.join(dumps, "g"),
               QWEN4EXP_F16FA_GGUF=os.path.join(art, "qwen4exp.gguf"),
               GLM5NEXT_F16FA_GGUF=os.path.join(art, "glm5next.gguf"),
               LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"), PARITY_LLAMA_COMMIT=commit,
               PARITY_KV="f16", PARITY_FA="on")
    p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=3600)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_(\w+)\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(wt, "tests"), os.path.join(tree, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__", "fixtures", "prompts", "strata_glm", "ggml_parity"))
    return tree


def make_dumps(name, model, run_name):
    root = os.path.join(scratch, name + "-dumps")
    shutil.rmtree(root, ignore_errors=True)
    for m in ("q", "g"):
        os.makedirs(os.path.join(root, m))
        for entry in os.listdir(os.path.join(art, m)):
            src = os.path.join(art, m, entry)
            if m == model and entry == run_name:
                shutil.copytree(src, os.path.join(root, m, entry))
            else:
                os.symlink(src, os.path.join(root, m, entry))
    return root


def report(label, rc, summary, failed):
    print(f"{label}\n    rc={rc} {' | '.join(summary)}")
    by = {}
    for t, mod, cls in failed:
        by.setdefault(mod.replace("_f16fa_parity", "").replace("_ref", " (property)"), []).append(f"{cls}.{t}")
    for mod, items in sorted(by.items()):
        print(f"    {mod}: " + ", ".join(sorted(items)))


print(f"llama.cpp {commit}; dumps {art}")
rc, summary, failed = run(make_tree("baseline"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = 0
for label, edits in MUTANTS:
    tree = make_tree(label.split()[0])
    for p2, o2, n2 in edits:
        fp = os.path.join(tree, p2)
        src = open(fp).read()
        assert src.count(o2) == 1, (label, p2, o2)
        open(fp, "w").write(src.replace(o2, n2))
    rc, summary, failed = run(tree)
    caught += rc != 0
    report(label, rc, summary, failed)
tree = make_tree("dumps")
for label, model, run_name, fn in DUMPS:
    root = make_dumps(label.split()[0], model, run_name)
    fn(root)
    rc, summary, failed = run(tree, root)
    caught += rc != 0
    report(label, rc, summary, failed)
    shutil.rmtree(root)
print(f"caught {caught}/{len(MUTANTS) + len(DUMPS)}")
