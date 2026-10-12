#!/usr/bin/env python3
"""Mutation check of the N12 ik_llama.cpp glm5next parity (gap 5): single-point mutations of the ik-semantics oracle
ref/glm5next_ik.py (O1-O7), of the test's own rules (T1-T3) and of the ik side, simulated by editing a copy of one
dumped run (D1-D4). Each mutant should make tests.test_ik_glm5next_parity fail; O4 is expected to survive (ik's L2
eps differs from llama.cpp's by an amount below every tolerance, ref/glm5next_ik.py item 3) and is kept to show it.
The repository is never edited: ref/ and tests/ are copied into <scratch>/<mutant>; a dump mutant gets a copy of the
one run directory it edits (the other runs are symlinked).

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n12-ik-mutation.py <repo root> <artifacts> <gguf-py> <scratch>
(<artifacts>: IK_GLM5NEXT_WORK of tools/tinygen/run_parity_ik_glm5next.sh, holding n12/ and ik.gguf).
Results: docs/evidence/2026-10-11-n12-ik-glm5next-cpu.md."""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

wt, art, gguf_py, scratch = sys.argv[1:5]
IK = "ref/glm5next_ik.py"
T = "tests/test_ik_glm5next_parity.py"
MUTANTS = [   # (label, [(path, old, new), ...])
    ("O1 SwiGLU clamp before the SiLU (llama.cpp's ggml_swiglu_clamp)",
     [(IK, "G.swiglu, G.gated_delta_rule = swiglu_post_silu, _gdr", "G.gated_delta_rule = _gdr")]),
    ("O2 no cell 0 from ik's tail padding (llama.cpp's sentinel)",
     [(IK, "    return [sorted(set(c) | {0}) if (int(t) + 1) % kpool != kpool - 1 else list(c) for c, t in zip(cells, pos)]",
       "    return [list(c) for c in cells]")]),
    ("O3 indexer LayerNorm eps 0 when the key is absent (llama.cpp's rule)",
     [(IK, "        hp[\"norm_eps\"] = hp[\"rms_eps\"]", "        hp[\"norm_eps\"] = 0.0")]),
    ("O4 KDA q/k L2 norm as llama.cpp: x / sqrt(|x|^2 + 1e-6) (expected to survive)",
     [(IK, "    if normalize_qk:\n        e = _CFG[\"rms_eps\"]", "    if False:\n        e = _CFG[\"rms_eps\"]"),
      (IK, "    return _ORIG[\"gated_delta_rule\"](q, k, v, g, beta, state=state, normalize_qk=False)",
       "    return _ORIG[\"gated_delta_rule\"](q, k, v, g, beta, state=state, normalize_qk=normalize_qk, eps=1e-6)")]),
    ("O5 MLA latent kept in f32 by the oracle (no f16 store)",
     [(IK, "    st[\"latent\"] = np.concatenate([st[\"latent\"], _cast(r[\"latent\"], cfg[\"latent\"])])",
       "    st[\"latent\"] = np.concatenate([st[\"latent\"], _cast(r[\"latent\"], \"f32\")])")]),
    ("O6 a pool visible one token before it is complete",
     [(IK, "        scores = G.scores_at(r[\"iq\"], r[\"wts\"], st[\"pooled\"], kp, pos)",
       "        scores = G.scores_at(r[\"iq\"], r[\"wts\"], st[\"pooled\"], kp, np.asarray(pos) + 1)")]),
    ("O7 flash attention emulated as exact f32 attention (no fa_chunk)",
     [(IK, "hp[\"k_mla\"], [\"fa_chunk\"] * T if cfg.get(\"fa\") else None)", "hp[\"k_mla\"], None)"),
      (T, "                modes = [\"fa_chunk\"] * n if self.cfg[\"fa\"] else None", "                modes = None")]),
    ("T1 test: the latent cache type taken from -ctk also with -fa off (llama.cpp's rule)",
     [(T, "        latent = self.flag(\"-ctk\", \"f16\") if fa else self.flag(\"-ctv\", \"f16\")",
       "        latent = self.flag(\"-ctk\", \"f16\")")]),
    ("T2 test: -ictk default f32 instead of ik's f16",
     [(T, "index=self.flag(\"-ictk\", \"f16\")", "index=self.flag(\"-ictk\", \"f32\")")]),
    ("T3 test: no tie replay", [(T, "        while cls._replay_first_tie():", "        while False:")]),
]


def load(path):
    m = json.load(open(path + ".json"))
    return m, np.fromfile(path + ".bin", dtype=np.float32)


def d1(root):            # one cached latent element (layer 7, decode graph 5) one binary16 spacing up
    p = os.path.join(root, "n12", "decode", "g005", "kv_cache-7__0")
    m, a = load(p)
    a = a.reshape(m["ne"][1], m["ne"][0])
    a[20, 5] = np.nextafter(np.float16(a[20, 5]), np.float16(np.inf))
    a.tofile(p + ".bin")


def d2(root):            # p64 layer 11: token 40 attends cell 24 instead of its first selected pool cell
    p = os.path.join(root, "n12", "p64", "g000", "dsa_top_k-11__0")
    m = json.load(open(p + ".json"))
    a = np.fromfile(p + ".bin", dtype=np.int32).reshape(m["ne"][1], m["ne"][0])
    row = a[40]
    j = next(i for i, c in enumerate(row) if c != 0 and c != 24)
    row[j] = 24 if 24 not in row else 25
    a.tofile(p + ".bin")


def d3(root):            # p64: the FFN output of layer 5 scaled by 1 + 2^-12
    p = os.path.join(root, "n12", "p64", "g000", "ffn_out-5__0")
    m, a = load(p)
    (a * np.float32(1 + 2.0 ** -12)).astype(np.float32).tofile(p + ".bin")


def d4(root):            # ub16: one row of the indexer cache (layer 3, graph 2) as f16 although -ictk f32
    p = os.path.join(root, "n12", "ub16", "g002", "dsa_cached_all-3__0")
    m, a = load(p)
    a = a.reshape(m["ne"][1], m["ne"][0])
    a[33] = a[33].astype(np.float16).astype(np.float32)
    a.tofile(p + ".bin")


DUMPS = [("D1 ik side: one cached latent element one binary16 spacing off (L7, decode graph 5)", "decode", d1),
         ("D2 ik side: one attended cell changed (L11, p64, token 40)", "p64", d2),
         ("D3 ik side: FFN output 2^-12 too large (L5, p64)", "p64", d3),
         ("D4 ik side: one indexer cache row rounded to f16 under -ictk f32 (L3, ub16 graph 2)", "ub16", d4)]
TESTS = ["tests.test_ik_glm5next_parity"]


def run(tree, dumps=art):
    env = dict(os.environ, IK_GLM5NEXT_DUMP=os.path.join(dumps, "n12"), IK_GLM5NEXT_GGUF=os.path.join(art, "ik.gguf"),
               LLAMA_CPP_GGUF_PY=gguf_py)
    p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=3600)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_\w+\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"), ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tree, "tests"))
    shutil.copy(os.path.join(wt, T), os.path.join(tree, T))
    return tree


def make_dumps(name, run_name):
    root = os.path.join(scratch, name + "-dumps")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "n12"))
    for entry in os.listdir(os.path.join(art, "n12")):
        src = os.path.join(art, "n12", entry)
        if entry == run_name:
            shutil.copytree(src, os.path.join(root, "n12", entry))
        else:
            os.symlink(src, os.path.join(root, "n12", entry))
    return root


def report(label, rc, summary, failed):
    print(f"{label}\n    rc={rc} {' | '.join(summary)}")
    if failed:
        print("    failed: " + ", ".join(f"{cls}.{t}" for t, cls in failed))


print(f"artifacts {art}")
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
for label, run_name, fn in DUMPS:
    root = make_dumps(label.split()[0], run_name)
    fn(root)
    rc, summary, failed = run(tree, root)
    caught += rc != 0
    report(label, rc, summary, failed)
    shutil.rmtree(root)
print(f"caught {caught}/{len(MUTANTS) + len(DUMPS)}")
