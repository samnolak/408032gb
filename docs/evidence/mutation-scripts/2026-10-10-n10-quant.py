#!/usr/bin/env python3
"""Mutation check of the N10 quantized-weight parity (gap 2): single-point mutations of the activation-quantization
oracle (ref/quant.py: Q1-Q6), of the dequantization of the pinned gguf-py (G1-G3, on a copy of gguf-py), of the tests'
tolerance (T1) and of the llama.cpp side, simulated by editing a copy of one dumped run (D1-D4 qwen4exp, D5-D8
glm5-next). Each mutant must make tests.test_qwen4exp_quant_parity, tests.test_glm5next_quant_parity and/or
tests.test_quant_ref fail. The repository and the pinned checkout are never
edited: ref/ and tests/ are copied into <scratch>/<mutant>, gguf-py into <scratch>/<mutant>-gguf-py, and a dump
mutant gets a copy of the one run directory it edits (everything else symlinked).

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant.py <repo root> <artifacts> <llama.cpp checkout>
       <scratch>
(<artifacts>: QUANT_WORK of tools/tinygen/run_parity_quant.sh, holding Q8_0/, Q4_K_M/ and glm5next/{Q8_0,Q4_K_M}/;
<llama.cpp checkout>: the one the dumps came from). Results: docs/evidence/2026-10-10-quant-fa-parity.md."""
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
RQ = "ref/quant.py"
TQ = "tests/test_qwen4exp_quant_parity.py"
GQ = "gguf/quants.py"          # relative to the gguf-py copy
MUTANTS = [   # (label, [(path, old, new), ...]); a path under gguf/ edits the gguf-py copy
    ("Q1 no activation quantization at all (dequantized weights only)",
     [(RQ, "        self.vdt = VEC_DOT[qtype] if act else None", "        self.vdt = None")]),
    ("Q2 Q8_0 activations by the scalar reference rule (id = 1/d, round half away from zero)",
     [(RQ, "    q = np.rint((b * inv.astype(np.float32)).astype(np.float32))",
       "    dd = (amax / np.float32(127)).astype(np.float32)\n"
       "    inv = np.where(dd != 0, np.float32(1) / np.where(dd != 0, dd, np.float32(1)), np.float32(0))\n"
       "    v = (b * inv.astype(np.float32)).astype(np.float32)\n"
       "    q = np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))")]),
    ("Q3 Q8_0 block scale kept in f32 (not rounded to f16)",
     [(RQ, "    d16 = (amax / np.float32(127)).astype(np.float16)", "    d16 = (amax / np.float32(127)).astype(np.float32)")]),
    ("Q4 k-quant activations quantized like Q8_0 (32-blocks, f16 scale) instead of Q8_K",
     [(RQ, "    elif vec_dot_type == \"Q8_K\":\n        q, d = q8_K(x)", "    elif vec_dot_type == \"Q8_K\":\n        q, d = q8_0(x)")]),
    ("Q5 F16 weights: activations not rounded to f16",
     [(RQ, "        return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float16).astype(np.float64)",
       "        return np.asarray(x, dtype=np.float64)")]),
    ("Q6 mul_mat_id: expert e reads the next expert's matrix",
     [(RQ, "            return QWeight(self.w[key], self.qtype, self.act)",
       "            return QWeight(self.w[(key + 1) % self.w.shape[0]], self.qtype, self.act)")]),
    ("G1 gguf-py Q4_K dequantize: the minimum added instead of subtracted",
     [(GQ, "        return (d * qs - dm).reshape((n_blocks, QK_K))", "        return (d * qs + dm).reshape((n_blocks, QK_K))")]),
    ("G2 gguf-py Q6_K dequantize: offset 31 instead of 32",
     [(GQ, "q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)",
       "q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(31)")]),
    ("G3 gguf-py Q5_0 dequantize: the fifth bit dropped",
     [(GQ, "qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)", "qs = ql.astype(np.int8) - np.int8(16)")]),
    ("T1 tests: TOL_OP 1e-2 (a tolerance that would hide the activation quantization)",
     [(TQ, "TOL_OP = 1e-5\n", "TOL_OP = 1e-2\n")]),
]
TESTS = ["tests.test_qwen4exp_quant_parity", "tests.test_glm5next_quant_parity", "tests.test_quant_ref"]


def load(path):
    m = json.load(open(path + ".json"))
    return m, np.fromfile(path + ".bin", dtype=np.float32 if m["type"] == "f32" else np.int32)


def d1(root):            # Q4_K_M pf64: one element of ffn_moe_down (layer 2) scaled by 1 + 2^-8
    p = os.path.join(root, "Q4_K_M", "pf64", "g000", "ffn_moe_down-2__0")
    m, a = load(p)
    a[1000] *= np.float32(1 + 2.0 ** -8)
    a.tofile(p + ".bin")


def d2(root):            # Q8_0 ub10 graph 3: token 4 of layer 5 routed to another expert set (ids rotated by one)
    p = os.path.join(root, "Q8_0", "ub10", "g003", "ffn_moe_topk-5__0")
    m, a = load(p)
    a = a.reshape(-1, m["ne"][0])
    a[4] = (a[4] + 1) % 8
    a.tofile(p + ".bin")


def d3(root):            # Q4_K_M decode graph 5: one element of hc_init one f32 spacing off
    p = os.path.join(root, "Q4_K_M", "decode", "g005", "hc_init__0")
    m, a = load(p)
    a[17] = np.nextafter(a[17], np.float32(np.inf))
    a.tofile(p + ".bin")


def d4(root):            # Q8_0 decode graph 7: result_output as a build without activation quantization would give it
    sys.path.insert(0, os.path.join(llama, "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(os.path.join(root, "Q8_0", "model.gguf"))
    t = next(t for t in r.tensors if t.name == "output.weight")
    w = dequantize(t.data, t.tensor_type).astype(np.float64)
    g = os.path.join(root, "Q8_0", "decode", "g007")
    _, x = load(os.path.join(g, "result_norm__0"))
    p = os.path.join(g, "result_output__0")
    m, _ = load(p)
    (x.astype(np.float64).reshape(-1, w.shape[1]) @ w.T).astype(np.float32).tofile(p + ".bin")


def d5(root):            # glm5-next Q4_K_M decode graph 10: one element of the DSA attention output (L7) 2^-6 too large
    p = os.path.join(root, "glm5next", "Q4_K_M", "decode", "g010", "kqv_out-7__0")
    m, a = load(p)
    a[30] *= np.float32(1 + 2.0 ** -6)
    a.tofile(p + ".bin")


def d6(root):            # glm5-next Q8_0 pf64: one element of kda_g2 (L4) shifted by 0.5 (the gate of kda_out)
    p = os.path.join(root, "glm5next", "Q8_0", "pf64", "g000", "kda_g2-4__0")
    m, a = load(p)
    a[777] += np.float32(0.5)
    a.tofile(p + ".bin")


def d7(root):            # glm5-next Q4_K_M ub10 graph 2: one element of the KDA input attn_norm (L1) 2^-6 too large
    p = os.path.join(root, "glm5next", "Q4_K_M", "ub10", "g002", "attn_norm-1__0")
    m, a = load(p)
    a[300] *= np.float32(1 + 2.0 ** -6)
    a.tofile(p + ".bin")


def d8(root):            # glm5-next Q8_0 decode graph 3: one element of inp_embd one f32 spacing off
    p = os.path.join(root, "glm5next", "Q8_0", "decode", "g003", "inp_embd__0")
    m, a = load(p)
    a[100] = np.nextafter(a[100], np.float32(np.inf))
    a.tofile(p + ".bin")


DUMPS = [("D1 llama side: one expert output element 2^-8 too large (Q4_K_M pf64, ffn_moe_down L2)", "Q4_K_M", "pf64", d1),
         ("D2 llama side: one token routed to other experts than the products used (Q8_0 ub10 g3, L5)", "Q8_0", "ub10",
          d2),
         ("D3 llama side: one embedding element one f32 spacing off (Q4_K_M decode g5, hc_init)", "Q4_K_M", "decode",
          d3),
         ("D4 llama side: result_output computed without activation quantization (Q8_0 decode g7)", "Q8_0", "decode",
          d4),
         ("D5 llama side, glm5-next: one DSA attention output element 2^-6 too large (Q4_K_M decode g10, kqv_out L7)",
          "glm5next/Q4_K_M", "decode", d5),
         ("D6 llama side, glm5-next: one KDA output-gate element off by 0.5 (Q8_0 pf64, kda_g2 L4)", "glm5next/Q8_0",
          "pf64", d6),
         ("D7 llama side, glm5-next: one KDA input element 2^-6 too large (Q4_K_M ub10 g2, attn_norm L1)",
          "glm5next/Q4_K_M", "ub10", d7),
         ("D8 llama side, glm5-next: one embedding element one f32 spacing off (Q8_0 decode g3, inp_embd)",
          "glm5next/Q8_0", "decode", d8)]


def run(tree, dumps=art, gguf_py=None):
    env = dict(os.environ, QWEN4EXP_QUANT_DUMP=dumps, GLM5NEXT_QUANT_DUMP=os.path.join(dumps, "glm5next"),
               LLAMA_CPP_GGUF_PY=gguf_py or os.path.join(llama, "gguf-py"), PARITY_LLAMA_COMMIT=commit)
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


def make_gguf_py(name):
    dst = os.path.join(scratch, name + "-gguf-py")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(llama, "gguf-py"), dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def make_dumps(name, qtype, run_name):
    """A dump root of symlinks, with a copy of the one run directory <qtype>/<run_name> (qtype "Q8_0", "Q4_K_M",
    "glm5next/Q8_0" or "glm5next/Q4_K_M")."""
    root = os.path.join(scratch, name + "-dumps")
    shutil.rmtree(root, ignore_errors=True)
    for qt in ("Q8_0", "Q4_K_M", "glm5next/Q8_0", "glm5next/Q4_K_M"):
        os.makedirs(os.path.join(root, qt))
        for entry in os.listdir(os.path.join(art, qt)):
            src = os.path.join(art, qt, entry)
            if qt == qtype and entry == run_name:
                shutil.copytree(src, os.path.join(root, qt, entry))
            else:
                os.symlink(src, os.path.join(root, qt, entry))
    return root


def report(label, rc, summary, failed):
    print(f"{label}\n    rc={rc} {' | '.join(summary)}")
    by = {}
    for t, mod, cls in failed:
        by.setdefault(mod.replace("_quant_parity", "").replace("quant_ref", "quant_ref (property)"), []).append(
            f"{cls}.{t}")
    for mod, items in sorted(by.items()):
        print(f"    {mod}: " + ", ".join(sorted(items)))


print(f"llama.cpp {commit}; dumps {art}")
rc, summary, failed = run(make_tree("baseline"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = 0
for label, edits in MUTANTS:
    name = label.split()[0]
    tree = make_tree(name)
    gpy = make_gguf_py(name) if any(p.startswith("gguf/") for p, _, _ in edits) else None
    for p2, o2, n2 in edits:
        fp = os.path.join(gpy if p2.startswith("gguf/") else tree, p2)
        src = open(fp).read()
        assert src.count(o2) == 1, (label, p2, o2)
        open(fp, "w").write(src.replace(o2, n2))
    rc, summary, failed = run(tree, gguf_py=gpy)
    caught += rc != 0
    report(label, rc, summary, failed)
    if gpy:
        shutil.rmtree(gpy)
tree = make_tree("dumps")
for label, qtype, run_name, fn in DUMPS:
    root = make_dumps(label.split()[0], qtype, run_name)
    fn(root)
    rc, summary, failed = run(tree, root)
    caught += rc != 0
    report(label, rc, summary, failed)
    shutil.rmtree(root)
print(f"caught {caught}/{len(MUTANTS) + len(DUMPS)}")
