#!/usr/bin/env python3
"""Mutation check of the N17 mimo2 oracle and tests: single-point mutations of ref/mimo2.py (the MTP head offset, the h
each head takes in process and in draft, the per-head seq_rm, the per-layer K/V positions, the f16 store, the attention
scale and the K / V head sizes) and of ref/fattn.py (the sinks of one_chunk, the f16 accumulator, Q to f16, the f32
exp argument, the AVX2 dot), and of the MTP test's rollback replay; each mutant must make tests.test_mimo2_mtp_heads_parity,
tests.test_mimo2_f16fa_parity, tests.test_mimo2_heads_ref and/or tests.test_fattn_sinks_ref fail. The repository is never
edited: ref/ and tests/*.py are copied into <scratch>/<mutant> and mutated there.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n17-mutations.py <repo root> <artifacts> <llama.cpp> <scratch>
(<artifacts>: MIMO2_N17_WORK of tools/tinygen/run_parity_mimo2_n17.sh, holding n17/; <llama.cpp>: the checkout the dumps
came from, for its gguf-py and commit). Results: docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md."""
import glob
import os
import re
import shutil
import subprocess
import sys

wt, art, llama, scratch = sys.argv[1:5]
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
R, FA = "ref/mimo2.py", "ref/fattn.py"
TH = "tests/test_mimo2_mtp_heads_parity.py"
MUTANTS = [
    ("H1 off-by-one head offset: head k computed with layer n_layer + (k + 1) mod H", R,
     "out of range [0, {hp.get('n_nextn')})\")\n    il = hp[\"n_layer\"] + offset\n",
     "out of range [0, {hp.get('n_nextn')})\")\n    il = hp[\"n_layer\"] + (offset + 1) % max(1, hp.get(\"n_nextn\", 1))\n"),
    ("H2 off-by-one head offset in draft: step k runs head min(k + 1, H - 1)", R,
     "out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=i if chain else 0, prec=prec)",
     "out = mtp_forward_chunk(ids, np.vstack(rows), W, hp, cache, offset=min(i + 1, n_heads - 1) if chain else 0, "
     "prec=prec)"),
    ("P1 wrong h source in process: head k > 0 takes head k-1's h_nextn (shifted) instead of the target's", R,
     "        out[head] = mtp_ubatches(tokens, h_rows, W, hp, cache, n_ubatch, head, prec, snap)\n",
     "        rows_k = h_rows if head == 0 else np.vstack([np.asarray(h_rows)[:1]] + "
     "[np.vstack([o[\"h_nextn\"] for o in out[head - 1]])[:-1]])\n"
     "        out[head] = mtp_ubatches(tokens, rows_k, W, hp, cache, n_ubatch, head, prec, snap)\n"),
    ("P2 wrong h source in draft: every row of step k paired with the pending target row", R,
     "ids, rows = [int(id_last)] + fed, list(chain_h)",
     "ids, rows = [int(id_last)] + fed, [pending] * (len(fed) + 1)"),
    ("P3 wrong h source in draft: the new token paired with the h of step k-2 (rows not chained one back)", R,
     "            chain_h.append(h)\n",
     "            chain_h.append(chain_h[-1])\n"),
    ("Q1 process without seq_rm before each head (heads 1.. append after head 0)", R,
     "        if n_heads > 1:\n            cache_truncate(cache, p0)\n",
     "        if False:\n            cache_truncate(cache, p0)\n"),
    ("Q2 draft step without seq_rm(pos0) (the prefix appended after the previous step's)", R,
     "        if chain:\n            cache_truncate(cache, pos0)\n",
     "        if False:\n            cache_truncate(cache, pos0)\n"),
    ("K1 per-layer K/V positions not recorded (every visible cell counted as foreign)", R,
     "        cache[\"P\"][il][idxs] = pos\n", ""),
    ("T1 test: after a rejection the pending h stays the call's last row (accept() ignored)", TH,
     "                    pending = cls.hB_at[p0 - 1]\n", "                    pending = pending\n"),
    ("F1 sink ignored in one_chunk flash attention", FA,
     "    if sink is not None:                                                  # O:9034-9056: on VKQ32 = f32(VKQ16)",
     "    if False:                                                             # O:9034-9056: on VKQ32 = f32(VKQ16)"),
    ("F2 sink applied before the cell loop instead of after it (as a first cell with a zero value row)", FA,
     "    M = np.full(G, -np.inf)\n    S = np.zeros(G, dtype=np.float32)\n",
     "    M = np.full(G, -np.inf) if sink is None else f32(sink)\n"
     "    S = np.zeros(G, dtype=np.float32) if sink is None else np.ones(G, dtype=np.float32)\n    sink = None\n"),
    ("R1 f16 rounding of the one_chunk accumulator skipped (ggml_vec_mad_f16 in f32)", FA,
     "        acc = f16(acc + vc[i][None, :] * vs[:, None])                     # ggml_vec_mad_f16",
     "        acc = acc + vc[i][None, :] * vs[:, None]                          # ggml_vec_mad_f16"),
    ("R2 f16 rounding of the cache store skipped (oracle keeps the f32 K / V)", R,
     "    ks, vs = (F.store(k, prec), F.store(v, prec)) if prec else (k, v)",
     "    ks, vs = (k, v)"),
    ("R3 Q not rounded to f16 in one_chunk", FA,
     "    qq = f16(qh)                                                          # Q_q = from_float(q) to F16",
     "    qq = np.asarray(qh, dtype=np.float64)"),
    ("E1 expf of the float64 difference (the f32 argument of O:8997, 9003 not rounded)", FA,
     "        ms = np.where(new, f32(np.exp(f32(np.where(new, M - s, 0.0)))), 1.0)\n"
     "        vs = np.where(new, 1.0, f32(np.exp(f32(np.where(new, 0.0, s - M)))))\n",
     "        ms = np.where(new, f32(np.exp(np.where(new, M - s, 0.0))), 1.0)\n"
     "        vs = np.where(new, 1.0, f32(np.exp(np.where(new, 0.0, s - M))))\n"),
    ("E2 attention scale float64 1/sqrt(d) instead of llama.cpp's f32 1.0f/sqrtf(d)", R,
     "    return F.kq_scale_f32(hp[\"k_dim\"]) if prec else 1.0 / np.sqrt(hp[\"k_dim\"])",
     "    return 1.0 / np.sqrt(hp[\"k_dim\"])"),
    ("E3 the AVX2 lanes of the f16 dot not modelled (double sum for every head size)", FA,
     "    n_simd = qq.shape[1] & ~(F16_STEP - 1)\n", "    n_simd = 0\n"),
    ("V1 K/V head-size confusion: the attention scale from the V head size (1/sqrt(16) for K 24)", R,
     "    return F.kq_scale_f32(hp[\"k_dim\"]) if prec else 1.0 / np.sqrt(hp[\"k_dim\"])",
     "    return F.kq_scale_f32(hp[\"v_dim\"]) if prec else 1.0 / np.sqrt(hp[\"k_dim\"])"),
    ("V2 K/V head-size confusion: the V rows of the cache sized by the K head", R,
     "        c[\"V\"][il] = np.zeros((size, hp[\"n_head_kv\"][il], hp[\"v_dim\"]))",
     "        c[\"V\"][il] = np.zeros((size, hp[\"n_head_kv\"][il], hp[\"k_dim\"]))"),
]
TESTS = ["tests.test_mimo2_mtp_heads_parity", "tests.test_mimo2_f16fa_parity", "tests.test_mimo2_heads_ref",
         "tests.test_fattn_sinks_ref"]


def run(tree):
    env = dict(os.environ, MIMO2_N17_DUMP=os.path.join(art, "n17"), LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"),
               PARITY_LLAMA_COMMIT=commit)
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
    os.makedirs(os.path.join(tree, "tests"))
    for f in glob.glob(os.path.join(wt, "tests", "*.py")):
        shutil.copy(f, os.path.join(tree, "tests"))
    return tree


print(f"llama.cpp {commit}; dumps {art}/n17")
rc, summary, failed = run(make_tree("baseline"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = 0
for label, path, old, new in MUTANTS:
    tree = make_tree(label.split()[0])
    fp = os.path.join(tree, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    open(fp, "w").write(src.replace(old, new))
    rc, summary, failed = run(tree)
    caught += rc != 0
    by_mod = {}
    for t, mod, cls in failed:
        by_mod.setdefault(mod, set()).add(f"{cls}.{t}" if not t.startswith("setUpClass") else f"{cls}.setUpClass")
    print(f"{label}\n    {path}: {old.strip()[:90]!r} -> {new.strip()[:90]!r}\n    rc={rc} {' | '.join(summary)}")
    for mod in sorted(by_mod):
        names = sorted(by_mod[mod])
        print(f"    {mod}: {len(names)} failing: " + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
    shutil.rmtree(tree, ignore_errors=True)
shutil.rmtree(os.path.join(scratch, "baseline"), ignore_errors=True)
print(f"caught {caught}/{len(MUTANTS)}")
