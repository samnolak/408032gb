#!/usr/bin/env python3
"""Mutation check of the N16 mimo2 oracle and tests (gap 12): single-point mutations of ref/mimo2.py (the window, the SWA
cache size and reuse, the purge, the rollback, the sinks, RoPE base, GQA, value scale, the MTP block), of the MXFP4 part
of ref/quant.py (E8M0 scale, nibble order, activation conversion), of a copy of the pinned gguf-py's MXFP4 dequantize,
and of the MTP test's pairing; each mutant must make tests.test_mimo2_decode_parity, tests.test_mimo2_mtp_parity,
tests.test_mimo2_chunked_ref, tests.test_mimo2_quant_parity and/or tests.test_mxfp4_ref fail. The repository is never
edited: ref/ and tests/*.py are copied into <scratch>/<mutant> and mutated there.

usage: python3 docs/evidence/mutation-scripts/2026-10-11-n16-mimo2-mutations.py <repo root> <artifacts> <llama.cpp> <scratch>
(<artifacts>: MIMO2_DECODE_WORK of tools/tinygen/run_parity_mimo2_decode.sh, holding n16/; <llama.cpp>: the checkout
the dumps came from, for its gguf-py and commit). Results: docs/evidence/2026-10-11-n16-mimo-decode-parity.md."""
import glob
import os
import re
import shutil
import subprocess
import sys

wt, art, llama, scratch = sys.argv[1:5]
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
R, Q = "ref/mimo2.py", "ref/quant.py"
TM = "tests/test_mimo2_mtp_parity.py"
GQ = "gguf-py/gguf/quants.py"            # the copy of the pinned gguf-py inside the mutant tree (G1 only)
MUTANTS = [
    ("W1 window: a key at distance n_swa still visible (<= instead of <)", R,
     "vis &= (qpos - kpos) < n_swa", "vis &= (qpos - kpos) <= n_swa"),
    ("W2 SWA cache without the 256-cell padding (n_swa + n_ubatch cells)", R,
     "size_swa = swa_size or pad(min(n_ctx, hp[\"n_swa\"] + n_ubatch), KV_PAD)",
     "size_swa = swa_size or min(n_ctx, hp[\"n_swa\"] + n_ubatch)"),
    ("W3 SWA cells never reused (find_slot takes empty cells only)", R,
     "if p0 < 0 or self.masked_swa(p0, pmax1):", "if p0 < 0:"),
    ("W4 no purge after an overwrite (KV:1160-1177)", R,
     "if rm_max >= 0 and self.seq_pos_min() <= rm_max:", "if False:"),
    ("W5 RoPE base of the global layers on the sliding-window layers", R,
     "base = hp[\"rope_base_swa\"] if swa else hp[\"rope_base\"]", "base = hp[\"rope_base\"]"),
    ("S1 sinks ignored", R,
     "kq, p, kqv = attend(q, pos, K, V, kpos, w.get(\"attn_sinks.weight\"),",
     "kq, p, kqv = attend(q, pos, K, V, kpos, None,"),
    ("S2 a layer without attn_sinks gets sink logits 0 instead of none", R,
     "sink = np.full(H, -np.inf) if sinks is None", "sink = np.zeros(H) if sinks is None"),
    ("R1 rollback length: the first rolled-back position is kept (seq_rm from n_keep + 1)", R,
     "    cache[\"base\"].seq_rm(n_keep, -1)\n    cache[\"swa\"].seq_rm(n_keep, -1)\n",
     "    cache[\"base\"].seq_rm(n_keep + 1, -1)\n    cache[\"swa\"].seq_rm(n_keep + 1, -1)\n"),
    ("R2 rollback frees the base cache only (the SWA cache keeps the rejected tokens)", R,
     "    cache[\"swa\"].seq_rm(n_keep, -1)\n", ""),
    ("R3 seq_rm does not move the head back", R,
     "if hit.size and hit[0] < self.head:", "if False:"),
    ("V1 value scale ignored", R,
     "out=att * hp[\"v_scale\"] if hp[\"v_scale\"] else att)", "out=att)"),
    ("G1 GQA tiled (h % n_kv) instead of grouped (h // r)", R,
     "grp = np.arange(H) // (H // K.shape[1])", "grp = np.arange(H) % K.shape[1]"),
    ("M1 MTP: eh_proj input [h ; e] instead of [e ; h]", R,
     "x = np.concatenate([e, h], axis=-1)", "x = np.concatenate([h, e], axis=-1)"),
    ("M2 MTP: head norm output_norm instead of layer_output_norm", R,
     "head_norm = w.get(\"nextn.shared_head_norm.weight\", w.get(\"layer_output_norm.weight\", W[\"output_norm.weight\"]))",
     "head_norm = W[\"output_norm.weight\"]"),
    ("M3 test: MTP token p paired with the target h of p instead of p-1 (full oracle)", TM,
     "h_prev = np.vstack([np.zeros((1, n_embd)), cls.hA[:-1]])", "h_prev = cls.hA"),
    ("X1 MXFP4: E8M0 scale 2^(e-127) instead of 2^(e-128)", Q,
     "(e - np.uint32(1)) << np.uint32(23)", "e << np.uint32(23)"),
    ("X2 MXFP4: high nibble first", Q,
     "return np.concatenate([qs & np.uint8(0x0F), qs >> np.uint8(4)], axis=-1)",
     "return np.concatenate([qs >> np.uint8(4), qs & np.uint8(0x0F)], axis=-1)"),
    ("X3 MXFP4: activations not converted to Q8_0 (vec_dot_type F32)", Q,
     "\"IQ4_NL\": \"Q8_0\", \"MXFP4\": \"Q8_0\",", "\"IQ4_NL\": \"Q8_0\", \"MXFP4\": None,"),
    ("P1 gguf-py copy: MXFP4 E8M0 scale 2^(e-127) in the dequantize the oracle's weights come from", GQ,
     "np.uint32(x - 1) << np.uint32(23)", "np.uint32(x) << np.uint32(23)"),
]
TESTS = ["tests.test_mimo2_decode_parity", "tests.test_mimo2_mtp_parity", "tests.test_mimo2_chunked_ref",
         "tests.test_mimo2_quant_parity", "tests.test_mxfp4_ref"]


def run(tree, gguf_py):
    env = dict(os.environ, MIMO2_DECODE_DUMP=os.path.join(art, "n16"), LLAMA_CPP_GGUF_PY=gguf_py,
               PARITY_LLAMA_COMMIT=commit)
    p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=3600)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_(\w+)\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name, with_gguf_py=False):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"), ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tree, "tests"))
    for f in glob.glob(os.path.join(wt, "tests", "*.py")):
        shutil.copy(f, os.path.join(tree, "tests"))
    if with_gguf_py:
        shutil.copytree(os.path.join(llama, "gguf-py"), os.path.join(tree, "gguf-py"),
                        ignore=shutil.ignore_patterns("__pycache__"))
    return tree


print(f"llama.cpp {commit}; dumps {art}/n16")
rc, summary, failed = run(make_tree("baseline"), os.path.join(llama, "gguf-py"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = 0
for label, path, old, new in MUTANTS:
    tree = make_tree(label.split()[0], path == GQ)
    fp = os.path.join(tree, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    open(fp, "w").write(src.replace(old, new))
    rc, summary, failed = run(tree, os.path.join(tree, "gguf-py") if path == GQ else os.path.join(llama, "gguf-py"))
    caught += rc != 0
    by_mod = {}
    for t, mod, cls in failed:
        by_mod.setdefault(mod, set()).add(f"{cls}.{t}" if not t.startswith("setUpClass") else f"{cls}.setUpClass")
    print(f"{label}\n    {path}: {old.strip()[:90]!r} -> {new.strip()[:90]!r}\n    rc={rc} {' | '.join(summary)}")
    for mod in sorted(by_mod):
        names = sorted(by_mod[mod])
        print(f"    {mod}: {len(names)} failing: " + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
print(f"caught {caught}/{len(MUTANTS)}")
