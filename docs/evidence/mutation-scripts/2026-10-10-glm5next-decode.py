#!/usr/bin/env python3
"""Mutation check of the glm5-next oracle's state carry and of its parity tests (N9): single-point mutations of the
incremental path of ref/glm5next.py (S1-S10), of its MTP block (T1-T2) and of the tests' pin-dependent rules and
pairing (U1-U3); each mutant must make tests.test_glm5next_decode_parity (per-graph llama.cpp dumps of several
ubatches, decode and rollback), tests.test_glm5next_mtp_parity (the MTP context, where the pin has the graph) and/or
tests.test_glm5next_chunked_ref (numpy-only continuity) fail. The repository is never edited: ref/ and tests/ are
copied into <scratch>/<mutant> and mutated there.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-glm5next-decode.py <repo root> <artifacts> <llama.cpp checkout>
       <scratch>
(<artifacts>: the directory tools/tinygen/run_parity_glm5next_decode.sh printed / GLM5NEXT_DECODE_WORK, holding n9/,
glm5next.gguf and glm5next_mtp.gguf; <llama.cpp checkout>: the one the dumps came from, for gguf-py and its commit).
Results: docs/evidence/2026-10-10-glm5next-decode-parity.md."""
import os
import re
import shutil
import subprocess
import sys

wt, art, llama, scratch = sys.argv[1:5]
commit = subprocess.run(["git", "-C", llama, "rev-parse", "HEAD"], capture_output=True, text=True,
                        check=True).stdout.strip()
R = "ref/glm5next.py"
TD = "tests/test_glm5next_decode_parity.py"
TM = "tests/test_glm5next_mtp_parity.py"
MUTANTS = [
    ("S1 KDA: state reset at every ubatch / call", R,
     "state=st[\"state\"], normalize_qk=True,", "state=None, normalize_qk=True,"),
    ("S2 KDA: conv history of the earlier ubatches dropped (zeros)", R,
     "raw, hist[c] = conv_hist(proj, cw, st[\"conv\"][c])",
     "raw, hist[c] = conv_hist(proj, cw, np.zeros_like(st[\"conv\"][c]))"),
    ("S3 KDA: conv history keeps the oldest rows instead of the newest", R,
     "return out, pad[pad.shape[0] - (K - 1):]", "return out, pad[:K - 1]"),
    ("S4 DSA: MLA latents of the earlier ubatches / calls forgotten (zeros)", R,
     "st[key] = np.concatenate([st[key], r[key]])",
     "st[key] = np.concatenate([np.zeros_like(st[key]) if key == \"latent\" else st[key], r[key]])"),
    ("S5 DSA: pooled keys of the earlier ubatches / calls forgotten (zeros)", R,
     "st[\"pooled\"] = np.concatenate([st[\"pooled\"], new.reshape(-1, hp[\"idx_dim\"])])",
     "st[\"pooled\"] = np.concatenate([np.zeros_like(st[\"pooled\"]), new.reshape(-1, hp[\"idx_dim\"])])"),
    ("S6 DSA: a pool straddling two ubatches loses the gate rows of its earlier members (zeros)", R,
     "new = indexer_pools(st[\"ik\"][b0 * kp:b1 * kp], st[\"ig\"][b0 * kp:b1 * kp],",
     "new = indexer_pools(st[\"ik\"][b0 * kp:b1 * kp], np.vstack([np.zeros((max(0, int(pos[0]) - b0 * kp), "
     "st[\"ig\"].shape[1])), st[\"ig\"][max(int(pos[0]), b0 * kp):b1 * kp]]),"),
    ("S7 DSA: incomplete tail taken relative to the ubatch", R,
     "for r, t in enumerate(np.asarray(pos).tolist()):", "for r, t in enumerate(range(len(pos))):"),
    ("S8 DSA: pool visibility relative to the ubatch", R,
     "visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)",
     "visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(len(pos))[:, None] + 1)"),
    ("S9 rollback: KDA conv history and state not rolled back (state after the rejected tokens)", R,
     "    cache.clear()\n", "    fresh[\"kda\"] = cache[\"kda\"]\n    cache.clear()\n"),
    ("S10 rollback: MLA latent / indexer cache not rolled back (rejected cells kept)", R,
     "    cache.clear()\n", "    fresh[\"dsa\"] = cache[\"dsa\"]\n    cache.clear()\n"),
    ("T1 MTP: eh_proj input concatenated as [h ; e] instead of [e ; h]", R,
     "x = np.concatenate([e, h], axis=-1)", "x = np.concatenate([h, e], axis=-1)"),
    ("T2 MTP: FFN residual taken from the normed input instead of eh_proj", R,
     "out[\"ffn_inp\"] = f = blk[\"out\"] + x", "out[\"ffn_inp\"] = f = blk[\"out\"] + an"),
    ("U1 test: gather-path tail slot live one cell too far (k <= n_tail)", TD,
     "live[n_top * kpool + k] = k < (t + 1) % kpool", "live[n_top * kpool + k] = k <= (t + 1) % kpool"),
    ("U2 test: k-pool padding rule of the other pin", TD,
     "SHARED_PAD_CELL = EC7630A", "SHARED_PAD_CELL = not EC7630A"),
    ("U3 test: MTP token p paired with the target h of p instead of p-1", TM,
     "cls.h_pair = np.vstack([np.zeros((1, E)), cls.H[:n_proc - 1]])", "cls.h_pair = cls.H[:n_proc]"),
]
TESTS = ["tests.test_glm5next_decode_parity", "tests.test_glm5next_mtp_parity", "tests.test_glm5next_chunked_ref"]


def run(tree):
    env = dict(os.environ, GLM5NEXT_DECODE_DUMP=os.path.join(art, "n9"),
               GLM5NEXT_TINY_GGUF=os.path.join(art, "glm5next.gguf"),
               GLM5NEXT_MTP_GGUF=os.path.join(art, "glm5next_mtp.gguf"),
               LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"), PARITY_LLAMA_COMMIT=commit)
    p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=1800)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_glm5next_(\w+)\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tree, "tests"))
    for f in ("test_glm5next_decode_parity.py", "test_glm5next_mtp_parity.py", "test_glm5next_chunked_ref.py"):
        shutil.copy(os.path.join(wt, "tests", f), os.path.join(tree, "tests", f))
    return tree


print(f"llama.cpp {commit}; dumps {art}/n9")
rc, summary, failed = run(make_tree("baseline"))
print(f"baseline (no mutation): rc={rc} {' | '.join(summary)}")
if rc != 0:
    sys.exit("baseline must pass; fix that first")
caught = caught_by_dumps = 0
for label, path, old, new in MUTANTS:
    tree = make_tree(label.split()[0])
    fp = os.path.join(tree, path)
    src = open(fp).read()
    assert src.count(old) == 1, (label, old)
    open(fp, "w").write(src.replace(old, new))
    rc, summary, failed = run(tree)
    caught += rc != 0
    dump_fail = sorted({f"{cls}.{t}" for t, mod, cls in failed if mod in ("decode_parity", "mtp_parity")})
    prop_fail = sorted({t for t, mod, cls in failed if mod == "chunked_ref"})
    caught_by_dumps += bool(dump_fail)
    print(f"{label}\n    {path}: {old.strip()!r} -> {new.strip()!r}\n    rc={rc} {' | '.join(summary)}")
    print("    dump parity failing: " + (", ".join(dump_fail) or "none"))
    print("    property failing: " + (", ".join(prop_fail) or "none"))
print(f"caught {caught}/{len(MUTANTS)}; caught by the dump parity tests alone {caught_by_dumps}/{len(MUTANTS)}")
