#!/usr/bin/env python3
"""Mutation check of the state carry of the qwen4exp oracle (N6): single-point mutations of the incremental path of
ref/qwen4exp.py (S1-S12) and of its MTP block (T1-T5); each mutant must make tests.test_qwen4exp_decode_parity
(per-graph llama.cpp dumps of several ubatches, decode and rollback), tests.test_qwen4exp_mtp_parity (the MTP context)
and/or tests.test_qwen4exp_chunked_ref fail. The repo is never edited: ref/ and the
test files are copied into <scratch dir>/<mutant> and mutated there.

usage: python3 tools/tinygen/mutate_qwen4exp_decode.py <repo root> <n6 dump root> <gguf> <llama.cpp gguf-py> <scratch>
(n6 dump root and gguf: the artifacts printed by tools/tinygen/run_parity_qwen4exp_decode.sh: <artifacts>/n6 and
<artifacts>/qwen4exp.gguf; the MTP model is taken as qwen4exp_mtp.gguf next to the gguf).
Results: docs/evidence/2026-10-10-qwen4exp-decode-parity.md."""
import os
import re
import shutil
import subprocess
import sys

wt, dump, gguf, gguf_py, scratch = sys.argv[1:6]
Q = "ref/qwen4exp.py"
MUTANTS = [
    ("S1 GDN: delta-rule state reset at every ubatch / call", Q,
     "out, state = gated_delta_rule(q, k, v, g, beta, state=state_in, normalize_qk=True, eps=eps)",
     "out, state = gated_delta_rule(q, k, v, g, beta, state=None, normalize_qk=True, eps=eps)"),
    ("S2 GDN: conv history of the earlier ubatches dropped (zeros)", Q,
     "conv_raw, conv_hist = short_conv_hist(qkv, w[\"ssm_conv1d.weight\"], conv_in)",
     "conv_raw, conv_hist = short_conv_hist(qkv, w[\"ssm_conv1d.weight\"], np.zeros_like(conv_in))"),
    ("S3 PLE: conv history of the earlier ubatches dropped (zeros)", Q,
     "full = np.vstack([hist_in, normed])", "full = np.vstack([np.zeros_like(hist_in), normed])"),
    ("S4 PLE: predecessors from earlier ubatches off by one (newest cached token skipped)", Q,
     "prefix = [int(t) for t in history[max(0, len(history) - n_prev):]]",
     "prefix = [int(t) for t in history[max(0, len(history) - n_prev - 1):len(history) - 1]]"),
    ("S5 QSA: pooled keys of earlier ubatches / calls forgotten (zeros)", Q,
     "idxc[\"pooled\"] = np.concatenate([idxc[\"pooled\"], new])",
     "idxc[\"pooled\"] = np.concatenate([np.zeros_like(idxc[\"pooled\"]), new])"),
    ("S6 QSA: a new pool rotated to its index within the ubatch, not its sequence position", Q,
     "new = imrope(new[:, None, :], pool_pos4(b1, kp)[b0:], **rope)[:, 0]",
     "new = imrope(new[:, None, :], pool_pos4(b1 - b0, kp), **rope)[:, 0]"),
    ("S7 KV cache: dense attention forgets the cells of earlier ubatches / calls", Q,
     "cells = [list(range(0, t + 1)) for t in np.asarray(pos).tolist()]",
     "cells = [list(range(int(pos[0]), t + 1)) for t in np.asarray(pos).tolist()]"),
    ("S8 RoPE: positions restart at 0 in every ubatch / call", Q,
     "p4 = text_pos4_at(pos)", "p4 = text_pos4_at(pos - pos[0])"),
    ("S9 QSA: incomplete tail taken relative to the ubatch", Q,
     "for r, t in enumerate(np.asarray(pos).tolist()):", "for r, t in enumerate(range(len(pos))):"),
    ("S10 QSA: pool visibility relative to the ubatch", Q,
     "visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.asarray(pos)[:, None] + 1)",
     "visible = (np.arange(pooled.shape[0])[None, :] + 1) * kpool <= (np.arange(len(pos))[:, None] + 1)"),
    ("S11 rollback: delta-rule state and conv history not rolled back (state after the rejected tokens)", Q,
     "    cache.clear()\n", "    fresh[\"gdn\"] = cache[\"gdn\"]\n    cache.clear()\n"),
    ("S12 rollback: PLE conv history not rolled back", Q,
     "    cache.clear()\n", "    fresh[\"ple\"] = cache[\"ple\"]\n    cache.clear()\n"),
    ("T1 MTP: eh_proj input concatenated as [h ; e] instead of [e ; h]", Q,
     "res = np.concatenate([e_rep, h_norm], axis=-1) @ np.asarray(w[\"nextn.eh_proj.weight\"]).T",
     "res = np.concatenate([h_norm, e_rep], axis=-1) @ np.asarray(w[\"nextn.eh_proj.weight\"]).T"),
    ("T2 MTP: h normed over the whole hc-wide row instead of per stream", Q,
     "h_norm = grouped_rms_norm(np.asarray(h_prev, dtype=np.float64).reshape(T, hc, n), w[\"nextn.hnorm.weight\"], eps)",
     "h_norm = rms_norm(np.asarray(h_prev, dtype=np.float64), eps).reshape(T, hc, n) * w[\"nextn.hnorm.weight\"]"),
    ("T3 MTP: embedding norm without its enorm gamma", Q,
     "e = rms_norm(W[\"token_embd.weight\"][tokens], eps, w[\"nextn.enorm.weight\"])",
     "e = rms_norm(W[\"token_embd.weight\"][tokens], eps)"),
    ("T4 MTP: the head mixer uses the trunk's output_hc_* instead of nextn.hc_head_*", Q,
     "head = hc_mix(res, w[\"nextn.hc_head_norm.weight\"], w[\"nextn.hc_head_down.weight\"], w[\"nextn.hc_head_up.weight\"],",
     "head = hc_mix(res, W[\"output_hc_norm.weight\"], W[\"output_hc_down.weight\"], W[\"output_hc_up.weight\"],"),
    ("T5 MTP: the next draft step's h is the projected input, not the block output", Q,
     "out[\"h_nextn\"] = res.reshape(T, hc * n)", "out[\"h_nextn\"] = out[\"eh_proj\"].reshape(T, hc * n)"),
]
TESTS = ["tests.test_qwen4exp_decode_parity", "tests.test_qwen4exp_mtp_parity", "tests.test_qwen4exp_chunked_ref"]


def run(tree):
    env = dict(os.environ, QWEN4EXP_DECODE_DUMP=dump, QWEN4EXP_TINY_GGUF=gguf, LLAMA_CPP_GGUF_PY=gguf_py,
               QWEN4EXP_MTP_GGUF=os.path.join(os.path.dirname(gguf), "qwen4exp_mtp.gguf"))
    p = subprocess.run([sys.executable, "-m", "unittest"] + TESTS, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=1800)
    out = p.stdout + p.stderr
    summary = [ln for ln in out.splitlines() if re.match(r"^(Ran \d+ tests|OK|FAILED)", ln)]
    # "FAIL: test_x (tests.test_qwen4exp_decode_parity.Decode.test_x)" and "ERROR: setUpClass (tests.<module>.<Class>)"
    failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_qwen4exp_(\w+)\.(\w+)[.)]", out, re.M)))
    return p.returncode, summary, failed


def make_tree(name):
    tree = os.path.join(scratch, name)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(os.path.join(wt, "ref"), os.path.join(tree, "ref"))
    os.makedirs(os.path.join(tree, "tests"))
    for f in ("test_qwen4exp_decode_parity.py", "test_qwen4exp_mtp_parity.py", "test_qwen4exp_chunked_ref.py",
              "test_qwen4exp_tiny_parity.py"):
        shutil.copy(os.path.join(wt, "tests", f), os.path.join(tree, "tests", f))
    return tree


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
