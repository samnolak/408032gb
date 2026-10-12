#!/usr/bin/env python3
"""N10 probe (gap 3, N7 finding 7): llama.cpp at the R1 default precision (-ctk f16 -ctv f16 -fa on) against llama.cpp
with f32 caches and flash attention off, on the same tiny model, tokens and call plan (tools/tinygen/run_parity_f16fa.sh
with PARITY_KV / PARITY_FA). Per run: the largest |difference of log-softmax| over the vocabulary and positions, the
greedy tokens that differ, and every discrete choice that differs between the two llama.cpp runs, read from the dumps:
MoE expert sets (ffn_moe_topk-<il>) and the cells the sparse attention kept (the selection masks, or the gathered
indices on the ec7630a glm5-next gather path).
Usage: 2026-10-10-n10-logprob-probe.py <work dir, f16 / fa on> <work dir, f32 / fa off>   (each holds q/ and g/)
"""
import json
import pathlib
import sys

import numpy as np


def dump(run_dir, g, name):
    p = run_dir / f"g{g:03d}"
    m = json.loads((p / f"{name}__0.json").read_text())
    dt = np.float32 if m["type"] == "f32" else np.int32
    return np.fromfile(p / f"{name}__0.bin", dtype=dt).reshape(m["ne"][::-1]), m


def has(run_dir, g, name):
    return (run_dir / f"g{g:03d}" / f"{name}__0.json").exists()


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def cells(run_dir, g, il, n, model):
    """Kept cells per row: from the selection mask, or (glm5-next gather path) the gathered indices with p > 0."""
    if model == "g" and has(run_dir, g, f"indexer_sel_idx-{il}"):
        idx = dump(run_dir, g, f"indexer_sel_idx-{il}")[0].reshape(n, -1)
        p = dump(run_dir, g, f"kq_soft_max_gathered-{il}")[0].reshape(n, -1, idx.shape[1])
        return [tuple(sorted(idx[r][(p[r] > 0).all(0)].tolist())) for r in range(n)]
    m = dump(run_dir, g, f"kq_mask_qsa-{il}" if model == "q" else f"kq_mask_dsa-{il}")[0].reshape(n, -1)
    return [tuple(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist()) for r in range(n)]


def main(a_root, b_root):
    for model, layers_sel in (("q", [3]), ("g", [3, 7, 11])):
        for run in ("pf64", "ub16", "ub10", "ub18", "decode", "rollback"):
            A, B = pathlib.Path(a_root, model, run), pathlib.Path(b_root, model, run)
            if not (A / "manifest.json").exists():
                continue
            man = json.loads((A / "manifest.json").read_text())
            la = np.fromfile(A / "logits.bin", dtype=np.float32)
            lb = np.fromfile(B / "logits.bin", dtype=np.float32)
            n_vocab = json.loads((A / "logits.json").read_text())["n_vocab"]
            la, lb = la.reshape(-1, n_vocab), lb.reshape(-1, n_vocab)
            d = np.abs(log_softmax(la) - log_softmax(lb)).max(1)
            greedy = int((la.argmax(1) != lb.argmax(1)).sum())
            flips, row0 = [], 0
            for g, gi in enumerate(man["graphs"]):
                call = man["calls"][gi["call"]]
                j = g - call["graphs"][0]
                n = min(man["n_ubatch"], call["n"] - j * man["n_ubatch"])
                pos0 = call["pos0"] + j * man["n_ubatch"]
                n_layer = 8 if model == "q" else 12
                for il in range(n_layer):
                    if not has(A, g, f"ffn_moe_topk-{il}"):
                        continue
                    ea = np.sort(dump(A, g, f"ffn_moe_topk-{il}")[0].reshape(n, -1), 1)
                    eb = np.sort(dump(B, g, f"ffn_moe_topk-{il}")[0].reshape(n, -1), 1)
                    for r in np.where((ea != eb).any(1))[0]:
                        flips.append(f"MoE L{il} pos {pos0 + r} (graph {g}): {ea[r].tolist()} vs {eb[r].tolist()}")
                for il in layers_sel:
                    ca, cb = cells(A, g, il, n, model), cells(B, g, il, n, model)
                    for r in range(n):
                        if ca[r] != cb[r]:
                            flips.append(f"{'QSA' if model == 'q' else 'DSA'} L{il} pos {pos0 + r} (graph {g}): "
                                         f"cells differ")
                row0 += n
            print(f"{'qwen4exp' if model == 'q' else 'glm5-next'} {run:8s} max |d log-softmax| {d.max():.3e} "
                  f"(row {int(d.argmax())}), greedy tokens differing {greedy} of {len(d)}, discrete choices "
                  f"differing {len(flips)}")
            for f in flips[:6]:
                print("    " + f)
            if len(flips) > 6:
                print(f"    ... {len(flips) - 6} more")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
