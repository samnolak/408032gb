#!/usr/bin/env python3
"""N10 gap 2 probe: the glm5-next KDA output projection (attn_output of gated = kda_normed * sigmoid(kda_g2), an unnamed
tensor the test must recompute) for one layer and graph, with the gate recomputed two ways: in float64 then rounded to
f32 once, and in f32 as ggml computes it (1 / (1 + exp(-x)) and the product in f32). For each: elements of kda_out
beyond ref/quant.py flip_bound at several windows (f32 spacings), the rows that differ from llama.cpp beyond TOL_OP, and
for the worst row of the float64 version the least-squares input change that explains its difference, in quantization
steps (an f16 block-scale flip of Q8_0 spreads over the 32 elements of the block in proportion to |q|). Read-only.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-gate-probe.py <repo root> <QUANT_WORK>/glm5next
       <gguf-py> <type> <run> <layer> <graph>"""
import sys

import numpy as np

wt, work, gguf_py, qt, run, il, g = sys.argv[1:8]
il, g = int(il), int(g)
sys.path.insert(0, wt)
from ref import glm5next as G, quant as QT  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402

sys.path.append(gguf_py)
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

W = QT.load(GGUFReader(f"{work}/{qt}/model.gguf"), dequantize, True)
R = Run(run, f"{work}/{qt}")
n = next(x for x in R.graphs if x["g"] == g)["n"]


def D(name):
    return R.dump(g, name)[0].reshape(n, -1)


normed, g2, got = D(f"kda_normed-{il}"), D(f"kda_g2-{il}"), D(f"kda_out-{il}")
w = G.layer_weights(W, il)["attn_output.weight"]
g32 = g2.astype(np.float32)
for label, gated in (("float64 gate rounded to f32", (normed * G.sigmoid(g2)).astype(np.float32)),
                     ("f32 gate as ggml", normed.astype(np.float32) * (np.float32(1) / (np.float32(1) + np.exp(-g32))))):
    want = gated @ w.T
    sc = np.abs(QT.act_quant(gated, w.vdt)) @ np.abs(w.w).T
    diff = np.abs(got - want)
    tol = 1e-5 * sc.max() + 1e-5 * np.abs(want).max()
    for ul in (8, 512, 4096):
        print(f"{qt} {run} L{il} g{g} {label}: elements beyond flip_bound({ul} spacings) + TOL_OP: "
              f"{int((diff > QT.flip_bound(gated, w, ul) + tol).sum())}")
    rows = np.where(diff.max(1) > 1e-5 * np.abs(want).max())[0]
    print(f"  rows beyond TOL_OP: {rows.tolist()}, max rel {float(diff.max() / np.abs(want).max()):.2e}")
gated = (normed * G.sigmoid(g2)).astype(np.float32)
r0 = int(np.argmax(np.abs(got - gated @ w.T).max(1)))
dx = np.linalg.lstsq(w.w, (got - gated @ w.T)[r0], rcond=None)[0]
amax = np.abs(gated[r0].reshape(-1, 32 if w.vdt == "Q8_0" else 256)).max(1)
blk = 32 if w.vdt == "Q8_0" else 256
big = np.argsort(-np.abs(dx))[:6]
print(f"  float64 gate, row {r0}: largest input changes (element: change in steps of its block) " +
      ", ".join(f"{int(j)}: {float(dx[j] / (amax[j // blk] / 127)):+.3f}" for j in big))
