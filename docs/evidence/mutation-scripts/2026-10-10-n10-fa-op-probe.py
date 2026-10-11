#!/usr/bin/env python3
"""N10 first probe (gap 3): ref/fattn.py on llama.cpp's own flash-attention inputs of the qwen4exp QSA layer 3 (the dumped
rotated q, the K / V cache views the attention read, the cells of the dumped selection mask) against the dumped attention
output, in each emulation mode. Normwise relative error, the largest error in binary16 spacings of the output element,
the share of elements more than half a spacing off; and whether the cache views hold binary16 values (0.0 = yes).
Usage: 2026-10-10-n10-fa-op-probe.py <run_parity_f16fa.sh work dir>/q <run> [max graph]
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from ref import fattn as F  # noqa: E402

root, run = pathlib.Path(sys.argv[1]), sys.argv[2]
g_max = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def load(g, name):
    d = root / run / f"g{g:03d}"
    k = 0
    while (d / f"{name}__{k}.json").exists():
        m = json.loads((d / f"{name}__{k}.json").read_text())
        yield m, np.fromfile(d / f"{name}__{k}.bin", dtype=np.float32).astype(np.float64)
        k += 1


man = json.loads((root / run / "manifest.json").read_text())
for g in range(min(len(man["graphs"]), g_max + 1)):
    il = 3
    q = [a.reshape(m["ne"][2], 4, 16) for m, a in load(g, f"Qcur-{il}") if m["ne"][1] == 4][0]
    T = q.shape[0]
    kc = [a.reshape(-1, 2, 16) for m, a in load(g, f"cache_k_l{il} (view)") if m["ne"][0] == 16 and m["ne"][1] == 2][0]
    vc = [a.reshape(-1, 2, 16) for m, a in load(g, f"cache_v_l{il} (view)") if m["ne"][0] == 16 and m["ne"][1] == 2][0]
    mask = next(load(g, f"kq_mask_qsa-{il}"))[1].reshape(T, -1)
    out = next(load(g, f"attn_pregate-{il}"))[1].reshape(T, 4, 16)
    cells = [np.where(np.isfinite(mask[t]) & (mask[t] > -1e30))[0] for t in range(T)]
    rep = (float(np.abs(kc - F.f16(kc)).max()), float(np.abs(vc - F.f16(vc)).max()))
    for mode in ("fa_chunk", "fa_tiled", "exact"):
        want = F.attend(q, kc, vc, cells, 1 / np.sqrt(16), mode)
        d = np.abs(out - want)
        ul = d / F.ulp16(want)
        print(f"g{g} L{il} T{T} {mode:9s} rel {d.max() / np.abs(want).max():.3e} max spacings {ul.max():.2f} "
              f"share > 0.5 spacing {np.mean(ul > 0.5):.4f}; K, V views minus f16(K, V): {rep[0]}, {rep[1]}")
