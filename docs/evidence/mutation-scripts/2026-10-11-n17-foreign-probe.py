#!/usr/bin/env python3
"""N17 finding probe: in the mtp3 run, the cells each head's process call wrote around the end of the 256-cell SWA ring,
then per head-0 row that reads K/V computed for another position (ref/mimo2.py attn["foreign"]): the cells whose layer-6
K/V position differs from the cell's position, and the relative difference of llama.cpp's kqv_out and logits from the
full-sequence oracle (A) and from the cell-accurate oracle (B).
usage: 2026-10-11-n17-foreign-probe.py <repo> <work>/n17 <llama.cpp> [run]"""
import os
import sys

repo, dump, llama = sys.argv[1:4]
run = sys.argv[4] if len(sys.argv) > 4 else "mtp3"
os.environ.update(MIMO2_N17_DUMP=dump, LLAMA_CPP_GGUF_PY=os.path.join(llama, "gguf-py"))
sys.path.insert(0, repo)
import numpy as np  # noqa: E402
from tests import test_mimo2_mtp_heads_parity as T  # noqa: E402

cls = T.MtpHeadsF16FA if run == "fa/mtp3" else T.MtpHeads
cls.setUpClass()
t = cls("test_head0_reads_foreign_kv")
nl = t.hp["n_layer"]
print(f"{run}: process calls whose heads wrote different SWA cells")
cells = {}
for gr in t.graphs:
    if gr["kind"] == "process":
        cells.setdefault((gr["rec"]["tok0"] - gr["head"] * gr["rec"]["n"], gr["p0"], gr["n"]), {})[gr["head"]] = \
            t.B[gr["g"]]["slots"]["swa"]
for (_, p0, n), per in sorted(cells.items(), key=lambda kv: kv[0][1]):
    if len({tuple(v) for v in per.values()}) > 1:
        print(f"  positions {p0}..{p0 + n - 1}: " + "; ".join(f"head {h} cells {per[h]}" for h in sorted(per)))
lo, _, _, _ = t.files()
print("graph kind    head pos  foreign cells (cell: holds pos / layer-6 K,V of pos)       kqv_out vs A    vs B   "
      "logits vs A    vs B")
for gr in t.graphs:
    a = t.B[gr["g"]]["attn"]
    for r in np.nonzero(a["foreign"])[0]:
        fc = np.nonzero(a["foreign_cells"][r])[0]
        desc = ", ".join(f"{c}: {a['kpos'][c]} / {a['kv_pos'][c]}" for c in fc)
        got = t.dump(gr["g"], f"kqv_out-{nl}")[0].reshape(gr["n"], -1)[r]
        wb = a["kqv"][r]
        eb = float(np.abs(got - wb).max() / np.abs(wb).max())
        if gr["kind"] == "process" and gr["kept"][r]:
            wa = t.A[0]["attn"]["kqv"][gr["pos"][r]]
            ea = f"{float(np.abs(got - wa).max() / np.abs(wa).max()):11.3e}"
        else:
            ea = "        n/a"
        if r in gr["out_idx"]:
            lr = lo[gr["out_rows"][list(gr["out_idx"]).index(r)]]
            lb = t.B[gr["g"]]["logits"][list(gr["out_idx"]).index(r)]
            elb = f"{float(np.abs(lr - lb).max() / np.abs(lb).max()):7.1e}"
            if gr["kind"] == "process" and gr["kept"][r]:
                la = t.A[0]["logits"][gr["pos"][r]]
                ela = f"{float(np.abs(lr - la).max() / np.abs(la).max()):11.3e}"
            else:
                ela = "        n/a"
        else:
            ela, elb = "        n/a", "    n/a"
        print(f"{gr['g']:5d} {gr['kind']:7s} {gr['head']:4d} {gr['pos'][r]:4d}  {desc:52s} {ea} {eb:7.1e} {ela} {elb}")
