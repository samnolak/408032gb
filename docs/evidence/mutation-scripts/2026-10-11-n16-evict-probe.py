#!/usr/bin/env python3
"""N16 probe: the evict run (prefill 280, llama_memory_seq_rm(0, 20, -1), a call of 24, 4 x 1) of
tools/tinygen/run_parity_mimo2_decode.sh. Prints, per re-decoded position, how many cells llama.cpp's first SWA layer
attended to (from the dumped kq_soft_max: probabilities > 0), how many positions the full-sequence oracle (A) attends
to, and the relative difference of kqv_out (that layer) and of the logits from (A) and from the cell-accurate
incremental oracle (B). usage: 2026-10-11-n16-evict-probe.py <repo> <MIMO2_DECODE_DUMP> <gguf-py>"""
import pathlib
import sys

import numpy as np

repo, dump, gguf_py = sys.argv[1:4]
sys.path.insert(0, repo)
import os  # noqa: E402
os.environ["MIMO2_DECODE_DUMP"], os.environ["LLAMA_CPP_GGUF_PY"] = dump, gguf_py
from ref import mimo2 as M  # noqa: E402
from tests import test_mimo2_decode_parity as D  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402

W, hp = D.model(pathlib.Path(dump, "n16.gguf"))
R = Run("evict", dump)
A = M.forward(R.final, W, hp)
cache, B = M.new_cache(hp, D.N_CTX, R.man["n_ubatch"]), {}
by_call = {}
for gr in R.graphs:
    by_call.setdefault(gr["call"], []).append(gr)
for c, call in enumerate(R.calls):
    if call["kind"] == "rollback":
        print(f"before the rollback: SWA cache holds positions {int(cache['swa'].pos.min())}..{int(cache['swa'].pos.max())}"
              f" ({int((cache['swa'].pos >= 0).sum())} of {cache['swa'].size} cells); rollback to {call['pos0']}")
        M.cache_truncate(cache, call["pos0"])
        print(f"after: SWA cache cells in use {int((cache['swa'].pos >= 0).sum())}, base cache "
              f"{int((cache['base'].pos >= 0).sum())}")
        continue
    for gr in by_call[c]:
        B[gr["g"]] = M.forward_chunk(gr["toks"], W, hp, cache)
il = 1
lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, -1).astype(np.float64)


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


print("pos  cells(L1, llama.cpp)  positions(L1, A)  kqv_out L1 vs A  vs B   logits vs A   vs B")
rb = next(c for c, call in enumerate(R.calls) if call["kind"] == "rollback")
for gr in R.graphs:
    if gr["call"] < rb:                     # the prefill before the rollback
        continue
    sm = R.dump(gr["g"], f"kq_soft_max-{il}")[0].reshape(hp["n_head"][il], gr["n"], -1)
    kqv = R.dump(gr["g"], f"kqv_out-{il}")[0].reshape(gr["n"], -1)
    for r, t in enumerate(gr["pos"]):
        if t > 31 and t < 44:
            continue
        b = B[gr["g"]]
        print(f"{t:3d}  {int((sm[0, r] > 0).sum()):20d}  {int((A['layers'][il]['attn']['probs'][0, t] > 0).sum()):16d}"
              f"  {rel(kqv[r], A['layers'][il]['attn']['kqv'][t]):15.3e}  {rel(kqv[r], b['layers'][il]['attn']['kqv'][r]):.1e}"
              f"  {rel(lo[gr['row0'] + r], A['logits'][t]):12.3e}  {rel(lo[gr['row0'] + r], b['logits'][r]):.1e}")
