#!/usr/bin/env python3
"""N12 probe (not a mutation check, after the audit of aaf7757): the two counts of rows where ik attends cell 0 only
through its tail padding. tests/test_ik_glm5next_parity.py counts rows with 0 in the (replayed) cells and 0 not in
the oracle's own reference choice (ref_cells); 2026-10-11-n12-deviations.py counts rows of ik's attended cells with 0,
0 not in the token's tail and pool 0 not selected. Prints both and the rows only the second one counts. Run from the
repository root with the environment of the test (IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY)."""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import test_ik_glm5next_parity as T  # noqa: E402
from ref import glm5next as G  # noqa: E402

c = T.IkP64
c.setUpClass()
kp = c.hp["kpool"]
test_n = probe_n = 0
extra = []
for gr in c.R.graphs:
    for il in c.DSA:
        ob = c.B[gr["g"]]["layers"][il]["dsa"]
        for r, t in enumerate(gr["pos"]):
            cells, ref, seen = ob["cells"][r], ob["ref_cells"][r], c.seen[il][t]
            a = 0 in cells and 0 not in ref
            b = 0 in seen and 0 not in G.tail_cells(t, kp) and not set(range(kp)) <= set(seen)
            test_n += a
            probe_n += b
            if b and not a:
                extra.append((il, t, "pool 0 in the oracle's own choice" if set(range(kp)) <= set(ref) else "other",
                              "replayed tie" if t in c.ov.get(("dsa", il), {}) else "not replayed"))
print(f"p64: test count {test_n}, deviations-probe count {probe_n} of {len(c.DSA) * 64} row-layers; "
      f"rows only the probe counts: {len(extra)}")
for e in extra:
    print("  layer %d, position %d: %s, %s" % e)
