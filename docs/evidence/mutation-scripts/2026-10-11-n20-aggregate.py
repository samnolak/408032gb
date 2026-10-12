#!/usr/bin/env python3
"""N20: from a driver log of tools/tinygen/run_parity_qwen4exp_rs_guard.sh (the tables tests/test_qwen4exp_multiseq_parity.py
prints at module teardown, N19 and N20 runs), print the worst relative error over all runs and components, the five
worst components, every divergence line and the QSA tie replays.
usage: python3 docs/evidence/mutation-scripts/2026-10-11-n20-aggregate.py <driver log>"""
import re
import sys

RUNS = ("multi_uni", "multi_str", "cp_uni", "cp_str", "rb_uni", "rb_str", "stale", "cp_pending", "cp_shared_rb",
        "last_call", "extra_cell", "cp_pending_order")
rows = []
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r"^(\w+)\s+(.+?)\s+(-?\d+)\s+(-?\d+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s*$", line)
        if m and m[1] in RUNS:
            rows.append((m[1], m[2].strip(), int(m[3]), int(m[4]), float(m[5]), float(m[6])))
vals = [r for r in rows if not (r[1].startswith("diverged") or r[1].startswith("qsa selection"))]
worst = {}
for run, what, il, g, mx, rel in vals:
    if what not in worst or rel > worst[what][5]:
        worst[what] = (run, what, il, g, mx, rel)
print(f"{len(rows)} table rows from {len({r[0] for r in rows})} runs; worst components:")
for run, what, il, g, mx, rel in sorted(worst.values(), key=lambda r: -r[5])[:5]:
    print(f"  {what:50s} {run:16s} layer {il:2d} graph {g:3d} max abs {mx:.3e} rel {rel:.3e}")
for run, what, il, g, mx, rel in rows:
    if what.startswith("diverged"):
        print(f"  {run}: {what} {mx:.3e} (graph {g})")
print("  tie replays:", sorted({w for r, w, *_ in rows if w.startswith("qsa selection")}))
