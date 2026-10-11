#!/usr/bin/env python3
"""N17: compare the dumps of tools/tinygen/run_parity_mimo2_n17.sh at the two llama.cpp pins, file by file (every
.bin of every graph directory, logits.bin, h_nextn.bin, h_in.bin), per run (mtp3 and fa/mtp3 also their MTP context).
usage: 2026-10-11-n17-pindiff.py <work-ec>/n17 <work-r1>/n17"""
import filecmp
import os
import sys

a_root, b_root = sys.argv[1:3]
RUNS = ["mtp3", "mtp3/mtp", "fa/mtp3", "fa/mtp3/mtp", "fa/pf64", "fa/ub64", "fa/decode", "fa/rollback", "fa/hd192"]
print("== pin dump comparison")
for run in RUNS:
    a, b = os.path.join(a_root, run), os.path.join(b_root, run)
    fa = {os.path.relpath(os.path.join(d, f), a) for d, _, fs in os.walk(a) for f in fs if f.endswith(".bin")
          and os.path.relpath(d, a).count(os.sep) == 0 and (d == a or os.path.basename(d).startswith("g"))}
    fb = {os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs if f.endswith(".bin")
          and os.path.relpath(d, b).count(os.sep) == 0 and (d == b or os.path.basename(d).startswith("g"))}
    both = sorted(fa & fb)
    diff = [f for f in both if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)]
    only = sorted({os.path.basename(f).split("__")[0] for f in fa ^ fb})
    top = [f for f in both if os.sep not in f]
    print(f"{run:12s} files in both {len(both):6d}, differing {len(diff):4d}; only at one pin {len(fa ^ fb):4d} "
          f"{only}; {', '.join(top)} differing: {[f for f in diff if os.sep not in f] or 'none'}")
