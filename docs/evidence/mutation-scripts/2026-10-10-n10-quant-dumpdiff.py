#!/usr/bin/env python3
"""N10 gap 2: compare the dumps of tools/tinygen/run_parity_quant.sh made at two llama.cpp pins, file by file (byte
equality of every .bin present in both), and list the names that differ or exist in one only. Read-only.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-dumpdiff.py <QUANT_WORK pin 1> <QUANT_WORK pin 2>"""
import collections
import filecmp
import os
import sys

a, b = sys.argv[1:3]


def bins(root):
    return {os.path.relpath(os.path.join(r, f), root) for r, _, fs in os.walk(root) for f in fs if f.endswith(".bin")}


fa, fb = bins(a), bins(b)
common = sorted(fa & fb)
diff = [f for f in common if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)]


def name(f):
    return os.path.basename(f).rsplit("__", 1)[0]


print(f".bin files: {len(fa)} and {len(fb)}, {len(common)} in both, {len(diff)} of them differ")
print("differing, by tensor name:", dict(collections.Counter(name(f) for f in diff)))
print("only in the first:", sorted({name(f) for f in fa - fb}))
print("only in the second:", sorted({name(f) for f in fb - fa}))
