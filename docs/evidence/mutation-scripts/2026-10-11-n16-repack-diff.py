#!/usr/bin/env python3
"""N16 probe: the MXFP4 runs of tools/tinygen/run_parity_mimo2_decode.sh on the repacked AVX2 path (pf64, decode) against
the same runs with --no-repack (pf64-nr, decode-nr), tensor file by tensor file, and their logits.
usage: 2026-10-11-n16-repack-diff.py <MIMO2_DECODE_DUMP>/mxfp4"""
import filecmp
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
for r in ("pf64", "decode"):
    a, b = root / r, root / f"{r}-nr"
    fs = sorted(p.relative_to(a) for p in a.glob("g*/*.bin"))
    diff = [p for p in fs if not filecmp.cmp(a / p, b / p, shallow=False)]
    by = {}
    for p in diff:
        nm = p.name.split("__")[0].rsplit("-", 1)[0]
        by[nm] = by.get(nm, 0) + 1
    la = np.fromfile(a / "logits.bin", dtype=np.float32)
    lb = np.fromfile(b / "logits.bin", dtype=np.float32)
    print(f"{r} vs {r}-nr: {len(fs)} tensor files, {len(diff)} differ; by name: {dict(sorted(by.items()))}; "
          f"logits max |diff| {np.abs(la - lb).max():.3e}")
