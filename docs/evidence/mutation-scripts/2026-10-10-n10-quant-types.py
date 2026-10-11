#!/usr/bin/env python3
"""N10 gap 2: per-tensor types of quantized GGUF files as the pinned gguf-py reads them (layer prefix folded, with
the tensor's ggml shape ne and the count), and the sha256 of each file. Read-only.

usage: python3 docs/evidence/mutation-scripts/2026-10-10-n10-quant-types.py <gguf-py> <file.gguf> [...]"""
import collections
import hashlib
import sys

sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader  # noqa: E402

for p in sys.argv[2:]:
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    r = GGUFReader(p)
    c = collections.Counter()
    for t in r.tensors:
        key = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        c[(key, t.tensor_type.name, tuple(int(x) for x in t.shape))] += 1
    print(f"{p.split('/')[-2]}/{p.split('/')[-1]}: {len(r.tensors)} tensors, sha256 {h}")
    for (k, ty, sh), m in sorted(c.items()):
        print(f"  {k:32s} {ty:6s} {str(sh):18s} x{m}")
