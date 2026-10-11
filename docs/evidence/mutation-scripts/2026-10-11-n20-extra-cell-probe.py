#!/usr/bin/env python3
"""N20: which recurrent rows the extra_cell run reads, per graph, from the oracle replay that the dump tests check
against llama.cpp (tests/test_qwen4exp_multiseq_parity.py Replay; test_carried_state passing means llama.cpp read the
same rows). For every graph: the recurrent range (head, n), the extra rows (cell, (plane, row) read, sequences), and
per sequence the (plane, row) it read and which kinds came from another prefix.
usage: QWEN4EXP_MULTISEQ_DUMP=<dumps> QWEN4EXP_TINY_GGUF=<gguf> LLAMA_CPP_GGUF_PY=<gguf-py> PARITY_LLAMA_COMMIT=<sha>
       [QWEN4EXP_RS_GUARD=1] python3 docs/evidence/mutation-scripts/2026-10-11-n20-extra-cell-probe.py <repo> [run]"""
import os
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
import tests.test_qwen4exp_multiseq_parity as B  # noqa: E402

run = sys.argv[2] if len(sys.argv) > 2 else "extra_cell"
W, hp = B.model(os.environ["QWEN4EXP_TINY_GGUF"])
R = B.Replay(pathlib.Path(os.environ["QWEN4EXP_MULTISEQ_DUMP"], run), W, hp, B.PIN)
print(f"{run} at {B.PIN}{' + guard' if B.GUARD else ''}; seq_rm results {[c['ok'] for c in R.man['calls'] if c['kind'] == 'seq_rm']}")
for gr in R.graphs:
    reads = {s: (gr["rs_reads"][s]["rows"], sorted({k for _, k in gr["rs_reads"][s]["stale"]})) for s in gr["seqs"]}
    print(f"graph {gr['g']} call {gr['call']}: seqs {gr['seqs']} x {gr['nst']}, recurrent head {gr['rs_head']} n {gr['rs_n']}, "
          f"extra {gr['rs_extra']}, reads (plane, row) {reads}, diverged {gr['diverged']}")
