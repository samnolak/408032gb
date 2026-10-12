#!/usr/bin/env python3
"""N20 oracle probe, copied from the N20 audit (<S>/audit-n20/probe/sim.py, evidence-auditor) with only this docstring
changed: a symbolic replay of ref/qwen4exp.py RecurrentCells, guard on and off, on 20000 random operation sequences
(4 sequences, n_rs_seq 3: decodes of 1-4 tokens per sequence, rollbacks of 1-3, seq_cp after a whole-sequence seq_rm
of the destination, as llama-server does). Every (row, plane) holds the token history of the state it stores; a
sequence must read exactly its own history. Counts the scenarios with a wrong read in each mode.
usage: python3 docs/evidence/mutation-scripts/2026-10-11-n20-oracle-probe.py <repo> [seed]"""
import random, sys
sys.path.insert(0, sys.argv[1])
from ref import qwen4exp as Q

def run(ops, guard, R=3, nseq=4):
    rc = Q.RecurrentCells(nseq, R, refuse_shared=True, guard=guard)
    H = {s: [] for s in range(nseq)}
    content = {}
    tok = [0]
    bad = []
    for op in ops:
        if op[0] == "d":
            parts = op[1]                       # [(seq, n)], equal n (one ubatch)
            n = parts[0][1]
            seqs = [s for s, _ in parts]
            rc.find_slot(seqs, [len(H[s]) + n - 1 for s in seqs], n)
            head, nn, rz = rc.head, rc.n, rc.rs_z
            reads = rc.s_copy()
            if rz >= 0:
                content[(rz, 0)] = ()
            got = [content.get((row, pl)) for pl, row in reads]
            for k, s in enumerate(seqs):
                if got[k] != tuple(H[s]):
                    bad.append((op, s, "read", reads[k]))
            for i in range(len(seqs), nn):          # extra cells: plane 0 <- what they read
                content[(head + i, 0)] = got[i]
            for k, s in enumerate(seqs):
                for _ in range(n):
                    tok[0] += 1; H[s].append(tok[0])
                for j in range(min(n, 1 + R)):
                    content[(head + k, j)] = tuple(H[s][:len(H[s]) - j])
        elif op[0] == "r":
            _, s, r = op
            if len(H[s]) > r and rc.seq_rm(s, len(H[s]) - r, -1):
                H[s] = H[s][:len(H[s]) - r]
        elif op[0] == "c":
            _, src, dst = op
            rc.seq_rm(dst, -1, -1); H[dst] = []
            rc.seq_cp(src, dst); H[dst] = list(H[src])
    return bad

def rand_ops(rng, nseq=4, R=3, L=12):
    ops = [("d", [(s, rng.randint(4, 6)) for s in range(1)])]
    ops = [("d", [(s, 6) for s in range(nseq)])]
    for _ in range(L):
        k = rng.random()
        if k < 0.5:
            seqs = rng.sample(range(nseq), rng.randint(1, nseq))
            n = rng.randint(1, R + 1)
            ops.append(("d", [(s, n) for s in seqs]))
        elif k < 0.8:
            ops.append(("r", rng.randrange(nseq), rng.randint(1, R)))
        else:
            a, b = rng.sample(range(nseq), 2)
            ops.append(("c", a, b))
    return ops

if __name__ == '__main__':
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    cnt = {True: 0, False: 0}; ex = {}
    N = 20000
    for t in range(N):
        ops = rand_ops(rng)
        for g in (False, True):
            b = run(ops, g)
            if b:
                cnt[g] += 1
                ex.setdefault(g, (ops, b))
    print(f"{N} random scenarios: unpatched oracle wrong reads in {cnt[False]}, guard oracle wrong reads in {cnt[True]}")
    if True in ex:
        ops, b = ex[True]
        print("first guard failure ops:", ops)
        print("bad:", b[:3])
    