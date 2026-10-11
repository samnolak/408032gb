#!/usr/bin/env python3
"""Call plans of the N19 runs (gap 10 of docs/reassessment/CORRECTNESS.md): several sequences in one llama.cpp qwen4exp
context, seq_cp, seq_rm and rollback edges, for tools/dump/dump_tensors.cpp (DUMP_CALLS several-sequence syntax) and
tests/test_qwen4exp_multiseq_parity.py, which imports RUNS for the expected behaviour of every run at both pins.

Steps of a plan:
  ("d", [(seq, n), (seq, n, n_wrong), ...])  one llama_process call; every part continues its sequence with n tokens,
                                              the last n_wrong of them "wrong" (a rejected draft);
  ("r", seq, n, ok_ec, ok_r1)                 llama_memory_seq_rm(mem, seq, pos - n, -1): a rollback of n positions,
                                              and whether llama.cpp accepts it at ec7630a / at the R1 pin 79e2e74;
  ("c", src, dst)                             llama_memory_seq_cp(mem, src, dst, -1, -1);
  ("x", seq)                                  llama_memory_seq_rm(mem, seq, -1, -1).
Token ids: sequence s's own token at position p is base_tokens(s)[p] (ids in [3, 96) with EOS (2) at EOS_AT[s], so
PLE predecessors cross EOS in every run); a wrong token is wrong(id) (shifted by 17 inside [3, 96)); a sequence that
receives a seq_cp continues with its own base tokens after the copied prefix. The generator applies a rollback to its
histories when ok_ec is true, so the same DUMP_TOKENS serve both pins (at a pin that refuses it, the following tokens
simply continue the unshortened sequence; the tests rebuild every sequence from what llama.cpp reported).
N20 (llama.cpp with patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch, QWEN4EXP_RS_GUARD=1 in the tests):
a run's "guard_ok" lists the seq_rm results of its "r" steps in order (default: ok_r1 of each), its
diverge["guard"] the divergences (default: none), "guard_log" how many refusals the patch's plane rule logs (default 0).

Usage: qwen4exp_multiseq_plans.py <run>   prints four lines: DUMP_CALLS, DUMP_TOKENS, then "np kvu n_rs_seq ubatch",
then the run's one-line description; qwen4exp_multiseq_plans.py --list prints the run names.
"""
import random
import sys

EOS = 2
N_BASE = 64
EOS_AT = {0: (7, 15, 21, 27), 1: (5, 13, 22), 2: (9, 17, 23, 30), 3: (11, 19, 26)}   # 3: N20 cp_extra only


def base_tokens(seq):
    rng = random.Random(1900 + seq)
    t = [rng.randrange(3, 96) for _ in range(N_BASE)]
    for p in EOS_AT[seq]:
        t[p] = EOS
    return t


def wrong(x):
    return (x - 3 + 17) % 93 + 3


def _rep(step, n):
    return [step] * n


_D3 = ("d", [(0, 1), (1, 1), (2, 1)])
_STEPS_MULTI = [
    ("d", [(0, 16), (1, 12), (2, 20)]),                  # one call: ubatches of 3 x 12, 2 x 4, 1 x 4 (unified)
    *_rep(_D3, 8),                                        # interleaved decode, one token per sequence and call
    *_rep(("d", [(0, 1), (2, 1)]), 2),                    # sequence 1 idle
    ("d", [(1, 2), (0, 1)]),                              # batch order 1, 0 and unequal lengths
    ("r", 2, 1, False, False),                            # n_rs_seq 0: every partial seq_rm is refused (RC:207-215)
    *_rep(("d", [(2, 1), (0, 1), (1, 1)]), 2),            # batch order 2, 0, 1
]
_STEPS_CP = [
    ("d", [(0, 22)]),
    ("c", 0, 1),                                          # whole-sequence copy: cells shared (unified)
    *_rep(("d", [(0, 1), (1, 1)]), 6),                    # both continue, with different tokens
    ("c", 0, 2),
    *_rep(_D3, 4),
    ("x", 1),
    *_rep(("d", [(0, 1), (2, 1)]), 2),
    ("x", 2),                                             # the last sharing sequence goes: every pool re-pooled (ec)
    *_rep(("d", [(0, 1)]), 6),
    ("c", 0, 1),                                          # sequence 1 reused for a second copy, and at once
    ("c", 0, 2),                                          # a third: both copies stale, their pools share reps
    ("d", [(1, 3), (0, 3), (2, 3)]),
]
_STEPS_RB = [
    ("d", [(0, 10), (1, 10), (2, 10)]),                   # -ub 8, n_rs_seq 3: split keeps the last 4 of each together
    ("d", [(0, 4, 2), (1, 4, 3), (2, 4, 1)]),             # drafts, 2 / 3 / 1 wrong
    ("r", 0, 2, True, True),
    ("r", 1, 3, True, True),
    ("r", 2, 1, True, True),
    ("d", [(0, 1), (2, 1)]),                              # sequence 1's pending rollback while its cell is in range
    ("d", [(1, 1)]),
    *_rep(_D3, 3),
    ("d", [(2, 4, 4)]),
    ("r", 2, 4, False, False),                            # 4 > n_rs_seq: refused, nothing changes
    ("r", 2, 3, True, True),
    ("r", 2, 1, False, False),                            # a second rollback before a decode: refused (pending)
    ("d", [(2, 1)]),
    ("d", [(0, 10, 3)]),                                  # one call, two ubatches (6 + 4), last 3 wrong
    ("r", 0, 3, True, True),                              # inside the multi-ubatch call: from its last ubatch
    *_rep(_D3, 2),
]

RUNS = {
    "multi_uni": dict(np=3, kvu=1, rs=0, ub=512, steps=_STEPS_MULTI,
                      doc="3 sequences, one KV stream (kv_unified): unequal prefill in one call, interleaved decode"),
    "multi_str": dict(np=3, kvu=0, rs=0, ub=512, steps=_STEPS_MULTI,
                      doc="the same plan with one KV stream per sequence (sequential split)"),
    "cp_uni": dict(np=3, kvu=1, rs=0, ub=512, steps=_STEPS_CP,
                   doc="seq_cp 0->1, 0->2 (shared cells), both continue, drop 1 and 2, copy twice again (kv_unified)"),
    "cp_str": dict(np=3, kvu=0, rs=0, ub=512, steps=_STEPS_CP,
                   doc="the same plan with one KV stream per sequence (cross-stream buffer copies)"),
    "rb_uni": dict(np=3, kvu=1, rs=3, ub=8, steps=_STEPS_RB,
                   doc="n_rs_seq 3, -ub 8: rollbacks of 1-3 per sequence, refused ones, inside a multi-ubatch call"),
    "rb_str": dict(np=3, kvu=0, rs=3, ub=8, steps=_STEPS_RB,
                   doc="the same plan with one KV stream per sequence"),
    "stale": dict(np=2, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 24)]),
        ("d", [(0, 1, 1)]),
        ("d", [(0, 1, 1)]),
        ("r", 0, 2, True, True),                          # across two one-token calls: accepted, stale planes
        *_rep(("d", [(0, 1)]), 3),
        ("d", [(1, 20)]),
        ("d", [(1, 1, 1)]),
        ("r", 1, 1, True, True),                          # one token after a one-token call: stale delta-rule plane
        *_rep(("d", [(1, 1)]), 3),
    ], doc="rollbacks deeper than the last call's last ubatch (accepted, silently read stale snapshot planes)",
        diverge={"ec": {0: 4, 1: 10}, "r1": {0: 4, 1: 10}, "guard": {}},
        guard_ok=[False, False], guard_log=2),            # N20: plane 2 / plane 1 not written by a one-token call
    "cp_pending": dict(np=2, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 20)]),
        ("d", [(0, 4, 2)]),
        ("r", 0, 2, True, True),
        ("c", 0, 1),                                      # copy while sequence 0's rollback is pending
        *_rep(("d", [(0, 1), (1, 1)]), 4),
    ], doc="seq_cp while a rollback is pending: the copy reads the state before the rollback",
        diverge={"ec": {1: 4}, "r1": {1: 4}, "guard": {}}),   # N20: the copy inherits the pending rollback
    "cp_shared_rb": dict(np=2, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 20)]),
        ("d", [(0, 4, 2)]),
        ("c", 0, 1),
        ("r", 0, 2, True, False),                         # one recurrent cell for both: ec accepts, R1 refuses
                                                          # (and ec with the N20 patch)
        *_rep(("d", [(0, 1), (1, 1)]), 3),
        ("d", [(1, 1), (0, 1)]),
    ], doc="a rollback of one of two sequences that share a recurrent cell after seq_cp"),
}


def plan(name, runs=None):
    """(DUMP_CALLS, DUMP_TOKENS) of a run of `runs` (default RUNS)."""
    r = (runs or RUNS)[name]
    base = {s: base_tokens(s) for s in range(r["np"])}
    hist = {s: [] for s in range(r["np"])}
    calls, toks = [], []
    for st in r["steps"]:
        if st[0] == "d":
            parts = []
            for part in st[1]:
                s, n = part[0], part[1]
                nw = part[2] if len(part) > 2 else 0
                p0 = len(hist[s])
                ids = [base[s][p] for p in range(p0, p0 + n)]
                for k in range(n - nw, n):
                    ids[k] = wrong(ids[k])
                hist[s] += ids
                toks += ids
                parts.append(f"{n}@{s}")
            calls.append("+".join(parts))
        elif st[0] == "r":
            _, s, n, ok_ec, _ok_r1 = st
            if ok_ec:
                hist[s] = hist[s][:len(hist[s]) - n]
            calls.append(f"r{n}@{s}")
        elif st[0] == "c":
            _, a, b = st
            hist[b] = list(hist[a])
            calls.append(f"c{a}:{b}")
        elif st[0] == "x":
            hist[st[1]] = []
            calls.append(f"x{st[1]}")
        else:
            raise ValueError(st)
    return ",".join(calls), ",".join(map(str, toks))


def main(argv, runs=None, doc=None):
    runs = runs or RUNS
    if len(argv) != 2:
        print(doc or __doc__, file=sys.stderr)
        return 2
    if argv[1] == "--list":
        print(" ".join(runs))
        return 0
    r = runs[argv[1]]
    calls, toks = plan(argv[1], runs)
    print(calls)
    print(toks)
    print(r["np"], r["kvu"], r["rs"], r["ub"])
    print(r["doc"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
