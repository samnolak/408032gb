#!/usr/bin/env python3
"""Call plans of the N20 runs: the recurrent rollback guard (patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch)
on llama.cpp qwen4exp, for tools/dump/dump_tensors.cpp (DUMP_CALLS several-sequence syntax) and
tests/test_qwen4exp_rs_guard_parity.py. Same step and token format as tools/tinygen/qwen4exp_multiseq_plans.py (N19):
("d", [(seq, n[, n_wrong])...]), ("r", seq, n, ok_ec, ok_r1), ("c", src, dst), ("x", seq). Every run states what
unpatched llama.cpp does at each pin (ok_ec / ok_r1, diverge["ec"] / ["r1"]: sequence -> first call whose graph reads a
recurrent row of another prefix) and what the guard patch does (guard_ok: the "r" steps' results in order,
diverge["guard"], guard_log: how many of its refusals come from the patch's new plane rule, which logs "refusing a
rollback"). The three runs cover the patch's pieces beyond N19's stale / cp_pending / cp_shared_rb runs:
  last_call         a rollback of exactly the tokens of the last call (plane n of an n-token ubatch is not written),
                    then the deepest exact rollbacks (n - 1 after n tokens, n_rs_seq after n_rs_seq + 1) still accepted
  extra_cell        a pending rollback applied by the extra-row copy (src/llama-graph.cpp build_rs) leaves plane 0 only:
                    a second rollback of that sequence before it decodes reads a plane of another row
  cp_pending_order  seq_cp while the source's rollback is pending, the copy decodes before the source
A copy INTO a sequence with a pending rollback (the patch also clears that index) has no run: llama.cpp's one-stream KV
seq_cp keeps the destination's own cells, so a caller clears the destination first (llama-server: seq_rm(dst, -1, -1)
before seq_cp), and that whole-sequence seq_rm already resets the index; the oracle unit tests cover it
(tests/test_qwen4exp_rs_guard_ref.py).
Usage: qwen4exp_rs_guard_plans.py <run> | --list (output as qwen4exp_multiseq_plans.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4exp_multiseq_plans as N19  # noqa: E402


def _rep(step, n):
    return [step] * n


RUNS = {
    "last_call": dict(np=1, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 16)]),
        ("d", [(0, 2, 2)]),
        ("r", 0, 2, True, True),                          # 2 = the last call's tokens: delta-rule plane 2 not written
        ("d", [(0, 3, 1)]),
        ("r", 0, 1, True, True),                          # 1 < 3: exact
        ("d", [(0, 4, 3)]),
        ("r", 0, 3, True, True),                          # 3 < 4 = n_rs_seq + 1: the deepest exact rollback
        ("d", [(0, 2, 1)]),
        ("r", 0, 1, True, True),                          # 1 < 2: exact
        *_rep(("d", [(0, 1)]), 2),
    ], doc="a rollback of all tokens of the last call (refused by the guard), then the deepest exact rollbacks",
        diverge={"ec": {0: 3}, "r1": {0: 3}, "guard": {}},
        guard_ok=[False, True, True, True], guard_log=1),
    "extra_cell": dict(np=3, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 10), (1, 10), (2, 10)]),
        ("d", [(0, 4), (1, 4, 2), (2, 4)]),
        ("r", 1, 2, True, True),                          # pending, exact (2 < 4)
        ("d", [(0, 1), (2, 1)]),                          # seq 1's cell lies between: the extra-row copy applies it
        ("r", 1, 1, True, True),                          # plane 1 of that row was not rewritten
        *_rep(("d", [(0, 1), (1, 1), (2, 1)]), 2),
    ], doc="a second rollback after the extra-row copy applied the first (only plane 0 is rewritten)",
        diverge={"ec": {1: 5}, "r1": {1: 5}, "guard": {}},
        guard_ok=[True, False], guard_log=1),
    "cp_pending_order": dict(np=2, kvu=1, rs=3, ub=512, steps=[
        ("d", [(0, 20)]),
        ("d", [(0, 4, 2)]),
        ("r", 0, 2, True, True),
        ("c", 0, 1),                                      # copy while seq 0's rollback is pending
        ("d", [(1, 1)]),                                  # the copy decodes first, in a cell of its own
        ("d", [(1, 1)]),
        ("d", [(0, 1)]),                                  # the source still reads its pending plane
        ("d", [(0, 1), (1, 1)]),
    ], doc="seq_cp while a rollback is pending, the copy decoding before the source",
        diverge={"ec": {1: 4}, "r1": {1: 4}, "guard": {}}),
}


def plan(name):
    return N19.plan(name, RUNS)


if __name__ == "__main__":
    sys.exit(N19.main(sys.argv, RUNS, __doc__))
