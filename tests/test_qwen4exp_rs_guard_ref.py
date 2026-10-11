"""N20: the oracle's model of the recurrent rollback guard (ref/qwen4exp.py RecurrentCells / MultiSeqContext guard=True,
mirroring patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch), numpy only, no llama.cpp needed.

GuardRules: the cell bookkeeping alone: which rollbacks the guard refuses (a plane the last write did not reach: the
tokens of the last ubatch, the extra-row copy, the gather moving a cell's rows), and that seq_cp hands the pending
rollback over; each against the unpatched rules, which accept the same rollbacks. GuardOracle: MultiSeqContext on the
random 4-layer model of tests/test_qwen4exp_chunked_ref.py: with the guard, every row of the N19 stale and cp_pending
scenarios equals the sequence computed alone (Q.forward) and nothing diverges; without it they diverge.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import qwen4exp as Q  # noqa: E402
from tests.test_qwen4exp_chunked_ref import random_model  # noqa: E402

TOL = 1e-6


def decoded(rc, seqs, n, n0):
    """One ubatch of n tokens per sequence of seqs, starting at position n0[s]: RC apply (find_slot) then s_copy."""
    rc.find_slot(seqs, [n0[s] + n - 1 for s in seqs], n)
    return rc.s_copy()


class GuardRules(unittest.TestCase):
    def rc(self, guard, size=2, n_rs_seq=3):
        return Q.RecurrentCells(size, n_rs_seq, refuse_shared=True, guard=guard)

    def test_guard_needs_the_shared_cell_refusal(self):
        with self.assertRaises(AssertionError):
            Q.RecurrentCells(2, 3, refuse_shared=False, guard=True)

    def test_a_rollback_needs_a_plane_the_last_ubatch_wrote(self):
        for guard in (False, True):
            rc = self.rc(guard)
            decoded(rc, [0], 10, {0: 0})                         # planes 0..3 written
            self.assertEqual(rc.rs_n[rc.tail[0]], 4)
            self.assertTrue(rc.seq_rm(0, 7, -1), "3 < 4: exact")
            decoded(rc, [0], 1, {0: 7})                          # one token: plane 0 only
            self.assertEqual(rc.rs_n[rc.tail[0]], 1)
            self.assertEqual(rc.seq_rm(0, 7, -1), not guard, "plane 1 was not written by the one-token ubatch")
            if not guard:
                continue
            decoded(rc, [0], 2, {0: 8})                          # planes 0, 1
            self.assertFalse(rc.seq_rm(0, 8, -1), "a rollback of all the ubatch's tokens: plane 2 not written")
            self.assertEqual((rc.pos[rc.tail[0]], rc.rs_idx[0]), (9, 0), "a refusal changes nothing")
            self.assertTrue(rc.seq_rm(0, 9, -1))
            self.assertEqual(rc.rs_idx[0], 1)

    def test_seq_cp_hands_the_pending_rollback_to_the_copy(self):
        for guard in (False, True):
            rc = self.rc(guard)
            decoded(rc, [0], 10, {0: 0})
            self.assertTrue(rc.seq_rm(0, 8, -1))
            rc.seq_cp(0, 1)
            self.assertEqual(rc.tail[1], rc.tail[0])
            self.assertEqual(rc.rs_idx[1], 2 if guard else 0)
            rows = decoded(rc, [0, 1], 1, {0: 8, 1: 8})          # copy on write: both read the shared cell's rows
            self.assertEqual(rows[0][0], 2)
            self.assertEqual(rows[1][0], 2 if guard else 0, "the copy reads the rolled-back plane only with the guard")

    def test_seq_cp_into_a_pending_rollback(self):
        """Recurrent cells only: no dump run, because llama.cpp's one-stream KV seq_cp keeps the destination's own
        cells, so callers clear the destination first (llama-server: seq_rm(dst, -1, -1), which resets the index)."""
        for guard in (False, True):
            rc = self.rc(guard)
            decoded(rc, [0, 1], 6, {0: 0, 1: 0})
            self.assertTrue(rc.seq_rm(1, 4, -1))                 # seq 1 pending 2
            rc.seq_cp(0, 1)
            self.assertEqual(rc.rs_idx[1], 0 if guard else 2, "the old destination's rollback is not the copy's")
            rc.seq_cp(1, 1)                                      # src == dst: nothing
            self.assertEqual(rc.rs_idx[1], 0 if guard else 2)
            rc2 = self.rc(guard)
            decoded(rc2, [1], 6, {1: 0})
            self.assertTrue(rc2.seq_rm(1, 4, -1))
            rc2.seq_cp(0, 1)                                     # a source without a cell: the copy is empty
            self.assertEqual((rc2.tail[1], rc2.rs_idx[1]), (-1, 0 if guard else 2))

    def test_the_extra_row_copy_consumes_every_sharers_rollback(self):
        """N20 audit (plan cp_extra): after seq_cp both sharers hold the pending rollback; when the shared cell is an
        extra cell whose rows moved, its plane 0 receives the rolled-back state for both, so both indices must go."""
        rc = self.rc(True, size=4)
        decoded(rc, [0, 1, 2, 3], 6, {s: 0 for s in range(4)})
        self.assertTrue(rc.seq_rm(1, 4, -1))                     # seq 1 pending 2
        self.assertTrue(rc.seq_rm(3, -1, -1))
        rc.seq_cp(1, 3)
        self.assertEqual((rc.tail[3], rc.rs_idx[1], rc.rs_idx[3]), (rc.tail[1], 2, 2))
        rows = decoded(rc, [0, 2], 3, {0: 6, 2: 6})
        c = rc.tail[1]
        self.assertEqual((c, rc.tail[3], rc.src0[c]), (2, 2, 1), "the shared cell moved to 2, its rows are row 1")
        self.assertEqual(rows[c - rc.head], (2, 1), "the extra row applies plane 2 of row 1")
        self.assertEqual((rc.rs_idx[1], rc.rs_idx[3], rc.rs_n[c]), (0, 0, 1))
        rows = decoded(rc, [1, 3], 4, {1: 4, 3: 4})              # both read plane 0 (copy on write for one of them)
        self.assertEqual(sorted(pl for pl, _ in rows[:2]), [0, 0])

    def test_the_extra_row_copy_leaves_plane_0_only(self):
        for guard in (False, True):
            rc = self.rc(guard, size=3)
            decoded(rc, [0, 1, 2], 4, {0: 0, 1: 0, 2: 0})
            self.assertEqual([rc.tail[s] for s in range(3)], [0, 1, 2])
            self.assertTrue(rc.seq_rm(1, 2, -1))                 # pending 2
            rows = decoded(rc, [0, 2], 1, {0: 4, 2: 4})
            # the gather put seq 2 into cell 1, so seq 1's cell (its rows still in row 1) is cell 2, an extra cell
            self.assertEqual((rc.head, rc.n, rc.tail[1], rc.src0[2]), (0, 3, 2, 1))
            self.assertEqual(rows[2], (2, 1), "the extra row reads plane 2 of row 1 (the pending rollback)")
            self.assertEqual(rc.rs_idx[1], 0)
            self.assertEqual(rc.rs_n[2], 1, "tracked in both modes, read by the guard only")
            self.assertEqual(rc.seq_rm(1, 1, -1), not guard, "plane 1 of row 2 holds seq 2's old plane")

    def test_an_extra_cell_that_did_not_move_keeps_its_planes(self):
        rc = self.rc(True, size=4)
        decoded(rc, [0, 1, 2, 3], 4, {s: 0 for s in range(4)})
        self.assertEqual([rc.tail[s] for s in range(4)], [0, 1, 2, 3])
        rows = decoded(rc, [0, 3], 1, {0: 4, 3: 4})
        # seq 3 is gathered into cell 1, seq 1 moves to cell 3 (its rows stay in row 1); seq 2 stays in cell 2
        self.assertEqual((rc.head, rc.n), (0, 4))
        self.assertEqual((rc.tail[1], rc.src0[3], rc.tail[2], rc.src0[2]), (3, 1, 2, 2))
        self.assertEqual(rows[2:], [(0, 2), (0, 1)])
        self.assertEqual((rc.rs_n[2], rc.rs_n[3]), (4, 1), "unmoved without a pending rollback: planes kept")
        self.assertTrue(rc.seq_rm(2, 1, -1), "3 < 4")
        self.assertFalse(rc.seq_rm(1, 3, -1), "seq 1's planes 1.. are now row 3's")


class GuardOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = random_model(seed=7)
        rng = np.random.default_rng(20)
        cls.base = {s: [int(x) for x in rng.integers(3, 40, 40)] for s in range(3)}

    def run_plan(self, M, plan):
        """("d", [(seq, n)]) / ("r", seq, n) / ("c", a, b): tokens from self.base by position. Returns the seq_rm
        results and the graph rows [(seq, pos, logits row, diverged)]."""
        oks, rows = [], []
        for st in plan:
            if st[0] == "d":
                parts = [(s, self.base[s][len(M.caches[s]["tokens"]):len(M.caches[s]["tokens"]) + n]) for s, n in st[1]]
                for G in M.decode(parts):
                    for s in G["seqs"]:
                        res = G["per_seq"][s]
                        rows += [(s, int(p), res["logits"][k], G["diverged"][s]) for k, p in enumerate(res["pos"])]
            elif st[0] == "r":
                oks.append(M.seq_rm(st[1], len(M.caches[st[1]]["tokens"]) - st[2], -1))
            else:
                M.seq_cp(st[1], st[2])
        return oks, rows

    def check_exact(self, M, rows):
        for s, p, lg, div in rows:
            self.assertFalse(div, (s, p))
            want = Q.forward(M.caches[s]["tokens"], self.W, self.hp)["logits"][p]
            self.assertLess(np.abs(lg - want).max(), 1e-8 + TOL * np.abs(want).max(), (s, p))

    def test_stale_rollbacks_are_refused_and_rows_stay_exact(self):
        plan = [("d", [(0, 12)]), ("d", [(0, 1)]), ("d", [(0, 1)]), ("r", 0, 2), ("d", [(0, 1)]),
                ("d", [(0, 2)]), ("r", 0, 2), ("d", [(0, 1)])]
        for pin in ("ec", "r1"):
            M = Q.MultiSeqContext(self.W, self.hp, 1, 64, True, 3, pin, 512, guard=True)
            oks, rows = self.run_plan(M, plan)
            self.assertEqual(oks, [False, False])
            self.check_exact(M, rows)
            M = Q.MultiSeqContext(self.W, self.hp, 1, 64, True, 3, pin, 512)
            oks, rows = self.run_plan(M, plan)
            self.assertEqual(oks, [True, True])
            self.assertTrue(any(div for *_, div in rows), "unpatched: the same plan diverges")

    def test_a_copy_made_while_a_rollback_is_pending_continues_the_kept_prefix(self):
        for order in ("together", "copy first"):
            dec = [("d", [(0, 1), (1, 1)])] * 3 if order == "together" else \
                [("d", [(1, 1)]), ("d", [(1, 1)]), ("d", [(0, 1)]), ("d", [(0, 1), (1, 1)])]
            plan = [("d", [(0, 10)]), ("d", [(0, 4)]), ("r", 0, 2), ("c", 0, 1)] + dec
            M = Q.MultiSeqContext(self.W, self.hp, 2, 64, True, 3, "r1", 512, guard=True)
            oks, rows = self.run_plan(M, plan)
            self.assertEqual(oks, [True])
            self.assertEqual(M.caches[1]["tokens"][:12], self.base[0][:12])
            self.check_exact(M, rows)
            M = Q.MultiSeqContext(self.W, self.hp, 2, 64, True, 3, "r1", 512)
            oks, rows = self.run_plan(M, plan)
            self.assertTrue(any(div for s, *_, div in rows if s == 1), "unpatched: the copy diverges")
            self.assertFalse(any(div for s, *_, div in rows if s == 0))


if __name__ == "__main__":
    unittest.main()
