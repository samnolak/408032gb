"""Property tests of the several-sequence oracle of ref/qwen4exp.py (N19; numpy only, no llama.cpp needed).

Q.split_equal (the replica of llama_batch_allocr::split_equal) against hand-walked examples of src/llama-batch.cpp and
its invariants on random batches; the refusal rules of Q.RecurrentCells (src/llama-memory-recurrent.cpp seq_rm at both
pins); and Q.MultiSeqContext on the random 4-layer model of tests/test_qwen4exp_chunked_ref.py: sequences in one
context compute exactly what each computes alone (no leakage in the oracle itself, unified and one stream per
sequence alike), seq_cp continues the copied history, a rollback inside the last call restores the kept prefix, and
the two stale cases (a rollback deeper than the last call's last ubatch, seq_cp while a rollback is pending) read the
rows the cell model says, which are the states of other prefixes. The comparison with llama.cpp is
tests/test_qwen4exp_multiseq_parity.py.
"""
import pathlib
import random
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import qwen4exp as Q  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402
from tests.test_qwen4exp_chunked_ref import random_model  # noqa: E402

TOL = 1e-10


def sizes(ubs, seq_of):
    return [len(i) for i, _ in ubs], [[seq_of[i[k * (len(i) // n)]] for k in range(n)] for i, n in ubs]


class SplitEqual(unittest.TestCase):
    def test_examples_walked_through_the_code(self):
        seq_of = [0] * 16 + [1] * 12 + [2] * 20
        self.assertEqual(sizes(Q.split_equal(seq_of, 512, False), seq_of), ([36, 8, 4], [[0, 1, 2], [0, 2], [2]]))
        # one stream per sequence: the sets of a ubatch must be consecutive ascending ids (src/llama-batch.cpp:569-572)
        self.assertEqual(sizes(Q.split_equal(seq_of, 512, True), seq_of), ([36, 4, 8], [[0, 1, 2], [0], [2]]))
        # n_rs_seq 3: the last 4 tokens of a sequence stay together (src/llama-batch.cpp:637-701)
        self.assertEqual(sizes(Q.split_equal([0] * 10, 8, True, 4), [0] * 10)[0], [6, 4])
        seq_of = [0] * 10 + [1] * 10 + [2] * 10
        self.assertEqual(sizes(Q.split_equal(seq_of, 8, False, 4), seq_of),
                         ([6, 6, 6, 4, 8], [[0, 1, 2]] * 3 + [[0], [1, 2]]))
        # batch order decides the order of the sets
        seq_of = [1, 1, 0]
        self.assertEqual(sizes(Q.split_equal(seq_of, 512, False), seq_of), ([2, 1], [[1, 0], [1]]))

    def test_invariants_on_random_batches(self):
        rng = random.Random(19)
        for _ in range(300):
            n_seq = rng.randint(1, 3)
            lens = {s: rng.randint(1, 20) for s in range(n_seq)}
            order = list(range(n_seq))
            rng.shuffle(order)
            seq_of = [s for s in order for _ in range(lens[s])]
            ub = rng.choice([4, 6, 8, 16, 512])
            keep = rng.choice([0, 0, 2, 4]) if ub > 4 else 0
            sequential = rng.random() < 0.5
            ubs = Q.split_equal(seq_of, ub, sequential, keep)
            flat = [i for idxs, _ in ubs for i in idxs]
            self.assertEqual(sorted(flat), list(range(len(seq_of))))
            for idxs, n in ubs:
                nst = len(idxs) // n
                self.assertEqual(nst * n, len(idxs))
                sets = [seq_of[idxs[k * nst]] for k in range(n)]
                self.assertEqual(len(set(sets)), n)
                for k in range(n):
                    self.assertEqual({seq_of[i] for i in idxs[k * nst:(k + 1) * nst]}, {sets[k]})
                    self.assertEqual(idxs[k * nst:(k + 1) * nst], sorted(idxs[k * nst:(k + 1) * nst]))
                if sequential:
                    self.assertEqual(sets, list(range(sets[0], sets[0] + n)))
            for s in lens:
                tail = [i for i, x in enumerate(seq_of) if x == s][-min(lens[s], keep):] if keep else []
                if tail:
                    self.assertEqual(len({j for j, (idxs, _) in enumerate(ubs) for i in idxs if i in tail}), 1,
                                     (seq_of, ub, keep, ubs))


class RecurrentRefusals(unittest.TestCase):
    def decoded(self, rc, seq, n0, n):
        rc.find_slot([seq], [n0 + n - 1])
        rc.s_copy()

    def test_rollback_rules(self):
        for pin in ("ec", "r1"):
            rc = Q.RecurrentCells(2, 3, refuse_shared=pin == "r1")
            self.decoded(rc, 0, 0, 10)
            self.assertFalse(rc.seq_rm(0, 6, -1), "4 > n_rs_seq")
            self.assertTrue(rc.seq_rm(0, 7, -1))
            self.assertEqual((rc.rs_idx[0], rc.pos[rc.tail[0]]), (3, 6))
            self.assertFalse(rc.seq_rm(0, 6, -1), "a pending rollback is single-use")
            self.decoded(rc, 0, 7, 1)                              # s_copy consumed it
            self.assertEqual(rc.rs_idx[0], 0)
            self.assertTrue(rc.seq_rm(0, 7, -1))
            self.assertTrue(rc.seq_rm(0, -1, -1), "the whole sequence")
            self.assertEqual((rc.rs_idx[0], rc.tail[0]), (0, -1))
            # one recurrent cell for two sequences after seq_cp
            self.decoded(rc, 0, 0, 10)
            rc.seq_cp(0, 1)
            self.assertEqual(rc.tail[1], rc.tail[0])
            self.assertEqual(rc.seq_rm(0, 8, -1), pin == "ec")
            if pin == "ec":
                self.assertEqual(rc.seq_pos_max(1), 7, "the other sequence's position moves back too")
        rc = Q.RecurrentCells(1, 0, refuse_shared=False)
        self.decoded(rc, 0, 0, 5)
        self.assertFalse(rc.seq_rm(0, 4, -1), "n_rs_seq 0: no partial rollback")


class MultiSeqOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = random_model(seed=7)
        rng = np.random.default_rng(19)
        cls.base = {s: [int(x) for x in rng.integers(3, 40, 40)] for s in range(3)}

    def ctx(self, unified=True, n_rs_seq=0, pin="ec", n_ubatch=512, n_seq=3):
        return Q.MultiSeqContext(self.W, self.hp, n_seq, 64, unified, n_rs_seq, pin, n_ubatch)

    def run_plan(self, M, plan):
        """plan: list of ("d", [(seq, n)]) / ("r", seq, n) / ("c", a, b) / ("x", seq); tokens from self.base by
        position (own sequence). Returns rows [(seq, pos, logits row, graph)] and the graphs."""
        rows, graphs = [], []
        for st in plan:
            if st[0] == "d":
                parts = [(s, self.base[s][len(M.caches[s]["tokens"]):len(M.caches[s]["tokens"]) + n]) for s, n in st[1]]
                for G in M.decode(parts):
                    graphs.append(G)
                    for s in G["seqs"]:
                        res = G["per_seq"][s]
                        for k, p in enumerate(res["pos"]):
                            rows.append((s, int(p), res["logits"][k], G))
            elif st[0] == "r":
                st_ok = M.seq_rm(st[1], len(M.caches[st[1]]["tokens"]) - st[2], -1)
                self.assertTrue(st_ok, st)
            elif st[0] == "c":
                M.seq_cp(st[1], st[2])
            else:
                M.seq_rm(st[1], -1, -1)
        return rows, graphs

    def alone(self, toks):
        return Q.forward(toks, self.W, self.hp)

    def test_sequences_are_independent(self):
        plan = [("d", [(0, 7), (1, 5), (2, 9)])] + [("d", [(0, 1), (1, 1), (2, 1)])] * 5 + \
               [("d", [(2, 1), (0, 1)]), ("d", [(1, 3), (0, 2)])]
        out = {}
        for unified in (True, False):
            M = self.ctx(unified=unified)
            rows, graphs = self.run_plan(M, plan)
            self.assertTrue(any(len(G["seqs"]) == 3 for G in graphs))
            for s, p, lg, G in rows:
                self.assertFalse(G["diverged"][s])
                want = self.alone(M.caches[s]["tokens"])["logits"][p]
                self.assertLess(np.abs(lg - want).max(), TOL, (unified, s, p))
            out[unified] = {(r[0], r[1]): r[2] for r in rows}
        self.assertEqual(sorted(out[True]), sorted(out[False]))
        self.assertLess(max(np.abs(out[True][k] - out[False][k]).max() for k in out[True]), TOL)

    def test_seq_cp_then_both_continue(self):
        for pin in ("ec", "r1"):
            M = self.ctx(pin=pin)
            plan = [("d", [(0, 9)]), ("c", 0, 1)] + [("d", [(0, 1), (1, 1)])] * 6 + [("x", 1)] + [("d", [(0, 1)])] * 3
            rows, graphs = self.run_plan(M, plan)
            seq1 = self.base[0][:9] + self.base[1][9:15]
            want1 = self.alone(seq1)["logits"]
            for s, p, lg, G in rows:
                want = want1[p] if s == 1 else self.alone(M.caches[0]["tokens"])["logits"][p]
                self.assertLess(np.abs(lg - want).max(), TOL, (pin, s, p))
            safe = [G["cache_safe"] for G in graphs]
            if pin == "ec":         # shared cells: every pool re-pooled while shared, then everything once more
                self.assertEqual(safe, [True] + [False] * 6 + [True] * 3)
                for G in graphs[1:8]:
                    self.assertEqual(G["new"], list(range(len(G["pools"]))))
            else:
                self.assertTrue(all(safe))
                self.assertLess(len(graphs[1]["new"]), len(graphs[1]["pools"]))

    def test_rollback_inside_the_last_call_restores_the_kept_prefix(self):
        M = self.ctx(n_rs_seq=3, n_ubatch=8, n_seq=2)
        plan = [("d", [(0, 9), (1, 6)]), ("d", [(0, 10)]), ("r", 0, 3), ("d", [(0, 1), (1, 1)]),
                ("d", [(1, 4)]), ("r", 1, 2), ("d", [(1, 1), (0, 1)])]
        rows, graphs = self.run_plan(M, plan)
        self.assertGreater(len([G for G in graphs if G["seqs"] == [0] and G["nst"] > 1]), 1, "a two-ubatch call")
        for s, p, lg, G in rows:
            self.assertFalse(G["diverged"][s])
        for s in (0, 1):
            toks = M.caches[s]["tokens"]
            last = [(p, lg) for s2, p, lg, G in rows if s2 == s][-1]
            self.assertLess(np.abs(last[1] - self.alone(toks)["logits"][last[0]]).max(), TOL)

    def test_rollback_deeper_than_the_last_ubatch_reads_stale_planes(self):
        M = self.ctx(n_rs_seq=3, n_seq=1)
        rows, graphs = self.run_plan(M, [("d", [(0, 8)]), ("d", [(0, 1)]), ("d", [(0, 1)]), ("r", 0, 2),
                                         ("d", [(0, 1)])])
        G = graphs[-1]
        self.assertTrue(G["diverged"][0])
        self.assertEqual(sorted({k for _, k in G["rs_reads"][0]["stale"]}), ["conv", "ple", "state"])
        A9 = self.alone(self.base[0][:9])
        for il in (0,):
            L = G["per_seq"][0]["layers"][il]["gdn"]
            # conv plane 2 of the last one-token call: the history before its token, i.e. of the first 9 positions
            self.assertLess(np.abs(L["conv_in"] - A9["layers"][il]["gdn"]["qkv"][6:9]).max(), TOL)
            # delta-rule plane 2 was last written by the 8-token call: the state 2 tokens before its end (6 tokens)
            Ga = A9["layers"][il]["gdn"]
            st6 = gated_delta_rule(Ga["q"][:6], Ga["k"][:6], Ga["v"][:6], Ga["g"][:6], Ga["beta"][:6],
                                   normalize_qk=True, eps=self.hp["eps"])[1]
            st8 = gated_delta_rule(Ga["q"][:8], Ga["k"][:8], Ga["v"][:8], Ga["g"][:8], Ga["beta"][:8],
                                   normalize_qk=True, eps=self.hp["eps"])[1]
            self.assertLess(np.abs(L["state_in"] - st6).max(), TOL)
            self.assertGreater(np.abs(L["state_in"] - st8).max(), 1e-3, "the kept prefix's state differs")
        M = self.ctx(n_rs_seq=3, n_seq=1)                          # one token back after a one-token call
        rows, graphs = self.run_plan(M, [("d", [(0, 8)]), ("d", [(0, 1)]), ("r", 0, 1), ("d", [(0, 1)])])
        self.assertEqual(sorted({k for _, k in graphs[-1]["rs_reads"][0]["stale"]}), ["state"],
                         "the conv history of slot 1 is the history before the token, the delta-rule state is not")

    def test_seq_cp_while_a_rollback_is_pending(self):
        M = self.ctx(n_rs_seq=3, n_seq=2)
        rows, graphs = self.run_plan(M, [("d", [(0, 8)]), ("d", [(0, 4)]), ("r", 0, 2), ("c", 0, 1),
                                         ("d", [(0, 1), (1, 1)])])
        G = graphs[-1]
        self.assertFalse(G["diverged"][0])
        self.assertTrue(G["diverged"][1])
        A12 = self.alone(self.base[0][:12])
        L = G["per_seq"][1]["layers"][0]["gdn"]
        self.assertLess(np.abs(L["conv_in"] - A12["layers"][0]["gdn"]["qkv"][9:12]).max(), TOL,
                        "the copy reads plane 0: the history of all 12 tokens, the 2 rolled back included")

    def test_cross_stream_copy(self):
        M = self.ctx(unified=False)
        self.run_plan(M, [("d", [(0, 6), (2, 3)]), ("c", 0, 1)])
        self.assertEqual(M.kv.pos[1], M.kv.pos[0])
        self.assertEqual([sorted(x) for x in M.kv.seqs[1] if x], [[1]] * 6)
        self.assertEqual(M.kv.head[1], M.kv.head[0])
        self.assertEqual(M.pending_copies, [(0, 1)])
        rows, graphs = self.run_plan(M, [("d", [(0, 1), (1, 1)])])
        self.assertEqual(M.pending_copies, [])
        for il in M.QSA:
            self.assertTrue(np.array_equal(graphs[0]["idx_raw"][il][1][:6], graphs[0]["idx_raw"][il][0][:6]))


if __name__ == "__main__":
    unittest.main()
