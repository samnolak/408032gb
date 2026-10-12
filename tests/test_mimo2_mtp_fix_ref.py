"""N21 property tests (numpy only, no GGUF, no llama.cpp): ref/mimo2.py mtp_process(fixed=True) (the order of
patches/llama.cpp/0001-mtp-heads-slot-replay.patch) and ref/mtp_cells.py (the placement-only model of a multi-head MTP
context under llama-server's calls), docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md.

The random 4-layer model with 3 MTP layers of tests/test_mimo2_heads_ref.py (model3)."""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import mimo2 as M  # noqa: E402
from ref import mtp_cells as C  # noqa: E402
from tests.test_mimo2_chunked_ref import rel  # noqa: E402
from tests.test_mimo2_heads_ref import model3, target_h  # noqa: E402

N_CTX = 64


def run_calls(W, hp, toks, hprev, calls, fixed, ring=16, ub=8):
    """mtp_process over consecutive calls of the given sizes; returns {head: [ubatch results]} and the cache."""
    cache = M.new_cache(hp, N_CTX, ub, mtp=True, swa_size=ring)
    outs, p = {k: [] for k in range(3)}, 0
    for n in calls:
        r = M.mtp_process(toks[p:p + n], hprev[p:p + n], W, hp, cache, ub, fixed=fixed)
        for k in range(3):
            outs[k] += r[k]
        p += n
    return outs, cache, p


class MtpFixRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = model3(1, n_swa=4)
        rng = np.random.default_rng(5)
        cls.toks = [int(x) for x in rng.integers(0, 20, size=48)]
        h = target_h(cls.W, cls.hp, cls.toks)
        cls.hprev = np.vstack([np.zeros((1, cls.hp["n_embd"])), h[:-1]])

    def full(self, k, p):
        return M.mtp_forward_chunk(self.toks[:p], self.hprev[:p], self.W, self.hp, offset=k)

    def check(self, calls, expect_upstream):
        """Upstream: the rows flagged (foreign K/V or an incomplete window) differ from the full-sequence block and
        the others equal it; fixed: no row flagged, every head equals the block."""
        res = {}
        for fixed in (False, True):
            outs, cache, p = run_calls(self.W, self.hp, self.toks, self.hprev, calls, fixed)
            flagged = {}
            for k in range(3):
                full = self.full(k, p)["logits"]
                got = np.vstack([o["logits"] for o in outs[k]])
                bad = np.concatenate([o["attn"]["foreign"] | o["attn"]["missing"] for o in outs[k]])
                flagged[k] = int(bad.sum())
                if bad.any():
                    self.assertGreater(rel(got[bad], full[bad]), 1e-3, (fixed, k))
                self.assertLess(rel(got[~bad], full[~bad]), 1e-10, (fixed, k))
            res[fixed] = (flagged, outs)
            if fixed:
                self.assertEqual(flagged, {0: 0, 1: 0, 2: 0})
        self.assertEqual({k: v > 0 for k, v in res[False][0].items()}, expect_upstream)
        return res

    def test_wrap_of_one_call(self):
        """N17's case on a 16-cell ring: the call of 4 at 14..17 wraps; upstream head 0 reads foreign K/V; fixed: every
        head writes cells [14, 15, 0, 1]."""
        calls = [1] * 14 + [4] + [1] * 12
        res = self.check(calls, {0: True, 1: False, 2: False})
        outs = res[True][1]
        cells = [[outs[k][14]["slots"]["swa"] for k in range(3)]]
        self.assertEqual(cells, [[[14, 15, 0, 1]] * 3])

    def test_call_of_several_ubatches(self):
        """A call of 20 tokens in ubatches of 8 after 10 tokens: upstream, head 0's later ubatches take cells the first
        ubatch's window needs before heads 1 and 2 run it (incomplete windows), and head 0 later reads foreign K/V;
        fixed: nothing flagged."""
        res = self.check([10, 20] + [1] * 10, {0: True, 1: True, 2: True})
        up = res[False][1]
        miss = {k: sorted(int(p) for o in up[k] for p, x in zip(o["pos"], o["attn"]["missing"]) if x) for k in range(3)}
        self.assertEqual(miss[0], [])
        self.assertTrue(miss[1] and miss[1] == miss[2] and min(miss[1]) == 10, miss)

    def test_fixed_equals_upstream_without_wrap(self):
        """Calls of at most one ubatch that wrap nothing: the fixed order computes the same numbers bit for bit."""
        calls = [8, 1, 1, 3, 1]
        a, _, _ = run_calls(self.W, self.hp, self.toks, self.hprev, calls, False, ring=32)
        b, _, _ = run_calls(self.W, self.hp, self.toks, self.hprev, calls, True, ring=32)
        for k in range(3):
            for x, y in zip(a[k], b[k]):
                self.assertTrue(np.array_equal(x["logits"], y["logits"]))
                self.assertEqual(x["slots"], y["slots"])

    def test_replay_refused_on_a_taken_cell(self):
        cache = M.new_cache(self.hp, N_CTX, 8, mtp=True, swa_size=16)
        o = M.mtp_forward_chunk(self.toks[:3], self.hprev[:3], self.W, self.hp, cache)
        with self.assertRaises(RuntimeError):
            M.mtp_forward_chunk(self.toks[3:6], self.hprev[3:6], self.W, self.hp, cache, replay=o["slots"])

    def test_cell_model_agrees_with_the_oracle(self):
        """ref/mtp_cells.py flags exactly the rows and cells ref/mimo2.py's cell oracle flags, both orders."""
        calls = [10, 20, 1, 1, 4, 1, 1, 1, 1, 4, 1]
        for fixed in (False, True):
            outs, _, _ = run_calls(self.W, self.hp, self.toks, self.hprev, calls, fixed)
            m = C.MtpCells(16, 4, 3, 8, fixed)
            p = 0
            for n in calls:
                m.process(p, n)
                p += n
            order = [(k, j) for k in range(3) for j in range(len(outs[k]))]
            ours = [o for o in m.log]
            # the oracle's results per head in call order; the cell model logs in execution order
            want = sorted(((int(outs[k][j]["pos"][0]), k), outs[k][j]) for k, j in order)
            got = sorted(((e[2][0], e[1]), e) for e in ours)
            self.assertEqual(len(want), len(got))
            for (kw, o), (kg, e) in zip(want, got):
                self.assertEqual(kw, kg)
                self.assertEqual(e[4], o["slots"]["swa"], (fixed, kw))
                self.assertEqual(list(e[3]), list(o["attn"]["foreign"]), (fixed, kw))
                self.assertEqual(list(e[5]), list(o["attn"]["missing"]), (fixed, kw))


class MtpCellsModel(unittest.TestCase):
    def test_ring_size(self):
        """MiMo-V2.6-Flash: window 128 (config.json sliding_window), n_ubatch 512: 768 cells; -ub 2048: 2304."""
        self.assertEqual(C.ring_size(262144, 128, 512), 768)
        self.assertEqual(C.ring_size(262144, 128, 2048), 2304)
        self.assertEqual(C.ring_size(512, 8, 64), 256)

    def test_n17_plan(self):
        """N17's mtp3 plan: upstream 11 head-0 rows at 257..263 read foreign K/V; fixed none."""
        one = [1] * 8
        plan = [240] + one + ["d3", 4, "r2", "d3", 4, "r1", "d3", 4, 1, 1, "d3", 4, "r3"] + [1] * 6

        def go(fixed):
            m = C.MtpCells(256, 8, 3, 64, fixed)
            pos = 0
            for s in plan:
                if s == "d3":
                    m.draft(pos, 3)
                elif isinstance(s, str):
                    pos -= int(s[1:])
                    m.accept(pos)
                else:
                    m.process(pos, s)
                    pos += s
            m.draft(pos, 3)
            return [(e[1], p) for e in m.log for p, x in zip(e[2], e[3]) if x]
        up = go(False)
        self.assertEqual(len(up), 11)
        self.assertEqual({h for h, _ in up}, {0})
        self.assertEqual(sorted({p for _, p in up}), list(range(257, 264)))
        self.assertEqual(go(True), [])

    def test_server_exposure_mimo_geometry(self):
        """llama-server-like decode at MiMo-V2.6-Flash geometry (768-cell ring, window 128, 3 heads, verify batches
        of 4): upstream, some head-0 drafts read foreign K/V and no head-1/2 draft does; with the fix none; with one
        head none; with a ring as large as the context (--swa-full) none."""
        up = C.summary(C.serve(700, 800, 0.3, 768, 128, seed=1))
        self.assertGreater(up["bad"][0], 0)
        self.assertEqual(up["bad"][1:], [0, 0])
        self.assertEqual(C.summary(C.serve(700, 800, 0.3, 768, 128, fixed=True, seed=1))["bad"], [0, 0, 0])
        self.assertEqual(C.summary(C.serve(700, 800, 0.3, 768, 128, n_heads=1, seed=1))["bad"], [0])
        self.assertEqual(C.summary(C.serve(700, 800, 0.3, 4096, 128, seed=1))["bad"], [0, 0, 0])

    def test_long_prompt_first_drafts(self):
        """A prompt of 5000 tokens in calls of 2048 (ubatches of 512): upstream, the first drafts after the prompt read
        foreign K/V in head 0; fixed none."""
        up = C.summary(C.serve(5000, 64, 0.6, 768, 128, seed=1))
        self.assertEqual(up["first_bad"][0], 5000)
        self.assertGreater(up["n_proc_missing"], 0)
        fx = C.summary(C.serve(5000, 64, 0.6, 768, 128, fixed=True, seed=1))
        self.assertEqual((fx["bad"], fx["n_proc_bad"], fx["n_proc_missing"]), ([0, 0, 0], 0, 0))


if __name__ == "__main__":
    unittest.main()
