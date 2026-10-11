"""Property tests of ref/mimo2.py's N17 additions: several MTP heads (nextn_layer_offset), the chained process / draft of
common/speculative.cpp, the per-layer record of which position each cell's K/V was computed for (the head-0 finding of
docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md) and the precision argument. numpy only: a random 4-layer model
with 3 MTP layers built here (tests/test_mimo2_chunked_ref.py random_model plus two more MTP layers), no GGUF.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import fattn as F  # noqa: E402
from ref import mimo2 as M  # noqa: E402
from tests.test_mimo2_chunked_ref import random_model, rel  # noqa: E402

N_CTX = 64


def model3(seed=0, n_swa=4):
    """random_model with MTP layers 4, 5, 6 (the first is random_model's own MTP layer)."""
    W, hp = random_model(seed, fused=True, sinks="swa", v_scale=0.75, mtp=True, n_swa=n_swa)
    rng = np.random.default_rng(seed + 100)
    L = hp["n_layer"]
    for il in (L + 1, L + 2):
        for name, a in list(W.items()):
            if name.startswith(f"blk.{L}."):
                W[name.replace(f"blk.{L}.", f"blk.{il}.", 1)] = a + 0.3 * rng.standard_normal(a.shape) * np.abs(a).mean()
    hp = dict(hp, n_nextn=3, n_all=L + 3, swa=hp["swa"] + [True, True], n_head=hp["n_head"] + hp["n_head"][-1:] * 2,
              n_head_kv=hp["n_head_kv"] + hp["n_head_kv"][-1:] * 2)
    return W, hp


def target_h(W, hp, tokens):
    return M.forward(tokens, W, hp)["h_nextn"]


class MtpHeadsRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = model3()
        rng = np.random.default_rng(5)
        cls.toks = [int(x) for x in rng.integers(0, 20, size=40)]
        cls.h = target_h(cls.W, cls.hp, cls.toks)
        cls.h_prev = np.vstack([np.zeros((1, cls.hp["n_embd"])), cls.h[:-1]])

    def test_offset_range(self):
        """graph_mtp asserts 0 <= nextn_layer_offset < n_layer_nextn (src/models/mimo2.cpp:267-270 @79e2e74)."""
        for off in (-1, 3):
            with self.assertRaises(ValueError):
                M.mtp_forward_chunk(self.toks[:3], self.h_prev[:3], self.W, self.hp, offset=off)
        self.assertEqual(M.new_cache(self.hp, N_CTX, 8, mtp=True)["layers"], [4, 5, 6])

    def test_head_k_is_layer_n_layer_plus_k(self):
        """Changing the weights of layer 5 changes head 1 only."""
        W2 = dict(self.W)
        W2["blk.5.attn_output.weight"] = self.W["blk.5.attn_output.weight"] * 1.5
        for head in range(3):
            a = M.mtp_forward_chunk(self.toks, self.h_prev, self.W, self.hp, offset=head)["logits"]
            b = M.mtp_forward_chunk(self.toks, self.h_prev, W2, self.hp, offset=head)["logits"]
            self.assertEqual(np.array_equal(a, b), head != 1, head)

    def test_chained_process_equals_every_head_over_the_sequence(self):
        """mtp_process over calls (seq_rm from the call's first position before each head, all heads with the target
        h) equals, for every head, the block over the whole sequence, while no call wraps the cache."""
        cache = M.new_cache(self.hp, N_CTX, 8, mtp=True)
        outs = {k: [] for k in range(3)}
        p = 0
        for n in (12, 5, 1, 1, 9, 3):
            r = M.mtp_process(self.toks[p:p + n], self.h_prev[p:p + n], self.W, self.hp, cache, 8)
            for k in range(3):
                outs[k] += r[k]
            p += n
        for k in range(3):
            full = M.mtp_forward_chunk(self.toks[:p], self.h_prev[:p], self.W, self.hp, offset=k)
            got = np.vstack([o["logits"] for o in outs[k]])
            self.assertLess(rel(got, full["logits"]), 1e-10, k)
            self.assertFalse(any(o["attn"]["foreign"].any() for o in outs[k]), k)

    def test_chained_draft(self):
        """Draft step k runs head k on [id_last, d_1 .. d_k] at pos0 .. pos0 + k with the rows [pending, h_0 .. h_k-1]
        (h_j: step j's h_nextn at its last row), after seq_rm(pos0); its last row equals a fresh run of that prefix."""
        cache = M.new_cache(self.hp, N_CTX, 8, mtp=True)
        M.mtp_process(self.toks[:20], self.h_prev[:20], self.W, self.hp, cache, 8)
        pending = self.h[19]
        snapshot = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in cache.items()}
        own, steps = M.mtp_draft(7, pending, self.W, self.hp, cache, 5)
        self.assertEqual(len(steps), 3)                                     # at most H steps (speculative.cpp:1485)
        for k, st in enumerate(steps):
            self.assertEqual(st["head"], k)
            self.assertEqual(st["pos0"], 20)
            self.assertEqual(st["tokens"], [7] + own[:k])
            want_rows = [pending] + [steps[j]["out"]["h_nextn"][-1] for j in range(k)]
            self.assertTrue(np.array_equal(st["rows"], np.vstack(want_rows)))
            fresh = M.new_cache(self.hp, N_CTX, 8, mtp=True)
            M.mtp_process(self.toks[:20], self.h_prev[:20], self.W, self.hp, fresh, 8)
            ref = M.mtp_forward_chunk(st["tokens"], st["rows"], self.W, self.hp, fresh, offset=k)
            self.assertLess(rel(st["out"]["logits"], ref["logits"]), 1e-12)
            self.assertEqual(own[k], int(np.argmax(st["out"]["logits"][-1])))
        self.assertEqual(len(cache["tokens"]), 23)
        self.assertEqual(snapshot["tokens"], self.toks[:20])

    def test_teacher_forced_draft_tokens(self):
        cache = M.new_cache(self.hp, N_CTX, 8, mtp=True)
        M.mtp_process(self.toks[:20], self.h_prev[:20], self.W, self.hp, cache, 8)
        own, steps = M.mtp_draft(7, self.h[19], self.W, self.hp, cache, 3, tokens=[11, 12])
        self.assertEqual([s["tokens"] for s in steps], [[7], [7, 11], [7, 11, 12]])

    def test_one_head_draft_is_n16(self):
        """One MTP layer: no seq_rm, step j feeds the previous argmax alone at pos0 + j with the previous h (N16)."""
        W1, hp1 = random_model(0, fused=True, sinks="swa", v_scale=0.75, mtp=True)
        h = target_h(W1, hp1, self.toks)
        hprev = np.vstack([np.zeros((1, hp1["n_embd"])), h[:-1]])
        cache = M.new_cache(hp1, N_CTX, 8, mtp=True)
        M.mtp_process(self.toks[:20], hprev[:20], W1, hp1, cache, 8)
        own, steps = M.mtp_draft(7, h[19], W1, hp1, cache, 4)
        self.assertEqual([(s["head"], s["pos0"], s["tokens"]) for s in steps],
                         [(0, 20, [7])] + [(0, 20 + j, [own[j - 1]]) for j in range(1, 4)])

    def test_head0_reads_foreign_kv_after_a_wrapping_call(self):
        """A 16-cell SWA cache: a call whose ubatch head 0 places across the end of the ring is placed by heads 1, 2
        from cell 0 after seq_rm; head 0's rows that see those positions then read K/V computed for other positions
        (attn["foreign"]) and differ from the block over the whole sequence; heads 1 and 2 do not."""
        W, hp = model3(1, n_swa=4)
        toks = self.toks[:30]
        h = target_h(W, hp, toks)
        hprev = np.vstack([np.zeros((1, hp["n_embd"])), h[:-1]])
        cache = M.new_cache(hp, N_CTX, 8, mtp=True, swa_size=16)
        outs, p = {k: [] for k in range(3)}, 0
        slots = []
        for n in [1] * 14 + [4] + [1] * 12:
            r = M.mtp_process(toks[p:p + n], hprev[p:p + n], W, hp, cache, 8)
            slots.append([r[k][0]["slots"]["swa"] for k in range(3)])
            for k in range(3):
                outs[k] += r[k]
            p += n
        self.assertIn([[14, 15, 0, 1], [0, 1, 2, 3], [0, 1, 2, 3]], slots)
        for k in range(3):
            full = M.mtp_forward_chunk(toks[:p], hprev[:p], W, hp, offset=k)
            got = np.vstack([o["logits"] for o in outs[k]])
            foreign = np.concatenate([o["attn"]["foreign"] for o in outs[k]])
            if k == 0:
                self.assertTrue(foreign.any())
                self.assertGreater(rel(got[foreign], full["logits"][foreign]), 1e-3)
                self.assertLess(rel(got[~foreign], full["logits"][~foreign]), 1e-10)
            else:
                self.assertFalse(foreign.any(), k)
                self.assertLess(rel(got, full["logits"]), 1e-10, k)

    def test_precision_argument(self):
        """prec None keeps the f32 contract; an f32 cache without flash attention only changes the scale to its f32
        value (~1e-8); the R1 default (f16, flash attention) rounds (the stored rows are f16 values) and stays close."""
        cache = M.new_cache(self.hp, N_CTX, 8)
        a = M.forward_chunk(self.toks[:9], self.W, self.hp, cache)
        b = M.forward_chunk(self.toks[:9], self.W, self.hp, M.new_cache(self.hp, N_CTX, 8), prec=F.precision("f32"))
        c = M.forward_chunk(self.toks[:9], self.W, self.hp, M.new_cache(self.hp, N_CTX, 8),
                            prec=F.precision("f16", True))
        self.assertLess(rel(b["logits"], a["logits"]), 1e-6)
        self.assertGreater(rel(c["logits"], a["logits"]), 1e-6)
        self.assertLess(rel(c["logits"], a["logits"]), 1e-2)
        self.assertEqual(M.kq_scale(dict(k_dim=24), F.precision("f16", True)), F.kq_scale_f32(24))
        self.assertEqual(M.kq_scale(dict(k_dim=24)), 1.0 / np.sqrt(24))
        st = c["layers"][1]["attn"]["stored"]["k"]
        self.assertTrue(np.array_equal(st, F.f16(st)))
        self.assertEqual(set(c["layers"][1]["attn"]["modes"]), {"fa_chunk"})


if __name__ == "__main__":
    unittest.main()
