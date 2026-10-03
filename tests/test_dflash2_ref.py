"""DFlash2 oracle (ref/dflash2.py) against the vendored vLLM reference (third_party/vllm-dflash2, ADR-011).

The reference functions are taken out of the unchanged vendored file by name (ast) and executed with torch on the CPU;
without torch those tests are skipped and only the definition-level checks run.
"""
import ast
import os
import unittest

import numpy as np

from ref import dflash2 as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "vllm-dflash2", "vllm", "model_executor", "models", "qwen3_dflash2.py")

try:
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # noqa: BLE001 - any import failure means "no reference run here"
    HAVE_TORCH = False


def reference_functions(*names):
    tree = ast.parse(open(SRC).read())
    ns = {"torch": torch, "F": F}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.Module([node], []), SRC, "exec"), ns)  # noqa: S102 - vendored, pinned, hash-checked source
    return [ns[n] for n in names]


class Definition(unittest.TestCase):
    """The oracle against the written-out definitions (the same loops vLLM's own tests use)."""

    def test_grouped_conv_definition(self):
        rng = np.random.default_rng(0)
        for block, taps, groups, gs, batch in ((8, 2, 5, 4, 3), (5, 3, 4, 2, 2), (8, 1, 3, 2, 2)):
            h = rng.standard_normal((batch * block, groups * gs))
            delta = rng.standard_normal((batch * block, taps, groups))
            base = rng.standard_normal((taps, groups * gs))
            got = D.grouped_conv(h, delta, base, block, gs).reshape(batch, block, groups, gs)
            hb, db, bb = h.reshape(batch, block, groups, gs), delta.reshape(batch, block, taps, groups), base.reshape(taps, groups, gs)
            want = np.zeros_like(hb)
            for pos in range(block):
                for tap in range(min(taps, pos + 1)):
                    want[:, pos] += (bb[tap] + db[:, pos, tap, :, None]) * hb[:, pos - tap]
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_conv_never_crosses_a_block_boundary(self):
        rng = np.random.default_rng(1)
        block, gs = 8, 4
        h = rng.standard_normal((2 * block, 16))
        delta, base = rng.standard_normal((2 * block, 2, 4)), rng.standard_normal((2, 16))
        a = D.grouped_conv(h, delta, base, block, gs)
        h2 = h.copy()
        h2[:block] += 1.0                                  # perturb the whole first block
        b = D.grouped_conv(h2, delta, base, block, gs)
        self.assertTrue(np.array_equal(a[block:], b[block:]))   # the second block does not see it
        self.assertFalse(np.allclose(a[:block], b[:block]))

    def test_walk_is_greedy_and_follows_the_previous_choice(self):
        rng = np.random.default_rng(2)
        b, steps, k = 3, 7, 4
        cand = rng.integers(0, 50, (b, steps, k))
        scores = rng.standard_normal((b, steps, k, k))
        tokens, picks = D.selector_walk(scores, cand)
        for r in range(b):
            prev = 0
            for l in range(steps):
                self.assertEqual(picks[r, l], int(np.argmax(scores[r, l, prev])))
                self.assertEqual(tokens[r, l], cand[r, l, picks[r, l]])
                prev = picks[r, l]

    def test_zero_codebooks_reduce_to_top1_of_the_head(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((2, 7, 40))
        cand, unary = D.top_k_candidates(logits, 16)
        scores = D.score_edges(np.zeros((40, 8)), np.zeros((40, 8)), cand, unary, rng.standard_normal((2, 7, 8)), np.array([1, 2]))
        tokens, _ = D.selector_walk(scores, cand)
        self.assertTrue(np.array_equal(tokens, logits.argmax(-1)))

    def test_context_kv_shapes_and_key_norm(self):
        rng = np.random.default_rng(4)
        layers, heads, hd, hid, taps, tgt, n = 3, 2, 4, 16, 3, 8, 5
        K, V = D.context_kv(rng.standard_normal((n, taps * tgt)), rng.standard_normal((hid, taps * tgt)), np.ones(hid),
                            rng.standard_normal((layers, heads * hd, hid)), rng.standard_normal((layers, heads * hd, hid)),
                            np.ones((layers, hd)), heads, hd, 1e-6)
        self.assertEqual(K.shape, (layers, n, heads, hd))
        self.assertEqual(V.shape, K.shape)
        np.testing.assert_allclose((K * K).mean(-1), 1.0, rtol=1e-4)     # keys are RMS-normed per head, values are not


@unittest.skipUnless(HAVE_TORCH, "torch is not installed: the vendored vLLM functions cannot be executed")
class AgainstVllm(unittest.TestCase):
    def test_grouped_conv(self):
        (ref,) = reference_functions("_grouped_conv")
        torch.manual_seed(0)
        for block, taps, groups, gs, batch in ((8, 2, 16, 16, 4), (8, 2, 256, 16, 2), (5, 3, 4, 2, 3)):
            h = torch.randn(batch * block, groups * gs, dtype=torch.float64)
            delta = torch.randn(batch * block, taps, groups, dtype=torch.float64)
            base = torch.randn(taps, groups * gs, dtype=torch.float64)
            want = ref(h, delta, base, block, groups, gs, taps).numpy()
            got = D.grouped_conv(h.numpy(), delta.numpy(), base.numpy(), block, gs)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_score_edges(self):
        (ref,) = reference_functions("_score_edges")
        torch.manual_seed(1)
        b, steps, k, rank, vocab = 3, 7, 16, 32, 97
        P, S = torch.randn(vocab, rank, dtype=torch.float64), torch.randn(vocab, rank, dtype=torch.float64)
        cand = torch.randint(vocab, (b, steps, k))
        unary = torch.randn(b, steps, k, dtype=torch.float64)
        hidden = torch.randn(b, steps, rank, dtype=torch.float64)
        anchors = torch.randint(vocab, (b,))
        want = ref(P, S, cand, unary, hidden, anchors, k).numpy()
        got = D.score_edges(P.numpy(), S.numpy(), cand.numpy(), unary.numpy(), hidden.numpy(), anchors.numpy())
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
