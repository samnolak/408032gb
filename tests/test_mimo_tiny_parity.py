"""ref/mimo.py and ref/router.py versus llama.cpp's real mimo2 graph on a tiny random model (CPU only).

Run: tools/tinygen/mimo2_tiny.py -> tools/dump/dump_tensors.cpp (-fa off -ctk f32 -ctv f32) -> this test.
Env: MIMO_TINY_DUMP, MIMO_TINY_GGUF, LLAMA_CPP_GGUF_PY.
Every comparison is recorded; the table printed at the end of the module lists, per component and layer, the max abs
and normwise relative error and how close it is to its tolerance (gap 14 of docs/reassessment/CORRECTNESS.md;
tolerances unchanged).
"""
import json
import os
import pathlib
import re
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref.mimo import attention, rope_neox  # noqa: E402
from ref.router import route  # noqa: E402

DUMP, GGUF, GGUF_PY = os.environ.get("MIMO_TINY_DUMP"), os.environ.get("MIMO_TINY_GGUF"), os.environ.get("LLAMA_CPP_GGUF_PY")


def dump(name):
    j = json.loads(pathlib.Path(DUMP, name + ".json").read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    a = np.fromfile(pathlib.Path(DUMP, name + ".bin"), dtype=dt).reshape(j["ne"][::-1]).astype(np.float64)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return a


TABLE = []    # (component, layer, max abs, rel, tol, compared): the per-layer error report (gap 14)


def record(what, il, mx, rel, tol, compared="rel"):
    TABLE.append((what, il, mx, rel, tol, compared))


def close(t, got, want, tol, what):
    """Normwise relative error < tol; the layer is the ` L<n>` in `what`. Sizes must match (no silent broadcasting)."""
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    t.assertEqual(got.size, want.size, f"{what}: size {got.shape} vs {want.shape}")
    got = got.reshape(want.shape)
    m = re.fullmatch(r"(.*) L(\d+)(.*)", what)
    comp, il = (m[1] + m[3], int(m[2])) if m else (what, -1)
    mx = float(np.max(np.abs(got - want)))
    err = mx / (np.max(np.abs(want)) + 1e-12)
    record(comp, il, mx, float(err), tol)
    t.assertLess(err, tol, f"{what}: rel err {err:.2e}")


def tearDownModule():
    if not TABLE:
        return
    print("\nper-layer errors (rel = max|got-want| / max|want|; 'of tol' = the compared error / tolerance):",
          file=sys.stderr)
    print(f"{'component':24s} {'layer':>5s} {'max abs':>10s} {'rel':>10s} {'tol':>12s} {'of tol':>7s}", file=sys.stderr)
    for what, il, mx, rel, tol, cmp in TABLE:
        frac = (rel if cmp == "rel" else mx) / tol
        print(f"{what:24s} {il:5d} {mx:10.3e} {rel:10.3e} {tol:8.0e} {cmp:3s} {frac:7.3f}", file=sys.stderr)
    worst = {}
    for row in TABLE:
        what, il, mx, rel, tol, cmp = row
        frac = (rel if cmp == "rel" else mx) / tol
        if il not in worst or frac > worst[il][0]:
            worst[il] = (frac, row)
    print("worst per layer:", file=sys.stderr)
    for il in sorted(worst):
        frac, (what, _, mx, rel, tol, cmp) = worst[il]
        print(f"  L{il:<3d} {what}: max abs {mx:.3e}, rel {rel:.3e}, {frac:.3f} of tol {tol:.0e} {cmp}", file=sys.stderr)


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, "run tools/tinygen/run_parity.sh mimo")
class MimoTinyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, GGUF_PY)
        from gguf import GGUFReader
        r = GGUFReader(GGUF)
        cls.W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
        f = {x.name: x for x in r.fields.values()}
        val = lambda k: f[k].parts[f[k].data[0]][0]  # noqa: E731
        cls.L, cls.H = int(val("mimo2.block_count")), int(val("mimo2.attention.head_count"))
        cls.swa = [bool(f["mimo2.attention.sliding_window_pattern"].parts[i][0])
                   for i in f["mimo2.attention.sliding_window_pattern"].data]
        cls.n_swa, cls.n_rot = int(val("mimo2.attention.sliding_window")), int(val("mimo2.rope.dimension_count"))
        cls.base, cls.base_swa = float(val("mimo2.rope.freq_base")), float(val("mimo2.rope.freq_base_swa"))
        cls.T = int(json.loads(pathlib.Path(DUMP, "logits.json").read_text())["n_tokens"])
        cls.Dk = int(val("mimo2.attention.key_length"))

    def test_rope(self):
        for il in range(self.L):
            base = self.base_swa if self.swa[il] else self.base
            for name in ("Qcur", "Kcur"):
                pre = dump(f"{name}-{il}__0").reshape(self.T, -1, self.Dk)
                close(self, rope_neox(pre, np.arange(self.T), self.n_rot, base), dump(f"{name}-{il}__1"), 1e-5,
                      f"{name} rope L{il}")

    def test_attention_swa_sinks_gqa(self):
        for il in range(self.L):
            q, k = dump(f"Qcur-{il}__1"), dump(f"Kcur-{il}__1")
            v = dump(f"Vcur-{il}__0").reshape(self.T, k.shape[1], -1)
            out = attention(q, k, v, self.W[f"blk.{il}.attn_sinks.weight"], self.n_swa if self.swa[il] else 0)
            close(self, out, dump(f"kqv_out-{il}__0"), 1e-5, f"kqv_out L{il} ({'swa' if self.swa[il] else 'global'})")
            close(self, out @ self.W[f"blk.{il}.attn_output.weight"].T, dump(f"attn_out-{il}__0"), 1e-5, f"attn_out L{il}")

    def test_sinks_matter(self):   # guard: the check above must fail without the sinks
        il = 1
        q, k = dump(f"Qcur-{il}__1"), dump(f"Kcur-{il}__1")
        v = dump(f"Vcur-{il}__0").reshape(self.T, k.shape[1], -1)
        no_sinks = attention(q, k, v, np.full(self.H, -1e30), self.n_swa)
        self.assertGreater(np.max(np.abs(no_sinks - dump(f"kqv_out-{il}__0"))), 1e-3)

    def test_router(self):
        for il in range(1, self.L):
            ids, w = route(dump(f"ffn_moe_logits-{il}__0"), self.W[f"blk.{il}.exp_probs_b.bias"], k=2, w_scale=1.0)
            got_ids = dump(f"ffn_moe_topk-{il}__0").astype(int)
            self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(got_ids, 1)), f"top-k L{il}")
            got_w = dump(f"ffn_moe_weights_norm-{il}__0").reshape(self.T, -1)
            pairs = [(dict(zip(ids[t], w[t])), dict(zip(got_ids[t], got_w[t]))) for t in range(self.T)]
            diffs = [abs(mine[e] - theirs.get(e, np.inf)) for mine, theirs in pairs for e in mine]
            record("router weights", il, float(max(diffs)), float(max(diffs) / (np.max(np.abs(got_w)) + 1e-12)),
                   1e-5, "abs")
            for mine, theirs in pairs:
                for e in mine:
                    self.assertAlmostEqual(mine[e], theirs[e], delta=1e-5)

    def test_plain_swiglu(self):
        for il in range(1, self.L):
            g, u = dump(f"ffn_moe_gate-{il}__0"), dump(f"ffn_moe_up-{il}__0")
            close(self, g / (1 + np.exp(-g)) * u, dump(f"ffn_moe_swiglu-{il}__0"), 1e-5, f"swiglu L{il}")


if __name__ == "__main__":
    unittest.main()
