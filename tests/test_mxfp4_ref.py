"""ref/quant.py MXFP4 (N16): the block layout, the E8M0 half scale, the dequantization and the block form of the CPU
product, against their definitions and (with LLAMA_CPP_GGUF_PY) against the pinned gguf-py's dequantize.

numpy only except test_dequant_is_gguf_py (skips cleanly without LLAMA_CPP_GGUF_PY; the N16 driver sets it).
"""
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import quant as QT  # noqa: E402

GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")


def random_blocks(rng, n_blocks, e_lo=0, e_hi=255):
    raw = rng.integers(0, 256, (n_blocks, QT.MXFP4_BLOCK_BYTES), dtype=np.uint8)
    raw[:, 0] = rng.integers(e_lo, e_hi, n_blocks)
    return raw.reshape(-1)


class Mxfp4Ref(unittest.TestCase):
    def test_e8m0_half_is_two_to_the_e_minus_128(self):
        e = np.arange(256, dtype=np.uint8)
        got = QT.e8m0_to_fp32_half(e).astype(np.float64)
        self.assertTrue(np.array_equal(got, np.ldexp(1.0, e.astype(int) - 128)))
        self.assertEqual(QT.e8m0_to_fp32_half(np.uint8(0)).view(np.uint32), 0x00200000)   # denormal 2^-128
        self.assertEqual(QT.e8m0_to_fp32_half(np.uint8(1)).view(np.uint32), 0x00400000)   # denormal 2^-127

    def test_block_layout_and_values(self):
        """A hand-made block: e = 130 (d = 4), qs[j] = (code of weight j + 16) << 4 | code of weight j."""
        codes = np.arange(32) % 16
        qs = (codes[:16] | (codes[16:] << 4)).astype(np.uint8)
        raw = np.concatenate([[130], qs]).astype(np.uint8)
        got = QT.mxfp4_dequant(raw)
        want = 4.0 * np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12] * 2)
        self.assertTrue(np.array_equal(got, want))
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(QT.MXFP4_BLOCK_BYTES, 17)
        self.assertEqual(QT.VEC_DOT["MXFP4"], "Q8_0")

    def test_vec_dot_is_the_dot_of_the_dequantized_operands(self):
        """The block form of ggml_vec_dot_mxfp4_q8_0 equals dequant(w) . act_quant(x, Q8_0): the CPU product is the
        product of the dequantized weight with the dequantized Q8_0 activation (what QWeight computes)."""
        rng = np.random.default_rng(5)
        for nb in (1, 2, 3, 8):
            for _ in range(20):
                raw = random_blocks(rng, nb, 100, 140)
                x = rng.standard_normal(nb * 32).astype(np.float32) * rng.uniform(0.01, 10)
                got = QT.mxfp4_q8_0_dot(raw, x)
                want = float(QT.mxfp4_dequant(raw).astype(np.float64) @ QT.act_quant(x, "Q8_0"))
                self.assertAlmostEqual(got, want, delta=1e-12 * max(1.0, abs(want)))
                w = QT.QWeight(QT.mxfp4_dequant(raw)[None], "MXFP4")
                self.assertAlmostEqual(float((w @ x)[0]), want, delta=1e-12 * max(1.0, abs(want)))

    @unittest.skipUnless(GGUF_PY, "set LLAMA_CPP_GGUF_PY (the pinned gguf-py) to compare with its dequantize")
    def test_dequant_is_gguf_py(self):
        """Every exponent 0..254 (denormal scales included) and random codes: ref/quant.py's dequantization equals
        gguf.quants.dequantize of the pinned gguf-py (gguf-py/gguf/quants.py class MXFP4), bit for bit."""
        sys.path.append(GGUF_PY)
        try:
            from gguf.constants import GGMLQuantizationType
            from gguf.quants import dequantize
        finally:
            sys.path.remove(GGUF_PY)
        rng = np.random.default_rng(11)
        raw = random_blocks(rng, 4096)
        raw.reshape(-1, 17)[:255, 0] = np.arange(255)
        with np.errstate(over="ignore"):            # beyond the f32 range (e >= 253) both give inf
            mine = QT.mxfp4_dequant(raw.reshape(64, -1))
            theirs = dequantize(raw.reshape(64, -1), GGMLQuantizationType.MXFP4)
        self.assertEqual(mine.shape, theirs.shape)
        self.assertTrue(np.array_equal(mine.view(np.uint32), np.asarray(theirs, dtype=np.float32).view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
