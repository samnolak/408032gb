"""ref/kda.py versus the real ggml_gated_delta_net (CPU backend, llama.cpp @ec7630a).

Build the harness first (tests/ggml_parity/build.sh), then:
  GGML_GDN_HARNESS=tests/ggml_parity/gdn_harness python3 -m unittest tests.test_kda_vs_ggml -v
The op does not normalise q/k (llama.cpp does that before calling it), so the oracle runs
with normalize_qk=False on pre-normalised inputs.
"""
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref.kda import gated_delta_rule, l2_norm  # noqa: E402

HARNESS = os.environ.get("GGML_GDN_HARNESS")


def run_ggml(q, k, v, g, beta, state, kda):
    T, Hk, S = q.shape
    Hv = v.shape[1]
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(fin, "wb") as f:
            f.write(struct.pack("<5i", T, Hk, Hv, S, int(kda)))
            for a in (q, k, v, g if kda else g[:, :, None], beta[:, :, None], state.transpose(0, 2, 1)):
                f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        subprocess.run([HARNESS, fin, fout], check=True)
        raw = np.fromfile(fout, dtype=np.float32)
    attn = raw[: T * Hv * S].reshape(T, Hv, S)
    st = raw[T * Hv * S:].reshape(Hv, S, S).transpose(0, 2, 1)
    return attn, st


@unittest.skipUnless(HARNESS, "set GGML_GDN_HARNESS (see tests/ggml_parity/build.sh)")
class OracleVsGgml(unittest.TestCase):
    def check(self, kda, T=9, Hk=2, Hv=4, S=16, seed=7):
        r = np.random.default_rng(seed)
        q = l2_norm(r.standard_normal((T, Hk, S)))
        k = l2_norm(r.standard_normal((T, Hk, S)))
        v = r.standard_normal((T, Hv, S))
        g = -r.uniform(0.01, 2.0, (T, Hv, S) if kda else (T, Hv))
        beta = 1 / (1 + np.exp(-r.standard_normal((T, Hv))))
        s0 = r.standard_normal((Hv, S, S)) * 0.1
        # feed the oracle exactly the float32 values ggml sees
        f32 = lambda a: a.astype(np.float32).astype(np.float64)  # noqa: E731
        o_ref, s_ref = gated_delta_rule(f32(q), f32(k), f32(v), f32(g), f32(beta), f32(s0), normalize_qk=False)
        o_g, s_g = run_ggml(q, k, v, g, beta, s0, kda)
        np.testing.assert_allclose(o_g, o_ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(s_g, s_ref, rtol=1e-4, atol=1e-5)
        return float(np.max(np.abs(o_g - o_ref)))

    def test_kda_per_channel_gate(self):
        print(f"\n  KDA max|ggml-oracle| = {self.check(kda=True):.3e}", end="")

    def test_gdn_scalar_gate(self):
        print(f"\n  GDN max|ggml-oracle| = {self.check(kda=False):.3e}", end="")

    def test_glm_head_geometry(self):
        # GLM-5.3-Flash KDA: 64 heads x 128 (CONFIRMED config.json); 4 heads keep the test fast
        print(f"\n  KDA S=128 max|ggml-oracle| = {self.check(kda=True, T=4, Hk=4, Hv=4, S=128):.3e}", end="")


if __name__ == "__main__":
    unittest.main()
