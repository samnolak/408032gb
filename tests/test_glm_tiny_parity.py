"""Oracles in ref/ versus llama.cpp's real glm5-next graph, on a tiny random model (CPU only).

Pipeline (tools/tinygen/run_parity.sh does all of it):
  1. tools/tinygen/glm5next_tiny.py writes a 4-layer GLM-5.3-Flash-shaped GGUF;
  2. tools/dump/dump_tensors.cpp runs llama.cpp @ec7630a on it and dumps named tensors;
  3. this test recomputes each operation from the dumped inputs and the GGUF weights.
Env: GLM_TINY_DUMP (dump dir), GLM_TINY_GGUF (model), LLAMA_CPP_GGUF_PY (gguf-py path).
The reference run uses f32 caches and no flash attention (-ctk f32 -ctv f32 -fa off): with the
default f16 cache llama.cpp rounds q, the cached latent and the attention probabilities to f16
(reproduced to 6e-8 in docs/evidence/2026-10-02-g1-parity.md), which is a precision choice of the
reference, not of the model.
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref.kda import gated_delta_rule  # noqa: E402
from ref.mhc import hc_post, hc_pre  # noqa: E402
from ref.router import route, swiglu_clamp  # noqa: E402

DUMP = os.environ.get("GLM_TINY_DUMP")
GGUF = os.environ.get("GLM_TINY_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")


def dump(name):
    j = json.loads(pathlib.Path(DUMP, name + ".json").read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    a = np.fromfile(pathlib.Path(DUMP, name + ".bin"), dtype=dt).reshape(j["ne"][::-1]).astype(np.float64)
    while a.ndim > 1 and a.shape[0] == 1:      # drop leading ggml dims of size 1 (ne3, ne2, ...)
        a = a[0]
    return a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class _TinyBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, GGUF_PY)
        from gguf import GGUFReader
        r = GGUFReader(GGUF)
        cls.W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
        cls.kv = {f.name: f for f in r.fields.values()}
        cls.T = int(json.loads(pathlib.Path(DUMP, "logits.json").read_text())["n_tokens"])
        n_layer = int(cls.kv["glm5-next.block_count"].parts[-1][0])
        hkv = [int(cls.kv["glm5-next.attention.head_count_kv"].parts[i][0])
               for i in cls.kv["glm5-next.attention.head_count_kv"].data]
        dense = int(cls.kv["glm5-next.leading_dense_block_count"].parts[-1][0])
        cls.N = n_layer
        cls.KDA = [i for i in range(n_layer) if hkv[i] == 0]
        cls.DSA = [i for i in range(n_layer) if hkv[i] > 0]
        cls.MOE = list(range(dense, n_layer))

    def streams_before(self, il):
        x = dump("hc_init__0") if il == 0 else dump(f"l_out-{il - 1}__0")
        return x.reshape(self.T, 4, -1)

    def assertClose(self, got, want, tol, what):
        err = np.max(np.abs(got - want)) / (np.max(np.abs(want)) + 1e-12)
        self.assertLess(err, tol, f"{what}: rel err {err:.2e}")
        return err


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, "run tools/tinygen/run_parity.sh")
class GlmTinyParity(_TinyBase):
    # ---- mHC: mixes -> pre, post, Sinkhorn comb, mixer input, new streams ----
    def test_mhc_attn(self):
        for il in range(self.N):
            s = self.streams_before(il)
            b = f"blk.{il}."
            mix, post, comb = hc_pre(s, self.W[b + "hc_attn_fn.weight"], self.W[b + "hc_attn_scale.weight"],
                                     self.W[b + "hc_attn_base.weight"], rms_eps=1e-5, hc_eps=1e-6, iters=20)
            self.assertClose(post, dump(f"hc_post-{il}__0"), 1e-5, f"hc_post L{il}")
            self.assertClose(comb, dump(f"hc_comb-{il}__0").transpose(0, 2, 1), 1e-5, f"hc_comb L{il}")
            self.assertClose(mix, dump(f"hc_attn_pre-{il}__0"), 1e-5, f"hc mixer input L{il}")
            mixer_out = dump(f"kda_out-{il}__0") if il in self.KDA else dump(f"attn_out-{il}__0")
            new = hc_post(mixer_out.reshape(self.T, -1), s, post, comb)
            self.assertClose(new, dump(f"hc_attn_post-{il}__0").reshape(self.T, 4, -1), 1e-5, f"hc_attn_post L{il}")

    # ---- KDA: gate formula, short conv, recurrence ----
    def test_kda_gate(self):
        for il in self.KDA:
            b = f"blk.{il}."
            x = dump(f"attn_norm-{il}__0")
            raw = (x @ self.W[b + "ssm_f_a.weight"].T) @ self.W[b + "ssm_f_b.weight"].T + self.W[b + "ssm_dt.bias"]
            H = self.W[b + "ssm_a"].shape[0]
            pre = raw.reshape(self.T, H, -1) * self.W[b + "ssm_a"][None, :, None]   # ssm_a holds A = -exp(A_log)
            g = -5.0 * sigmoid(-pre)
            self.assertClose(g, dump(f"kda_g1-{il}__0").reshape(self.T, H, -1), 1e-5, f"kda gate L{il}")

    def test_kda_short_conv(self):
        for il in self.KDA:
            b = f"blk.{il}."
            x = dump(f"attn_norm-{il}__0")
            for q in ("q", "k", "v"):
                proj = x @ self.W[b + f"attn_{q}.weight"].T                       # (T, d_inner)
                w = self.W[b + f"ssm_conv1d_{q}.weight"].reshape(proj.shape[1], -1)  # (d_inner, d_conv)
                d = w.shape[1]
                pad = np.vstack([np.zeros((d - 1, proj.shape[1])), proj])         # zero initial state
                y = np.stack([np.sum(pad[t:t + d].T * w, axis=1) for t in range(self.T)])
                y = y * sigmoid(y)                                                # silu
                got = dump(f"kda_{q}_conv-{il}__0").reshape(self.T, -1)
                self.assertClose(y, got, 1e-5, f"kda {q} conv L{il}")

    def test_kda_recurrence(self):
        for il in self.KDA:
            q = dump(f"kda_q_conv-{il}__0")
            k = dump(f"kda_k_conv-{il}__0")
            v = dump(f"kda_v_conv-{il}__0")
            g = dump(f"kda_g1-{il}__0")
            beta = dump(f"kda_beta-{il}__0")[:, :, 0]
            out, _ = gated_delta_rule(q, k, v, g, beta, normalize_qk=True)
            got = dump(f"kda_scan_out-{il}__0")
            self.assertClose(out, got, 1e-4, f"kda scan L{il}")

    # ---- MoE: router and clamped SwiGLU ----
    def test_router(self):
        for il in self.MOE:
            logits = dump(f"ffn_moe_logits-{il}__0")
            ids, w = route(logits, self.W[f"blk.{il}.exp_probs_b.bias"], k=2, w_scale=2.5)
            got_ids = dump(f"ffn_moe_topk-{il}__0").astype(int)
            got_w = dump(f"ffn_moe_weights_scaled-{il}__0")[:, :, 0]
            self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(got_ids, 1)), f"top-k ids L{il}")
            for t in range(self.T):
                mine = dict(zip(ids[t], w[t]))
                theirs = dict(zip(got_ids[t], got_w[t]))
                for e in mine:
                    self.assertAlmostEqual(mine[e], theirs[e], delta=1e-5)

    def test_swiglu_clamp(self):
        for il in self.MOE:
            gate, up = dump(f"ffn_moe_gate-{il}__0"), dump(f"ffn_moe_up-{il}__0")
            self.assertGreater(np.mean(np.abs(gate) > 0.5), 0.05, "clamp must actually trigger")
            self.assertClose(swiglu_clamp(gate, up, 0.5), dump(f"ffn_moe_swiglu_limited-{il}__0"), 1e-5,
                             f"swiglu clamp L{il}")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, "run tools/tinygen/run_parity.sh")
class GlmTinyDsaParity(_TinyBase):
    """DSA layers: indexer pooling, scores, cell selection, absorbed MLA attention."""

    def w(self, n):
        return self.W[f"blk.{self.IL}.{n}"]

    def test_dsa_indexer_and_selection(self):
        for self.IL in self.DSA:
            self._indexer_and_selection()

    def test_dsa_mla_attention(self):
        for self.IL in self.DSA:
            self._mla_attention()

    def _indexer_and_selection(self):
        from ref.dsa import indexer_pools, indexer_scores, layer_norm, select_cells
        x = dump(f"attn_norm-{self.IL}__0")
        ik = layer_norm(x @ self.w("indexer.attn_k.weight").T, self.w("indexer.k_norm.weight"),
                        self.w("indexer.k_norm.bias"), eps=0.0)
        self.assertClose(ik, dump(f"indexer_k-{self.IL}__0"), 1e-4, "indexer key")
        ig = x @ self.w("indexer_compressor_gate.weight").T
        pooled = indexer_pools(ik, ig, self.w("indexer_compressor_ape.weight"))
        got_pool = dump(f"indexer_pool_k_new-{self.IL}__0")[:pooled.shape[0]]
        self.assertClose(pooled, got_pool, 1e-4, "pooled keys")
        qr = dump(f"q_resid-{self.IL}__0")
        n_idx = self.w("indexer.proj.weight").shape[0]
        iq = (qr @ self.w("indexer.attn_q_b.weight").T).reshape(self.T, n_idx, -1)
        self.assertClose(iq, dump(f"indexer_q-{self.IL}__0"), 1e-4, "indexer query")
        wts = (x @ self.w("indexer.proj.weight").T) / np.sqrt(iq.shape[-1] * n_idx)
        sc = indexer_scores(iq, wts, pooled)
        got = dump(f"indexer_score-{self.IL}__0")[:, :pooled.shape[0]]
        fin = np.isfinite(sc)
        self.assertTrue(np.array_equal(fin, np.isfinite(got) & (got > -1e30)), "pool visibility")
        self.assertClose(sc[fin], got[fin], 1e-4, "indexer scores")
        cells = select_cells(sc, top_k=8)
        mask = dump(f"kq_mask_dsa-{self.IL}__0")
        ties = 0
        for t in range(self.T):
            seen = sorted(np.where(np.isfinite(mask[t]) & (mask[t] > -1e30))[0].tolist())
            if cells[t] == seen:
                continue
            # ggml_top_k uses std::partial_sort (ggml-cpu ops.cpp): order among equal scores is
            # unspecified and differs by backend. Accept a different choice only if it is another
            # valid top-k: same tail, same number of pools, and every chosen pool scores >= every
            # visible pool left out.
            tail0 = ((t + 1) // 4) * 4
            self.assertEqual([c for c in seen if c >= tail0], list(range(tail0, t + 1)), f"tail t={t}")
            chosen = sorted({c // 4 for c in seen if c < tail0})
            vis = np.where(np.isfinite(sc[t]))[0]
            left = [b for b in vis if b not in chosen]
            self.assertEqual(len(chosen), min(2, len(vis)), f"pool count t={t}")
            if left:
                self.assertGreaterEqual(min(sc[t, chosen]), max(sc[t, left]) - 1e-6, f"not a top-k t={t}")
            ties += 1
        self.tie_tokens = ties

    def _mla_attention(self):
        from ref.dsa import mla_attention, rms_norm
        x = dump(f"attn_norm-{self.IL}__0")
        qr = rms_norm(x @ self.w("attn_q_a.weight").T, self.w("attn_q_a_norm.weight"), 1e-5)
        self.assertClose(qr, dump(f"q_resid-{self.IL}__0"), 1e-5, "q_resid")
        H = self.w("attn_k_b.weight").shape[0]
        q = (qr @ self.w("attn_q_b.weight").T).reshape(self.T, H, -1)
        q_abs = np.einsum("hld,thd->thl", self.w("attn_k_b.weight"), q)
        self.assertClose(q_abs, dump(f"q_absorbed-{self.IL}__0"), 1e-5, "q absorbed")
        lat = rms_norm(x @ self.w("attn_kv_a_mqa.weight").T, self.w("attn_kv_a_norm.weight"), 1e-5)
        self.assertClose(lat, dump(f"kv_cmpr-{self.IL}__0").reshape(self.T, -1), 1e-5, "latent")
        mask = dump(f"kq_mask_dsa-{self.IL}__0")
        cells = [np.where(np.isfinite(mask[t]) & (mask[t] > -1e30))[0] for t in range(self.T)]
        out = mla_attention(q_abs, lat, cells, self.w("attn_v_b.weight"), k_mla=q.shape[-1])
        self.assertClose(out, dump(f"kqv_out-{self.IL}__0"), 1e-4, "kqv_out")
        self.assertClose(out @ self.w("attn_output.weight").T, dump(f"attn_out-{self.IL}__0"), 1e-4, "attn_out")
