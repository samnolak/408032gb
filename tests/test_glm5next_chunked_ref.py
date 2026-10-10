"""Property tests of the incremental path of ref/glm5next.py (numpy only, no llama.cpp needed).

State continuity: running a sequence as consecutive chunks (several ubatches, then one token per call) with the state
the oracle carries between chunks (G.forward_chunk: KDA conv history and state, MLA latent cache, indexer key | gate
rows and pooled keys; see ref/glm5next.py "incremental path") must give the rows of the whole-sequence forward
(causality), and a rollback (G.cache_truncate) must leave exactly the state of the kept prefix. The model is a random
5-layer glm5-next built here with the GGUF tensor shapes of tools/tinygen/glm5next_tiny.py: KDA on layers 0, 2, 3
(layer 0 with the dense lead FFN), DSA on layers 1 and 4 (kpool 2, budget 2 pools + tail), MoE with a shared expert,
and an MTP block as layer 5 (DSA + MoE + nextn.*, as tools/tinygen/glm5next_tiny.py --mtp; drawn after the trunk):
MtpChunkedEqualsOneShot checks G.mtp_forward_chunk the same way.
The comparison against llama.cpp is tests/test_glm5next_decode_parity.py.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import glm5next as G  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402


def random_model(seed=0):
    """Weights (numpy shapes = GGUF ne reversed, names as in the GGUF) and hyper-parameters of a tiny glm5-next."""
    rng = np.random.default_rng(seed)
    E, V, hc, H, D, K = 32, 40, 4, 2, 8, 4
    ql, kl, km, vm = 8, 8, 8, 8
    nh, d, kp, top_k = 2, 8, 2, 4
    X, F, n_used, L = 4, 8, 2, 5
    kda = [True, False, True, True, False]
    hp = dict(arch="glm5-next", n_layer=L, n_nextn=1, n_embd=E, n_head=H, kda=kda + [False], rms_eps=1e-5,
              norm_eps=0.0, kda_head_dim=D, d_conv=K, kda_lower_bound=-5.0, k_mla=km, v_mla=vm, kv_lora=kl, q_lora=ql,
              dense_lead=1, n_expert_used=n_used, w_scale=2.5, norm_w=True, clamp_exp=[0.5] * (L + 1),
              clamp_shexp=[0.5] * (L + 1),
              idx_heads=nh, idx_dim=d, idx_top_k=top_k, kpool=kp, select_tail=True, hc=hc, sinkhorn_iters=20,
              hc_eps=1e-6)

    def t(*shape, scale=None, ones=False):
        if ones:
            return 1.0 + 0.05 * rng.standard_normal(shape)
        return (scale if scale is not None else 1.0 / np.sqrt(shape[-1])) * rng.standard_normal(shape)

    W = {"token_embd.weight": t(V, E, scale=1.0), "output_norm.weight": t(E, ones=True), "output.weight": t(V, E)}
    mix = (2 + hc) * hc
    for il in range(L + 1):                                        # layer L: the MTP block
        b = f"blk.{il}."
        W[b + "attn_norm.weight"], W[b + "ffn_norm.weight"] = t(E, ones=True), t(E, ones=True)
        if il == L:
            W.update({b + "nextn.eh_proj.weight": t(E, 2 * E), b + "nextn.enorm.weight": t(E, ones=True),
                      b + "nextn.hnorm.weight": t(E, ones=True)})
        for k in ("attn", "ffn") if il < L else ():
            W.update({b + f"hc_{k}_fn.weight": t(mix, hc * E, scale=0.3), b + f"hc_{k}_base.weight": t(mix, scale=0.5),
                      b + f"hc_{k}_scale.weight": t(3, ones=True)})
        if il < L and kda[il]:
            for c in "qkv":
                W[b + f"ssm_conv1d_{c}.weight"] = t(H * D, 1, K, scale=0.5)
                W[b + f"attn_{c}.weight"] = t(H * D, E)
            W.update({b + "ssm_f_a.weight": t(D, E), b + "ssm_f_b.weight": t(H * D, D), b + "ssm_beta.weight": t(H, E),
                      b + "ssm_a": t(H, scale=0.5), b + "ssm_dt.bias": t(H * D, scale=0.5),
                      b + "ssm_g_a.weight": t(D, E), b + "ssm_g_b.weight": t(H * D, D),
                      b + "ssm_norm.weight": t(D, ones=True), b + "attn_output.weight": t(E, H * D)})
        else:
            W.update({b + "attn_q_a_norm.weight": t(ql, ones=True), b + "attn_kv_a_norm.weight": t(kl, ones=True),
                      b + "attn_q_a.weight": t(ql, E), b + "attn_q_b.weight": t(H * km, ql),
                      b + "attn_kv_a_mqa.weight": t(kl, E), b + "attn_k_b.weight": t(H, kl, km),
                      b + "attn_v_b.weight": t(H, vm, kl), b + "attn_output.weight": t(E, H * vm),
                      b + "indexer.k_norm.weight": t(d, ones=True), b + "indexer.k_norm.bias": t(d, scale=0.1),
                      b + "indexer.proj.weight": t(nh, E), b + "indexer.attn_k.weight": t(d, E),
                      b + "indexer.attn_q_b.weight": t(nh * d, ql), b + "indexer_compressor_gate.weight": t(d, E),
                      b + "indexer_compressor_ape.weight": t(kp, d, scale=0.5)})
        if il < hp["dense_lead"]:
            W.update({b + "ffn_gate.weight": t(3 * E, E), b + "ffn_up.weight": t(3 * E, E),
                      b + "ffn_down.weight": t(E, 3 * E)})
        else:
            W.update({b + "ffn_gate_inp.weight": t(X, E), b + "exp_probs_b.bias": t(X, scale=0.1),
                      b + "ffn_gate_exps.weight": t(X, F, E), b + "ffn_up_exps.weight": t(X, F, E),
                      b + "ffn_down_exps.weight": t(X, E, F), b + "ffn_gate_shexp.weight": t(F, E),
                      b + "ffn_up_shexp.weight": t(F, E), b + "ffn_down_shexp.weight": t(E, F)})
    return W, hp


W, HP = random_model()
TOKENS = [int(x) for x in np.random.default_rng(1).integers(0, 40, size=23)]
FULL = G.forward(TOKENS, W, HP)


class ChunkedEqualsOneShot(unittest.TestCase):
    def assertRows(self, outs, sizes, tol=1e-10):
        p = 0
        for o, n in zip(outs, sizes):
            np.testing.assert_allclose(o["logits"], FULL["logits"][p:p + n], rtol=0, atol=tol)
            for il in range(HP["n_layer"]):
                np.testing.assert_allclose(o["layers"][il]["l_out"], FULL["layers"][il]["l_out"][p:p + n],
                                           rtol=0, atol=tol)
                if not HP["kda"][il]:
                    self.assertEqual(o["layers"][il]["dsa"]["cells"], FULL["layers"][il]["dsa"]["cells"][p:p + n])
            p += n

    def test_ubatches_straddling_pools(self):
        sizes = [5, 3, 7, 8]                                      # kpool 2: pools straddle 5|3 and 15|8
        self.assertRows(G.forward_chunks(TOKENS, W, HP, sizes), sizes)

    def test_token_by_token_decode(self):
        sizes = [6] + [1] * 17
        self.assertRows(G.forward_chunks(TOKENS, W, HP, sizes), sizes)

    def test_budget_binds_and_tail(self):
        """The selection drops visible pools for late tokens and always keeps the incomplete tail."""
        cells = FULL["layers"][1]["dsa"]["cells"]
        self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(cells)))
        for t, c in enumerate(cells):
            self.assertTrue(set(G.tail_cells(t, HP["kpool"])) <= set(c))
            self.assertEqual(G.tie_variant(FULL["layers"][1]["dsa"]["scores"][t], c, t, HP["idx_top_k"],
                                           HP["kpool"]), (True, ""))

    def test_carried_state(self):
        """After chunks, the cache holds the full-sequence state: KDA conv history = last d_conv-1 projected rows,
        KDA state = the scan's final state, latent / indexer rows of every cell, the pooled key of every complete pool."""
        cache = G.new_cache()
        for n, i in ((7, 0), (1, 7), (9, 8), (6, 17)):
            G.forward_chunk(TOKENS[i:i + n], W, HP, cache)
        for il in range(HP["n_layer"]):
            if HP["kda"][il]:
                Kf, st = FULL["layers"][il]["kda"], cache["kda"][il]
                for c in "qkv":
                    np.testing.assert_allclose(st["conv"][c], Kf[f"proj_{c}"][-(HP["d_conv"] - 1):], atol=1e-12)
                np.testing.assert_allclose(st["state"], Kf["state"], atol=1e-10)
            else:
                Df, st = FULL["layers"][il]["dsa"], cache["dsa"][il]
                np.testing.assert_allclose(st["latent"], Df["latent"], atol=1e-12)
                np.testing.assert_allclose(st["ik"], Df["ik"], atol=1e-12)
                np.testing.assert_allclose(st["pooled"], Df["pooled"], atol=1e-12)
                self.assertEqual(st["pooled"].shape[0], len(TOKENS) // HP["kpool"])

    def test_rollback_leaves_the_prefix_state(self):
        """A draft whose last two tokens are wrong, truncated back, then the right tokens: the rows after the
        rollback equal the one-shot forward, and the truncated cache equals the cache of the prefix alone."""
        cache = G.new_cache()
        G.forward_chunk(TOKENS[:12], W, HP, cache)
        wrong = TOKENS[12:14] + [(x + 7) % 40 for x in TOKENS[14:16]]
        G.forward_chunk(wrong, W, HP, cache)                     # completes pool 7 (14, 15) with the wrong tokens
        G.cache_truncate(cache, 14, W, HP)
        ref = G.new_cache()
        G.forward_chunk(TOKENS[:14], W, HP, ref)
        for il in range(HP["n_layer"]):
            for part in ("kda", "dsa"):
                if il in ref[part]:
                    for key, v in ref[part][il].items():
                        if isinstance(v, dict):
                            for c in v:
                                np.testing.assert_array_equal(cache[part][il][key][c], v[c])
                        else:
                            np.testing.assert_array_equal(cache[part][il][key], v)
        outs = [G.forward_chunk([tok], W, HP, cache) for tok in TOKENS[14:]]
        for j, o in enumerate(outs):
            np.testing.assert_allclose(o["logits"], FULL["logits"][14 + j:15 + j], rtol=0, atol=1e-10)

    def test_kda_scan_continuity(self):
        """The KDA scan split at any point with the carried state equals the one-shot scan (ref/kda.py)."""
        K = FULL["layers"][0]["kda"]
        out1, s1 = gated_delta_rule(K["q"][:9], K["k"][:9], K["v"][:9], K["g"][:9], K["beta"][:9], eps=G.L2_EPS)
        out2, s2 = gated_delta_rule(K["q"][9:], K["k"][9:], K["v"][9:], K["g"][9:], K["beta"][9:], state=s1,
                                    eps=G.L2_EPS)
        np.testing.assert_allclose(np.concatenate([out1, out2]), K["scan"], atol=1e-12)
        np.testing.assert_allclose(s2, K["state"], atol=1e-12)


class MtpChunkedEqualsOneShot(unittest.TestCase):
    """The MTP block over its own cache: chunks and one-token steps give the rows of one chunk; the h fed is the
    trunk's h_nextn (result_norm) of the previous position, zeros at the start."""

    def test_chunks_and_decode(self):
        h = np.vstack([np.zeros((1, HP["n_embd"])), FULL["result_norm"][:-1]])
        one = G.mtp_forward_chunk(TOKENS, h, W, HP, G.new_cache())
        cache, p = G.new_cache(), 0
        for n in (7, 1, 1, 6, 1, 7):
            o = G.mtp_forward_chunk(TOKENS[p:p + n], h[p:p + n], W, HP, cache)
            np.testing.assert_allclose(o["logits"], one["logits"][p:p + n], rtol=0, atol=1e-10)
            np.testing.assert_allclose(o["h_nextn"], one["h_nextn"][p:p + n], rtol=0, atol=1e-10)
            self.assertEqual(o["dsa"]["cells"], one["dsa"]["cells"][p:p + n])
            p += n
        self.assertEqual(p, len(TOKENS))
        self.assertEqual(sorted(cache["dsa"]), [HP["n_layer"]], "the MTP cache holds the MTP layer alone")
        self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(one["dsa"]["cells"])), "MTP DSA never sparse")


if __name__ == "__main__":
    unittest.main()
