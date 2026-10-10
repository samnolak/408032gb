"""Property tests of the incremental path of ref/qwen4exp.py (numpy only, no llama.cpp needed).

State continuity: running a sequence as consecutive chunks (several ubatches, then one token per call) with the state
the oracle carries between chunks (Q.forward_chunk; the same state llama.cpp carries, see ref/qwen4exp.py
"incremental path") must give the rows of the whole-sequence forward (causality), and a rollback (Q.cache_truncate)
must leave exactly the state of the kept prefix. The model is a random 4-layer qwen4exp built here with the GGUF
tensor shapes of tools/tinygen/qwen4exp_tiny.py: gated delta net on layers 0 and 2, PLE on layer 2, QSA on layer 1
(compress ratio 2, budget 2 pools), dense full attention on layer 3. The comparison against llama.cpp is
tests/test_qwen4exp_decode_parity.py.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import qwen4exp as Q  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402


def random_model(seed=0):
    """Weights (numpy shapes = GGUF ne reversed, names as in the GGUF) and hyper-parameters of a tiny qwen4exp."""
    rng = np.random.default_rng(seed)
    E, V, hc, lr = 32, 40, 2, 4
    H, Hkv, D = 2, 1, 8
    S, Hk, Hv, Kc = 8, 1, 2, 4
    X, F, Fs = 4, 8, 8
    nh, d, kp, top_k = 2, 8, 2, 4
    ple_vocab = [11, 13, 17, 19]
    ple_dim = E // len(ple_vocab)
    hp = dict(n_layer=4, n_embd=E, eps=1e-6, hc=hc, recurrent=[True, False, True, False], compress=[0, kp, 0, 0],
              n_head=H, n_head_kv=Hkv, head_dim=D, n_rot=8, rope_sections=[1, 1, 1, 1], rope_base=100.0,
              idx_heads=nh, idx_dim=d, idx_top_k=top_k, ssm_state=S, ssm_groups=Hk, ssm_v_heads=Hv,
              n_expert_used=2, ple_layers=[2], ple_ngram=3, ple_heads_per_ngram=2, ple_eos=2,
              ple_multipliers=[0x1A2B3C4D5E1, 0x0F1E2D3C4B5, 0x13579BDF023],
              ple_offsets=[int(x) for x in np.cumsum([0] + ple_vocab[:-1])], ple_vocab_sizes=ple_vocab, kpool=kp)

    def t(*shape, scale=None, ones=False):
        if ones:
            return 1.0 + 0.05 * rng.standard_normal(shape)
        return (scale if scale is not None else 1.0 / np.sqrt(shape[-1])) * rng.standard_normal(shape)

    W = {"token_embd.weight": t(V, E, scale=1.0), "output.weight": t(V, E),
         "output_hc_norm.weight": t(hc, E, ones=True), "output_hc_down.weight": t(lr, hc * E),
         "output_hc_up.weight": t(hc * E, lr), "per_layer_token_embd.weight": t(sum(ple_vocab), ple_dim, scale=1.0)}
    kd, vd = S * Hk, S * Hv
    for il in range(hp["n_layer"]):
        b = f"blk.{il}."
        for k in ("attn", "ffn"):
            W[b + f"hc_{k}_norm.weight"] = t(hc, E, ones=True)
            W[b + f"hc_{k}_down.weight"] = t(lr, hc * E)
            W[b + f"hc_{k}_up.weight"] = t(hc * E, lr)
            W[b + f"hc_{k}_inject.weight"] = t(hc, hc * E)
        if hp["recurrent"][il]:
            W.update({b + "attn_qkv.weight": t(2 * kd + vd, E), b + "attn_gate.weight": t(vd, E),
                      b + "ssm_conv1d.weight": t(2 * kd + vd, Kc, scale=0.5), b + "ssm_dt.bias": t(Hv, scale=0.5),
                      b + "ssm_a": -np.exp(0.5 * rng.standard_normal(Hv)), b + "ssm_beta.weight": t(Hv, E),
                      b + "ssm_alpha.weight": t(Hv, E), b + "ssm_norm.weight": t(S, ones=True),
                      b + "ssm_out.weight": t(E, vd)})
        else:
            W.update({b + "attn_q.weight": t(2 * D * H, E), b + "attn_k.weight": t(D * Hkv, E),
                      b + "attn_v.weight": t(D * Hkv, E), b + "attn_output.weight": t(E, D * H),
                      b + "attn_q_norm.weight": t(D, ones=True), b + "attn_k_norm.weight": t(D, ones=True),
                      b + "indexer.q_proj.weight": t(nh * d, E), b + "indexer.k_proj.weight": t(d, E),
                      b + "indexer.q_norm.weight": t(d, ones=True), b + "indexer.k_norm.weight": t(d, ones=True)})
        if il in hp["ple_layers"]:
            W.update({b + "ple_key.weight": t(hc * E, E), b + "ple_value.weight": t(E, E),
                      b + "ple_norm_key.weight": t(hc, E, ones=True), b + "ple_norm_query.weight": t(hc, E, ones=True),
                      b + "ple_norm_conv.weight": t(hc, E, ones=True),
                      b + "ple_conv1d.weight": t(hc * E, 3, scale=0.5)})
        W.update({b + "ffn_gate_inp.weight": t(X, E), b + "ffn_gate_exps.weight": t(X, F, E),
                  b + "ffn_up_exps.weight": t(X, F, E), b + "ffn_down_exps.weight": t(X, E, F),
                  b + "ffn_gate_inp_shexp.weight": t(E), b + "ffn_gate_shexp.weight": t(Fs, E),
                  b + "ffn_up_shexp.weight": t(Fs, E), b + "ffn_down_shexp.weight": t(E, Fs)})
    return W, hp


def tokens(seed=1, T=24):
    rng = np.random.default_rng(seed)
    toks = [int(x) for x in rng.integers(3, 40, T)]
    for p in (7, 12, 13, 19):                       # EOS (id 2), once repeated, inside every chunking below
        toks[p] = 2
    return toks


CHUNKINGS = ([24], [8, 8, 8], [5, 1, 1, 9, 8], [3, 7, 4, 10], [11] + [1] * 13, [1] * 24)


def rows_of(out, il, path):
    x = out["layers"][il]
    for k in path:
        x = x[k]
    return np.asarray(x)


class ChunkedEqualsFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.hp = random_model()
        cls.toks = tokens()
        cls.full = Q.forward(cls.toks, cls.W, cls.hp)

    def assertRows(self, got, want, what):
        err = float(np.abs(np.asarray(got) - np.asarray(want)).max())
        self.assertLess(err, 1e-10 * max(1.0, float(np.abs(want).max())), what)

    def test_every_chunking_reproduces_the_full_forward(self):
        paths = [("combine_ffn",), ("combine_attn",), ("mix_attn", "mixed"), ("moe", "out")]
        for sizes in CHUNKINGS:
            outs = Q.forward_chunks(self.toks, self.W, self.hp, sizes)
            for o in outs:
                p = o["pos"]
                for il in range(self.hp["n_layer"]):
                    for path in paths:
                        self.assertRows(rows_of(o, il, path), rows_of(self.full, il, path)[p], f"{sizes} L{il} {path}")
                    if self.hp["recurrent"][il]:
                        self.assertRows(o["layers"][il]["gdn"]["scan"], self.full["layers"][il]["gdn"]["scan"][p],
                                        f"{sizes} L{il} gdn scan")
                    else:
                        self.assertRows(o["layers"][il]["attn"]["pregate"],
                                        self.full["layers"][il]["attn"]["pregate"][p], f"{sizes} L{il} attention")
                        if self.hp["compress"][il]:
                            want = [self.full["layers"][il]["attn"]["cells"][t] for t in p]
                            self.assertEqual(o["layers"][il]["attn"]["cells"], want, f"{sizes} L{il} QSA cells")
                for il in self.hp["ple_layers"]:
                    self.assertRows(o["layers"][il]["ple"]["conv_out"], self.full["layers"][il]["ple"]["conv_out"][p],
                                    f"{sizes} L{il} PLE conv")
                self.assertTrue(np.array_equal(o["ple_rows"], self.full["ple_rows"][p]), f"{sizes} PLE rows")
                self.assertRows(o["logits"], self.full["logits"][p], f"{sizes} logits")

    def test_carried_state_is_the_state_of_the_prefix(self):
        hp, full = self.hp, self.full
        for sizes in ([5, 1, 1, 9, 8], [11] + [1] * 13):
            cache, i = Q.new_cache(), 0
            for n in sizes:
                o = Q.forward_chunk(self.toks[i:i + n], self.W, hp, cache)
                i += n
                for il in range(hp["n_layer"]):
                    if hp["recurrent"][il]:
                        G = full["layers"][il]["gdn"]
                        _, s = gated_delta_rule(G["q"][:i], G["k"][:i], G["v"][:i], G["g"][:i], G["beta"][:i],
                                                normalize_qk=True, eps=hp["eps"])
                        self.assertRows(cache["gdn"][il]["state"], s, f"L{il} delta-rule state after {i}")
                        K1 = cache["gdn"][il]["conv"].shape[0]
                        want = np.vstack([np.zeros((K1, G["qkv"].shape[1])), G["qkv"][:i]])[-K1:]
                        self.assertRows(cache["gdn"][il]["conv"], want, f"L{il} conv history after {i}")
                        self.assertRows(o["layers"][il]["gdn"]["state"], s, f"L{il} returned state after {i}")
                    elif hp["compress"][il]:
                        k_raw = full["layers"][il]["attn"]["k_raw"]
                        n_pool = i // hp["kpool"]
                        self.assertRows(cache["idx"][il]["pooled"], full["layers"][il]["attn"]["pooled"][:n_pool],
                                        f"L{il} pooled keys after {i}")
                        self.assertRows(cache["idx"][il]["k_raw"], k_raw[:i], f"L{il} raw keys after {i}")
                for il in hp["ple_layers"]:
                    P = full["layers"][il]["ple"]
                    w = Q.layer_weights(self.W, il)
                    normed = Q.grouped_rms_norm(P["gated"], w["ple_norm_conv.weight"], hp["eps"])
                    normed = normed.reshape(len(self.toks), -1)
                    Kh = cache["ple"][il]["hist"].shape[0]
                    want = np.vstack([np.zeros((Kh, normed.shape[1])), normed[:i]])[-Kh:]
                    self.assertRows(cache["ple"][il]["hist"], want, f"L{il} PLE history after {i}")
                self.assertEqual(cache["tokens"], self.toks[:i])

    def test_rollback_then_continue_equals_the_kept_sequence(self):
        # draft 4 tokens after a prefix of 11, reject the last 2, continue with the true tokens one per call
        hp, W, toks = self.hp, self.W, self.toks
        cache = Q.new_cache()
        Q.forward_chunk(toks[:11], W, hp, cache)
        draft = toks[11:13] + [(toks[13] + 5) % 37 + 3, (toks[14] + 7) % 37 + 3]
        self.assertNotEqual(draft[2:], toks[13:15])
        d = Q.forward_chunk(draft, W, hp, cache)
        self.assertRows(d["logits"][:2], self.full["logits"][11:13], "accepted draft rows")
        Q.cache_truncate(cache, 13, W, hp)
        self.assertEqual(cache["tokens"], toks[:13])
        for p in range(13, len(toks)):
            o = Q.forward_chunk(toks[p:p + 1], W, hp, cache)
            self.assertRows(o["logits"], self.full["logits"][p:p + 1], f"logits after the rollback, pos {p}")
            for il in range(hp["n_layer"]):
                self.assertRows(o["layers"][il]["combine_ffn"], self.full["layers"][il]["combine_ffn"][p:p + 1],
                                f"L{il} pos {p}")

    def test_guards(self):
        """The random model must exercise what the tests above claim: sparse QSA, EOS resets, pools that straddle a
        chunk boundary, a non-zero carried state."""
        hp, full = self.hp, self.full
        cells = full["layers"][1]["attn"]["cells"]
        self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(cells)), "QSA never drops a causal cell")
        no_reset = Q.ple_rows(self.toks, hp["ple_multipliers"], hp["ple_offsets"], hp["ple_vocab_sizes"],
                              hp["ple_ngram"], hp["ple_heads_per_ngram"], -1)
        self.assertTrue((no_reset != full["ple_rows"]).any(), "no EOS reset")
        self.assertTrue(any(s % hp["kpool"] for s in np.cumsum([5, 1, 1, 9, 8])[:-1]), "no pool straddles a chunk")
        self.assertGreater(float(np.abs(full["layers"][0]["gdn"]["state"]).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
