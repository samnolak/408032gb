"""ref/qwen4exp.py versus llama.cpp's real qwen4exp graph on a tiny random model (CPU only).

Pipeline (tools/tinygen/run_parity_qwen4exp.sh does all of it):
  1. tools/tinygen/qwen4exp_tiny.py writes an 8-layer qwen4exp GGUF (6 gated-delta-net layers, PLE on layer 1,
     a QSA full-attention layer 3, a dense full-attention layer 7, MoE + shared expert everywhere);
  2. tools/dump/dump_tensors.cpp runs llama.cpp @ec7630a on 64 fixed token ids (three of them EOS, so the PLE
     n-gram reset is exercised) and dumps the tensors named by cb(..., "<name>", il) in src/models/qwen4exp.cpp;
  3. this test recomputes every component from the dumped inputs and the GGUF weights (per-component checks),
     then runs the whole oracle from the token ids alone and compares every layer output and the logits.
Env: QWEN4EXP_TINY_DUMP (dump dir, holds tokens.txt), QWEN4EXP_TINY_GGUF (model), LLAMA_CPP_GGUF_PY (gguf-py).

Precision contract: f32 weights and caches, flash attention off (-ctk f32 -ctv f32 -fa off), as in
tests/test_glm_tiny_parity.py. The oracle runs in float64. Error metric: max|got - want| / max|want| over the
compared tensor (normwise relative error); tolerance TOL = 1e-4 for every f32 path (agent contract: fp32
1e-4), exact equality for copies (embedding gather, PLE rows), integer ids and selections. The per-component
table printed at the end lists max abs, rel and the argmax of the error for every layer.

Names: cb() renames a tensor it is called on twice, so the surviving name is the later one: kqv_out ->
attn_pregate (G:920, 1003), v_conv -> v_conv_predelta (G:1107, 1124), the second hc_combine -> l_last (G:495,
498), ffn_moe_out of build_moe_ffn -> ffn_moe_out (same name, G:1162). G = src/models/qwen4exp.cpp @ec7630a.
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import qwen4exp as Q  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402

DUMP = os.environ.get("QWEN4EXP_TINY_DUMP")
GGUF = os.environ.get("QWEN4EXP_TINY_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
TOL = 1e-4
ARCH = "qwen4exp"


def dump(name, k=0):
    j = json.loads(pathlib.Path(DUMP, f"{name}__{k}.json").read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    a = np.fromfile(pathlib.Path(DUMP, f"{name}__{k}.bin"), dtype=dt).reshape(j["ne"][::-1]).astype(np.float64)
    while a.ndim > 1 and a.shape[0] == 1:      # drop leading ggml dims of size 1 (ne3, ne2, ...)
        a = a[0]
    return a


def read_hparams(fields):
    def val(key):
        f = fields[f"{ARCH}.{key}"]
        return f.parts[f.data[0]][0]

    def arr(key):
        f = fields.get(f"{ARCH}.{key}")
        return [] if f is None else [f.parts[i][0] for i in f.data]

    L = int(val("block_count"))
    hp = dict(
        n_layer=L, n_embd=int(val("embedding_length")), eps=float(val("attention.layer_norm_rms_epsilon")),
        hc=int(val("hyper_connection.count")),
        recurrent=[bool(x) for x in arr("attention.recurrent_layers")],
        compress=[int(x) for x in arr("attention.compress_ratios")],
        n_head=int(val("attention.head_count")), n_head_kv=int(val("attention.head_count_kv")),
        head_dim=int(val("attention.key_length")), n_rot=int(val("rope.dimension_count")),
        rope_sections=[int(x) for x in arr("rope.dimension_sections")], rope_base=float(val("rope.freq_base")),
        idx_heads=int(val("attention.indexer.head_count")), idx_dim=int(val("attention.indexer.key_length")),
        idx_top_k=int(val("attention.indexer.top_k")),
        ssm_state=int(val("ssm.state_size")), ssm_groups=int(val("ssm.group_count")),
        ssm_v_heads=int(val("ssm.time_step_rank")), n_expert_used=int(val("expert_used_count")),
        ple_layers=[int(x) for x in arr("ple.layers")], ple_ngram=int(val("ple.ngram_size")),
        ple_heads_per_ngram=int(val("ple.heads_per_ngram")), ple_eos=int(val("ple.eos_token_id")),
        ple_multipliers=[int(x) for x in arr("ple.layer_multipliers")],
        ple_offsets=[int(x) for x in arr("ple.head_offsets")],
        ple_vocab_sizes=[int(x) for x in arr("ple.head_vocab_sizes")],
    )
    hp["kpool"] = max(hp["compress"])
    return hp


class _Base(unittest.TestCase):
    table = []        # (component, layer, max abs, rel, argmax) for the report

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, GGUF_PY)
        from gguf import GGUFReader
        r = GGUFReader(GGUF)
        cls.W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
        cls.hp = read_hparams({f.name: f for f in r.fields.values()})
        cls.tokens = [int(x) for x in pathlib.Path(DUMP, "tokens.txt").read_text().split(",")]
        cls.T = int(json.loads(pathlib.Path(DUMP, "logits.json").read_text())["n_tokens"])
        assert cls.T == len(cls.tokens)
        cls.N = cls.hp["n_layer"]
        cls.LIN = [i for i in range(cls.N) if cls.hp["recurrent"][i]]
        cls.ATT = [i for i in range(cls.N) if not cls.hp["recurrent"][i]]
        cls.QSA = [i for i in cls.ATT if cls.hp["compress"][i] > 0]

    def close(self, what, il, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.shape, want.shape, f"{what} L{il}: shape {got.shape} vs {want.shape}")
        d = np.abs(got - want)
        mx = float(d.max())
        rel = mx / (float(np.abs(want).max()) + 1e-30)
        _Base.table.append((what, il, mx, rel, np.unravel_index(int(d.argmax()), d.shape)))
        self.assertLess(rel, tol, f"{what} L{il}: rel err {rel:.2e} (abs {mx:.2e})")
        return rel

    def w(self, il):
        return Q.layer_weights(self.W, il)

    def res_in(self, il):
        """Residual entering layer il (before PLE), from the dumps."""
        x = dump("hc_init") if il == 0 else dump(f"l_last-{il - 1}")
        return x.reshape(self.T, self.hp["hc"], -1)

    def res_attn_mixer_in(self, il):
        """Residual the attention mixer reads: after PLE on the PLE layer (G:453-455, 1457)."""
        res = self.res_in(il)
        if il in self.hp["ple_layers"]:
            res = res + dump(f"ple_gated_value-{il}").reshape(res.shape) + dump(f"ple_conv_out-{il}").reshape(res.shape)
        return res


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, "set QWEN4EXP_TINY_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: "
                                                "run tools/tinygen/run_parity_qwen4exp.sh")
class Qwen4expTinyParity(_Base):
    # ---- embedding, PLE ----
    def test_embedding_hc_init(self):
        want = Q.hc_init(self.W["token_embd.weight"][self.tokens], self.hp["hc"])
        self.close("hc_init (exact)", -1, dump("hc_init").reshape(want.shape), want, tol=1e-300)

    def test_ple_hash_rows(self):
        hp = self.hp
        rows = Q.ple_rows(self.tokens, hp["ple_multipliers"], hp["ple_offsets"], hp["ple_vocab_sizes"],
                          hp["ple_ngram"], hp["ple_heads_per_ngram"], hp["ple_eos"])
        want = Q.ple_embed(self.W["per_layer_token_embd.weight"], rows)
        self.close("ple_embd = table[hash rows] (exact)", -1, dump("ple_embd"), want, tol=1e-300)
        # guard: the EOS reset must change some rows, else the test cannot see it
        no_reset = Q.ple_rows(self.tokens, hp["ple_multipliers"], hp["ple_offsets"], hp["ple_vocab_sizes"],
                              hp["ple_ngram"], hp["ple_heads_per_ngram"], -1)
        self.assertGreater(int((no_reset != rows).any(axis=1).sum()), 0, "no token exercises the EOS reset")

    def test_ple_module(self):
        for il in self.hp["ple_layers"]:
            w = self.w(il)
            res = self.res_in(il)
            p = Q.ple_forward(dump("ple_embd"), res, w["ple_key.weight"], w["ple_value.weight"],
                              w["ple_norm_key.weight"], w["ple_norm_query.weight"], w["ple_norm_conv.weight"],
                              w["ple_conv1d.weight"], self.hp["eps"], self.hp["ple_ngram"])
            self.close("ple_gate", il, dump(f"ple_gate-{il}").reshape(p["gate"].shape), p["gate"])
            self.close("ple_gated_value", il, dump(f"ple_gated_value-{il}").reshape(p["gated"].shape), p["gated"])
            self.close("ple_conv_out", il, dump(f"ple_conv_out-{il}").reshape(p["conv_out"].shape), p["conv_out"])
            # the PLE output itself has no name: check it through the grouped norm the mixer applies to it
            xn = Q.grouped_rms_norm(p["out"], w["hc_attn_norm.weight"], self.hp["eps"]).reshape(self.T, -1)
            self.close("ple out via hc_norm", il, dump(f"hc_norm-{il}", 0), xn)

    # ---- hyper-connections ----
    def _mixer(self, il, k, res, names, w_norm, w_down, w_up, w_inject):
        m = Q.hc_mix(res, w_norm, w_down, w_up, w_inject, self.hp["eps"])
        self.close(f"hc_norm[{k}]", il, dump(names[0], k), m["norm"])
        self.close(f"hc_gate[{k}]", il, dump(names[1], k), m["gate"])
        self.close(f"hc_mixed[{k}]", il, dump(names[2], k if il >= 0 else 0), m["mixed"])
        if w_inject is not None:
            self.close(f"hc_inject[{k}]", il, dump(names[3], k), m["inject"])
        return m

    def test_hc_mixers(self):
        for il in range(self.N):
            w = self.w(il)
            n = [f"hc_norm-{il}", f"hc_gate-{il}", f"hc_mixed-{il}", f"hc_inject-{il}"]
            self._mixer(il, 0, self.res_attn_mixer_in(il), n, w["hc_attn_norm.weight"], w["hc_attn_down.weight"],
                        w["hc_attn_up.weight"], w["hc_attn_inject.weight"])
            res = dump(f"hc_combine-{il}").reshape(self.T, self.hp["hc"], -1)
            self._mixer(il, 1, res, n, w["hc_ffn_norm.weight"], w["hc_ffn_down.weight"],
                        w["hc_ffn_up.weight"], w["hc_ffn_inject.weight"])
        # the final mixer is the output norm (G:514-519): no inject, result_norm is its mixed output
        res = dump(f"l_last-{self.N - 1}").reshape(self.T, self.hp["hc"], -1)
        self._mixer(-1, 0, res, ["hc_norm", "hc_gate", "result_norm"], self.W["output_hc_norm.weight"],
                    self.W["output_hc_down.weight"], self.W["output_hc_up.weight"], None)

    def test_hc_combine(self):
        for il in range(self.N):
            res = self.res_attn_mixer_in(il)
            blk = dump(f"linear_attn_out-{il}") if il in self.LIN else dump(f"attn_output-{il}")
            new = Q.hc_combine(res, blk.reshape(self.T, -1), dump(f"hc_inject-{il}", 0))
            self.close("hc_combine (attn)", il, dump(f"hc_combine-{il}").reshape(new.shape), new)
            new2 = Q.hc_combine(new, dump(f"ffn_out-{il}"), dump(f"hc_inject-{il}", 1))
            self.close("hc_combine (ffn) = l_last", il, dump(f"l_last-{il}").reshape(new2.shape), new2)

    # ---- gated delta net ----
    def test_gdn_projections_and_gates(self):
        for il in self.LIN:
            g = Q.gdn_layer(dump(f"hc_mixed-{il}", 0), self.w(il), self.hp)
            T, Hv = self.T, self.hp["ssm_v_heads"]
            self.close("linear_attn_qkv_mixed", il, dump(f"linear_attn_qkv_mixed-{il}"), g["qkv"])
            self.close("z", il, dump(f"z-{il}"), g["z"])
            self.close("beta_sigmoid", il, dump(f"beta_sigmoid-{il}").reshape(T, Hv), g["beta"])
            self.close("alpha", il, dump(f"alpha-{il}"), g["alpha"])
            self.close("a_softplus", il, dump(f"a_softplus-{il}"), g["a_softplus"])
            self.close("gate (log decay)", il, dump(f"gate-{il}"), g["g"])
            self.close("conv_output_raw", il, dump(f"conv_output_raw-{il}"), g["conv_raw"])
            self.close("conv_output_silu", il, dump(f"conv_output_silu-{il}"), g["conv"])

    def test_gdn_l2norm_scan_and_output(self):
        hp = self.hp
        for il in self.LIN:
            w = self.w(il)
            T, S, Hk, Hv = self.T, hp["ssm_state"], hp["ssm_groups"], hp["ssm_v_heads"]
            conv = dump(f"conv_output_silu-{il}")
            kd = S * Hk
            q = dump(f"q_conv_predelta-{il}").reshape(T, -1, S)
            k = dump(f"k_conv_predelta-{il}").reshape(T, -1, S)
            v = dump(f"v_conv_predelta-{il}").reshape(T, Hv, S)
            from ref.kda import l2_norm
            self.close("q l2 norm", il, q[:, :Hk], l2_norm(conv[:, :kd].reshape(T, Hk, S), hp["eps"]))
            self.close("k l2 norm", il, k[:, :Hk], l2_norm(conv[:, kd:2 * kd].reshape(T, Hk, S), hp["eps"]))
            self.close("v split", il, v, conv[:, 2 * kd:].reshape(T, Hv, S))
            g = dump(f"gate-{il}")
            beta = dump(f"beta_sigmoid-{il}").reshape(T, Hv)
            out, _ = gated_delta_rule(q, k, v, g, beta, normalize_qk=False)
            self.close("gdn scan (attn_output)", il, dump(f"attn_output-{il}").reshape(out.shape), out)
            z = dump(f"z-{il}").reshape(T, Hv, S)
            fin = Q.gdn_output_norm(dump(f"attn_output-{il}").reshape(T, Hv, S), w["ssm_norm.weight"], z,
                                    hp["eps"]).reshape(T, -1)
            self.close("final_output (gated norm)", il, dump(f"final_output-{il}"), fin)
            self.close("linear_attn_out", il, dump(f"linear_attn_out-{il}"),
                       dump(f"final_output-{il}") @ w["ssm_out.weight"].T)

    def test_gdn_heads_differ(self):     # guard: H_v > H_k, so the head mapping must matter
        self.assertGreater(self.hp["ssm_v_heads"], self.hp["ssm_groups"])

    # ---- full attention ----
    def _rope(self):
        hp = self.hp
        return dict(n_rot=hp["n_rot"], sections=hp["rope_sections"], base=hp["rope_base"])

    def test_attn_projections_norms_rope(self):
        hp = self.hp
        for il in self.ATT:
            w = self.w(il)
            T, H, Hkv, D = self.T, hp["n_head"], hp["n_head_kv"], hp["head_dim"]
            x = dump(f"hc_mixed-{il}", 0)
            qf = x @ w["attn_q.weight"].T
            self.close("Qcur_full", il, dump(f"Qcur_full-{il}"), qf)
            q_raw = qf.reshape(T, H, 2 * D)[..., :D]
            self.close("Qcur_reshaped (q half)", il, dump(f"Qcur_reshaped-{il}"), q_raw)
            self.close("gate_reshaped (gate half)", il, dump(f"gate_reshaped-{il}"),
                       qf.reshape(T, H, 2 * D)[..., D:].reshape(T, H * D))
            qn = Q.rms_norm(q_raw, hp["eps"], w["attn_q_norm.weight"])
            self.close("Qcur_normed", il, dump(f"Qcur_normed-{il}"), qn)
            kl = x @ w["attn_k.weight"].T
            self.close("Kcur (proj)", il, dump(f"Kcur-{il}", 0), kl)
            kn = Q.rms_norm(kl.reshape(T, Hkv, D), hp["eps"], w["attn_k_norm.weight"])
            self.close("Kcur_normed", il, dump(f"Kcur_normed-{il}"), kn)
            self.close("Vcur", il, dump(f"Vcur-{il}", 0), x @ w["attn_v.weight"].T)
            pos = Q.text_pos4(T)
            self.close("Qcur (IMRoPE)", il, dump(f"Qcur-{il}"),
                       Q.imrope(dump(f"Qcur_normed-{il}"), pos, **self._rope()))
            self.close("Kcur (IMRoPE)", il, dump(f"Kcur-{il}", 1),
                       Q.imrope(dump(f"Kcur_normed-{il}"), pos, **self._rope()))

    def test_rope_sections_matter(self):    # guard: the config must put a pair in the 4th section
        sec = Q.imrope_sector_map(self.hp["n_rot"], self.hp["rope_sections"])
        self.assertIn(3, sec.tolist())
        self.assertTrue(set(sec.tolist()) >= {0, 1, 2})

    def _selected_cells(self, il):
        mask = dump(f"kq_mask_qsa-{il}")[:, :]
        return [np.where(np.isfinite(mask[t]) & (mask[t] > -1e30))[0] for t in range(self.T)]

    def _tie_variants(self, scores, seen, mine, il):
        """Assert every token whose reference selection differs from the oracle's is a tie variant."""
        n = 0
        for t in range(self.T):
            s = sorted(int(c) for c in seen[t])
            if s == mine[t]:
                continue
            ok, why = Q.qsa_tie_variant(scores[t], s, t, self.hp["idx_top_k"], self.hp["kpool"])
            self.assertTrue(ok, f"QSA selection L{il} t={t}: {why}; reference {s}, oracle {mine[t]}")
            n += 1
        return n

    def test_qsa_indexer_and_selection(self):
        hp = self.hp
        kp, nh, d = hp["kpool"], hp["idx_heads"], hp["idx_dim"]
        for il in self.QSA:
            w = self.w(il)
            T = self.T
            x = dump(f"hc_mixed-{il}", 0)
            k_raw = x @ w["indexer.k_proj.weight"].T
            self.close("indexer_k_raw", il, dump(f"indexer_k_raw-{il}"), k_raw)
            pooled = Q.qsa_pooled_keys(dump(f"indexer_k_raw-{il}"), w["indexer.k_norm.weight"], kp, hp["eps"],
                                       **self._rope())
            n = pooled.shape[0]
            self.close("indexer_pool_k_new", il, dump(f"indexer_pool_k_new-{il}")[:n], pooled)
            gathered = dump(f"indexer_k-{il}").reshape(-1, d)        # [d, 1, n_pool] padded to 64 pools
            self.close("indexer_k (gathered pools)", il, gathered[:n], pooled)
            iq = Q.imrope(Q.rms_norm((x @ w["indexer.q_proj.weight"].T).reshape(T, nh, d), hp["eps"],
                                     w["indexer.q_norm.weight"]), Q.text_pos4(T), **self._rope())
            self.close("indexer_q", il, dump(f"indexer_q-{il}"), iq)
            sc = Q.qsa_scores(dump(f"indexer_q-{il}"), gathered[:n], kp)
            got = dump(f"indexer_score-{il}")
            fin = np.isfinite(sc)
            self.assertTrue(np.array_equal(fin, np.isfinite(got[:, :n])), "pool visibility")
            self.assertFalse(np.isfinite(got[:, n:]).any(), "padded pools must stay masked")
            self.close("indexer_score (visible)", il, got[:, :n][fin], sc[fin])
            # selection: top pools + tail. A different choice is accepted only if the oracle judges it another
            # valid top-k (Q.qsa_tie_variant: same tail, boundary scores tie within f32 rounding); counted.
            mine = Q.qsa_select(sc, hp["idx_top_k"], kp)
            ties = self._tie_variants(sc, self._selected_cells(il), mine, il)
            _Base.table.append((f"qsa selection: tokens differing only by tie order = {ties}", il, 0.0, 0.0, ()))
            dropped = sum(len(c) < t + 1 for t, c in enumerate(mine))
            _Base.table.append((f"qsa selection: tokens attending to fewer than all causal cells = {dropped}",
                                il, 0.0, 0.0, ()))
            # guard: the budget must bind, i.e. some token sees more pools than it may select
            self.assertGreater(int(fin.sum(axis=1).max()), hp["idx_top_k"] // kp)

    def test_attention_core_and_gate(self):
        hp = self.hp
        for il in self.ATT:
            w = self.w(il)
            T, D = self.T, hp["head_dim"]
            q, k = dump(f"Qcur-{il}"), dump(f"Kcur-{il}", 1)
            v = dump(f"Vcur-{il}", 0).reshape(T, hp["n_head_kv"], D)
            cells = self._selected_cells(il) if il in self.QSA else None
            pre = Q.attention(q, k, v, cells, 1.0 / np.sqrt(D))
            kind = "qsa" if il in self.QSA else "dense causal"
            self.close(f"attn_pregate = kqv_out ({kind})", il, dump(f"attn_pregate-{il}"), pre)
            gate = dump(f"gate_reshaped-{il}")
            self.close("gate_sigmoid", il, dump(f"gate_sigmoid-{il}"), Q.sigmoid(gate))
            self.close("attn_gated", il, dump(f"attn_gated-{il}"), Q.gate_attention(dump(f"attn_pregate-{il}"), gate))
            self.close("attn_output (wo)", il, dump(f"attn_output-{il}"),
                       dump(f"attn_gated-{il}") @ w["attn_output.weight"].T)

    def test_qsa_mask_is_sparse(self):    # guard: QSA must actually drop causal cells somewhere
        for il in self.QSA:
            cells = self._selected_cells(il)
            self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(cells)))

    # ---- MoE ----
    def test_moe_router(self):
        k = self.hp["n_expert_used"]
        for il in range(self.N):
            x = dump(f"hc_mixed-{il}", 1)
            w = self.w(il)
            logits = x @ w["ffn_gate_inp.weight"].T
            self.close("ffn_moe_logits", il, dump(f"ffn_moe_logits-{il}"), logits)
            probs, ids, _ = Q.route_softmax(dump(f"ffn_moe_logits-{il}"), k)
            self.close("ffn_moe_probs (softmax)", il, dump(f"ffn_moe_probs-{il}"), probs)
            got_ids = dump(f"ffn_moe_topk-{il}").astype(int)
            self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(got_ids, 1)), f"top-k ids L{il}")
            got_w = dump(f"ffn_moe_weights_norm-{il}").reshape(self.T, k)
            self.close("ffn_moe_weights_norm", il, got_w, Q.route_softmax(dump(f"ffn_moe_logits-{il}"), k, got_ids)[2])
            self.assertFalse(any(n.startswith(f"ffn_moe_weights_scaled-{il}__") for n in os.listdir(DUMP)),
                             "qwen4exp must not scale the routing weights")

    def test_moe_experts_and_shared(self):
        k = self.hp["n_expert_used"]
        for il in range(self.N):
            x = dump(f"hc_mixed-{il}", 1)
            w = self.w(il)
            ids = dump(f"ffn_moe_topk-{il}").astype(int)
            m = Q.moe_layer(x, w, k, ids_override=ids)
            self.close("ffn_moe_gate", il, dump(f"ffn_moe_gate-{il}"), m["gate"])
            self.close("ffn_moe_up", il, dump(f"ffn_moe_up-{il}"), m["up"])
            self.close("ffn_moe_swiglu", il, dump(f"ffn_moe_swiglu-{il}"),
                       Q.swiglu(dump(f"ffn_moe_gate-{il}"), dump(f"ffn_moe_up-{il}")))
            self.close("ffn_moe_down", il, dump(f"ffn_moe_down-{il}"), m["down"])
            self.close("ffn_moe_out (routed)", il, dump(f"ffn_moe_out-{il}"), m["routed"])
            self.close("ffn_shexp", il, dump(f"ffn_shexp-{il}"), m["shexp"])
            self.close("shared_expert_gate_sigmoid", il, dump(f"shared_expert_gate_sigmoid-{il}").reshape(-1),
                       m["sh_gate"])
            self.close("ffn_out (routed + gated shared)", il, dump(f"ffn_out-{il}"), m["out"])

    # ---- head ----
    def test_logits(self):
        rn = dump("result_norm")
        want = rn @ self.W["output.weight"].T
        self.close("result_output", -1, dump("result_output"), want)
        n_vocab = self.W["output.weight"].shape[0]
        logits = np.fromfile(pathlib.Path(DUMP, "logits.bin"), dtype=np.float32).reshape(self.T, n_vocab)
        self.close("logits.bin", -1, logits, want)

    # ---- the whole model from token ids ----
    def test_end_to_end(self):
        """Run the oracle from the token ids alone. Discrete choices (MoE top-k, QSA pools) are the oracle's
        own; only where the reference picked another member of a tie (validated on the oracle's own scores by
        Q.qsa_tie_variant) is its choice replayed."""
        ref = Q.forward(self.tokens, self.W, self.hp)
        overrides = {}
        for il in self.QSA:
            seen = [sorted(int(c) for c in cells) for cells in self._selected_cells(il)]
            a = ref["layers"][il]["attn"]
            if seen != a["cells"]:
                self._tie_variants(a["scores"], seen, a["cells"], il)
                overrides[("qsa", il)] = seen
        if overrides:
            ref = Q.forward(self.tokens, self.W, self.hp, overrides)
        for il in range(self.N):
            got_ids = np.sort(dump(f"ffn_moe_topk-{il}").astype(int), 1)
            self.assertTrue(np.array_equal(np.sort(ref["layers"][il]["moe"]["ids"], 1), got_ids), f"e2e top-k L{il}")
            want = ref["layers"][il]["combine_ffn"]
            self.close("e2e l_last", il, dump(f"l_last-{il}").reshape(want.shape), want)
        self.close("e2e result_norm", -1, dump("result_norm"), ref["result_norm"])
        n_vocab = self.W["output.weight"].shape[0]
        logits = np.fromfile(pathlib.Path(DUMP, "logits.bin"), dtype=np.float32).reshape(self.T, n_vocab)
        self.close("e2e logits", -1, logits, ref["logits"])
        self.assertTrue(np.array_equal(logits.argmax(1), ref["logits"].argmax(1)), "e2e greedy tokens")
        _Base.table.append((f"e2e: QSA tie replays = {len(overrides)} layer(s)", -1, 0.0, 0.0, ()))

    @classmethod
    def tearDownClass(cls):
        if not _Base.table:
            return
        print("\nper-component errors (rel = max|got-want| / max|want|):", file=sys.stderr)
        print(f"{'component':48s} {'layer':>5s} {'max abs':>10s} {'rel':>10s}  argmax", file=sys.stderr)
        for what, il, mx, rel, arg in _Base.table:
            print(f"{what:48s} {il:5d} {mx:10.3e} {rel:10.3e}  {tuple(int(a) for a in arg)}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
