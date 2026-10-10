"""The qwen4exp MTP block (nextn_predict_layers = 1) of llama.cpp against ref/qwen4exp.py mtp_forward_chunk (CPU).

Model: the N2 tiny model with an MTP block appended (tools/tinygen/qwen4exp_tiny.py --mtp: block_count 9, layer 8 is
a full-attention QSA layer with blk.8.nextn.*; the trunk weights equal the default model's). The run
(tools/tinygen/run_parity_qwen4exp_decode.sh, directory "mtp") drives the MTP block the way common/speculative.cpp
does at llama.cpp @ec7630a (tools/dump/dump_tensors.cpp, DUMP_MTP=1):
  * the target context exports h_nextn, the hc-wide residual before the final mixer (G:501-506);
  * after every target call a second context of type LLAMA_CONTEXT_TYPE_MTP, whose memory holds the MTP layer alone
    (src/llama-model.cpp:2713-2720), processes the same tokens at the same positions, token p paired with the target
    h_nextn of p-1 (common/speculative.cpp:1511-1529): a 48-token call (-ub 16: 3 graphs), then 8 one-token calls;
  * then 4 greedy draft steps (common/speculative.cpp:1700-1704): the target's greedy token after position 55 with
    the target row of 55, then each drafted token at the next position with the MTP's own h_nextn of the step before.
Every MTP graph is compared with (A) [full]: mtp_forward_chunk over all processed positions at once, fed by the
full-sequence target oracle (rows of the draft steps have no (A)), and (B) [chunk]: the same calls replayed graph by
graph through the MTP context's own KV and k-pool cache, the draft chain fed with the oracle's own h_nextn.
G = src/models/qwen4exp.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d.
Env: QWEN4EXP_DECODE_DUMP (root; the run is <root>/mtp, its MTP context in <root>/mtp/mtp), QWEN4EXP_MTP_GGUF,
LLAMA_CPP_GGUF_PY. Tolerances as tests/test_qwen4exp_decode_parity.py (normwise relative 1e-4; exact gathers and
ids; QSA ties replayed only when both selections are valid top-k sets).
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
from tests.test_qwen4exp_tiny_parity import read_hparams  # noqa: E402

DUMP = os.environ.get("QWEN4EXP_DECODE_DUMP")
GGUF = os.environ.get("QWEN4EXP_MTP_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
TOL = 1e-4
SKIP = ("set QWEN4EXP_DECODE_DUMP, QWEN4EXP_MTP_GGUF, LLAMA_CPP_GGUF_PY: "
        "run tools/tinygen/run_parity_qwen4exp_decode.sh")
TABLE = []


def load_model():
    sys.path.append(GGUF_PY)
    try:
        from gguf import GGUFReader
    finally:
        sys.path.remove(GGUF_PY)
    r = GGUFReader(GGUF)
    W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
    fields = {f.name: f for f in r.fields.values()}
    hp = read_hparams(fields)
    f = fields["qwen4exp.nextn_predict_layers"]
    hp["n_layer_nextn"] = int(f.parts[f.data[0]][0])
    # block_count counts the MTP block: n_layer() = n_layer_all - n_layer_nextn (src/llama-model.cpp:1327-1330,
    # src/llama-hparams.cpp:348-350)
    hp["n_layer"] -= hp["n_layer_nextn"]
    return W, hp


def load(d, g, name, k=0):
    p = pathlib.Path(d, f"g{g:03d}")
    j = json.loads((p / f"{name}__{k}.json").read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64)


def graphs_of(man):
    """Graphs of the calls of a manifest: dict(g, call, kind, p0, n, toks, row0)."""
    ub, out = man["n_ubatch"], []
    for c, call in enumerate(man["calls"]):
        g0, g1 = call["graphs"]
        for j, g in enumerate(range(g0, g1)):
            n = min(ub, call["n"] - j * ub)
            out.append(dict(g=g, call=c, kind=call["kind"], p0=call["pos0"] + j * ub, n=n,
                            row0=call["tok0"] + j * ub, toks=call.get("tokens", [])[j * ub:j * ub + n]))
    return out


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Mtp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not pathlib.Path(DUMP, "mtp", "mtp", "manifest.json").exists():
            raise unittest.SkipTest(f"no MTP run under {DUMP}")
        cls.W, cls.hp = W, hp = load_model()
        cls.dir = pathlib.Path(DUMP, "mtp")
        cls.mdir = cls.dir / "mtp"
        cls.man = json.loads((cls.dir / "manifest.json").read_text())
        cls.mman = json.loads((cls.mdir / "manifest.json").read_text())
        cls.toks = [int(x) for x in (cls.dir / "tokens.txt").read_text().split(",")]
        cls.T = len(cls.toks)
        cls.E = cls.mman["n_embd_out"]
        cls.V = W["output.weight"].shape[0]
        cls.il = hp["n_layer"]
        cls.tg = graphs_of(cls.man)
        cls.mg = graphs_of(cls.mman)
        cls.A_tgt = Q.forward(cls.toks, W, hp)
        cls.H = cls.A_tgt["layers"][hp["n_layer"] - 1]["combine_ffn"].reshape(cls.T, -1)
        cls.seen = {}
        for gr in cls.mg:
            m = load(cls.mdir, gr["g"], f"kq_mask_qsa-{cls.il}").reshape(gr["n"], -1)
            for r in range(gr["n"]):
                cls.seen[(gr["g"], r)] = sorted(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist())
        cls._oracles({})
        cls.ties, cls.not_ties, ov_b, ov_a = 0, [], {}, {}
        for gr in cls.mg:
            ob = cls.B[gr["g"]]["attn"]
            for r in range(gr["n"]):
                t, s = gr["p0"] + r, cls.seen[(gr["g"], r)]
                if s == ob["cells"][r]:
                    continue
                ok, why = Q.qsa_tie_variant(ob["scores"][r], s, t, hp["idx_top_k"], hp["kpool"])
                own, why_own = Q.qsa_tie_variant(ob["scores"][r], ob["cells"][r], t, hp["idx_top_k"], hp["kpool"])
                if not (ok and own):
                    cls.not_ties.append(f"pos {t} graph {gr['g']}: reference {why or 'valid'}, oracle "
                                        f"{why_own or 'valid'}; reference {s}, oracle {ob['cells'][r]}")
                    continue
                cls.ties += 1
                ov_b.setdefault(gr["g"], {})[t] = s
                if gr["kind"] == "process":
                    ov_a[t] = s
        if cls.ties:
            cls._oracles(ov_b, ov_a)

    @classmethod
    def _oracles(cls, ov_b, ov_a=None):
        """(A): the processed positions in one chunk; (B): graph by graph, with the draft chain."""
        W, hp, E = cls.W, cls.hp, cls.E
        n_proc = sum(c["n"] for c in cls.mman["calls"] if c["kind"] == "process")
        proc = [t for c in cls.mman["calls"] if c["kind"] == "process" for t in c["tokens"]]
        cls.h_pair = np.vstack([np.zeros((1, E)), cls.H[:n_proc - 1]])          # token p pairs with h of p-1
        ov = {("qsa", cls.il): ov_a} if ov_a else None
        cls.A = Q.mtp_forward_chunk(proc, cls.h_pair, W, hp, Q.new_cache(), ov)
        cls.B, cls.h_expected, cache, h_chain = {}, {}, Q.new_cache(), None
        for gr in cls.mg:
            if gr["kind"] == "process":
                h = cls.h_pair[gr["p0"]:gr["p0"] + gr["n"]]
            else:
                h = cls.H[n_proc - 1:n_proc] if h_chain is None else h_chain
            ovg = {("qsa", cls.il): ov_b[gr["g"]]} if gr["g"] in ov_b else None
            cls.B[gr["g"]] = o = Q.mtp_forward_chunk(gr["toks"], h, W, hp, cache, ovg)
            cls.h_expected[gr["g"]] = h
            if gr["kind"] == "draft":
                h_chain = o["h_nextn"]

    def close(self, what, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{what} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max())
        rel = mx / (float(np.abs(want).max()) + 1e-30)
        TABLE.append(("mtp", what, self.il, g, mx, rel))
        self.assertLess(rel, tol, f"MTP {what} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, gr, got, key):
        """Compare an MTP graph tensor with (A) for processed rows and (B); key selects the oracle field."""
        b = key(self.B[gr["g"]])
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b))
        if gr["kind"] == "process":
            self.close(f"{what} [full]", gr["g"], got, key(self.A)[gr["p0"]:gr["p0"] + gr["n"]])
        self.close(f"{what} [chunk]", gr["g"], got, b)

    def d(self, g, name, k=0):
        return load(self.mdir, g, name, k)

    # ---- the target side ----
    def test_target_h_nextn_and_logits(self):
        """The rows the MTP context is fed: the target's h_nextn (before its final mixer) and its logits."""
        h = np.fromfile(self.dir / "h_nextn.bin", dtype=np.float32).reshape(self.T, self.E)
        self.close("target h_nextn rows (h_nextn.bin) [full]", -1, h, self.H)
        lo = np.fromfile(self.dir / "logits.bin", dtype=np.float32).reshape(self.T, self.V)
        self.close("target logits [full]", -1, lo, self.A_tgt["logits"])
        for gr in self.tg:
            got = load(self.dir, gr["g"], "h_nextn")
            self.close("target h_nextn tensor [full]", gr["g"], got, self.H[gr["p0"]:gr["p0"] + gr["n"]])

    # ---- MTP inputs ----
    def test_mtp_inputs_and_projection(self):
        """h fed with each MTP token (pairing of common/speculative.cpp), e/h norms and eh_proj (G:558-589)."""
        h_in = np.fromfile(self.mdir / "h_in.bin", dtype=np.float32).reshape(-1, self.E)
        for gr in self.mg:
            g, n = gr["g"], gr["n"]
            rows = h_in[gr["row0"]:gr["row0"] + n]
            self.close(f"h fed ({gr['kind']}) [chunk]", g, rows, self.h_expected[g])
            if gr["kind"] == "process":
                self.close("h fed (process) [full]", g, rows, self.h_pair[gr["p0"]:gr["p0"] + n])
            self.close("mtp_h_input in the graph [chunk]", g, self.d(g, "mtp_h_input (reshaped)"), self.h_expected[g])
            emb = self.W["token_embd.weight"][gr["toks"]]
            self.close("mtp_tok_embd (exact)", g, self.d(g, f"mtp_tok_embd-{self.il}"), emb, tol=1e-300)
            self.both("mtp_hnorm (per-stream norm of h)", gr, self.d(g, f"mtp_hnorm-{self.il}"), lambda o: o["h_norm"])
            self.both("mtp_enorm (normed embedding, repeated)", gr, self.d(g, f"mtp_enorm-{self.il}"),
                      lambda o: o["e_norm"])
            self.both("mtp_eh_proj", gr, self.d(g, f"mtp_eh_proj-{self.il}"), lambda o: o["eh_proj"])

    # ---- the MTP layer ----
    def test_mtp_layer(self):
        il = self.il
        k = self.hp["n_expert_used"]
        for gr in self.mg:
            g, n = gr["g"], gr["n"]
            self.both("hc_norm (attention mixer)", gr, self.d(g, f"hc_norm-{il}", 0), lambda o: o["mix_attn"]["norm"])
            self.both("hc_mixed (attention mixer)", gr, self.d(g, f"hc_mixed-{il}", 0),
                      lambda o: o["mix_attn"]["mixed"])
            self.both("hc_inject (attention mixer)", gr, self.d(g, f"hc_inject-{il}", 0),
                      lambda o: o["mix_attn"]["inject"])
            self.both("Qcur (IMRoPE)", gr, self.d(g, f"Qcur-{il}"), lambda o: o["attn"]["q"])
            self.both("Kcur (IMRoPE)", gr, self.d(g, f"Kcur-{il}", 1), lambda o: o["attn"]["k"])
            self.both("Vcur", gr, self.d(g, f"Vcur-{il}", 0), lambda o: o["attn"]["v"])
            self.both("attn_pregate (qsa)", gr, self.d(g, f"attn_pregate-{il}"), lambda o: o["attn"]["pregate"])
            self.both("attn_output (wo)", gr, self.d(g, f"attn_output-{il}"), lambda o: o["attn"]["out"])
            self.both("hc_combine (attn)", gr, self.d(g, f"hc_combine-{il}", 0), lambda o: o["combine_attn"])
            self.both("hc_norm (MoE mixer)", gr, self.d(g, f"hc_norm-{il}", 1), lambda o: o["mix_ffn"]["norm"])
            self.both("hc_mixed (MoE mixer)", gr, self.d(g, f"hc_mixed-{il}", 1), lambda o: o["mix_ffn"]["mixed"])
            ids = np.sort(self.d(g, f"ffn_moe_topk-{il}").reshape(n, k).astype(int), 1)
            self.assertTrue(np.array_equal(ids, np.sort(self.B[g]["moe"]["ids"], 1)), f"MTP top-k g{g}")
            self.both("ffn_out (routed + gated shared)", gr, self.d(g, f"ffn_out-{il}"), lambda o: o["moe"]["out"])
            self.both("hc_combine (ffn)", gr, self.d(g, f"hc_combine-{il}", 1), lambda o: o["combine_ffn"])

    def test_mtp_qsa_pool_cache(self):
        """The MTP context's own k-pool cache: pools completed by process and draft calls, read back later."""
        hp, il = self.hp, self.il
        kp, d = hp["kpool"], hp["idx_dim"]
        self.assertEqual(self.not_ties, [], "QSA selections that are no valid top-k by the oracle's scores")
        pooled_in_draft = 0
        for gr in self.mg:
            g, p0, n = gr["g"], gr["p0"], gr["n"]
            Ab, Aa = self.B[g]["attn"], self.A["attn"]
            self.both("indexer_k_raw", gr, self.d(g, f"indexer_k_raw-{il}"), lambda o: o["attn"]["k_raw"])
            b0, b1 = Ab["pool_range"]
            self.assertEqual((b0, b1), (p0 // kp, (p0 + n) // kp))
            new = self.d(g, f"indexer_pool_k_new-{il}").reshape(-1, d)
            self.assertEqual(new.shape[0], max(b1 - b0, 1, n // kp + 1))
            if b1 > b0:
                self.close("indexer_pool_k_new (pools completed here) [chunk]", g, new[:b1 - b0], Ab["pooled_new"])
                if gr["kind"] == "process":
                    self.close("indexer_pool_k_new (pools completed here) [full]", g, new[:b1 - b0],
                               Aa["pooled"][b0:b1])
                pooled_in_draft += gr["kind"] == "draft"
            got_k = self.d(g, f"indexer_k-{il}").reshape(-1, d)
            self.close("indexer_k (pool cache) [chunk]", g, got_k[:b1], Ab["pooled"])
            if gr["kind"] == "process":
                self.close("indexer_k (pool cache) [full]", g, got_k[:b1], Aa["pooled"][:b1])
            sc = self.d(g, f"indexer_score-{il}").reshape(n, -1)
            fin = np.isfinite(Ab["scores"])
            self.assertTrue(np.array_equal(np.isfinite(sc[:, :b1]), fin), f"pool visibility g{g}")
            self.assertFalse(np.isfinite(sc[:, b1:]).any(), f"padded pools must stay masked g{g}")
            self.close("indexer_score (visible) [chunk]", g, sc[:, :b1][fin], Ab["scores"][fin])
            for r in range(n):
                self.assertEqual(self.seen[(g, r)], Ab["cells"][r], f"MTP QSA cells pos {p0 + r} g{g}")
        TABLE.append(("mtp", f"qsa selection: tie replays = {self.ties}", il, -1, 0.0, 0.0))
        self.assertGreater(pooled_in_draft, 0, "no pool completed by a draft step")
        self.assertTrue(any(len(c) < p + 1 for p, c in enumerate(self.A["attn"]["cells"])), "QSA never sparse")

    def test_mtp_head_and_draft_chain(self):
        """h_nextn (the next step's h), the head mixer, logits; the greedy draft chain."""
        il = self.il
        lo = np.fromfile(self.mdir / "logits.bin", dtype=np.float32).reshape(-1, self.V).astype(np.float64)
        h_out = np.fromfile(self.mdir / "h_nextn.bin", dtype=np.float32).reshape(-1, self.E)
        for gr in self.mg:
            g, n, r0 = gr["g"], gr["n"], gr["row0"]
            self.both("h_nextn (MTP residual)", gr, self.d(g, f"h_nextn-{il}"), lambda o: o["h_nextn"])
            self.both("h_nextn rows (h_nextn.bin)", gr, h_out[r0:r0 + n], lambda o: o["h_nextn"])
            self.both("hc_norm (head mixer)", gr, self.d(g, f"hc_norm-{il}", 2), lambda o: o["head"]["norm"])
            self.both("result_norm", gr, self.d(g, "result_norm"), lambda o: o["result_norm"])
            self.both("result_output", gr, self.d(g, "result_output"), lambda o: o["logits"])
            self.both("logits.bin", gr, lo[r0:r0 + n], lambda o: o["logits"])
            self.assertTrue(np.array_equal(lo[r0:r0 + n].argmax(1), self.B[g]["logits"].argmax(1)), f"argmax g{g}")
        drafts = [gr for gr in self.mg if gr["kind"] == "draft"]
        self.assertEqual(drafts[0]["toks"], [int(self.A_tgt["logits"][-1].argmax())], "first draft token")
        for prev, cur in zip(drafts, drafts[1:]):
            self.assertEqual(cur["toks"], [int(self.B[prev["g"]]["logits"][0].argmax())], f"draft token g{cur['g']}")

    def test_run_exercises_its_mechanisms(self):
        kinds = [(c["kind"], c["n"]) for c in self.mman["calls"]]
        self.assertIn(("process", 48), kinds)
        self.assertEqual(sum(1 for gr in self.mg if gr["call"] == 0), 3, "the 48-token call runs as 3 ubatches")
        self.assertGreaterEqual(sum(1 for k, n in kinds if k == "process" and n == 1), 1)
        self.assertGreaterEqual(sum(1 for k, n in kinds if k == "draft"), 2, "a draft chain needs 2 steps or more")
        self.assertEqual(self.mman["n_ubatch"], 16)


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, mx, rel in TABLE:
        if what not in worst or rel > worst[what][3]:
            worst[what] = (il, g, mx, rel)
    print("\nMTP: worst error per component over graphs (rel = max|got-want| / max|want|):", file=sys.stderr)
    print(f"{'component':58s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for what, (il, g, mx, rel) in worst.items():
        print(f"{what:58s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
