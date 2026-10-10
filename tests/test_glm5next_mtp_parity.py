"""The glm5-next MTP block (nextn_predict_layers = 1) of llama.cpp against ref/glm5next.py mtp_forward_chunk (CPU).

Only where llama.cpp has the graph: the R1 pin 79e2e74 does (graph_mtp, src/models/glm5-next.cpp:543-667 @79e2e74);
the AGENTS.md pin ec7630a does not (src/models/glm5-next.cpp:189-191 @ec7630a throws "GLM5-Next NextN graph not
implemented yet"), so there tools/tinygen/run_parity_glm5next_decode.sh does not produce the run and these tests skip.
Model: the G1 tiny model with an MTP block appended (tools/tinygen/glm5next_tiny.py --layers 12 --mtp: block_count 13,
layer 12 is a DSA + MoE layer with blk.12.nextn.{eh_proj, enorm, hnorm}, no mHC; the trunk weights equal the default
model's). The run (directory "mtp") drives the MTP block the way common/speculative.cpp does
(tools/dump/dump_tensors.cpp, DUMP_MTP=1):
  * the target context exports h_nextn, its post-norm hidden state rms_norm(hc_mean(streams)) * output_norm
    (src/models/glm5-next.cpp:780-787 @79e2e74);
  * after every target call a second context of type LLAMA_CONTEXT_TYPE_MTP, whose memory holds the MTP layer alone
    (src/llama-model.cpp:2503-2512 @79e2e74), processes the same tokens at the same positions, token p paired with the
    target h_nextn of p-1: a 48-token call (-ub 16: 3 graphs), then 8 one-token calls;
  * then 4 greedy draft steps: the target's greedy token after position 55 with the target row of 55, then each
    drafted token at the next position with the MTP's own h_nextn of the step before.
Every MTP graph is compared with (A) [full]: mtp_forward_chunk over all processed positions at once, fed by the
full-sequence target oracle (rows of the draft steps have no (A)), and (B) [chunk]: the same calls replayed graph by
graph through the MTP context's own MLA latent and k-pool cache, the draft chain fed with the oracle's own h_nextn.
Env: GLM5NEXT_DECODE_DUMP (root; the run is <root>/mtp, its MTP context in <root>/mtp/mtp), GLM5NEXT_MTP_GGUF,
LLAMA_CPP_GGUF_PY, PARITY_LLAMA_COMMIT. Tolerances as tests/test_glm5next_decode_parity.py (normwise relative 1e-4;
exact embedding gather; MoE ids as sets; DSA ties replayed only when both selections are valid top-k sets).
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import glm5next as G  # noqa: E402
from tests.test_glm5next_decode_parity import SHARED_PAD_CELL, gather_live, mla_probs  # noqa: E402

DUMP = os.environ.get("GLM5NEXT_DECODE_DUMP")
GGUF = os.environ.get("GLM5NEXT_MTP_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
LLAMA_COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a640789c393694fb194f1bbbf0369fc62d")
TOL = 1e-4
SKIP = ("set GLM5NEXT_DECODE_DUMP, GLM5NEXT_MTP_GGUF, LLAMA_CPP_GGUF_PY: "
        "run tools/tinygen/run_parity_glm5next_decode.sh")
TABLE = []


def load_model():
    sys.path.append(GGUF_PY)
    try:
        from gguf import GGUFReader
    finally:
        sys.path.remove(GGUF_PY)
    r = GGUFReader(GGUF)
    W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
    return W, G.hparams({f.name: f for f in r.fields.values()})


def load(d, g, name, k=0):
    p = pathlib.Path(d, f"g{g:03d}")
    j = json.loads((p / f"{name}__{k}.json").read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64)


def has(d, g, name, k=0):
    return pathlib.Path(d, f"g{g:03d}", f"{name}__{k}.json").exists()


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
            raise unittest.SkipTest(f"no MTP run under {DUMP} (llama.cpp {LLAMA_COMMIT[:7]}: no glm5-next MTP graph "
                                    "at ec7630a, see the module docstring)")
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
        kp = hp["kpool"]
        cls.n_top = hp["idx_top_k"] // kp
        cls.n_sel = kp * cls.n_top + (kp - 1 if hp["select_tail"] else 0)
        cls.tg = graphs_of(cls.man)
        cls.mg = graphs_of(cls.mman)
        cls.A_tgt = G.forward(cls.toks, W, hp)
        cls.H = cls.A_tgt["result_norm"]                    # the target's h_nextn rows
        cls.seen, cls.gather = {}, {}
        for gr in cls.mg:
            g, n = gr["g"], gr["n"]
            cls.gather[g] = has(cls.mdir, g, f"indexer_sel_idx-{cls.il}")
            if cls.gather[g]:
                idx = load(cls.mdir, g, f"indexer_sel_idx-{cls.il}").reshape(n, -1).astype(int)
                p = load(cls.mdir, g, f"kq_soft_max_gathered-{cls.il}").reshape(n, -1, idx.shape[1])
                for r in range(n):
                    cls.seen[(g, r)] = sorted(idx[r][(p[r] > 0).all(0)].tolist())
            else:
                m = load(cls.mdir, g, f"kq_mask_dsa-{cls.il}").reshape(n, -1)
                for r in range(n):
                    cls.seen[(g, r)] = sorted(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist())
        cls._oracles({})
        cls.ties, cls.not_ties, ov_b, ov_a = 0, [], {}, {}
        for gr in cls.mg:
            ob = cls.B[gr["g"]]["dsa"]
            for r in range(gr["n"]):
                t, s = gr["p0"] + r, cls.seen[(gr["g"], r)]
                if s == ob["cells"][r]:
                    continue
                args = (t, hp["idx_top_k"], kp, hp["select_tail"])
                ok, why = G.tie_variant(ob["scores"][r], s, *args)
                own, why_own = G.tie_variant(ob["scores"][r], ob["cells"][r], *args)
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
        ov = {("dsa", cls.il): ov_a} if ov_a else None
        cls.A = G.mtp_forward_chunk(proc, cls.h_pair, W, hp, G.new_cache(), ov)
        cls.B, cls.h_expected, cache, h_chain = {}, {}, G.new_cache(), None
        for gr in cls.mg:
            if gr["kind"] == "process":
                h = cls.h_pair[gr["p0"]:gr["p0"] + gr["n"]]
            else:
                h = cls.H[n_proc - 1:n_proc] if h_chain is None else h_chain
            ovg = {("dsa", cls.il): ov_b[gr["g"]]} if gr["g"] in ov_b else None
            cls.B[gr["g"]] = o = G.mtp_forward_chunk(gr["toks"], h, W, hp, cache, ovg)
            cls.h_expected[gr["g"]] = h
            if gr["kind"] == "draft":
                h_chain = o["h_nextn"]

    def close(self, what, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{what} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
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
        """The rows the MTP context is fed: the target's post-norm h_nextn, and its logits."""
        h = np.fromfile(self.dir / "h_nextn.bin", dtype=np.float32).reshape(self.T, self.E)
        self.close("target h_nextn rows (h_nextn.bin) [full]", -1, h, self.H)
        lo = np.fromfile(self.dir / "logits.bin", dtype=np.float32).reshape(self.T, self.V)
        self.close("target logits [full]", -1, lo, self.A_tgt["logits"])
        for gr in self.tg:
            got = load(self.dir, gr["g"], "h_nextn")
            self.close("target h_nextn tensor [full]", gr["g"], got, self.H[gr["p0"]:gr["p0"] + gr["n"]])

    # ---- MTP inputs ----
    def test_mtp_inputs_and_projection(self):
        """h fed with each MTP token (the pairing of common/speculative.cpp), e/h norms and eh_proj."""
        h_in = np.fromfile(self.mdir / "h_in.bin", dtype=np.float32).reshape(-1, self.E)
        il = self.il
        for gr in self.mg:
            g, n = gr["g"], gr["n"]
            rows = h_in[gr["row0"]:gr["row0"] + n]
            self.close(f"h fed ({gr['kind']}) [chunk]", g, rows, self.h_expected[g])
            if gr["kind"] == "process":
                self.close("h fed (process) [full]", g, rows, self.h_pair[gr["p0"]:gr["p0"] + n])
            emb = self.W["token_embd.weight"][gr["toks"]]
            self.close("mtp_tok_embd (exact)", g, self.d(g, f"mtp_tok_embd-{il}"), emb, tol=1e-300)
            self.both("mtp_enorm", gr, self.d(g, f"mtp_enorm-{il}"), lambda o: o["e_norm"])
            self.both("mtp_hnorm", gr, self.d(g, f"mtp_hnorm-{il}"), lambda o: o["h_norm"])
            self.both("mtp_eh_proj ([e ; h])", gr, self.d(g, f"mtp_eh_proj-{il}"), lambda o: o["eh_proj"])

    # ---- the MTP layer ----
    def test_mtp_layer(self):
        il, k = self.il, self.hp["n_expert_used"]
        for gr in self.mg:
            g, n = gr["g"], gr["n"]
            self.both("mtp_attn_norm", gr, self.d(g, f"mtp_attn_norm-{il}"), lambda o: o["attn_norm"])
            for name, key in (("q_resid", "q_resid"), ("kv_cmpr", "latent"), ("q_absorbed", "q_abs"),
                              ("indexer_k", "ik"), ("indexer_gate", "ig"), ("indexer_q", "iq"),
                              ("indexer_weights", "wts"), ("kqv_out", "kqv")):
                self.both(f"{name} (MTP DSA)", gr, self.d(g, f"{name}-{il}"), lambda o, key=key: o["dsa"][key])
            self.both("mtp_attn_out", gr, self.d(g, f"mtp_attn_out-{il}"), lambda o: o["dsa"]["out"])
            self.both("mtp_ffn_inp (residual)", gr, self.d(g, f"mtp_ffn_inp-{il}"), lambda o: o["ffn_inp"])
            self.both("mtp_ffn_norm", gr, self.d(g, f"mtp_ffn_norm-{il}"), lambda o: o["ffn_norm"])
            ids = np.sort(self.d(g, f"ffn_moe_topk-{il}").reshape(n, k).astype(int), 1)
            self.assertTrue(np.array_equal(ids, np.sort(self.B[g]["ffn"]["ids"], 1)), f"MTP top-k g{g}")
            if gr["kind"] == "process":
                want = np.sort(self.A["ffn"]["ids"][gr["p0"]:gr["p0"] + n], 1)
                self.assertTrue(np.array_equal(ids, want), f"MTP top-k (full) g{g}")
            self.both("mtp_ffn_moe_out (routed)", gr, self.d(g, f"mtp_ffn_moe_out-{il}"),
                      lambda o: o["ffn"]["moe_out"])
            self.both("mtp_ffn_shexp (shared, clamped)", gr, self.d(g, f"mtp_ffn_shexp-{il}"),
                      lambda o: o["ffn"]["shexp"])
            self.both("mtp_ffn_out", gr, self.d(g, f"mtp_ffn_out-{il}"), lambda o: o["ffn"]["out"])
            self.both("mtp_post_ffn (residual)", gr, self.d(g, f"mtp_post_ffn-{il}"), lambda o: o["post_ffn"])

    def test_mtp_pool_cache_and_selection(self):
        """The MTP context's own k-pool cache: pools completed by process and draft calls, read back later."""
        hp, il = self.hp, self.il
        kp, d = hp["kpool"], hp["idx_dim"]
        self.assertEqual(self.not_ties, [], "DSA selections that are no valid top-k by the oracle's scores")
        pooled_in_draft = 0
        for gr in self.mg:
            g, p0, n = gr["g"], gr["p0"], gr["n"]
            Db, Da = self.B[g]["dsa"], self.A["dsa"]
            b0, b1 = Db["pool_range"]
            self.assertEqual((b0, b1), (p0 // kp, (p0 + n) // kp))
            new = self.d(g, f"indexer_pool_k_new-{il}").reshape(-1, d)
            self.assertEqual(new.shape[0], max(b1 - b0, 1, n // kp + 1))
            if b1 > b0:
                self.close("indexer_pool_k_new (pools completed here) [chunk]", g, new[:b1 - b0], Db["pooled_new"])
                if gr["kind"] == "process":
                    self.close("indexer_pool_k_new (pools completed here) [full]", g, new[:b1 - b0],
                               Da["pooled"][b0:b1])
                pooled_in_draft += gr["kind"] == "draft"
            n_pad = new.shape[0] - (b1 - b0)
            ik_all = np.concatenate([np.asarray(Da["ik"])[:p0], np.asarray(Db["ik"])])     # cell = position
            if SHARED_PAD_CELL:
                cells = [0 if b1 > b0 else p0] * n_pad
            else:
                reps = {kp * j + kp - 1 for j in range(b1)}
                cells = [c for c in range(p0 + n) if c not in reps][:n_pad]
                self.assertEqual(len(cells), n_pad)
            for r, c in zip(range(b1 - b0, new.shape[0]), cells):
                self.close("indexer_pool_k_new (padding rows)", g, new[r], ik_all[c])
            got_k = self.d(g, f"indexer_pool_k-{il}").reshape(-1, d)
            if b1:
                self.close("indexer_pool_k (pool cache) [chunk]", g, got_k[:b1], Db["pooled"])
                if gr["kind"] == "process":
                    self.close("indexer_pool_k (pool cache) [full]", g, got_k[:b1], Da["pooled"][:b1])
            sc = self.d(g, f"indexer_score-{il}").reshape(n, -1)
            fin = np.isfinite(Db["scores"])
            self.assertTrue(np.array_equal(np.isfinite(sc[:, :b1]) & (sc[:, :b1] > -1e30), fin),
                            f"pool visibility g{g}")
            self.assertFalse((np.isfinite(sc[:, b1:]) & (sc[:, b1:] > -1e30)).any(),
                             f"padded pools must stay masked g{g}")
            self.close("indexer_score (visible) [chunk]", g, sc[:, :b1][fin], Db["scores"][fin])
            for r in range(n):
                self.assertEqual(self.seen[(g, r)], Db["cells"][r], f"MTP DSA cells pos {p0 + r} g{g}")
            if self.gather[g]:
                idx = self.d(g, f"indexer_sel_idx-{il}").reshape(n, -1).astype(int)
                p = self.d(g, f"kq_soft_max_gathered-{il}").reshape(n, hp["n_head"], self.n_sel)
                lat = np.concatenate([np.asarray(Da["latent"])[:p0], np.asarray(Db["latent"])])
                for r in range(n):
                    live = gather_live(p0 + r, kp, self.n_top, self.n_sel)
                    self.assertTrue(np.array_equal((p[r] > 0).all(0), live), f"live slots pos {p0 + r} g{g}")
                    want = mla_probs(Db["q_abs"][r], lat, idx[r][live], hp["k_mla"])
                    self.close("kq_soft_max_gathered (live slots)", g, p[r][:, live], want)
        TABLE.append(("mtp", f"dsa selection: tie replays = {self.ties}", il, -1, 0.0, 0.0))
        self.assertGreater(pooled_in_draft, 0, "no pool completed by a draft step")
        self.assertTrue(any(len(c) < p + 1 for p, c in enumerate(self.A["dsa"]["cells"])), "DSA never sparse")

    def test_mtp_head_and_draft_chain(self):
        """h_nextn (the next step's h), the head, logits; the greedy draft chain."""
        lo = np.fromfile(self.mdir / "logits.bin", dtype=np.float32).reshape(-1, self.V).astype(np.float64)
        h_out = np.fromfile(self.mdir / "h_nextn.bin", dtype=np.float32).reshape(-1, self.E)
        for gr in self.mg:
            g, n, r0 = gr["g"], gr["n"], gr["row0"]
            self.both("h_nextn rows (h_nextn.bin)", gr, h_out[r0:r0 + n], lambda o: o["h_nextn"])
            # the MTP context is masked (crop_after_nextn false), so the tensor named "h_nextn" is renamed
            # "mtp_shared_head_norm" by the next cb (src/models/glm5-next.cpp:650-656 @79e2e74): one tensor
            self.both("mtp_shared_head_norm (= h_nextn)", gr, self.d(g, "mtp_shared_head_norm"), lambda o: o["h_nextn"])
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
        self.assertEqual(self.hp["n_nextn"], 1)


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, mx, rel in TABLE:
        if what not in worst or rel > worst[what][3]:
            worst[what] = (il, g, mx, rel)
    print(f"\nMTP: worst error per component over graphs (rel = max|got-want| / max|want|), llama.cpp "
          f"{LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'component':58s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for what, (il, g, mx, rel) in worst.items():
        print(f"{what:58s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
