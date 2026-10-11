"""The mimo2 MTP (NextN) block against ref/mimo2.py mtp_forward_chunk (N16, gap 12 of docs/reassessment/CORRECTNESS.md).

Both llama.cpp pins have the mimo2 MTP graph (graph_mtp, src/models/mimo2.cpp:272-407 @ec7630a; the same code at
79e2e74). tools/tinygen/run_parity_mimo2_decode.sh runs n16.gguf (tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa
--value-scale 0.75 --mtp: the MTP block is layer 6, a sliding-window layer with fused attn_qkv, sinks, a dense FFN,
nextn.eh_proj / enorm / hnorm and layer_output_norm) with DUMP_MTP=1 DUMP_MTP_DRAFT=4 (tools/dump/dump_tensors.cpp,
driven like common/speculative.cpp): the target context runs prefill 240 (-ub 64) and 24 one-token calls, exporting
h_nextn (the last layer's l_out, before the output norm, mimo2.cpp:244-246); after every target call the MTP context
processes the same tokens at the same positions, token p paired with the target's h_nextn of p-1 (zeros at 0); then
four greedy draft steps (token = argmax of the previous step, h = the previous step's MTP h_nextn). The MTP context's
memory holds the MTP layer alone (src/llama-model.cpp:2807-2812) in a 256-cell SWA cache, which wraps at position 256.

Compared, every MTP graph row by row, with
  (A) [full]: mtp_forward_chunk over all processed positions at once (cache None), fed by the full-sequence target
      oracle's h_nextn (process rows only);
  (B) [chunk]: the incremental target oracle (forward_chunk) feeding the incremental MTP oracle with its own cell model
      of the MTP context's cache, draft steps included (their token and h come from (B) itself).
Components: mtp_tok_embd (exact), the h row fed (h_in.bin), mtp_hnorm, mtp_enorm, mtp_concat, mtp_eh_proj,
mtp_attn_norm, mtp_wqkv, mtp_Qcur / mtp_Kcur (RoPE), mtp_Vcur, kq-6 (all n_kv cells) and kq_soft_max-6 (0 on masked
cells), kqv_out-6, mtp_attn_out, mtp_attn_out_scaled, mtp_ffn_inp, mtp_ffn_norm, ffn_up / ffn_gate / ffn_swiglu of the
dense FFN, mtp_ffn_out, mtp_post_ffn, h_nextn (the graph's output rows, and h_nextn.bin), mtp_shared_head_norm (the
head norm: layer_output_norm here, the middle of the fallback chain mimo2.cpp:392-394), result_output and logits.bin;
the target's h_nextn.bin; the draft tokens equal (B)'s argmax chain. Tolerance 1e-4 normwise, as
tests/test_mimo2_decode_parity.py.
Env: MIMO2_DECODE_DUMP, LLAMA_CPP_GGUF_PY, PARITY_LLAMA_COMMIT.
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import mimo2 as M  # noqa: E402
from tests import test_mimo2_decode_parity as D  # noqa: E402
from tests.test_qwen4exp_decode_parity import Run  # noqa: E402

DUMP, GGUF_PY, TOL = D.DUMP, D.GGUF_PY, D.TOL
TABLE = []


@unittest.skipUnless(DUMP and GGUF_PY, D.SKIP)
class Mtp(unittest.TestCase):
    RUN = "mtp"

    @classmethod
    def setUpClass(cls):
        if not pathlib.Path(DUMP, cls.RUN, "mtp", "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} (with an MTP context) under {DUMP}")
        cls.W, cls.hp = D.model(pathlib.Path(DUMP, "n16.gguf"))
        hp, W = cls.hp, cls.W
        cls.T = Run(cls.RUN, DUMP)                                     # the target context
        cls.dir = pathlib.Path(DUMP, cls.RUN, "mtp")
        cls.man = json.loads((cls.dir / "manifest.json").read_text())
        n_embd = hp["n_embd"]
        # (A) target and MTP over the whole processed sequence
        A = M.forward(cls.T.final, W, hp)
        cls.hA = A["h_nextn"]
        h_prev = np.vstack([np.zeros((1, n_embd)), cls.hA[:-1]])
        cls.A = M.mtp_forward_chunk(cls.T.final, h_prev, W, hp)
        # (B) the incremental target, then the MTP context call by call
        tcache, hB, logits_B = M.new_cache(hp, D.N_CTX, cls.T.man["n_ubatch"]), [], []
        for gr in cls.T.graphs:
            b = M.forward_chunk(gr["toks"], W, hp, tcache)
            hB.append(b["h_nextn"])
            logits_B.append(b["logits"])
        cls.hB, cls.logits_tB = np.vstack(hB), np.vstack(logits_B)
        ub = cls.man["n_ubatch"]
        mcache = M.new_cache(hp, D.N_CTX, ub, mtp=True)
        cls.graphs, cls.B, cls.draft_tokens = [], {}, []
        pending, last = np.zeros(n_embd), cls.logits_tB[-1]
        prev_out = None
        for c, call in enumerate(cls.man["calls"]):
            n, p0 = call["n"], call["pos0"]
            if call["kind"] == "process":
                toks = call["tokens"]
                rows = np.vstack([pending[None]] + ([cls.hB[p0:p0 + n - 1]] if n > 1 else []))
                pending = cls.hB[p0 + n - 1]
            else:                                                       # draft: (B)'s own chain
                tok = int(np.argmax(last if prev_out is None else prev_out["logits"][-1]))
                cls.draft_tokens.append(tok)
                toks = [tok]
                rows = (pending if prev_out is None else prev_out["h_nextn"][-1])[None]
            g0, g1 = call["graphs"]
            for j, g in enumerate(range(g0, g1)):
                m = min(ub, n - j * ub)
                out = M.mtp_forward_chunk(toks[j * ub:j * ub + m], rows[j * ub:j * ub + m], W, hp, mcache)
                cls.B[g] = out
                cls.graphs.append(dict(g=g, call=c, kind=call["kind"], p0=p0 + j * ub, n=m,
                                       pos=list(range(p0 + j * ub, p0 + j * ub + m)), toks=toks[j * ub:j * ub + m],
                                       row0=call["tok0"] + j * ub, h=rows[j * ub:j * ub + m]))
                if call["kind"] == "draft":
                    prev_out = out
        cls.mcache = mcache

    # ---- helpers ----
    def dump(self, g, name, k=0):
        p = self.dir / f"g{g:03d}"
        j = json.loads((p / f"{name}__{k}.json").read_text())
        dt = np.float32 if j["type"] == "f32" else np.int32
        return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64), j["ne"]

    def close(self, what, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{what} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max())
        rel = mx / (float(np.abs(want).max()) + 1e-30)
        TABLE.append((what, g, mx, rel))
        self.assertLess(rel, tol, f"MTP {what} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, gr, got, a_full, b_rows):
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        if gr["kind"] == "process":
            self.close(f"{what} [full]", gr["g"], got, np.asarray(a_full)[gr["pos"]])
        self.close(f"{what} [chunk]", gr["g"], got, b_rows)

    # ---- tests ----
    def test_target_h_nextn(self):
        """The target's exported h_nextn rows (the last l_out, pre-norm) equal (A) and (B)."""
        h = np.fromfile(pathlib.Path(DUMP, self.RUN, "h_nextn.bin"), dtype=np.float32).reshape(-1, self.hp["n_embd"])
        self.assertEqual(len(h), len(self.T.final))
        self.close("target h_nextn.bin [full]", -1, h, self.hA)
        self.close("target h_nextn.bin [chunk]", -1, h, self.hB)

    def test_graphs_tokens_and_pairing(self):
        """One graph per ubatch of every MTP call; the tokens of each graph (mtp_tok_embd, exact); the h row fed with
        each token (h_in.bin) is the target's h_nextn of the previous position (process) or the previous step's MTP
        h_nextn (draft); the draft tokens are (B)'s greedy chain."""
        self.assertEqual(len(self.man["graphs"]), len(self.graphs))
        h_in = np.fromfile(self.dir / "h_in.bin", dtype=np.float32).reshape(-1, self.hp["n_embd"])
        for gr in self.graphs:
            got = self.dump(gr["g"], f"mtp_tok_embd-{self.hp['n_layer']}")[0].reshape(gr["n"], -1)
            self.close("mtp_tok_embd (exact)", gr["g"], got, self.W["token_embd.weight"][gr["toks"]], tol=1e-300)
            rows = h_in[gr["row0"]:gr["row0"] + gr["n"]]
            if gr["kind"] == "process":
                want = np.vstack([np.zeros((1, self.hp["n_embd"])), self.hA])[gr["pos"]]
                self.close("h_in (row fed) [full]", gr["g"], rows, want)
            self.close("h_in (row fed) [chunk]", gr["g"], rows, gr["h"])
        drafts = [c["tokens"][0] for c in self.man["calls"] if c["kind"] == "draft"]
        self.assertEqual(len(drafts), 4)
        self.assertEqual(drafts, self.draft_tokens)

    def test_mtp_block(self):
        hp, il = self.hp, self.hp["n_layer"]
        lo = np.fromfile(self.dir / "logits.bin", dtype=np.float32).reshape(-1, self.W["output.weight"].shape[0])
        hn = np.fromfile(self.dir / "h_nextn.bin", dtype=np.float32).reshape(-1, hp["n_embd"])
        A = self.A
        for gr in self.graphs:
            g, b, n = gr["g"], self.B[gr["g"]], gr["n"]
            a_att, b_att = A["attn"], b["attn"]
            D_ = lambda nm, k=0: self.dump(g, nm, k)[0]                 # noqa: E731
            self.both("mtp_enorm", gr, D_(f"mtp_enorm-{il}"), A["e_norm"], b["e_norm"])
            self.both("mtp_hnorm", gr, D_(f"mtp_hnorm-{il}"), A["h_norm"], b["h_norm"])
            self.both("mtp_concat [e ; h]", gr, D_(f"mtp_concat-{il}"), np.concatenate([A["e_norm"], A["h_norm"]], 1),
                      np.concatenate([b["e_norm"], b["h_norm"]], 1))
            self.both("mtp_eh_proj", gr, D_(f"mtp_eh_proj-{il}"), A["eh_proj"], b["eh_proj"])
            self.both("mtp_attn_norm", gr, D_(f"mtp_attn_norm-{il}"), A["attn_norm"], b["attn_norm"])
            fa = np.concatenate([a_att[x].reshape(len(a_att[x]), -1) for x in ("q_pre", "k_pre", "v")], 1)
            fb = np.concatenate([b_att[x].reshape(n, -1) for x in ("q_pre", "k_pre", "v")], 1)
            self.both("mtp_wqkv (fused projection)", gr, D_(f"mtp_wqkv-{il}"), fa, fb)
            self.both("mtp_Qcur (RoPE)", gr, D_(f"mtp_Qcur-{il}"), a_att["q"], b_att["q"])
            self.both("mtp_Kcur (RoPE)", gr, D_(f"mtp_Kcur-{il}"), a_att["k"], b_att["k"])
            self.both("mtp_Vcur", gr, D_(f"mtp_Vcur-{il}"), a_att["v"], b_att["v"])
            # the MTP context's cache: kq on every cell, the softmax 0 on the masked ones
            H = hp["n_head"][il]
            kq, ne = self.dump(g, f"kq-{il}")
            self.assertEqual(ne[0], b_att["n_kv"])
            self.close("kq-6 (all n_kv cells) [chunk]", g, kq.reshape(H, n, -1), b_att["kq"])
            sm = D_(f"kq_soft_max-{il}").reshape(H, n, -1)
            vis = M.visible(gr["pos"], b_att["kpos"], hp["n_swa"])
            self.assertTrue((sm[:, ~vis] == 0).all() and (sm[:, vis] > 0).all(), f"MTP mask g{g}")
            self.close("kq_soft_max-6 (cells) [chunk]", g, sm, b_att["probs"])
            if gr["kind"] == "process":
                self.close("kq_soft_max-6 (cells) [full]", g, sm, D.place_full(a_att["probs"], b_att["kpos"], gr["pos"]))
            self.both("kqv_out-6", gr, D_(f"kqv_out-{il}"), a_att["kqv"], b_att["kqv"])
            self.both("mtp_attn_out", gr, D_(f"mtp_attn_out-{il}"), a_att["attn_out"], b_att["attn_out"])
            self.both("mtp_attn_out_scaled", gr, D_(f"mtp_attn_out_scaled-{il}"), a_att["out"], b_att["out"])
            self.both("mtp_ffn_inp", gr, D_(f"mtp_ffn_inp-{il}"), A["ffn_inp"], b["ffn_inp"])
            self.both("mtp_ffn_norm", gr, D_(f"mtp_ffn_norm-{il}"), A["ffn_norm"], b["ffn_norm"])
            for nm, key in (("ffn_up", "up"), ("ffn_gate", "gate"), ("ffn_swiglu", "act")):
                self.both(f"{nm}-6 (dense)", gr, D_(f"{nm}-{il}"), A["ffn"][key], b["ffn"][key])
            self.both("mtp_ffn_out", gr, D_(f"mtp_ffn_out-{il}"), A["ffn"]["out"], b["ffn"]["out"])
            self.both("mtp_post_ffn", gr, D_(f"mtp_post_ffn-{il}"), A["post_ffn"], b["post_ffn"])
            self.both("h_nextn (graph output)", gr, D_("h_nextn"), A["h_nextn"], b["h_nextn"])
            self.both("mtp_shared_head_norm", gr, D_("mtp_shared_head_norm"), A["head_norm"], b["head_norm"])
            self.both("result_output", gr, D_("result_output"), A["logits"], b["logits"])
            self.both("logits.bin", gr, lo[gr["row0"]:gr["row0"] + n], A["logits"], b["logits"])
            self.both("h_nextn.bin", gr, hn[gr["row0"]:gr["row0"] + n], A["h_nextn"], b["h_nextn"])
            self.assertTrue(np.array_equal(lo[gr["row0"]:gr["row0"] + n].argmax(1), b["logits"].argmax(1)),
                            f"MTP greedy tokens g{g}")

    def test_run_exercises_its_mechanisms(self):
        """The MTP context's SWA cache (256 cells) wrapped (some ubatch written to cells other than its positions),
        the head norm is layer_output_norm (not output_norm), the process calls cover every target position."""
        il = self.hp["n_layer"]
        self.assertTrue(self.hp["swa"][il])
        self.assertEqual(self.mcache["swa"].size, 256)
        self.assertTrue(any(gr["pos"] != self.B[gr["g"]]["slots"]["swa"] for gr in self.graphs))
        self.assertIn("blk.6.layer_output_norm.weight", self.W)
        self.assertNotIn("blk.6.nextn.shared_head_norm.weight", self.W)
        self.assertFalse(np.allclose(self.W["blk.6.layer_output_norm.weight"], self.W["output_norm.weight"]))
        proc = [p for gr in self.graphs if gr["kind"] == "process" for p in gr["pos"]]
        self.assertEqual(proc, list(range(len(self.T.final))))


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for what, g, mx, rel in TABLE:
        if what not in worst or rel > worst[what][2]:
            worst[what] = (g, mx, rel)
    print(f"\nMTP (layer 6): worst error per component over the MTP graphs, llama.cpp {D.LLAMA_COMMIT[:7]}:",
          file=sys.stderr)
    print(f"{'component':44s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for what, (g, mx, rel) in worst.items():
        print(f"{what:44s} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
