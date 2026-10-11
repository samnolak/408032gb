"""N17: the three MTP heads of a mimo2 (MiMo-V2.x-Flash) model, nextn_layer_offset 0, 1 and 2, driven like
common/speculative.cpp's chain_heads and llama-server's seq_rm around it, against ref/mimo2.py (mtp_process,
mtp_draft, mtp_forward_chunk(offset=k)). N16 (tests/test_mimo2_mtp_parity.py) covers one head (offset 0).

The real MiMo-V2.6-Flash config has num_nextn_predict_layers 3 (third_party/hf/mimo-v2.6-flash-chat/config.json) and
llama.cpp's converter writes 3 MTP layers (conversion/mimo.py:22-24 @79e2e74). tools/tinygen/run_parity_mimo2_n17.sh
writes n17.gguf (tools/tinygen/mimo2_tiny.py --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3: the N16 trunk
and MTP block 6, plus MTP blocks 7 and 8) and runs it with DUMP_MTP=1 (tools/dump/dump_tensors.cpp) on the plan
  240, 8 x 1, d3, 4 (2 wrong), r2, d3, 4 (1 wrong), r1, d3, 4, 2 x 1, d3, 4 (3 wrong), r3, 6 x 1, then DUMP_MTP_DRAFT=3
(-ub 64, -c 512): "d3" is a 3-step draft at the current position (the next token of the sequence as the target's
sampled token, then llama-server's seq_rm of the draft positions, server-context.cpp:3423 @79e2e74), "4" the target's
verify batch, "r<n>" the rejection: seq_rm of both contexts (server-context.cpp:4505 @79e2e74, 3998 @ec7630a) and the
pending h row of the last kept position (speculative.cpp accept(), :1814-1827 @79e2e74, 1733-1746 @ec7630a). Every
target call is processed by head 0, 1, 2 in turn, each after seq_rm from the call's first position
(speculative.cpp:1598-1621 @79e2e74); a draft step k runs head k on the whole draft prefix (:1700-1708, 1767-1777
@79e2e74). The run reaches position 266: both the target's and the MTP context's SWA
caches (256 cells) wrap.

Compared, every MTP graph row by row, per head:
  (A) [full]: mtp_forward_chunk(final tokens, offset=k) over all positions at once, fed by the full target oracle's
      h_nextn (the process rows whose position is kept; every head in process takes the TARGET's h);
  (B) [chunk]: the incremental target oracle feeding ref/mimo2.py mtp_process / mtp_draft over the cell model of the
      MTP context's cache (one cell set shared by the three head layers), the server's seq_rm steps replayed from the
      MTP manifest; draft steps take llama.cpp's draft tokens (asserted equal to the oracle's own argmax) and the
      oracle's own h rows.
Checked: mtp_tok_embd (exact), the h row fed (h_in.bin) and, on llama.cpp's own numbers bit for bit, which h each head
takes (process: the target's h_nextn of the previous position for all heads; draft step k: the pending target row, then
the h_nextn each earlier step output at its last row), every block component of graph_mtp, kq-<il> on all n_kv cells
and kq_soft_max-<il> (0 on masked cells), result_output, logits.bin / h_nextn.bin of the output rows, greedy tokens.
Tolerance 1e-4 normwise (tests/test_mimo2_decode_parity.py). Env: MIMO2_N17_DUMP (the driver's <work>/n17),
LLAMA_CPP_GGUF_PY, PARITY_LLAMA_COMMIT.
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

DUMP = os.environ.get("MIMO2_N17_DUMP")
GGUF_PY = D.GGUF_PY
LLAMA_COMMIT = D.LLAMA_COMMIT
SKIP = "set MIMO2_N17_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_mimo2_n17.sh"
N_CTX = D.N_CTX
TABLE = []        # (run, component, head, graph, max abs, rel)


def load_mtp_run(root, run):
    """The MTP context's manifest and directory of a run."""
    d = pathlib.Path(root, run, "mtp")
    return d, json.loads((d / "manifest.json").read_text())


class _Heads:
    RUN = "mtp3"
    PREC = None                 # ref/fattn.py precision of the run (None: f32 caches, flash attention off)
    TOL = D.TOL
    TABLE = TABLE
    ROOT = None                 # dump root (None: MIMO2_N17_DUMP); N21 runs (tests/test_mimo2_mtp_fix_parity.py) set it
    MODEL = "n17.gguf"
    FIXED = False               # N21: the run's process calls follow the patched order (DUMP_MTP_FIX, slot replay)

    @classmethod
    def prec_full(cls, pos_modes):
        return None if cls.PREC is None else dict(cls.PREC, modes=pos_modes)

    @classmethod
    def setUpClass(cls):
        root = cls.ROOT or DUMP
        if not (root and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(root, cls.RUN, "mtp", "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} (with an MTP context) under {root}")
        cls.W, cls.hp = D.model(pathlib.Path(root, cls.MODEL))
        W, hp = cls.W, cls.hp
        cls.H = hp["n_nextn"]
        cls.T = Run(cls.RUN, root)
        cls.dir, cls.man = load_mtp_run(root, cls.RUN)
        assert bool(cls.man.get("slot_replay")) == cls.FIXED, (cls.RUN, cls.man.get("slot_replay"))
        n_embd, ub = hp["n_embd"], cls.T.man["n_ubatch"]
        prec = cls.PREC
        # ---- target: (A) and (B), the h rows per call and per position
        tmodes = {}
        for gr in cls.T.graphs:
            for p, k in zip(gr["pos"], gr["kept"]):
                if k:
                    tmodes[p] = cls.mode_of(len(gr["pos"]))
        A = M.forward(cls.T.final, W, hp, prec=cls.prec_full(tmodes))
        cls.hA = A["h_nextn"]
        tcache, call_rows, cls.hB_at, cls.tB = M.new_cache(hp, N_CTX, ub), {}, {}, {}
        by_call = {}
        for gr in cls.T.graphs:
            by_call.setdefault(gr["call"], []).append(gr)
        for c, call in enumerate(cls.T.calls):
            if call["kind"] == "rollback":
                M.cache_truncate(tcache, call["pos0"])
                continue
            rows = []
            for gr in by_call[c]:
                b = M.forward_chunk(gr["toks"], W, hp, tcache, prec=prec)
                cls.tB[gr["g"]] = b
                rows.append(b["h_nextn"])
                for r, p in enumerate(gr["pos"]):
                    cls.hB_at[p] = b["h_nextn"][r]
            call_rows[c] = np.vstack(rows)
        # kept flags of every target decode call's rows (a process row is final iff the target row of the call is)
        call_kept = {}
        for gr in cls.T.graphs:
            call_kept.setdefault(gr["call"], []).extend(gr["kept"])
        # ---- MTP (A): every head over the final sequence with the target's h
        h_prev_A = np.vstack([np.zeros((1, n_embd)), cls.hA[:-1]])
        mmodes = {}
        cls.A = {}
        # ---- MTP (B): replay the MTP manifest
        mcache = M.new_cache(hp, N_CTX, ub, mtp=True)
        pending = np.zeros(n_embd)
        tq = [c for c, call in enumerate(cls.T.calls) if call["kind"] == "decode"]
        cls.graphs, cls.B, cls.drafts, cls.stored = [], {}, [], {}
        mc, i, ti = cls.man["calls"], 0, 0
        while i < len(mc):
            call = mc[i]
            kind, p0 = call["kind"], call["pos0"]
            if kind == "process" and cls.FIXED and cls.H > 1:
                # N21 order: per chunk of ub tokens, heads 0 .. H-1, one record (one graph) each
                tc = tq[ti]
                ti += 1
                tcall = cls.T.calls[tc]
                n = tcall["n"]
                assert tcall["pos0"] == p0, (cls.RUN, i, tcall, call)
                n_chunks = -(-n // ub)
                recs = mc[i:i + cls.H * n_chunks]
                want = [("process", p0 + c * ub, h, min(ub, n - c * ub)) for c in range(n_chunks) for h in range(cls.H)]
                assert [(r["kind"], r["pos0"], r["head"], r["n"]) for r in recs] == want, (cls.RUN, i, recs)
                assert len(mcache["tokens"]) == p0, (cls.RUN, i, len(mcache["tokens"]), p0)
                toks = [t for r in recs if r["head"] == 0 for t in r["tokens"]]
                rows = np.vstack([pending[None]] + ([call_rows[tc][:n - 1]] if n > 1 else []))
                res = M.mtp_process(toks, rows, W, hp, mcache, ub, prec=prec, snap=True, fixed=True)
                for k, rec in enumerate(recs):
                    c, head = divmod(k, cls.H)
                    g0, g1 = rec["graphs"]
                    assert g1 - g0 == 1, (cls.RUN, rec)
                    m = rec["n"]
                    pos = list(range(p0 + c * ub, p0 + c * ub + m))
                    for p in pos:
                        mmodes[p] = cls.mode_of(m)
                    cls.B[g0] = out = res[head][c]
                    cls.stored[g0] = out["snap"]
                    cls.graphs.append(dict(g=g0, rec=rec, kind="process", head=head, p0=pos[0], n=m, pos=pos,
                                           toks=rec["tokens"], row0=rec["tok0"], h=rows[c * ub:c * ub + m],
                                           kept=list(call_kept[tc][c * ub:c * ub + m]),
                                           out_rows=[rec["out0"] + r for r in range(m)], out_idx=list(range(m)),
                                           call_p0=p0, call_n=n))
                pending = call_rows[tc][-1]
                i += len(recs)
            elif kind == "process":
                tc = tq[ti]
                ti += 1
                tcall = cls.T.calls[tc]
                assert (tcall["pos0"], tcall["n"]) == (p0, call["n"]), (cls.RUN, i, tcall, call)
                n = call["n"]
                rows = np.vstack([pending[None]] + ([call_rows[tc][:n - 1]] if n > 1 else []))
                heads = mc[i:i + (cls.H if cls.H > 1 else 1)]
                assert [h["kind"] for h in heads] == ["process"] * len(heads) and \
                    [h["head"] for h in heads] == list(range(len(heads))), (cls.RUN, i, heads)
                assert len(mcache["tokens"]) == p0, (cls.RUN, i, len(mcache["tokens"]), p0)
                res = M.mtp_process(heads[0]["tokens"], rows, W, hp, mcache, ub, prec=prec, snap=True)
                for rec in heads:
                    head = rec["head"]
                    g0, g1 = rec["graphs"]
                    assert g1 - g0 == len(res[head]), (cls.RUN, rec)
                    for j, g in enumerate(range(g0, g1)):
                        m = min(ub, n - j * ub)
                        pos = list(range(p0 + j * ub, p0 + j * ub + m))
                        for p in pos:
                            mmodes[p] = cls.mode_of(m)
                        cls.B[g] = out = res[head][j]
                        cls.stored[g] = out["snap"]
                        cls.graphs.append(dict(g=g, rec=rec, kind="process", head=head, p0=pos[0], n=m, pos=pos,
                                               toks=rec["tokens"][j * ub:j * ub + m], row0=rec["tok0"] + j * ub,
                                               h=rows[j * ub:j * ub + m],
                                               kept=list(call_kept[tc][j * ub:j * ub + m]),
                                               out_rows=[rec["out0"] + j * ub + r for r in range(m)],
                                               out_idx=list(range(m)), call_p0=p0, call_n=n))
                pending = call_rows[tc][-1]
                i += len(heads)
            elif kind in ("rollback", "seq_rm"):
                M.cache_truncate(mcache, p0)
                if kind == "rollback":
                    pending = cls.hB_at[p0 - 1]
                i += 1
            elif kind == "draft":
                steps = [mc[i]]
                while i + len(steps) < len(mc) and mc[i + len(steps)]["kind"] == "draft" and \
                        len(mc[i + len(steps)]["tokens"]) == len(steps) + 1 and cls.H > 1:
                    steps.append(mc[i + len(steps)])
                id_last = steps[0]["tokens"][0]
                fed = [s["tokens"][-1] for s in steps[1:]]
                own, res = M.mtp_draft(id_last, pending, W, hp, mcache, len(steps), tokens=fed, prec=prec, snap=True)
                lo = np.fromfile(cls.dir / "logits.bin", dtype=np.float32).reshape(-1, W["output.weight"].shape[0])
                ll = [int(np.argmax(lo[s["out0"] + len(s["outputs"]) - 1])) for s in steps]
                cls.drafts.append(dict(pos0=p0, id_last=id_last, own=own, llama=ll, recs=steps))
                for k, (rec, st) in enumerate(zip(steps, res)):
                    g0, g1 = rec["graphs"]
                    assert g1 - g0 == 1, rec
                    cls.B[g0] = st["out"]
                    cls.stored[g0] = st["out"]["snap"]
                    cls.graphs.append(dict(g=g0, rec=rec, kind="draft", head=rec["head"], p0=st["pos0"],
                                           n=len(rec["tokens"]), pos=list(range(st["pos0"], st["pos0"] + rec["n"])),
                                           toks=rec["tokens"], row0=rec["tok0"], h=st["rows"],
                                           kept=[False] * rec["n"], out_rows=[rec["out0"] + r for r in
                                                                              range(len(rec["outputs"]))],
                                           out_idx=rec["outputs"], step=k))
                i += len(steps)
            else:
                raise AssertionError(f"unknown MTP call kind {kind}")
        cls.mcache = mcache
        for head in range(cls.H):
            cls.A[head] = M.mtp_forward_chunk(cls.T.final, h_prev_A, W, hp, offset=head,
                                              prec=cls.prec_full(mmodes))
        cls.graphs.sort(key=lambda gr: gr["g"])

    @classmethod
    def mode_of(cls, n_rows):
        from ref import fattn as F
        return F.mode_for(F.kv_f16(cls.PREC), bool(cls.PREC and cls.PREC.get("fa")), n_rows) if cls.PREC else "exact"

    # ---- helpers ----
    def dump(self, g, name, k=0):
        p = self.dir / f"g{g:03d}"
        j = json.loads((p / f"{name}__{k}.json").read_text())
        dt = np.float32 if j["type"] == "f32" else np.int32
        return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64), j["ne"]

    def has(self, g, name, k=0):
        return (self.dir / f"g{g:03d}" / f"{name}__{k}.json").exists()

    def close(self, what, head, g, got, want, tol=None):
        tol = self.TOL if tol is None else tol
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} head {head} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        self.TABLE.append((self.RUN, what, head, g, mx, rel))
        self.assertLess(rel, tol, f"{self.RUN} {what} head {head} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, gr, got, a_full, b_rows, tol=None, clean=None):
        """(A) on the kept process rows (and, clean given, only the rows whose attention read no foreign K/V: see
        test_head0_reads_foreign_kv), (B) on every row."""
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        kept = np.array(gr["kept"])
        if clean is not None:
            kept = kept & np.asarray(clean)
        if gr["kind"] == "process" and kept.any():
            self.close(f"{what} [full]", gr["head"], gr["g"], got[kept],
                       np.asarray(a_full)[np.array(gr["pos"])[kept]], tol)
        self.close(f"{what} [chunk]", gr["head"], gr["g"], got, b_rows, tol)

    def files(self):
        n_vocab, n_embd = self.W["output.weight"].shape[0], self.hp["n_embd"]
        lo = np.fromfile(self.dir / "logits.bin", dtype=np.float32).reshape(-1, n_vocab)
        hn = np.fromfile(self.dir / "h_nextn.bin", dtype=np.float32).reshape(-1, n_embd)
        h_in = np.fromfile(self.dir / "h_in.bin", dtype=np.float32).reshape(-1, n_embd)
        h_tgt = np.fromfile(pathlib.Path(self.ROOT or DUMP, self.RUN, "h_nextn.bin"), dtype=np.float32).reshape(-1,
                                                                                                         n_embd)
        return lo, hn, h_in, h_tgt

    # ---- tests ----
    def test_graphs_heads_and_pairing(self):
        """One graph per ubatch of every MTP call; the graph of head k names layer n_layer + k (the offset reached the
        graph); the tokens (mtp_tok_embd, exact); the h rows fed (h_in.bin) against (B) and (A); and, bit for bit on
        llama.cpp's own numbers, which h each head takes: in process every head the target's h_nextn of the previous
        position (the first token the pending row: the previous call's last row, after a rejection the row of the last
        kept position); in draft step k the pending target row, then the h_nextn row each earlier step output at its
        last row, with the tokens [id_last, the earlier steps' argmax]."""
        hp, nl = self.hp, self.hp["n_layer"]
        self.assertEqual(len(self.man["graphs"]), len(self.graphs))
        self.assertEqual([gr["g"] for gr in self.graphs], list(range(len(self.graphs))))
        lo, hn, h_in, h_tgt = self.files()
        n_exact = 0
        for gr in self.graphs:
            g, head = gr["g"], gr["head"]
            self.assertTrue(self.has(g, f"mtp_tok_embd-{nl + head}"), f"g{g}: no layer {nl + head}")
            for other in range(self.H):
                if other != head:
                    self.assertFalse(self.has(g, f"mtp_tok_embd-{nl + other}"), f"g{g}: layer {nl + other}")
            got = self.dump(g, f"mtp_tok_embd-{nl + head}")[0].reshape(gr["n"], -1)
            self.close("mtp_tok_embd (exact)", head, g, got, self.W["token_embd.weight"][gr["toks"]], tol=1e-300)
            rows = h_in[gr["row0"]:gr["row0"] + gr["n"]]
            if gr["kind"] == "process":
                want_a = np.vstack([np.zeros((1, hp["n_embd"])), self.hA])[gr["pos"]]
                kept = np.array(gr["kept"])
                if kept.any():
                    self.close("h_in (row fed) [full]", head, g, rows[kept], want_a[kept])
            self.close("h_in (row fed) [chunk]", head, g, rows, gr["h"])
        # bit-exact pairing on llama.cpp's numbers: the target rows of every decode call, in call order (the MTP
        # process records come as one group of H heads per target decode call, in the same order)
        dec, row = [], 0
        for c in self.T.calls:
            if c["kind"] == "decode":
                dec.append((c["pos0"], h_tgt[row:row + c["n"]]))
                row += c["n"]
        pend = np.zeros(hp["n_embd"], dtype=np.float32)
        tgt_at, n_grp, chain_rows, chain_toks = {}, -1, None, None
        done = None                     # rows of the current target call the process records have covered (head 0)
        for i, rec in enumerate(self.man["calls"]):
            kind = rec["kind"]
            if kind == "rollback":
                pend = tgt_at[rec["pos0"] - 1]
                continue
            if kind == "seq_rm":
                continue
            rows = h_in[rec["tok0"]:rec["tok0"] + rec["n"]]
            if kind == "process":
                if rec["head"] == 0 and (done is None or done == len(dec[n_grp][1])):
                    n_grp += 1
                    done = 0
                p0, tgt = dec[n_grp]
                off = rec["pos0"] - p0          # 0, or the chunk's offset in the N21 order
                self.assertEqual(off, done if self.FIXED else 0, f"process call {i}")
                full = np.vstack([pend[None]] + ([tgt[:-1]] if len(tgt) > 1 else []))
                want = full[off:off + rec["n"]]
                self.assertTrue(np.array_equal(rows, want), f"process call {i} head {rec['head']}: h rows")
                n_exact += rows.size
                if rec["head"] == self.H - 1:
                    done += rec["n"] if self.FIXED else len(tgt)
                    if done == len(tgt):
                        for k in range(len(tgt)):
                            tgt_at[p0 + k] = tgt[k]
                        pend = tgt[-1]
            elif kind == "draft":
                if rec["head"] == 0:
                    chain_rows, chain_toks = [pend], [rec["tokens"][0]]
                self.assertEqual(rec["tokens"], chain_toks, f"draft call {i}: tokens")
                self.assertTrue(np.array_equal(rows, np.vstack(chain_rows)), f"draft call {i}: h rows")
                self.assertEqual(rec["outputs"], [rec["n"] - 1], f"draft call {i}: outputs")
                n_exact += rows.size
                chain_rows = chain_rows + [hn[rec["out0"]]]
                chain_toks = chain_toks + [int(np.argmax(lo[rec["out0"]]))]
        self.assertEqual(n_grp + 1, len(dec))
        self.TABLE.append((self.RUN, f"h rows fed, bit-exact pairing checked on {n_exact} elements", -1, -1, 0.0, 0.0))

    def test_mtp_heads_block(self):
        """Every graph_mtp component per head and graph against (B) and, for kept process rows, (A); logits and h_nextn
        of the output rows; the greedy token of every output row."""
        hp, nl = self.hp, self.hp["n_layer"]
        lo, hn, _, _ = self.files()
        for gr in self.graphs:
            g, head, b, n = gr["g"], gr["head"], self.B[gr["g"]], gr["n"]
            il = nl + head
            A = self.A[head]
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
            tol_att = self.tol_after_attention()
            # rows whose attention read only own K/V, with every position of their window held by a cell (N21)
            cl = ~(np.asarray(b_att["foreign"]) | np.asarray(b_att["missing"]))
            self.both("kqv_out", gr, D_(f"kqv_out-{il}"), a_att["kqv"], b_att["kqv"], tol_att, cl)
            self.both("mtp_attn_out", gr, D_(f"mtp_attn_out-{il}"), a_att["attn_out"], b_att["attn_out"], tol_att, cl)
            self.both("mtp_attn_out_scaled", gr, D_(f"mtp_attn_out_scaled-{il}"), a_att["out"], b_att["out"], tol_att,
                      cl)
            self.both("mtp_ffn_inp", gr, D_(f"mtp_ffn_inp-{il}"), A["ffn_inp"], b["ffn_inp"], tol_att, cl)
            self.both("mtp_ffn_norm", gr, D_(f"mtp_ffn_norm-{il}"), A["ffn_norm"], b["ffn_norm"], tol_att, cl)
            for nm, key in (("ffn_up", "up"), ("ffn_gate", "gate"), ("ffn_swiglu", "act")):
                self.both(f"{nm} (dense)", gr, D_(f"{nm}-{il}"), A["ffn"][key], b["ffn"][key], tol_att, cl)
            self.both("mtp_ffn_out", gr, D_(f"mtp_ffn_out-{il}"), A["ffn"]["out"], b["ffn"]["out"], tol_att, cl)
            self.both("mtp_post_ffn", gr, D_(f"mtp_post_ffn-{il}"), A["post_ffn"], b["post_ffn"], tol_att, cl)
            # the output rows (graph outputs, logits.bin, h_nextn.bin): rows out_idx of the ubatch
            oi = np.array(gr["out_idx"])
            sub = dict(gr, kept=list(np.array(gr["kept"])[oi]), pos=list(np.array(gr["pos"])[oi]))
            co = cl[oi]
            self.both("h_nextn (graph output)", sub, D_("h_nextn"), A["h_nextn"], b["h_nextn"][oi], tol_att, co)
            self.both("mtp_shared_head_norm", sub, D_("mtp_shared_head_norm"), A["head_norm"], b["head_norm"][oi],
                      tol_att, co)
            self.both("result_output", sub, D_("result_output"), A["logits"], b["logits"][oi], tol_att, co)
            rows = lo[gr["out_rows"]]
            self.both("logits.bin", sub, rows, A["logits"], b["logits"][oi], tol_att, co)
            self.both("h_nextn.bin", sub, hn[gr["out_rows"]], A["h_nextn"], b["h_nextn"][oi], tol_att, co)
            self.assertTrue(np.array_equal(rows.argmax(1), b["logits"][oi].argmax(1)), f"MTP greedy tokens g{g}")

    def tol_after_attention(self):
        return self.TOL

    def test_kv_cells(self):
        """The MTP context's cache, per head: n_kv, kq-<il> (K . q) on every one of the n_kv cells (empty, stale and
        masked ones included: the cells each head's layer holds, written by that head only), kq_soft_max-<il> exactly
        0 on the masked cells, > 0 on the visible ones, equal to (B)."""
        hp, nl = self.hp, self.hp["n_layer"]
        for gr in self.graphs:
            g, head, n = gr["g"], gr["head"], gr["n"]
            il = nl + head
            b_att = self.B[g]["attn"]
            H = hp["n_head"][il]
            kq, ne = self.dump(g, f"kq-{il}")
            self.assertEqual(ne[0], b_att["n_kv"], f"n_kv head {head} g{g}")
            self.close("kq (all n_kv cells) [chunk]", head, g, kq.reshape(H, n, -1), b_att["kq"])
            sm = self.dump(g, f"kq_soft_max-{il}")[0].reshape(H, n, -1)
            vis = M.visible(gr["pos"], b_att["kpos"], hp["n_swa"])
            self.assertTrue((sm[:, ~vis] == 0).all() and (sm[:, vis] > 0).all(), f"MTP mask head {head} g{g}")
            self.close("kq_soft_max (cells) [chunk]", head, g, sm, b_att["probs"])

    def test_head0_reads_foreign_kv(self):
        """Finding (docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md): process() re-places every call once per head
        after seq_rm(first position) (speculative.cpp:1598-1617 @79e2e74), and each head writes only its own layer.
        seq_rm moves the cache head back to the first freed cell (src/llama-kv-cache.cpp:410-419 @ec7630a, 428-437
        @79e2e74), so a call whose ubatch head 0 wrapped around the end of the 256-cell SWA ring (cells 253, 254, 255,
        0) is placed by heads 1 and 2 from cell 0 (cells 0, 1, 2, 3); the cell positions follow the last head. Head 0's
        layer then holds, under cells 0..3, the K/V of positions 256, 1, 2, 3 where the cells say 253..256, and every
        later head-0 row whose window reaches them attends to those. llama.cpp equals the cell-accurate oracle (B) there
        (test_kv_cells, test_mtp_heads_block) and differs from the full-sequence oracle (A); no other head and no other
        row is affected (they are compared with (A) in test_mtp_heads_block)."""
        nl = self.hp["n_layer"]
        per_call = {}
        for gr in self.graphs:
            if gr["kind"] == "process":
                per_call.setdefault((gr["rec"]["tok0"] - gr["head"] * gr["rec"]["n"], gr["p0"]), {})[gr["head"]] = \
                    self.B[gr["g"]]["slots"]["swa"]
        split = [(k, v) for k, v in per_call.items() if len({json.dumps(c) for c in v.values()}) > 1]
        self.assertEqual(len(split), 1, split)
        (_, p0), cells = split[0]
        self.assertEqual(p0, 253)
        self.assertEqual(cells[0], [253, 254, 255, 0])
        self.assertTrue(all(cells[h] == [0, 1, 2, 3] for h in range(1, self.H)), cells)
        foreign = [(gr, np.nonzero(self.B[gr["g"]]["attn"]["foreign"])[0]) for gr in self.graphs]
        foreign = [(gr, r) for gr, r in foreign if r.size]
        self.assertTrue(foreign, "no row read foreign K/V")
        self.assertEqual({gr["head"] for gr, _ in foreign}, {0})
        worst_kqv = worst_lo = 0.0
        lo, _, _, _ = self.files()
        n_rows = 0
        for gr, rows in foreign:
            g, il = gr["g"], nl
            n_rows += len(rows)
            if gr["kind"] != "process":
                continue
            got = self.dump(g, f"kqv_out-{il}")[0].reshape(gr["n"], -1)
            for r in rows:
                if not gr["kept"][r]:
                    continue
                want = self.A[0]["attn"]["kqv"][gr["pos"][r]]
                worst_kqv = max(worst_kqv, float(np.abs(got[r] - want).max() / np.abs(want).max()))
                if r in gr["out_idx"]:
                    lr = lo[gr["out_rows"][list(gr["out_idx"]).index(r)]]
                    wl = self.A[0]["logits"][gr["pos"][r]]
                    worst_lo = max(worst_lo, float(np.abs(lr - wl).max() / np.abs(wl).max()))
        self.assertGreater(worst_kqv, 1e-2)
        self.assertEqual(sorted({gr["pos"][r] for gr, rows in foreign for r in rows}), list(range(257, 264)))
        self.assertEqual(n_rows, 11)
        self.TABLE.append((self.RUN, f"head 0 kqv_out vs (A), {n_rows} rows reading foreign K/V (expected to differ)",
                           0, -1, 0.0, worst_kqv))
        self.TABLE.append((self.RUN, "head 0 logits vs (A), the same rows (expected to differ)", 0, -1, 0.0, worst_lo))
        self.foreign_positions = sorted({gr["pos"][r] for gr, rows in foreign for r in rows})
        print(f"\n{self.RUN}: head-0 rows reading foreign K/V: {len(foreign)} graphs, positions "
              f"{self.foreign_positions}; worst kqv_out vs (A) {worst_kqv:.3e}, logits {worst_lo:.3e}", file=sys.stderr)

    def test_draft_tokens(self):
        """Every draft step's greedy token: llama.cpp's (argmax of its logits) equals the oracle's own argmax."""
        self.assertGreaterEqual(len(self.drafts), getattr(self, "MIN_DRAFTS", 5))
        for d in self.drafts:
            self.assertEqual(len(d["own"]), self.H, d)
            self.assertEqual(d["llama"], d["own"], f"draft at {d['pos0']}: llama.cpp {d['llama']} vs oracle {d['own']}")

    def test_run_exercises_its_mechanisms(self):
        """Three heads, each with process and draft graphs; the MTP SWA cache (256 cells) wrapped; heads 1 and 2 of a
        process call write the same cells (seq_rm before each head, then the same placement; head 0 differs once,
        test_head0_reads_foreign_kv); the rejections (2, 1, 3 positions) and the drafts at the expected positions; the
        heads differ."""
        hp, nl = self.hp, self.hp["n_layer"]
        self.assertEqual(self.H, 3)
        self.assertEqual(hp["n_layer"], 6)
        self.assertTrue(all(hp["swa"][nl + k] for k in range(self.H)))
        for head in range(self.H):
            kinds = {gr["kind"] for gr in self.graphs if gr["head"] == head}
            self.assertEqual(kinds, {"process", "draft"}, f"head {head}")
        self.assertEqual(self.mcache["swa"].size, 256)
        self.assertTrue(any(gr["pos"] != self.B[gr["g"]]["slots"]["swa"] for gr in self.graphs), "no wrap")
        slots = {}
        for gr in self.graphs:
            if gr["kind"] == "process":
                slots.setdefault((gr["p0"], gr["n"]), {}).setdefault(gr["head"], []).append(
                    self.B[gr["g"]]["slots"]["swa"])
        for key, per_head in slots.items():      # heads 1 .. H-1 always agree (head 0: test_head0_reads_foreign_kv)
            self.assertEqual(len({json.dumps(v[-1]) for h, v in per_head.items() if h > 0}), 1,
                             f"cells differ between heads 1 .. H-1 at {key}")
        rbs = [(c["pos0"], c["n"]) for c in self.man["calls"] if c["kind"] == "rollback"]
        self.assertEqual(rbs, [(250, 2), (253, 1), (260, 3)])
        self.assertEqual([d["pos0"] for d in self.drafts], [248, 250, 253, 259, 266])
        last = {head: self.A[head]["logits"] for head in range(self.H)}
        self.assertGreater(np.abs(last[1] - last[0]).max(), 1e-2)
        self.assertGreater(np.abs(last[2] - last[1]).max(), 1e-2)


class _HeadsF16FA(_Heads):
    """The same run at the R1 default precision (-ctk f16 -ctv f16 -fa on; the MTP context gets the same cache types
    and flash-attention flag, tools/dump/dump_tensors.cpp common_context_params_to_llama): the oracles at
    ref/fattn.py precision("f16", True); TOL16 = 16 U16 normwise everywhere but the token rows (the h each head takes
    comes from the f16 target already); no kq / kq_soft_max nodes, so the MTP caches are checked through their views
    and the flash-attention op on llama.cpp's own inputs (tests/test_mimo2_f16fa_parity.py)."""
    RUN = "fa/mtp3"

    @classmethod
    def setUpClass(cls):
        from ref import fattn as F
        cls.PREC = F.precision("f16", True)
        cls.TOL = 16 * F.U16
        super().setUpClass()

    def test_kv_cells(self):
        """The MTP caches through the views: (a) the rows graph g wrote (B's cells) are f16 of its own dumped
        mtp_Kcur / mtp_Vcur, every element; (b) every one of the n_kv cells the head's attention read (get_k / get_v
        views: the head's own layer) against the rows the oracle holds there (tests/f16fa.py cache_check)."""
        from tests.f16fa import cache_check, cache_share, store_check
        hp, nl = self.hp, self.hp["n_layer"]
        stats, n_store = {}, 0
        self.CACHE_AGREE = 0.5
        for gr in self.graphs:
            g, head, n = gr["g"], gr["head"], gr["n"]
            il = nl + head
            a = self.B[g]["attn"]
            Hkv = hp["n_head_kv"][il]
            Kst, Vst = self.stored[g]
            for kv, Dd, cur, st in (("k", hp["k_dim"], "mtp_Kcur", Kst), ("v", hp["v_dim"], "mtp_Vcur", Vst)):
                m = json.loads((self.dir / f"g{g:03d}" / f"cache_{kv}_l{il} (view)__1.json").read_text())
                self.assertEqual((m["ggml_type"], m["ne"][2]), ("f16", a["n_kv"]), f"cache_{kv}_l{il} g{g}")
                whole = self.dump(g, f"cache_{kv}_l{il} (view)", 0)[0].reshape(-1, Hkv * Dd)
                view = self.dump(g, f"cache_{kv}_l{il} (view)", 1)[0].reshape(-1, Hkv, Dd)
                n_store += store_check(self, kv.upper(), head, g, whole[a["idxs"]],
                                       self.dump(g, f"{cur}-{il}")[0].reshape(n, -1))
                cache_check(self, kv.upper(), head, g, view, st[:a["n_kv"]], stats, self.TOL)
        self.TABLE.append((self.RUN, f"cache store: {n_store} elements equal to the rounded pre-store value", -1, -1,
                           0.0, 0.0))
        cache_share(self, stats)

    def test_flash_attention_op(self):
        """ref/fattn.py attend() with the head's sinks on llama.cpp's own q (mtp_Qcur), K / V views and the cells B's
        cell model makes visible: kqv_out within one binary16 spacing (one_chunk) or 1e-5 normwise (tiled)."""
        from ref import fattn as F
        from tests.f16fa import ulp_diff
        hp, nl = self.hp, self.hp["n_layer"]
        n_half = n_all = 0
        for gr in self.graphs:
            g, head, n = gr["g"], gr["head"], gr["n"]
            il = nl + head
            a = self.B[g]["attn"]
            Hkv = hp["n_head_kv"][il]
            q = self.dump(g, f"mtp_Qcur-{il}")[0].reshape(n, hp["n_head"][il], hp["k_dim"])
            k = self.dump(g, f"cache_k_l{il} (view)", 1)[0].reshape(-1, Hkv, hp["k_dim"])
            v = self.dump(g, f"cache_v_l{il} (view)", 1)[0].reshape(-1, Hkv, hp["v_dim"])
            vis = M.visible(gr["pos"], a["kpos"], hp["n_swa"])
            cells = [np.nonzero(vis[t])[0] for t in range(n)]
            mode = self.mode_of(n)
            want = F.attend(q, k, v, cells, M.kq_scale(hp, self.PREC), mode,
                            M.layer_weights(self.W, il)["attn_sinks.weight"])
            got = self.dump(g, f"kqv_out-{il}")[0].reshape(want.shape)
            if mode == "fa_chunk":
                u = ulp_diff(got, want)
                n_half += int((u > 0.5).sum())
                n_all += u.size
                self.assertLessEqual(float(u.max()), 1.0, f"flash attention head {head} g{g}: {u.max():.2f} spacings")
            else:
                self.close(f"flash attention op ({mode}, own inputs)", head, g, got, want, 1e-5)
        self.TABLE.append((self.RUN, f"flash attention op (one_chunk): elements > 0.5 spacing {n_half}/{n_all}", -1,
                           -1, 0.0, n_half / max(n_all, 1)))
        self.assertLessEqual(n_half, 1e-3 * n_all)


def _cls(name, base=_Heads, **kw):
    return unittest.skipUnless(DUMP and GGUF_PY, SKIP)(type(name, (base, unittest.TestCase), kw))


MtpHeads = _cls("MtpHeads")
MtpHeadsF16FA = _cls("MtpHeadsF16FA", _HeadsF16FA)


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, head, g, mx, rel in (r[:6] for r in TABLE):
        key = (run, what, head)
        if key not in worst or rel > worst[key][2]:
            worst[key] = (g, mx, rel)
    print(f"\nMTP heads: worst error per run, component and head over the MTP graphs, llama.cpp {LLAMA_COMMIT[:7]}:",
          file=sys.stderr)
    print(f"{'run':8s} {'component':44s} {'head':>4s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, what, head), (g, mx, rel) in worst.items():
        print(f"{run:8s} {what:44s} {head:4d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)
    per_layer = {}
    for run, what, head, g, mx, rel in (r[:6] for r in TABLE):
        if g < 0 or "expected to differ" in what or head < 0:
            continue
        if (run, head) not in per_layer or rel > per_layer[(run, head)][3]:
            per_layer[(run, head)] = (what, g, mx, rel)
    print(f"\nMTP heads: worst error per run and layer (head k = layer n_layer + k; the head-0 rows that read foreign "
          f"K/V against (B) only), llama.cpp {LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'run':8s} {'layer':>5s} {'component':44s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, head), (what, g, mx, rel) in sorted(per_layer.items()):
        print(f"{run:8s} {6 + head:5d} {what:44s} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
