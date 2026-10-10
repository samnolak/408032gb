"""Several ubatches, token-by-token decode and a rollback of llama.cpp's glm5-next graph against ref/glm5next.py.

G1 (tests/test_glm_tiny_parity.py) covers one 64-token prefill ubatch from an empty cache, op by op. Here the G1 tiny
model (tools/tinygen/glm5next_tiny.py --layers 12, seed 0: KDA layers, DSA layers 3/7/11, a dense lead layer, MoE)
runs one 64-token sequence through llama.cpp on the CPU in five ways (tools/tinygen/run_parity_glm5next_decode.sh).
tools/dump/dump_tensors.cpp with DUMP_PER_GRAPH=1 writes every computed graph (one per ubatch) into its own directory
and manifest.json maps the llama_process calls to those graphs:
  ub16      one call, -ub 16: 4 ubatches; their boundaries coincide with the k-pool boundaries (kpool 4)
  ub10      one call, -ub 10: 7 ubatches; pools straddle ubatch boundaries, the last ubatch has 4 tokens
  ub18      one call, -ub 18: 18 + 18 + 18 + 10; ubatches of more than 16 tokens after the first, straddling pools
  decode    prefill 48 tokens in one call, then 16 calls of one token each (n_tokens = 1 per llama_process)
  rollback  n_rs_seq = 3: prefill 48, a 4-token draft whose last two tokens are wrong, llama_memory_seq_rm of those
            two (a partial rollback from the snapshots), then the remaining 14 tokens one per call
Every graph is compared, row by row, with
  (A) the full-sequence oracle G.forward(final tokens) at the graph's positions (ref/dsa.py prefill selection, KDA from
      a zero state): an incremental result must equal the full recompute (causality); rows of the rejected draft
      tokens have no (A);
  (B) the incremental oracle G.forward_chunk replaying the same calls (rollback = G.cache_truncate), which carries the
      state llama.cpp carries (ref/glm5next.py "incremental path"). The KDA state a graph wrote is compared too
      (new_state-<il>, n_rs_seq = 0 only, src/models/delta-net-base.cpp:546-560); the conv history and the state a
      graph read have no named tensor in glm5-next.cpp, so they are checked through the rows that read them (the
      first d_conv-1 rows of kda_*_conv and the scan output of every graph with p0 > 0).
"G" = src/models/glm5-next.cpp, "M" = src/llama-memory-hybrid-idx.cpp at llama.cpp
ec7630a640789c393694fb194f1bbbf0369fc62d unless a line says 79e2e74 (the R1 pin of tools/engines/pins.json).
Env: GLM5NEXT_DECODE_DUMP (root holding one directory per run), GLM5NEXT_TINY_GGUF (model), LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT (the commit the dumps come from; default ec7630a).

Pin differences the test follows (PARITY_LLAMA_COMMIT):
  * ec7630a has a second DSA path for small ubatches: n_tokens <= 16 and n_kv > n_sel (G:324-331) gathers the
    selected latents (indexer_sel_idx, kv_gathered, kq_soft_max_gathered, kqv_gathered; G:883-886, 971-996) instead
    of the additive mask over all cells (kq_mask_dsa, G:889-918, 998-1005). 79e2e74 has only the mask path
    (G:990-1078 @79e2e74). The gather path's dead slots (pools not yet visible, missing tail cells) are masked by a
    host-side mask (M:875-883, 978-1011, 1013-1040) that is not dumped: the test derives it from the rule and checks
    it against the dumped softmax (exactly 0 on dead slots).
  * the padding rows of indexer_pool_k_new: ec7630a re-pools one shared cell (the first member of pool 0 if a pool
    is new, else the first token of the ubatch, M:947-967); 79e2e74 gives every padded entry its own cell, the
    smallest cells that are no complete pool's rep (M:888-905 @79e2e74). A padded entry has kpool equal members, so
    its pooled key is that cell's indexer key.
Tolerances as in N6: normwise relative error max|got - want| / max|want| < 1e-4 for f32 paths; exact for the
embedding gather; MoE ids equal as sets; DSA selection equal, or another valid top-k by the oracle's own scores
(G.tie_variant), counted and replayed (0 expected). The table printed at the end has the worst error per run and
component with the layer and graph where it occurred.
Layouts (checked by the tests): new_state is [S_v, S_v, H_v] with S[i][j] at [j*S_v + i]
(ggml/src/ggml-cpu/ops.cpp:11228-11229), so a dump reads as the oracle's state[h].T; hc_comb is [dst, src, T] with
dst fastest (ref/mhc.py); KV cell = position for this single sequence (the selection test would fail otherwise).
"""
import json
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref import fattn as F  # noqa: E402
from ref import glm5next as G  # noqa: E402
from ref.kda import gated_delta_rule  # noqa: E402
from ref.router import moe_tie_variant  # noqa: E402

DUMP = os.environ.get("GLM5NEXT_DECODE_DUMP")
GGUF = os.environ.get("GLM5NEXT_TINY_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
LLAMA_COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a640789c393694fb194f1bbbf0369fc62d")
EC7630A = LLAMA_COMMIT.startswith("ec7630a")
GATHER_MAX_UB = 16 if EC7630A else 0     # G:326-331: max_ub = 16; no gather path at 79e2e74
SHARED_PAD_CELL = EC7630A
TOL = 1e-4
SKIP = ("set GLM5NEXT_DECODE_DUMP, GLM5NEXT_TINY_GGUF, LLAMA_CPP_GGUF_PY: "
        "run tools/tinygen/run_parity_glm5next_decode.sh")

_MODEL = {}
TABLE = []        # (run, component, layer, graph, max abs, rel, argmax)


def model(path=None):
    """Weights and hyper-parameters of the GGUF at path (default GGUF), read once per path."""
    path = path or GGUF
    if path not in _MODEL:
        sys.path.append(GGUF_PY)
        try:
            from gguf import GGUFReader
        finally:
            sys.path.remove(GGUF_PY)
        r = GGUFReader(path)
        _MODEL[path] = ({t.name: np.array(t.data, dtype=np.float64) for t in r.tensors},
                        G.hparams({f.name: f for f in r.fields.values()}))
    return _MODEL[path]


class Run:
    """One dumped run: manifest, graphs with their positions."""

    def __init__(self, name, root=None):
        self.name = name
        self.dir = pathlib.Path(root or DUMP, name)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.fed = [int(x) for x in (self.dir / "tokens.txt").read_text().split(",")]
        ub = self.man["n_ubatch"]
        self.graphs, self.calls = [], self.man["calls"]
        seq, row = [], 0
        for c, call in enumerate(self.calls):
            if call["kind"] == "rollback":
                assert call["pos0"] <= len(seq)
                seq = seq[:call["pos0"]]
                for gr in self.graphs:               # rows at or after the cut were rejected
                    gr["kept"] = [k and p < call["pos0"] for k, p in zip(gr["kept"], gr["pos"])]
                continue
            assert call["pos0"] == len(seq), (name, c, call, len(seq))
            g0, g1 = call["graphs"]
            for j, g in enumerate(range(g0, g1)):
                n = min(ub, call["n"] - j * ub)
                t0 = call["tok0"] + j * ub
                p0 = call["pos0"] + j * ub
                self.graphs.append(dict(g=g, call=c, p0=p0, n=n, pos=list(range(p0, p0 + n)),
                                        toks=self.fed[t0:t0 + n], row0=row, kept=[True] * n))
                row += n
            seq = seq + self.fed[call["tok0"]:call["tok0"] + call["n"]]
        self.final = seq
        self.n_rows = row

    def dump(self, g, name, k=0):
        p = self.dir / f"g{g:03d}"
        j = json.loads((p / f"{name}__{k}.json").read_text())
        dt = np.float32 if j["type"] == "f32" else np.int32
        return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64), j["ne"]

    def has(self, g, name, k=0):
        return (self.dir / f"g{g:03d}" / f"{name}__{k}.json").exists()


def gather_live(t, kpool, n_top, n_sel):
    """Live slots of the ec7630a gather path for the token at position t (M:996-1001, 1017-1036): the first
    min(nv, n_top) pool slots (top_k sorted by descending score, G:861-868), then the tail slots k < (t+1) % kpool."""
    nv = (t + 1) // kpool
    live = np.zeros(n_sel, dtype=bool)
    live[:min(nv, n_top) * kpool] = True
    for k in range(n_sel - n_top * kpool):
        live[n_top * kpool + k] = k < (t + 1) % kpool
    return live


def mla_probs(q_abs, latent, cells, k_mla):
    """Attention probabilities (H, len(cells)) of one token over its cells (ref/dsa.py mla_attention)."""
    sc = np.einsum("hl,sl->hs", q_abs, latent[cells]) / np.sqrt(k_mla)
    p = np.exp(sc - sc.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


class _RunParity:
    RUN = None
    # what a subclass may change (tests/test_*_f16fa_parity.py): the dump root and the model, the precision the
    # oracles run at (ref/fattn.py precision; None = the f32 / flash-attention-off contract), the tolerance, the table
    ROOT, GGUF_PATH, PREC, TOL, TABLE, SKIP = DUMP, GGUF, None, TOL, TABLE, SKIP
    # tie replays: DSA pools tied within f32 rounding (G.tie_variant); MoE expert sets are never replayed here (None)
    TIE_RTOL, MOE_TIE_RTOL = 1e-6, None

    @classmethod
    def setUpClass(cls):
        if not (cls.ROOT and cls.GGUF_PATH and GGUF_PY):
            raise unittest.SkipTest(cls.SKIP)
        if not pathlib.Path(cls.ROOT, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {cls.ROOT}")
        cls.W, cls.hp = model(cls.GGUF_PATH)
        hp = cls.hp
        cls.R = Run(cls.RUN, cls.ROOT)
        cls.KDA = [i for i in range(hp["n_layer"]) if hp["kda"][i]]
        cls.DSA = [i for i in range(hp["n_layer"]) if not hp["kda"][i]]
        cls.MOE = list(range(hp["dense_lead"], hp["n_layer"]))
        kp = hp["kpool"]
        cls.n_top = hp["idx_top_k"] // kp
        cls.n_sel = kp * cls.n_top + (kp - 1 if hp["select_tail"] else 0)      # G:328-330
        # the cells each token attended to, from whichever DSA path the graph took
        cls.seen, cls.gather = {il: {} for il in cls.DSA}, {}
        for gr in cls.R.graphs:
            g, n = gr["g"], gr["n"]
            cls.gather[g] = cls.R.has(g, f"indexer_sel_idx-{cls.DSA[0]}")
            for il in cls.DSA:
                if cls.gather[g]:
                    idx = cls.R.dump(g, f"indexer_sel_idx-{il}")[0].reshape(n, -1).astype(int)
                    p = cls.R.dump(g, f"kq_soft_max_gathered-{il}")[0].reshape(n, -1, idx.shape[1])
                    for r in range(n):
                        cls.seen[il][(g, r)] = sorted(idx[r][(p[r] > 0).all(0)].tolist())
                else:
                    m = cls.R.dump(g, f"kq_mask_dsa-{il}")[0].reshape(n, -1)
                    for r in range(n):
                        cls.seen[il][(g, r)] = sorted(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist())
        # replay a reference choice among tied candidates (validated on the oracle's own scores), as N2/N6 do: a DSA
        # selection among pools tied within TIE_RTOL, and (MOE_TIE_RTOL not None) an expert set among experts tied
        # within MOE_TIE_RTOL; both choices must be valid top-k sets by the oracle's scores, else a bug in the oracle's
        # own choice would be replayed away. The first differing (graph, layer, kind) in computation order is replayed
        # and the oracles recomputed, until none is left; a difference that is no tie is kept in not_ties (reported by
        # test_dsa_pool_cache_and_selection / test_moe_and_ffn).
        cls.ties, cls.moe_ties, cls.not_ties, cls.ov_a, cls.ov_b = 0, 0, [], {}, {}
        cls._oracles(cls.ov_b, cls.ov_a)
        while cls._replay_first_tie():
            cls._oracles(cls.ov_b, cls.ov_a)

    @classmethod
    def _replay_first_tie(cls):
        hp, kp = cls.hp, cls.hp["kpool"]
        k = hp["n_expert_used"]
        for gr in cls.R.graphs:
            g = gr["g"]
            for il in range(hp["n_layer"]):
                kinds = ["dsa"] if il in cls.DSA else []
                if il in cls.MOE and cls.MOE_TIE_RTOL is not None:
                    kinds.append("moe")
                for kind in kinds:
                    rows, bad = [], []
                    for r, t in enumerate(gr["pos"]):
                        if kind == "dsa":
                            ob = cls.B[g]["layers"][il]["dsa"]
                            ref, own = cls.seen[il][(g, r)], ob["cells"][r]
                            if ref == own:
                                continue
                            args = (t, hp["idx_top_k"], kp, hp["select_tail"], cls.TIE_RTOL)
                            (ok, why), (ok2, why2) = G.tie_variant(ob["scores"][r], ref, *args), \
                                G.tie_variant(ob["scores"][r], own, *args)
                        else:
                            ob = cls.B[g]["layers"][il]["ffn"]
                            ids = cls.R.dump(g, f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k)
                            ref, own = sorted(ids[r].astype(int).tolist()), sorted(int(e) for e in ob["ids"][r])
                            if ref == own:
                                continue
                            (ok, why), (ok2, why2) = moe_tie_variant(ob["sel"][r], ref, k, cls.MOE_TIE_RTOL), \
                                moe_tie_variant(ob["sel"][r], own, k, cls.MOE_TIE_RTOL)
                        rows.append((r, t, ref))
                        if not (ok and ok2):
                            bad.append(f"{kind} L{il} pos {t} graph {g}: reference {why or 'valid'}, oracle "
                                       f"{why2 or 'valid'}; reference {ref}, oracle {own}")
                    if bad:
                        cls.not_ties += bad
                        return False
                    if not rows:
                        continue
                    for r, t, ref in rows:
                        cls.ov_b.setdefault((kind, il), {})[(g, r)] = ref
                        if gr["kept"][r]:
                            cls.ov_a.setdefault((kind, il), {})[t] = ref
                    if kind == "dsa":
                        cls.ties += len(rows)
                    else:
                        cls.moe_ties += len(rows)
                    return True
        return False

    @classmethod
    def _oracles(cls, overrides_b, overrides_a=None):
        W, hp = cls.W, cls.hp
        cls.A = G.forward(cls.R.final, W, hp, overrides_a or None, prec=cls.prec_full())
        cls.B, cache = {}, G.new_cache()
        by_call = {}
        for gr in cls.R.graphs:
            by_call.setdefault(gr["call"], []).append(gr)
        done = []
        for c, call in enumerate(cls.R.calls):
            if call["kind"] == "rollback":
                # the kept prefix is rebuilt with the replays of the graphs that computed it
                ov = {}
                for gr in done:
                    for key, v in overrides_b.items():
                        ov.setdefault(key, {}).update({gr["pos"][r]: x for (g, r), x in v.items() if g == gr["g"]})
                G.cache_truncate(cache, call["pos0"], W, hp, ov or None, prec=cls.PREC)
                continue
            for gr in by_call[c]:
                ov = {k: {gr["pos"][r]: cells for (g, r), cells in v.items() if g == gr["g"]}
                      for k, v in overrides_b.items()}
                cls.B[gr["g"]] = G.forward_chunk(gr["toks"], W, hp, cache, ov, prec=cls.prec_graph(gr))
                done.append(gr)

    @classmethod
    def graph_mode(cls, gr):
        """With PREC: the attention mode llama.cpp takes for the graph's DSA layers (ref/fattn.py mode_for, or the
        gather path)."""
        if cls.gather[gr["g"]]:          # ec7630a gather path: f32 mul_mat / soft_max over the cached latents
            return "exact"
        return F.mode_for(F.kv_f16(cls.PREC), cls.PREC.get("fa", False), gr["n"])

    @classmethod
    def prec_graph(cls, gr):
        """Oracle precision for the replay of graph gr (None without PREC)."""
        if not cls.PREC:
            return None
        return dict(cls.PREC, modes={p: cls.graph_mode(gr) for p in gr["pos"]})

    @classmethod
    def prec_full(cls):
        """Oracle precision for the full-sequence oracle (A): every position in the mode of the last graph that
        computed it (None without PREC)."""
        if not cls.PREC:
            return None
        return dict(cls.PREC, modes={p: cls.graph_mode(gr) for gr in cls.R.graphs for p in gr["pos"]})

    # ---- helpers ----
    def close(self, what, il, g, got, want, tol=None):
        tol = self.TOL if tol is None else tol
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} L{il} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        self.TABLE.append((self.RUN, what, il, g, mx, rel,
                           np.unravel_index(int(d.argmax()), d.shape) if d.size else ()))
        self.assertLess(rel, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, il, gr, got, a_full, b_rows, tol=None):
        """Compare a per-token tensor of graph gr with (A) the full-sequence rows (kept rows only) and (B)."""
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        kept = np.array(gr["kept"])
        if kept.any():
            idx = np.array(gr["pos"])[kept]
            self.close(f"{what} [full]", il, gr["g"], got[kept], np.asarray(a_full)[idx], tol)
        self.close(f"{what} [chunk]", il, gr["g"], got, b_rows, tol)

    def prefix_state(self, il, n):
        K = self.A["layers"][il]["kda"]
        if n == 0:
            H, D = self.hp["n_head"], self.hp["kda_head_dim"]
            return np.zeros((H, D, D))
        return gated_delta_rule(K["q"][:n], K["k"][:n], K["v"][:n], K["g"][:n], K["beta"][:n], normalize_qk=True,
                                eps=G.L2_EPS)[1]

    # ---- tests ----
    def test_graphs_and_token_mapping(self):
        """One graph per ubatch, in order; each graph's embedding rows are exactly its tokens (proves the mapping
        of graphs to positions that every other test relies on)."""
        R, hp = self.R, self.hp
        ub = R.man["n_ubatch"]
        self.assertEqual(len(R.man["graphs"]), len(R.graphs))
        for c, call in enumerate(R.calls):
            if call["kind"] == "decode":
                g0, g1 = call["graphs"]
                self.assertEqual(g1 - g0, -(-call["n"] // ub), f"call {c}")
        self.assertEqual([gr["g"] for gr in R.graphs], list(range(len(R.graphs))))
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32)
        self.assertEqual(lo.size, R.n_rows * self.W["output.weight"].shape[0])
        for gr in R.graphs:
            got, ne = R.dump(gr["g"], "hc_init")
            self.assertEqual(ne[2], gr["n"])
            want = G.hc_init(self.W["token_embd.weight"][gr["toks"]], hp["hc"])
            self.close("hc_init (exact)", -1, gr["g"], got, want, tol=1e-300)

    def test_mhc(self):
        """mHC per layer: post weights, Sinkhorn comb, mixer inputs, new streams (attention and FFN sides)."""
        R, hc = self.R, self.hp["hc"]
        for gr in R.graphs:
            g, b, n = gr["g"], self.B[gr["g"]], gr["n"]
            for il in range(self.hp["n_layer"]):
                La, Lb = self.A["layers"][il], b["layers"][il]
                for k, side in ((0, "attn"), (1, "ffn")):
                    self.both(f"hc_post ({side})", il, gr, R.dump(g, f"hc_post-{il}", k)[0], La[f"{side}_post_w"],
                              Lb[f"{side}_post_w"])
                    comb = R.dump(g, f"hc_comb-{il}", k)[0].reshape(n, hc, hc).transpose(0, 2, 1)
                    self.both(f"hc_comb ({side}, Sinkhorn)", il, gr, comb, La[f"{side}_comb"], Lb[f"{side}_comb"])
                self.both("hc_attn_pre (mixer input)", il, gr, R.dump(g, f"hc_attn_pre-{il}")[0], La["attn_pre"],
                          Lb["attn_pre"])
                self.both("hc_attn_post", il, gr, R.dump(g, f"hc_attn_post-{il}")[0], La["hc_attn_post"],
                          Lb["hc_attn_post"])
                self.both("hc_ffn_pre (mixer input)", il, gr, R.dump(g, f"hc_ffn_pre-{il}")[0], La["ffn_pre"],
                          Lb["ffn_pre"])
                self.both("l_out (layer output streams)", il, gr, R.dump(g, f"l_out-{il}")[0], La["l_out"],
                          Lb["l_out"])

    def test_kda_layers(self):
        """KDA: short convs over the carried conv history, gate, beta, scan from the carried state, output gate."""
        R, hp = self.R, self.hp
        K1 = hp["d_conv"] - 1
        hist_rows = 0
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.KDA:
                Ka, Kb = self.A["layers"][il]["kda"], b["layers"][il]["kda"]
                self.both("attn_norm (KDA input)", il, gr, R.dump(g, f"attn_norm-{il}")[0],
                          self.A["layers"][il]["attn_norm"], b["layers"][il]["attn_norm"])
                for c in "qkv":
                    got = R.dump(g, f"kda_{c}_conv-{il}")[0]
                    self.both(f"kda_{c}_conv", il, gr, got, Ka[c], Kb[c])
                    if gr["p0"] > 0:     # the rows whose window reaches into the history carried from earlier graphs
                        m = min(K1, gr["n"])
                        sub = dict(gr, pos=gr["pos"][:m], kept=gr["kept"][:m])
                        self.both(f"kda_{c}_conv (rows reading the conv history)", il, sub,
                                  got.reshape(np.shape(Kb[c]))[:m], Ka[c], Kb[c][:m])
                        hist_rows += 1
                self.both("kda_g1 (decay gate)", il, gr, R.dump(g, f"kda_g1-{il}")[0], Ka["g"], Kb["g"])
                self.both("kda_beta", il, gr, R.dump(g, f"kda_beta-{il}")[0], Ka["beta"], Kb["beta"])
                self.both("kda_scan_out", il, gr, R.dump(g, f"kda_scan_out-{il}")[0], Ka["scan"], Kb["scan"])
                if gr["p0"] > 0:         # the first row reads the carried state
                    sub = dict(gr, pos=gr["pos"][:1], kept=gr["kept"][:1])
                    got = R.dump(g, f"kda_scan_out-{il}")[0].reshape(np.shape(Kb["scan"]))[:1]
                    self.both("kda_scan_out (first row, reads the carried state)", il, sub, got, Ka["scan"],
                              Kb["scan"][:1])
                # kda_normed = rms_norm(scan per head) * ssm_norm rescales every head row to O(1): a row whose RMS
                # is far below the tensor's maximum carries the upstream f32 error amplified by that ratio
                # (measured at ec7630a: ub18 L10 pos 15 head 3, scan RMS 1.6e-3 vs max |scan| 0.50, row error 3.0e-4
                # relative -> 9.7e-5 normwise after the norm). So the norm is checked as an op on the reference's
                # own scan (G:751-754), and end to end through kda_out, whose projection does not rescale rows.
                scan = R.dump(g, f"kda_scan_out-{il}")[0].reshape(np.shape(Kb["scan"]))
                want = G.rms_norm(scan, hp["rms_eps"], self.W[f"blk.{il}.ssm_norm.weight"])
                self.close("kda_normed (op on the dumped scan)", il, g, R.dump(g, f"kda_normed-{il}")[0], want)
                self.both("kda_out", il, gr, R.dump(g, f"kda_out-{il}")[0], Ka["out"], Kb["out"])
        if len(R.graphs) > 1:
            self.assertGreater(hist_rows, 0)

    def test_kda_state_written_back(self):
        """new_state (n_rs_seq = 0): the KDA state a graph writes for the next one, (B) the incremental oracle's
        carry and (A) the prefix state of the full sequence; with n_rs_seq > 0 the graph writes snapshots instead."""
        R = self.R
        n_checked = 0
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.KDA:
                if R.man["n_rs_seq"] > 0:
                    self.assertFalse(R.has(g, f"new_state-{il}"), "n_rs_seq > 0 writes snapshots, not new_state")
                    continue
                Kb = b["layers"][il]["kda"]
                s = R.dump(g, f"new_state-{il}")[0].reshape(Kb["state"].shape).transpose(0, 2, 1)
                self.close("new_state (KDA state written back) [chunk]", il, g, s, Kb["state"])
                if all(gr["kept"]):
                    self.close("new_state (KDA state written back) [full]", il, g, s,
                               self.prefix_state(il, gr["p0"] + gr["n"]))
                n_checked += 1
        if R.man["n_rs_seq"] == 0:
            self.assertGreater(n_checked, 0)

    def test_dsa_projections(self):
        """Per-token DSA inputs: q_resid, the latent written to the MLA cache, absorbed q, indexer key | gate written
        to the indexer cache, indexer query and head weights."""
        R = self.R
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.DSA:
                Da, Db = self.A["layers"][il]["dsa"], b["layers"][il]["dsa"]
                for name, key in (("q_resid", "q_resid"), ("kv_cmpr (MLA latent)", "latent"),
                                  ("q_absorbed", "q_abs"), ("indexer_k (layer-normed key)", "ik"),
                                  ("indexer_gate", "ig"), ("indexer_q", "iq"), ("indexer_weights", "wts")):
                    self.both(name, il, gr, R.dump(g, f"{name.split()[0]}-{il}")[0], Da[key], Db[key])

    def test_dsa_pool_cache_and_selection(self):
        """k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph
        (indexer_pool_k_new, padded to a stable count), every complete pool is read back from the cache
        (indexer_pool_k), scores and the selection use them; the attention runs over the selected cells of the MLA
        cache on the path the pin takes for the ubatch size."""
        R, hp = self.R, self.hp
        kp, d = hp["kpool"], hp["idx_dim"]
        self.assertEqual(self.not_ties, [], "DSA selections (or MoE expert sets) that are no valid top-k by the "
                                            "oracle's scores")
        pooled_in_decode = straddle = 0
        for gr in R.graphs:
            g, p0, n = gr["g"], gr["p0"], gr["n"]
            b = self.B[g]
            self.assertEqual(self.gather[g], n <= GATHER_MAX_UB, f"g{g}: DSA path for {n} tokens at {LLAMA_COMMIT}")
            for il in self.DSA:
                Db, Da = b["layers"][il]["dsa"], self.A["layers"][il]["dsa"]
                b0, b1 = Db["pool_range"]
                self.assertEqual((b0, b1), (p0 // kp, (p0 + n) // kp))
                new = R.dump(g, f"indexer_pool_k_new-{il}")[0].reshape(-1, d)
                # n_new_g = max(n_new, 1, min(n_tokens/kpool + 1, ...)) (M:765-770)
                self.assertEqual(new.shape[0], max(b1 - b0, 1, n // kp + 1))
                if b1 > b0:
                    self.close("indexer_pool_k_new (pools completed here) [chunk]", il, g, new[:b1 - b0],
                               Db["pooled_new"])
                    if all(gr["kept"][:(b1 * kp) - p0]):
                        # before the store: pooled_raw (equal to pooled unless the cache is f16)
                        self.close("indexer_pool_k_new (pools completed here) [full]", il, g, new[:b1 - b0],
                                   Da.get("pooled_raw", Da["pooled"])[b0:b1])
                    pooled_in_decode += n == 1
                    straddle += b0 * kp < p0
                n_pad = new.shape[0] - (b1 - b0)
                # KV cell = position: cells before the graph from (A), the graph's own cells from (B)
                ik_all = F.store(np.concatenate([np.asarray(Da["ik"])[:p0], np.asarray(Db["ik"])]), self.PREC)
                if SHARED_PAD_CELL:
                    cells = [0 if b1 > b0 else p0] * n_pad
                else:
                    reps = {kp * j + kp - 1 for j in range(b1)}
                    cells = [c for c in range(p0 + n) if c not in reps][:n_pad]
                    self.assertEqual(len(cells), n_pad)
                    self.assertFalse(set(cells) & reps, f"padded cells {cells} include a rep, g{g}")
                    self.assertEqual(len(set(cells)), n_pad, f"padded cells {cells} not distinct, g{g}")
                for r, c in zip(range(b1 - b0, new.shape[0]), cells):
                    self.close("indexer_pool_k_new (padding rows)", il, g, new[r], ik_all[c])
                got_k = R.dump(g, f"indexer_pool_k-{il}")[0].reshape(-1, d)        # padded to 64 pools
                self.assertGreaterEqual(got_k.shape[0], b1 + 1)
                if b1:
                    self.close("indexer_pool_k (pool cache, all complete pools) [chunk]", il, g, got_k[:b1],
                               Db["pooled"])
                    if all(gr["kept"]):
                        self.close("indexer_pool_k (pool cache, all complete pools) [full]", il, g, got_k[:b1],
                                   Da["pooled"][:b1])
                sc = R.dump(g, f"indexer_score-{il}")[0].reshape(n, -1)
                fin = np.isfinite(Db["scores"])
                self.assertTrue(np.array_equal(np.isfinite(sc[:, :b1]) & (sc[:, :b1] > -1e30), fin),
                                f"pool visibility g{g}")
                self.assertFalse((np.isfinite(sc[:, b1:]) & (sc[:, b1:] > -1e30)).any(),
                                 f"padded pools must stay masked g{g}")
                self.close("indexer_score (visible) [chunk]", il, g, sc[:, :b1][fin], Db["scores"][fin])
                for r, t in enumerate(gr["pos"]):
                    self.assertEqual(self.seen[il][(g, r)], Db["cells"][r], f"DSA cells L{il} pos {t} g{g}")
                    if gr["kept"][r]:
                        self.assertEqual(self.seen[il][(g, r)], Da["cells"][t], f"DSA cells (full) L{il} pos {t}")
                if self.gather[g]:
                    self._gather_path(gr, il, Da, Db)
                self.both("kqv_out (sparse MLA)", il, gr, R.dump(g, f"kqv_out-{il}")[0], Da["kqv"], Db["kqv"])
                self.both("attn_out (wo)", il, gr, R.dump(g, f"attn_out-{il}")[0], Da["out"], Db["out"])
        self.TABLE.append((self.RUN, f"dsa selection: tie replays = {self.ties}", self.DSA[0], -1, 0.0, 0.0, ()))
        if self.RUN in ("decode", "rollback"):
            self.assertGreater(pooled_in_decode, 0, "no pool completed by a one-token call")
        if self.RUN in ("ub10", "ub18", "decode", "rollback"):
            self.assertGreater(straddle, 0, "no pool straddles two graphs")
        # the budget binds: some token sees more pools than it may select
        last = R.graphs[-1]
        self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(self.A["layers"][self.DSA[0]]["dsa"]["cells"])
                            if t >= last["p0"]))

    def _gather_path(self, gr, il, Da, Db):
        """ec7630a small-ubatch path: the live slots follow the host-side mask rule, the gathered latents are the
        cached latents of the selected cells, the softmax over them matches the oracle's attention, dead slots 0."""
        R, hp, g, n = self.R, self.hp, gr["g"], gr["n"]
        idx = R.dump(g, f"indexer_sel_idx-{il}")[0].reshape(n, -1).astype(int)
        self.assertEqual(idx.shape[1], self.n_sel)
        p = R.dump(g, f"kq_soft_max_gathered-{il}")[0].reshape(n, hp["n_head"], self.n_sel)
        kv = R.dump(g, f"kv_gathered-{il}")[0].reshape(n, self.n_sel, -1)
        # cell = position; the latents as the MLA cache holds them (f16 with an f16 cache)
        lat_b = F.store(np.concatenate([np.asarray(Da["latent"])[:gr["p0"]], np.asarray(Db["latent"])]), self.PREC)
        for r, t in enumerate(gr["pos"]):
            live = gather_live(t, hp["kpool"], self.n_top, self.n_sel)
            self.assertTrue(np.array_equal((p[r] > 0).all(0), live), f"live slots L{il} pos {t} g{g}")
            self.assertFalse(p[r][:, ~live].any(), f"dead slots must have probability 0, L{il} pos {t} g{g}")
            cells = idx[r][live]
            self.close("kv_gathered (live slots = cached latents)", il, g, kv[r][live], lat_b[cells])
            want = mla_probs(Db["q_abs"][r], lat_b, cells, hp["k_mla"])
            self.close("kq_soft_max_gathered (live slots)", il, g, p[r][:, live], want)

    def test_moe_and_ffn(self):
        R, hp = self.R, self.hp
        k = hp["n_expert_used"]
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            kept = np.array(gr["kept"])
            for il in range(hp["n_layer"]):
                Fa, Fb = self.A["layers"][il]["ffn"], b["layers"][il]["ffn"]
                self.both("ffn_norm (FFN input)", il, gr, R.dump(g, f"ffn_norm-{il}")[0],
                          self.A["layers"][il]["ffn_norm"], b["layers"][il]["ffn_norm"])
                if il in self.MOE:
                    ids = R.dump(g, f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k).astype(int)
                    self.assertTrue(np.array_equal(np.sort(ids, 1), np.sort(Fb["ids"], 1)), f"top-k L{il} g{g}")
                    if kept.any():
                        want = np.sort(Fa["ids"][np.array(gr["pos"])[kept]], 1)
                        self.assertTrue(np.array_equal(np.sort(ids, 1)[kept], want), f"top-k (full) L{il} g{g}")
                    wd = R.dump(g, f"ffn_moe_weights_scaled-{il}")[0].reshape(gr["n"], k)
                    # weights in the reference's id order, matched by expert id
                    wb = np.array([[dict(zip(Fb["ids"][r], Fb["weights"][r]))[e] for e in ids[r]]
                                   for r in range(gr["n"])])
                    self.close("ffn_moe_weights_scaled [chunk]", il, g, wd, wb)
                    self.both("ffn_moe_out (routed)", il, gr, R.dump(g, f"ffn_moe_out-{il}")[0], Fa["moe_out"],
                              Fb["moe_out"])
                    self.both("ffn_shexp (shared, clamped)", il, gr, R.dump(g, f"ffn_shexp-{il}")[0], Fa["shexp"],
                              Fb["shexp"])
                kind = "routed + shared" if il in self.MOE else "dense, clamped"
                self.both(f"ffn_out ({kind})", il, gr, R.dump(g, f"ffn_out-{il}")[0], Fa["out"], Fb["out"])
        if self.MOE_TIE_RTOL is not None:
            self.TABLE.append((self.RUN, f"moe experts: tie replays = {self.moe_ties}", self.MOE[0], -1, 0.0, 0.0,
                               ()))

    def test_head_and_logits(self):
        R = self.R
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab).astype(np.float64)
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            self.both("hc_head (stream mean)", -1, gr, R.dump(g, "hc_head")[0], self.A["hc_head"], b["hc_head"])
            self.both("result_norm", -1, gr, R.dump(g, "result_norm")[0], self.A["result_norm"], b["result_norm"])
            self.both("result_output", -1, gr, R.dump(g, "result_output")[0], self.A["logits"], b["logits"])
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            self.both("logits.bin", -1, gr, rows, self.A["logits"], b["logits"])
            self.assertTrue(np.array_equal(rows.argmax(1), b["logits"].argmax(1)), f"greedy tokens g{g}")

    def test_run_exercises_its_mechanisms(self):
        R = self.R
        self.assertEqual(R.final, self.final_tokens())
        sizes = [gr["n"] for gr in R.graphs]
        if self.RUN == "ub16":
            self.assertEqual(sizes, [16] * 4)
        if self.RUN == "ub10":
            self.assertEqual(sizes, [10] * 6 + [4])
        if self.RUN == "ub18":
            self.assertEqual(sizes, [18] * 3 + [10])
        if self.RUN == "decode":
            self.assertEqual(sizes, [48] + [1] * 16)
        paths = [self.gather[gr["g"]] for gr in R.graphs]
        if not EC7630A:
            self.assertFalse(any(paths), "79e2e74 has no gather path")
        elif self.RUN == "ub18":     # the mask path continuing a cache (p0 > 0), then the gather path
            self.assertEqual(paths, [False, False, False, True])
        elif self.RUN == "decode":   # the mask path for the prefill, the gather path for every decode step
            self.assertEqual(paths, [False] + [True] * 16)
        if self.RUN == "rollback":
            self.assertEqual(R.man["n_rs_seq"], 3)
            kinds = [c["kind"] for c in R.calls]
            self.assertIn("rollback", kinds)
            rb = R.calls[kinds.index("rollback")]
            draft = R.calls[kinds.index("rollback") - 1]
            self.assertEqual(rb["pos0"], draft["pos0"] + draft["n"] - rb["n"])
            rejected = R.fed[draft["tok0"] + draft["n"] - rb["n"]:draft["tok0"] + draft["n"]]
            self.assertNotEqual(rejected, R.final[rb["pos0"]:rb["pos0"] + rb["n"]], "the rejected tokens are right")
            after = [gr for gr in R.graphs if gr["call"] > kinds.index("rollback")]
            self.assertEqual(after[0]["p0"], rb["pos0"])
            # the draft completed pool 12 (positions 48..51) with the wrong tokens; it is re-pooled after the rollback
            kp = self.hp["kpool"]
            redo = [gr for gr in after if (gr["p0"] + gr["n"]) // kp > gr["p0"] // kp and gr["p0"] < draft["pos0"] + draft["n"]]
            self.assertTrue(redo, "the pool the rejected tokens completed is never re-pooled")

    def final_tokens(self):
        """The 64-token sequence every run must end with (tools/tinygen/run_parity_glm5next_decode.sh)."""
        return [int(x) for x in pathlib.Path(self.ROOT, "tokens.txt").read_text().split(",")]


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Ubatch16(_RunParity, unittest.TestCase):
    RUN = "ub16"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Ubatch10(_RunParity, unittest.TestCase):
    RUN = "ub10"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Ubatch18(_RunParity, unittest.TestCase):
    RUN = "ub18"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Decode(_RunParity, unittest.TestCase):
    RUN = "decode"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Rollback(_RunParity, unittest.TestCase):
    RUN = "rollback"


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, mx, rel, arg in TABLE:
        key = (run, what)
        if key not in worst or rel > worst[key][3]:
            worst[key] = (il, g, mx, rel, arg)
    print(f"\nworst error per run and component over layers and graphs (rel = max|got-want| / max|want|), "
          f"llama.cpp {LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'run':8s} {'component':58s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, what), (il, g, mx, rel, arg) in worst.items():
        print(f"{run:8s} {what:58s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)
    per_layer = {}
    for run, what, il, g, mx, rel, arg in TABLE:
        if il not in per_layer or rel > per_layer[il][4]:
            per_layer[il] = (run, what, g, mx, rel)
    print(f"\nworst error per layer over runs, components and graphs (layer -1 = embedding and head), llama.cpp "
          f"{LLAMA_COMMIT[:7]}:", file=sys.stderr)
    print(f"{'layer':>5s} {'run':8s} {'component':58s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for il in sorted(per_layer):
        run, what, g, mx, rel = per_layer[il]
        print(f"{il:5d} {run:8s} {what:58s} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
