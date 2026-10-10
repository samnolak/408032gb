"""Several ubatches, token-by-token decode and a rollback of llama.cpp's qwen4exp graph against ref/qwen4exp.py.

N2 (tests/test_qwen4exp_tiny_parity.py) covers one 64-token prefill ubatch from an empty cache. Here the N2 tiny
model (tools/tinygen/qwen4exp_tiny.py, seed 0) runs one 64-token sequence (6 EOS) through llama.cpp @ec7630a on the
CPU in four ways (tools/tinygen/run_parity_qwen4exp_decode.sh). tools/dump/dump_tensors.cpp with DUMP_PER_GRAPH=1
writes every computed graph (one per ubatch) into its own directory and manifest.json maps the llama_process calls to
those graphs:
  ub16      one call, -ub 16: 4 ubatches; their boundaries coincide with the k-pool boundaries (compress ratio 4)
  ub10      one call, -ub 10: 7 ubatches; pools straddle ubatch boundaries, the last ubatch has 4 tokens
  decode    prefill 48 tokens in one call, then 16 calls of one token each (n_tokens = 1 per llama_process)
  rollback  n_rs_seq = 3: prefill 48, a 4-token draft whose last two tokens are wrong, llama_memory_seq_rm of those
            two (a partial rollback from the snapshots, src/llama-memory-recurrent.cpp:199-215), then the remaining
            14 tokens one per call
Every graph is compared, row by row, with
  (A) the full-sequence oracle Q.forward(final tokens) at the graph's positions: an incremental result must equal the
      full recompute (causality); rows of the rejected draft tokens have no (A);
  (B) the incremental oracle Q.forward_chunk replaying the same calls (rollback = Q.cache_truncate), which carries
      the state llama.cpp carries (ref/qwen4exp.py "incremental path"). The state a graph read is compared too:
      conv_state_at-<il> (GDN conv history; on the PLE layer occurrence 0 is the PLE conv history, G:1423 runs
      before the mixer), state_predelta-<il> (delta-rule state, G:1073-1074), new_state-<il> (n_rs_seq = 0 only,
      src/models/delta-net-base.cpp:546-560) and indexer_k-3 (pooled keys gathered from the k-pool cache, G:797).
G = src/models/qwen4exp.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d.
Env: QWEN4EXP_DECODE_DUMP (root holding one directory per run), QWEN4EXP_TINY_GGUF (model), LLAMA_CPP_GGUF_PY.

Tolerances as in N2: normwise relative error max|got - want| / max|want| < 1e-4 for f32 paths; exact (0) for the
embedding and PLE gathers (the host-side hash, its predecessors read from the KV cells of earlier ubatches/calls) and
for the zero state at the sequence start; MoE ids equal as sets; QSA selection equal, or another valid top-k by the
oracle's own scores (Q.qsa_tie_variant), counted and replayed like N2 (0 expected). The table printed at the end has
the worst error per run and component with the layer and graph where it occurred.

Layouts (checked by the tests): conv_state_at is [K-1, C] with time fastest (rows of the conv input), the delta-rule
state is [S_v, S_v, H_v] with S[i][j] at [j*S_v + i] (ggml/src/ggml-cpu/ops.cpp:11228-11229), so a dump reads as the
oracle's state[h].T; KV cell = position for this single sequence (the selection test would fail otherwise).
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
from tests.test_qwen4exp_tiny_parity import read_hparams  # noqa: E402

DUMP = os.environ.get("QWEN4EXP_DECODE_DUMP")
GGUF = os.environ.get("QWEN4EXP_TINY_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
# llama.cpp commit the dumps come from (set by tools/tinygen/run_parity_qwen4exp_decode.sh); only the padding rows of
# indexer_pool_k_new depend on it, see the check below
LLAMA_COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a640789c393694fb194f1bbbf0369fc62d")
SHARED_PAD_CELL = LLAMA_COMMIT.startswith("ec7630a")
TOL = 1e-4
SKIP = ("set QWEN4EXP_DECODE_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: "
        "run tools/tinygen/run_parity_qwen4exp_decode.sh")

_MODEL = {}
TABLE = []        # (run, component, layer, graph, max abs, rel, argmax)


def model():
    if not _MODEL:
        sys.path.append(GGUF_PY)
        try:
            from gguf import GGUFReader
        finally:
            sys.path.remove(GGUF_PY)
        r = GGUFReader(GGUF)
        _MODEL["W"] = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
        _MODEL["hp"] = read_hparams({f.name: f for f in r.fields.values()})
    return _MODEL["W"], _MODEL["hp"]


class Run:
    """One dumped run: manifest, graphs with their positions, the replayed oracles."""

    def __init__(self, name):
        self.name = name
        self.dir = pathlib.Path(DUMP, name)
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


class _RunParity:
    RUN = None

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")
        cls.W, cls.hp = model()
        hp = cls.hp
        cls.R = Run(cls.RUN)
        cls.LIN = [i for i in range(hp["n_layer"]) if hp["recurrent"][i]]
        cls.ATT = [i for i in range(hp["n_layer"]) if not hp["recurrent"][i]]
        cls.QSA = [i for i in cls.ATT if hp["compress"][i] > 0]
        cls.seen = {il: {} for il in cls.QSA}         # (graph, row) -> cells the reference attended to
        for gr in cls.R.graphs:
            for il in cls.QSA:
                m, ne = cls.R.dump(gr["g"], f"kq_mask_qsa-{il}")
                m = m.reshape(gr["n"], -1)
                for r in range(gr["n"]):
                    cls.seen[il][(gr["g"], r)] = sorted(np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0].tolist())
        cls._oracles({})
        # replay a reference choice among exactly tied pools (validated on the oracle's own scores), as N2 does;
        # both selections must be valid top-k sets for the token's absolute position, else a bug in the oracle's
        # own selection (e.g. its tail) would be replayed away; any other difference is kept for
        # test_qsa_pool_cache_and_selection to report
        cls.ties, cls.not_ties, overrides_a, overrides_b = 0, [], {}, {}
        for gr in cls.R.graphs:
            for il in cls.QSA:
                ob = cls.B[gr["g"]]["layers"][il]["attn"]
                for r, t in enumerate(gr["pos"]):
                    s = cls.seen[il][(gr["g"], r)]
                    if s == ob["cells"][r]:
                        continue
                    ok, why = Q.qsa_tie_variant(ob["scores"][r], s, t, hp["idx_top_k"], hp["kpool"])
                    own, why_own = Q.qsa_tie_variant(ob["scores"][r], ob["cells"][r], t, hp["idx_top_k"], hp["kpool"])
                    if not (ok and own):
                        cls.not_ties.append(f"L{il} pos {t} graph {gr['g']}: reference {why or 'valid'}, oracle "
                                            f"{why_own or 'valid'}; reference {s}, oracle {ob['cells'][r]}")
                        continue
                    cls.ties += 1
                    overrides_b.setdefault(("qsa", il), {})[(gr["g"], r)] = s
                    if gr["kept"][r]:
                        overrides_a.setdefault(("qsa", il), {})[t] = s
        if cls.ties:
            cls._oracles(overrides_b, overrides_a)

    @classmethod
    def _oracles(cls, overrides_b, overrides_a=None):
        W, hp = cls.W, cls.hp
        ov_a = None
        if overrides_a:
            base = Q.forward(cls.R.final, W, hp)
            ov_a = {k: [v.get(t, c) for t, c in enumerate(base["layers"][k[1]]["attn"]["cells"])]
                    for k, v in overrides_a.items()}
        cls.A = Q.forward(cls.R.final, W, hp, ov_a)
        cls.B, cache = {}, Q.new_cache()
        by_call = {}
        for gr in cls.R.graphs:
            by_call.setdefault(gr["call"], []).append(gr)
        for c, call in enumerate(cls.R.calls):
            if call["kind"] == "rollback":
                Q.cache_truncate(cache, call["pos0"], W, hp)
                continue
            for gr in by_call[c]:
                ov = {k: {gr["pos"][r]: cells for (g, r), cells in v.items() if g == gr["g"]}
                      for k, v in overrides_b.items()}
                cls.B[gr["g"]] = Q.forward_chunk(gr["toks"], W, hp, cache, ov)

    # ---- helpers ----
    def close(self, what, il, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} L{il} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        TABLE.append((self.RUN, what, il, g, mx, rel, np.unravel_index(int(d.argmax()), d.shape) if d.size else ()))
        self.assertLess(rel, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def both(self, what, il, gr, got, a_full, b_rows, tol=TOL):
        """Compare a per-token tensor of graph gr with (A) the full-sequence rows (kept rows only) and (B)."""
        got = np.asarray(got, dtype=np.float64).reshape(np.shape(b_rows))
        kept = np.array(gr["kept"])
        if kept.any():
            idx = np.array(gr["pos"])[kept]
            self.close(f"{what} [full]", il, gr["g"], got[kept], np.asarray(a_full)[idx], tol)
        self.close(f"{what} [chunk]", il, gr["g"], got, b_rows, tol)

    def prefix_state(self, il, n):
        G = self.A["layers"][il]["gdn"]
        if n == 0:
            S = self.hp["ssm_state"]
            return np.zeros((self.hp["ssm_v_heads"], S, S))
        return gated_delta_rule(G["q"][:n], G["k"][:n], G["v"][:n], G["g"][:n], G["beta"][:n],
                                normalize_qk=True, eps=self.hp["eps"])[1]

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
            emb = self.W["token_embd.weight"][gr["toks"]]
            got, ne = R.dump(gr["g"], "hc_init")
            self.assertEqual(ne[2], gr["n"])
            self.close("hc_init (exact)", -1, gr["g"], got, Q.hc_init(emb, hp["hc"]), tol=1e-300)

    def test_ple_rows_across_calls(self):
        """PLE rows (host-side hash) of every graph, exact: predecessors of the first tokens of a ubatch or a decode
        step come from the KV cells of earlier ubatches/calls (EOS reset included)."""
        R, hp = self.R, self.hp
        cross_eos = 0
        for gr in R.graphs:
            got, _ = R.dump(gr["g"], "ple_embd")
            b = self.B[gr["g"]]
            self.both("ple_embd = table[rows] (exact)", -1, gr, got, self.A["ple_embd"], b["ple_embd"], tol=1e-300)
            self.assertTrue(np.array_equal(b["ple_rows"][np.array(gr["kept"])],
                                           self.A["ple_rows"][np.array(gr["pos"])[np.array(gr["kept"])]]))
            hist = R.final[max(0, gr["p0"] - hp["ple_ngram"] + 1):gr["p0"]]
            cross_eos += gr["p0"] > 0 and hp["ple_eos"] in hist
        self.assertGreater(cross_eos, 0, "no ubatch/step reads an EOS predecessor from an earlier graph")

    def test_carried_state(self):
        """The state each graph read: GDN conv history, delta-rule state, PLE conv history; and the state it wrote
        (new_state). (A) from the full-sequence oracle's rows before the graph, (B) the incremental oracle's carry."""
        R, hp = self.R, self.hp
        ple = hp["ple_layers"]
        nonzero = 0
        for gr in R.graphs:
            g, p0, n = gr["g"], gr["p0"], gr["n"]
            b = self.B[g]
            for il in self.LIN:
                Lb, La = b["layers"][il]["gdn"], self.A["layers"][il]["gdn"]
                a, ne = R.dump(g, f"conv_state_at-{il}", 1 if il in ple else 0)
                got = a.reshape(ne[1], ne[0]).T                         # [K-1, C], time fastest in ggml
                K1 = got.shape[0]
                want_a = np.vstack([np.zeros((K1, La["qkv"].shape[1])), La["qkv"][:p0]])[-K1:]
                s, _ = R.dump(g, f"state_predelta-{il}")
                s = s.reshape(Lb["state_in"].shape).transpose(0, 2, 1)  # ggml keeps S[i][j] at [j*S_v + i]
                if p0 == 0:
                    self.assertFalse(got.any() or s.any(), f"L{il}: non-zero state at the sequence start")
                    continue
                nonzero += 1
                self.close("conv_state_at (GDN conv history) [full]", il, g, got, want_a)
                self.close("conv_state_at (GDN conv history) [chunk]", il, g, got, Lb["conv_in"])
                self.close("state_predelta (delta-rule state) [full]", il, g, s, self.prefix_state(il, p0))
                self.close("state_predelta (delta-rule state) [chunk]", il, g, s, Lb["state_in"])
                if R.man["n_rs_seq"] == 0:
                    ns, _ = R.dump(g, f"new_state-{il}")
                    ns = ns.reshape(Lb["state"].shape).transpose(0, 2, 1)
                    self.close("new_state (written back) [chunk]", il, g, ns, Lb["state"])
                    if all(gr["kept"]):
                        self.close("new_state (written back) [full]", il, g, ns, self.prefix_state(il, p0 + n))
                else:
                    self.assertFalse(R.has(g, f"new_state-{il}"), "n_rs_seq > 0 writes snapshots, not new_state")
            for il in ple:
                Pb, Pa = b["layers"][il]["ple"], self.A["layers"][il]["ple"]
                a, ne = R.dump(g, f"conv_state_at-{il}", 0)
                got = a.reshape(ne[1], ne[0]).T
                if p0 == 0:
                    self.assertFalse(got.any(), f"L{il}: non-zero PLE history at the sequence start")
                    continue
                w = Q.layer_weights(self.W, il)
                normed = Q.grouped_rms_norm(Pa["gated"], w["ple_norm_conv.weight"], hp["eps"]).reshape(
                    len(R.final), -1)
                Kh = got.shape[0]
                want_a = np.vstack([np.zeros((Kh, normed.shape[1])), normed[:p0]])[-Kh:]
                self.close("conv_state_at (PLE conv history) [full]", il, g, got, want_a)
                self.close("conv_state_at (PLE conv history) [chunk]", il, g, got, Pb["hist_in"])
        self.assertGreater(nonzero, 0)

    def test_gdn_layers(self):
        R = self.R
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.LIN:
                Lb, La = b["layers"][il]["gdn"], self.A["layers"][il]["gdn"]
                self.both("conv_output_raw", il, gr, R.dump(g, f"conv_output_raw-{il}")[0], La["conv_raw"],
                          Lb["conv_raw"])
                self.both("conv_output_silu", il, gr, R.dump(g, f"conv_output_silu-{il}")[0], La["conv"], Lb["conv"])
                self.both("gdn scan (attn_output)", il, gr, R.dump(g, f"attn_output-{il}")[0], La["scan"], Lb["scan"])
                self.both("final_output (gated norm)", il, gr, R.dump(g, f"final_output-{il}")[0], La["final"],
                          Lb["final"])
                self.both("linear_attn_out", il, gr, R.dump(g, f"linear_attn_out-{il}")[0], La["out"], Lb["out"])

    def test_ple_module(self):
        R = self.R
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.hp["ple_layers"]:
                Pb, Pa = b["layers"][il]["ple"], self.A["layers"][il]["ple"]
                self.both("ple_gate", il, gr, R.dump(g, f"ple_gate-{il}")[0], Pa["gate"], Pb["gate"])
                self.both("ple_gated_value", il, gr, R.dump(g, f"ple_gated_value-{il}")[0], Pa["gated"], Pb["gated"])
                self.both("ple_conv_out", il, gr, R.dump(g, f"ple_conv_out-{il}")[0], Pa["conv_out"], Pb["conv_out"])

    def test_attention_over_the_kv_cache(self):
        """Rotated q/k at absolute positions and the attention output over the cached cells (QSA layer and the dense
        causal layer)."""
        R = self.R
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            for il in self.ATT:
                Ab, Aa = b["layers"][il]["attn"], self.A["layers"][il]["attn"]
                self.both("Qcur (IMRoPE)", il, gr, R.dump(g, f"Qcur-{il}")[0], Aa["q"], Ab["q"])
                self.both("Kcur (IMRoPE)", il, gr, R.dump(g, f"Kcur-{il}", 1)[0], Aa["k"], Ab["k"])
                self.both("Vcur", il, gr, R.dump(g, f"Vcur-{il}", 0)[0], Aa["v"], Ab["v"])
                kind = "qsa" if il in self.QSA else "dense causal"
                self.both(f"attn_pregate ({kind})", il, gr, R.dump(g, f"attn_pregate-{il}")[0], Aa["pregate"],
                          Ab["pregate"])
                self.both("attn_output (wo)", il, gr, R.dump(g, f"attn_output-{il}")[0], Aa["out"], Ab["out"])

    def test_qsa_pool_cache_and_selection(self):
        """k-pool cache across ubatches and calls: the pools a graph completes are pooled in that graph
        (indexer_pool_k_new, padded to a stable count), every complete pool is read back from the cache
        (indexer_k), scores and the selection use them."""
        R, hp = self.R, self.hp
        kp, d = hp["kpool"], hp["idx_dim"]
        self.assertEqual(self.not_ties, [], "QSA selections that are no valid top-k by the oracle's scores")
        pooled_in_decode = straddle = 0
        for gr in R.graphs:
            g, p0, n = gr["g"], gr["p0"], gr["n"]
            b = self.B[g]
            for il in self.QSA:
                Ab, Aa = b["layers"][il]["attn"], self.A["layers"][il]["attn"]
                w = Q.layer_weights(self.W, il)
                self.both("indexer_k_raw", il, gr, R.dump(g, f"indexer_k_raw-{il}")[0], Aa["k_raw"], Ab["k_raw"])
                b0, b1 = Ab["pool_range"]
                self.assertEqual((b0, b1), (p0 // kp, (p0 + n) // kp))
                new, ne = R.dump(g, f"indexer_pool_k_new-{il}")
                new = new.reshape(-1, d)
                # n_new_g = max(n_new, 1, min(n_tokens/kpool + 1, ...)) (src/llama-memory-hybrid-idx.cpp:769-770)
                self.assertEqual(new.shape[0], max(b1 - b0, 1, n // kp + 1))
                if b1 > b0:
                    self.close("indexer_pool_k_new (pools completed here) [chunk]", il, g, new[:b1 - b0],
                               Ab["pooled_new"])
                    if all(gr["kept"][:(b1 * kp) - p0]):
                        self.close("indexer_pool_k_new (pools completed here) [full]", il, g, new[:b1 - b0],
                                   Aa["pooled"][b0:b1])
                    pooled_in_decode += n == 1
                    straddle += b0 * kp < p0
                # ec7630a: padded rows re-pool a cell whose slot is never read: the first member of pool 0 if some
                # pool is new, else the first token of the ubatch; rotated at position 0
                # (src/llama-memory-hybrid-idx.cpp:947-967; npos zero-filled at 898)
                # At 79e2e74 every padded entry takes its own cell instead: the smallest cells that are no complete
                # pool's rep ("entries sharing one would write it from several threads in the scatter",
                # src/llama-memory-hybrid-idx.cpp:890-905 @79e2e74); a pool's rep is its last member's cell
                # (:857-859), here kp*j + kp - 1 for the b1 complete pools; the padded slots are still never read.
                n_pad = new.shape[0] - (b1 - b0)
                if SHARED_PAD_CELL:
                    k_pad = Aa["k_raw"][0] if b1 > b0 else Ab["k_raw"][0]
                    pad = Q.rms_norm(k_pad, hp["eps"], w["indexer.k_norm.weight"])
                    for r in range(b1 - b0, new.shape[0]):
                        self.close("indexer_pool_k_new (padding rows)", il, g, new[r], pad)
                else:
                    reps = {kp * j + kp - 1 for j in range(b1)}
                    cells = [c for c in range(p0 + n) if c not in reps][:n_pad]
                    self.assertEqual(len(cells), n_pad)
                    # KV cell = position: cells before the graph from (A), the graph's own cells from (B)
                    k_all = np.concatenate([np.asarray(Aa["k_raw"])[:p0], np.asarray(Ab["k_raw"])])
                    for r, c in zip(range(b1 - b0, new.shape[0]), cells):
                        pad = Q.rms_norm(k_all[c], hp["eps"], w["indexer.k_norm.weight"])
                        self.close("indexer_pool_k_new (padding rows, own non-rep cells)", il, g, new[r], pad)
                got_k = R.dump(g, f"indexer_k-{il}")[0].reshape(-1, d)        # padded to 64 pools
                self.assertGreaterEqual(got_k.shape[0], b1 + 1)
                self.close("indexer_k (pool cache, all complete pools) [chunk]", il, g, got_k[:b1], Ab["pooled"])
                if all(gr["kept"]):
                    self.close("indexer_k (pool cache, all complete pools) [full]", il, g, got_k[:b1],
                               Aa["pooled"][:b1])
                self.both("indexer_q", il, gr, R.dump(g, f"indexer_q-{il}")[0], Aa["iq"], Ab["iq"])
                sc = R.dump(g, f"indexer_score-{il}")[0].reshape(n, -1)
                fin = np.isfinite(Ab["scores"])
                self.assertTrue(np.array_equal(np.isfinite(sc[:, :b1]), fin), f"pool visibility g{g}")
                self.assertFalse(np.isfinite(sc[:, b1:]).any(), f"padded pools must stay masked g{g}")
                self.close("indexer_score (visible) [chunk]", il, g, sc[:, :b1][fin], Ab["scores"][fin])
                for r, t in enumerate(gr["pos"]):
                    self.assertEqual(self.seen[il][(g, r)], Ab["cells"][r], f"QSA cells L{il} pos {t} g{g}")
                    if gr["kept"][r]:
                        self.assertEqual(self.seen[il][(g, r)], Aa["cells"][t], f"QSA cells (full) L{il} pos {t}")
        TABLE.append((self.RUN, f"qsa selection: tie replays = {self.ties}", 3, -1, 0.0, 0.0, ()))
        if self.RUN in ("decode", "rollback"):
            self.assertGreater(pooled_in_decode, 0, "no pool completed by a one-token call")
        if self.RUN in ("ub10", "decode", "rollback"):
            self.assertGreater(straddle, 0, "no pool straddles two graphs")
        # the budget binds: some token sees more pools than it may select
        self.assertTrue(any(len(c) < t + 1 for t, c in enumerate(self.A["layers"][self.QSA[0]]["attn"]["cells"])
                            if t >= R.graphs[-1]["p0"]))

    def test_moe_ids_and_combines(self):
        R = self.R
        k = self.hp["n_expert_used"]
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            kept = np.array(gr["kept"])
            for il in range(self.hp["n_layer"]):
                ids = np.sort(R.dump(g, f"ffn_moe_topk-{il}")[0].reshape(gr["n"], k).astype(int), 1)
                self.assertTrue(np.array_equal(ids, np.sort(b["layers"][il]["moe"]["ids"], 1)), f"top-k L{il} g{g}")
                if kept.any():
                    want = np.sort(self.A["layers"][il]["moe"]["ids"][np.array(gr["pos"])[kept]], 1)
                    self.assertTrue(np.array_equal(ids[kept], want), f"top-k (full) L{il} g{g}")
                self.both("ffn_out (routed + gated shared)", il, gr, R.dump(g, f"ffn_out-{il}")[0],
                          self.A["layers"][il]["moe"]["out"], b["layers"][il]["moe"]["out"])
                self.both("hc_combine (attn)", il, gr, R.dump(g, f"hc_combine-{il}")[0],
                          self.A["layers"][il]["combine_attn"], b["layers"][il]["combine_attn"])
                self.both("l_last (layer output)", il, gr, R.dump(g, f"l_last-{il}")[0],
                          self.A["layers"][il]["combine_ffn"], b["layers"][il]["combine_ffn"])

    def test_head_and_logits(self):
        R = self.R
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_rows, n_vocab).astype(np.float64)
        for gr in R.graphs:
            g, b = gr["g"], self.B[gr["g"]]
            self.both("result_norm", -1, gr, R.dump(g, "result_norm")[0], self.A["result_norm"], b["result_norm"])
            self.both("result_output", -1, gr, R.dump(g, "result_output")[0], self.A["logits"], b["logits"])
            rows = lo[gr["row0"]:gr["row0"] + gr["n"]]
            self.both("logits.bin", -1, gr, rows, self.A["logits"], b["logits"])
            self.assertTrue(np.array_equal(rows.argmax(1), b["logits"].argmax(1)), f"greedy tokens g{g}")

    def test_run_exercises_its_mechanisms(self):
        R, hp = self.R, self.hp
        self.assertEqual(R.final, self.final_tokens())
        if self.RUN == "ub16":
            self.assertEqual([gr["n"] for gr in R.graphs], [16] * 4)
        if self.RUN == "ub10":
            self.assertEqual([gr["n"] for gr in R.graphs], [10] * 6 + [4])
        if self.RUN == "decode":
            self.assertEqual([gr["n"] for gr in R.graphs], [48] + [1] * 16)
            self.assertIn(hp["ple_eos"], [gr["toks"][0] for gr in R.graphs[1:]], "no EOS decoded as a single token")
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

    def final_tokens(self):
        """The 64-token sequence every run must end with (tools/tinygen/run_parity_qwen4exp_decode.sh)."""
        return [int(x) for x in pathlib.Path(DUMP, "tokens.txt").read_text().split(",")]


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Ubatch16(_RunParity, unittest.TestCase):
    RUN = "ub16"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class Ubatch10(_RunParity, unittest.TestCase):
    RUN = "ub10"


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
    print("\nworst error per run and component over layers and graphs (rel = max|got-want| / max|want|):",
          file=sys.stderr)
    print(f"{'run':8s} {'component':58s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, what), (il, g, mx, rel, arg) in worst.items():
        print(f"{run:8s} {what:58s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
