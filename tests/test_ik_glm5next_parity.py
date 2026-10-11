"""ik_llama.cpp's glm5next graph against ref/glm5next_ik.py (N12, gap 5 of docs/reassessment/CORRECTNESS.md).

The G1 / N9 tiny GLM model, written in the dialect R1 runs ik_llama.cpp on (tools/tinygen/glm5next_tiny.py --layers 12
--dialect ik: architecture "glm5next"), is run by ik_llama.cpp at the pin of tools/engines/pins.json on the CPU and
dumped graph by graph by tools/dump/ik_dump_tensors.cpp (tools/tinygen/run_parity_ik_glm5next.sh). Runs (64 tokens):
  p64      one 64-token ubatch, DSA k-pool selection on (--dsa)
  ub16     one call, -ub 16: 4 ubatches
  decode   prefill 48, then 16 calls of one token
  dense    without --dsa (ik's default: every MLA layer attends to all earlier cells)
  ictk16   the indexer key | gate cache in f16 (-ictk f16, ik's default)
  fa16     -fa on -ctk f16 -ctv f16 --dsa, the precision of scripts/r1/configs/ik-llamacpp-server.json
  fa16dec  the same as prefill 48 + 16 single-token calls
  alias    the same weights under llama.cpp's name "glm5-next" (logits only)
The oracle is ref/glm5next.py with ik's deviations (ref/glm5next_ik.py docstring, items 1-5): post-SiLU SwiGLU clamp,
indexer LayerNorm eps = RMS eps, ik's L2 eps, cell 0 attended where ik pads the DSA tail with it, the MLA latent
stored in the -ctv type with -fa off (the -ctk type with -fa on) and the indexer cache in the -ictk type.
Checks, per graph and layer:
  * local, from ik's own dumped inputs, at TOL_OP = 1e-5 (the N10 operation tolerance): the latent written to the cache
    (exact: the cache type's rounding of ik's kv_compressed), the DSA projections from ik's attn_norm (absorbed q,
    indexer key with ik's eps), the pooled keys from ik's cached key | gate rows, the indexer scores from ik's q,
    weights and pooled keys, the attention output from ik's absorbed q, its cached latents and the cells it
    attended (with -fa off; with -fa on TOL_FA), the FFN / MoE output from ik's ffn_norm (post-SiLU clamp);
  * the selection: the cells each token attended (dsa_top_k restricted to the causal range) equal the oracle's, or
    are another valid top-k among pools tied within TIE_RTOL by the oracle's own scores (G.tie_variant), replayed
    as in N9 (counted); the run must contain rows where cell 0 is attended only through ik's tail padding;
  * chained, the incremental oracle from token ids along the same calls: TOL = 1e-4 (N9) up to the first DSA layer's
    input, TOL_F16 = 16 * 2^-11 (N10) after it, where the latent is rounded to f16 and an f32 difference can move a
    value across an f16 rounding boundary; the logits likewise;
  * the "glm5-next" alias gives bit-identical logits.
Env: IK_GLM5NEXT_DUMP (directory with one directory per run), IK_GLM5NEXT_GGUF (the --dialect ik model),
LLAMA_CPP_GGUF_PY. "B" = src/graphs/build_glm5next.cpp, "L" = src/llama.cpp of ik_llama.cpp @5194a9e.
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
from ref import glm5next_ik as K  # noqa: E402

DUMP = os.environ.get("IK_GLM5NEXT_DUMP")
GGUF = os.environ.get("IK_GLM5NEXT_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
TOL, TOL_OP, TOL_F16 = 1e-4, 1e-5, 16 * 2.0 ** -11
TOL_FA = 2.0 ** -11                # local attention with -fa on, emulated by ref/fattn.py fa_chunk (ref/glm5next_ik.py
                                   # item 6): one f16 unit roundoff, the accumulator's order of operations not reproduced
TIE_RTOL = 1e-6
SKIP = "set IK_GLM5NEXT_DUMP, IK_GLM5NEXT_GGUF, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_ik_glm5next.sh"
TABLE = []                         # (run, check, layer, graph, rel err)
_MODEL = {}


def model(path):
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


def cast(x, t):
    x = np.asarray(x, dtype=np.float64)
    return x.astype(np.float32).astype(np.float16).astype(np.float64) if t == "f16" else x.astype(np.float32)


class Run:
    """One dumped run: manifest (tools/dump/ik_dump_tensors.cpp), flags, graphs with their positions."""

    def __init__(self, name, root):
        self.name, self.dir = name, pathlib.Path(root, name)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.flags = (self.dir / "flags.txt").read_text().split()
        self.toks = [int(x) for x in pathlib.Path(root, "tokens.txt").read_text().split(",")]
        ub, self.graphs = self.man["n_ubatch"], []
        for c, call in enumerate(self.man["calls"]):
            g0, g1 = call["graphs"]
            for j, g in enumerate(range(g0, g1)):
                n = min(ub, call["n"] - j * ub)
                p0 = call["pos0"] + j * ub
                self.graphs.append(dict(g=g, call=c, n=n, pos=list(range(p0, p0 + n)), toks=self.toks[p0:p0 + n]))

    def flag(self, name, default):
        return self.flags[self.flags.index(name) + 1] if name in self.flags else default

    def cfg(self):
        """ref/glm5next_ik.config of the run's flags (ref/glm5next_ik.py item 5; -ictk defaults to f16, C:495)."""
        fa = self.flag("-fa", "on") == "on"                       # ik's default is flash attention on (C:445)
        latent = self.flag("-ctk", "f16") if fa else self.flag("-ctv", "f16")
        return K.config(dsa="--dsa" in self.flags, latent=latent, index=self.flag("-ictk", "f16"), fa=fa)

    def dump(self, g, name):
        p = self.dir / f"g{g:03d}"
        j = json.loads((p / f"{name}__0.json").read_text())
        dt = np.float32 if j["type"] == "f32" else np.int32
        return np.fromfile(p / f"{name}__0.bin", dtype=dt).astype(np.float64), j["ne"]

    def logits(self):
        return np.fromfile(self.dir / "logits.bin", dtype=np.float32).astype(np.float64).reshape(len(self.toks), -1)


def rel(got, want):
    want = np.asarray(want, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64).reshape(want.shape)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


class _IkRun:
    RUN = None

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")
        cls.W, hp0 = model(GGUF)
        cls.hp = hp = K.ik_hparams(hp0)
        cls.R = Run(cls.RUN, DUMP)
        cls.cfg = cls.R.cfg()
        cls.KDA = [i for i in range(hp["n_layer"]) if hp["kda"][i]]
        cls.DSA = [i for i in range(hp["n_layer"]) if not hp["kda"][i]]
        cls.first_dsa = cls.DSA[0]
        # the cells each token attended: ik's dsa_top_k (B:203-215) within the causal range (the base KQ mask keeps
        # later cells masked, ggml/src/iqk/iqk_cpu_ops.cpp:1106-1108); every earlier cell without --dsa
        cls.seen = {il: {} for il in cls.DSA}
        for gr in cls.R.graphs:
            for il in cls.DSA:
                if cls.cfg["dsa"]:
                    c, ne = cls.R.dump(gr["g"], f"dsa_top_k-{il}")
                    c = c.reshape(ne[1], ne[0]).astype(int)
                    for r, t in enumerate(gr["pos"]):
                        cls.seen[il][t] = sorted({int(x) for x in c[r] if x <= t})
                else:
                    for t in gr["pos"]:
                        cls.seen[il][t] = list(range(t + 1))
        cls.ties, cls.not_ties, cls.ov = 0, [], {}
        cls._oracle()
        while cls._replay_first_tie():
            cls._oracle()

    @classmethod
    def _oracle(cls):
        """The incremental oracle along the run's graphs (one ubatch per graph), with the replays in cls.ov."""
        cache, cls.B = G.new_cache(), {}
        with K.semantics(cls.cfg, cls.hp):
            for gr in cls.R.graphs:
                ov = {k: {t: v[t] for t in gr["pos"] if t in v} for k, v in cls.ov.items()}
                cls.B[gr["g"]] = G.forward_chunk(gr["toks"], cls.W, cls.hp, cache, ov or None)

    @classmethod
    def _replay_first_tie(cls):
        hp, kp = cls.hp, cls.hp["kpool"]
        for gr in cls.R.graphs:
            for il in cls.DSA:
                ob = cls.B[gr["g"]]["layers"][il]["dsa"]
                rows, bad = [], []
                for r, t in enumerate(gr["pos"]):
                    got, own = cls.seen[il][t], ob["cells"][r]
                    if got == own:
                        continue
                    args = (t, hp["idx_top_k"], kp, True, TIE_RTOL)
                    ok_own, why_own = G.tie_variant(ob["scores"][r], ob["ref_cells"][r], *args)
                    ok = False
                    for cand in (got, sorted(set(got) - {0})):     # ik's set before its tail padding (item 4)
                        if K.ik_cells([cand], [t], kp)[0] == got and G.tie_variant(ob["scores"][r], cand, *args)[0]:
                            ok = True
                    rows.append((t, got))
                    if not (ok and ok_own):
                        bad.append(f"DSA L{il} pos {t} graph {gr['g']}: ik {got}, oracle {own} "
                                   f"(oracle's own choice {why_own or 'valid'}, ik's {'valid' if ok else 'no tie'})")
                if bad:
                    cls.not_ties += bad
                    return False
                if rows:
                    for t, got in rows:
                        cls.ov.setdefault(("dsa", il), {})[t] = got
                    cls.ties += len(rows)
                    return True
        return False

    def record(self, what, il, g, err, tol):
        TABLE.append((self.RUN, what, il, g, err))
        self.assertLess(err, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {err:.3e} >= {tol:.1e}")

    def chain_tol(self, il, before_mixer=False):
        if il < self.first_dsa or (il == self.first_dsa and before_mixer):
            return TOL
        return TOL_F16

    # ---- tests ----
    def test_graphs_and_embedding(self):
        ub = self.R.man["n_ubatch"]
        for call in self.R.man["calls"]:
            g0, g1 = call["graphs"]
            self.assertEqual(g1 - g0, -(-call["n"] // ub), f"{self.RUN}: graphs of call {call}")
        for gr in self.R.graphs:
            e, ne = self.R.dump(gr["g"], "inp_embd")
            want = self.W["token_embd.weight"][gr["toks"]].astype(np.float32)
            self.assertTrue(np.array_equal(e.reshape(want.shape), want), f"{self.RUN} inp_embd graph {gr['g']}")

    def test_selection(self):
        self.assertEqual(self.not_ties, [], "\n".join(self.not_ties))
        if not self.cfg["dsa"]:
            return
        kp, padded = self.hp["kpool"], 0
        for gr in self.R.graphs:
            for il in self.DSA:
                ob = self.B[gr["g"]]["layers"][il]["dsa"]
                for r, t in enumerate(gr["pos"]):
                    self.assertEqual(self.seen[il][t], ob["cells"][r], f"{self.RUN} DSA L{il} pos {t}")
                    padded += 0 in ob["cells"][r] and 0 not in ob["ref_cells"][r]
        TABLE.append((self.RUN, f"DSA rows attending cell 0 by tail padding / tie replays: {padded} / {self.ties}",
                      -1, -1, 0.0))
        self.assertGreater(padded, 0, f"{self.RUN}: no row exercises ik's tail padding (item 4)")

    def test_cache_stores(self):
        """The MLA cache holds the -ctv (-fa off) or -ctk (-fa on) type's rounding of ik's own latent, the indexer cache
        the -ictk type's rounding of ik's own key | gate (item 5); cell = position. Every graph reads, for every
        earlier position, exactly the row the graph that computed that position wrote."""
        d = self.hp["idx_dim"]
        written = {}                                          # (kind, il) -> rows stored so far, by position
        for gr in self.R.graphs:
            p0, p1 = gr["pos"][0], gr["pos"][-1] + 1
            for il in self.DSA:
                rows = {}
                if self.cfg["dsa"]:
                    k = self.R.dump(gr["g"], f"dsa_indexer_k-{il}")[0].reshape(gr["n"], d)
                    gate = self.R.dump(gr["g"], f"dsa_indexer_gate-{il}")[0].reshape(gr["n"], d)
                    allr, ane = self.R.dump(gr["g"], f"dsa_cached_all-{il}")
                    rows["indexer"] = (allr.reshape(ane[1], ane[0]),
                                       cast(np.concatenate([k, gate], axis=1), self.cfg["index"]))
                kc, ne = self.R.dump(gr["g"], f"kv_compressed-{il}")
                cache, cne = self.R.dump(gr["g"], f"kv_cache-{il}")
                rows["latent"] = (cache.reshape(cne[1], cne[0]), cast(kc.reshape(gr["n"], ne[0]), self.cfg["latent"]))
                for kind, (got, new) in rows.items():
                    old = written.get((kind, il), np.zeros((0, new.shape[1])))
                    want = np.concatenate([old[:p0], new])
                    self.assertEqual(old.shape[0], p0, f"{self.RUN} {kind} L{il} graph {gr['g']}: positions")
                    self.assertTrue(np.array_equal(got[:p1], want), f"{self.RUN} {kind} cache L{il} graph {gr['g']}: "
                                    f"rows {sorted(set(np.where((got[:p1] != want).any(1))[0].tolist()))[:8]} differ "
                                    f"from what was stored")
                    written[(kind, il)] = want

    def test_dsa_local(self):
        """DSA from ik's own inputs: projections, pooled keys, scores, attention (module docstring)."""
        hp, kp, d, nh = self.hp, self.hp["kpool"], self.hp["idx_dim"], self.hp["idx_heads"]
        for gr in self.R.graphs:
            g, n = gr["g"], gr["n"]
            p_end = gr["pos"][-1] + 1
            for il in self.DSA:
                w = G.layer_weights(self.W, il)
                x = self.R.dump(g, f"attn_norm-{il}")[0].reshape(n, -1)
                r = G.dsa_project(x, w, hp)
                qd, qne = self.R.dump(g, f"Qcur-{il}")                # [kv_lora, T, H]
                qd = qd.reshape(qne[2], qne[1], qne[0]).transpose(1, 0, 2)
                self.record("Qcur (local)", il, g, rel(qd, r["q_abs"]), TOL_OP)
                cache, cne = self.R.dump(g, f"kv_cache-{il}")
                lat = cache.reshape(cne[1], cne[0])
                cells = [self.seen[il][t] for t in gr["pos"]]
                modes = ["fa_chunk"] * n if self.cfg["fa"] else None
                kqv = G.mla_attention(qd, lat, cells, np.asarray(w["attn_v_b.weight"]), hp["k_mla"], modes)
                tol = TOL_FA if self.cfg["fa"] else TOL_OP
                self.record("kqv_out (local)", il, g, rel(self.R.dump(g, f"kqv_out-{il}")[0],
                                                          kqv @ w["attn_output.weight"].T), tol)
                if not self.cfg["dsa"]:
                    continue
                self.record("indexer key (local, ik eps)", il, g,
                            rel(self.R.dump(g, f"dsa_indexer_k-{il}")[0], r["ik"]), TOL_OP)
                allr, ane = self.R.dump(g, f"dsa_cached_all-{il}")    # [2d, n_kv]: key | gate per cell
                allr = allr.reshape(ane[1], ane[0])[:p_end]
                nb = p_end // kp
                pooled = G.indexer_pools(allr[:nb * kp, :d], allr[:nb * kp, d:], w["indexer_compressor_ape.weight"], kp)
                pk, pne = self.R.dump(g, f"dsa_indexer_k_pooled-{il}")
                self.record("pooled keys (local)", il, g, rel(pk.reshape(pne[1], pne[0])[:nb], pooled), TOL_OP)
                iq = self.R.dump(g, f"dsa_indexer_q-{il}")[0].reshape(n, nh, d)
                wts = self.R.dump(g, f"dsa_indexer_weights-{il}")[0].reshape(n, nh)
                sc, sne = self.R.dump(g, f"dsa_indexer_score-{il}")
                sc = sc.reshape(sne[1], sne[0])[:, :nb]
                want = G.scores_at(iq, wts, pk.reshape(pne[1], pne[0])[:nb], kp, gr["pos"])
                self.assertTrue(np.array_equal(np.isfinite(sc), np.isfinite(want)), f"{self.RUN} visible pools L{il}")
                fin = np.isfinite(want)
                if fin.any():
                    self.record("indexer scores (local)", il, g, rel(sc[fin], want[fin]), TOL_OP)

    def test_ffn_local(self):
        """FFN / MoE from ik's own ffn_norm: post-SiLU clamp (item 1)."""
        for gr in self.R.graphs:
            for il in range(self.hp["n_layer"]):
                x = self.R.dump(gr["g"], f"ffn_norm-{il}")[0].reshape(gr["n"], -1)
                with K.semantics(self.cfg, self.hp):
                    f = G.ffn_layer(x, G.layer_weights(self.W, il), self.hp, il)
                self.record("ffn_out (local)", il, gr["g"], rel(self.R.dump(gr["g"], f"ffn_out-{il}")[0], f["out"]),
                            TOL_OP)

    def test_chain(self):
        """The incremental oracle from token ids (with the replays), component by component, and the logits."""
        for gr in self.R.graphs:
            g, B = gr["g"], self.B[gr["g"]]
            for il, L in enumerate(B["layers"]):
                self.record("hc_attn_pre", il, g, rel(self.R.dump(g, f"hc_attn_pre-{il}")[0], L["attn_pre"]),
                            self.chain_tol(il, True))
                if self.hp["kda"][il]:
                    self.record("KDA out", il, g, rel(self.R.dump(g, f"linear_attn_out-{il}")[0], L["kda"]["out"]),
                                self.chain_tol(il))
                else:
                    self.record("DSA kqv_out", il, g, rel(self.R.dump(g, f"kqv_out-{il}")[0], L["dsa"]["out"]),
                                self.chain_tol(il))
                self.record("mHC attn post", il, g, rel(self.R.dump(g, f"attn_out-{il}")[0], L["hc_attn_post"]),
                            self.chain_tol(il))
                self.record("ffn_out", il, g, rel(self.R.dump(g, f"ffn_out-{il}")[0], L["ffn"]["out"]),
                            self.chain_tol(il))
                self.record("l_out", il, g, rel(self.R.dump(g, f"l_out-{il}")[0], L["l_out"]), self.chain_tol(il))
        got = self.R.logits()
        want = np.concatenate([self.B[gr["g"]]["logits"] for gr in self.R.graphs])
        self.record("logits", -1, -1, rel(got, want), TOL_F16)
        same = int((got.argmax(1) == want.argmax(1)).sum())
        TABLE.append((self.RUN, f"greedy argmax equal on {same}/{len(got)} rows", -1, -1, 0.0))


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkP64(_IkRun, unittest.TestCase):
    RUN = "p64"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkUb16(_IkRun, unittest.TestCase):
    RUN = "ub16"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkDecode(_IkRun, unittest.TestCase):
    RUN = "decode"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkDense(_IkRun, unittest.TestCase):
    RUN = "dense"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkIctk16(_IkRun, unittest.TestCase):
    RUN = "ictk16"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkFa16(_IkRun, unittest.TestCase):
    RUN = "fa16"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkFa16Decode(_IkRun, unittest.TestCase):
    RUN = "fa16dec"


@unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)
class IkAlias(unittest.TestCase):
    def test_glm5_next_name_gives_identical_logits(self):
        for run in ("alias", "p64"):
            if not pathlib.Path(DUMP, run, "logits.bin").exists():
                self.skipTest(f"no run {run} under {DUMP}")
        a, b = Run("alias", DUMP).logits(), Run("p64", DUMP).logits()
        self.assertTrue(np.array_equal(a, b), f"alias vs glm5next: max diff {np.abs(a - b).max():.3e}")


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, err in TABLE:
        k = (run, what)
        if k not in worst or err > worst[k][0]:
            worst[k] = (err, il, g)
    print("\nworst normwise relative error per run and check (layer, graph):", file=sys.stderr)
    for (run, what), (err, il, g) in worst.items():
        where = "" if il < 0 and g < 0 else f"  (L{il}, g{g})"
        print(f"  {run:8s} {what:44s} {err:.3e}{where}" if err else f"  {run:8s} {what}", file=sys.stderr)
