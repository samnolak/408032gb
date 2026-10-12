"""Several sequences in one llama.cpp qwen4exp context, seq_cp, seq_rm and rollback edges against ref/qwen4exp.py (N19,
gap 10 of docs/reassessment/CORRECTNESS.md).

N6 (tests/test_qwen4exp_decode_parity.py) covers one sequence. Here the same tiny model (tools/tinygen/qwen4exp_tiny.py,
seed 0: GDN layers 0-2 and 4-6, PLE on layer 1, QSA on layer 3 with kpool 4 and 2 pools + tail, dense attention on
layer 7) runs the nine plans of tools/tinygen/qwen4exp_multiseq_plans.py through llama.cpp on the CPU
(tools/tinygen/run_parity_qwen4exp_multiseq.sh; -np 2 or 3, f32 caches, flash attention off), dumped graph by graph by
tools/dump/dump_tensors.cpp with the several-sequence DUMP_CALLS steps (parts "n@s" joined by "+", "r<n>@<s>",
"c<a>:<b>", "x<s>") and manifest.json recording every llama_process call, the graphs it computed, every seq_rm result
and llama_memory_seq_pos_min / _max of every sequence after every step. Each graph is compared row by row with
  (A) per sequence alone: Q.forward(final tokens of that sequence) at the row's position (one entry per "life" of a
      sequence id: a seq_cp into it or dropping it starts a new one). Only for rows not rolled back later and rows of
      a sequence that has not read a recurrent state of another prefix ("diverged", see below);
  (B) Q.MultiSeqContext replaying the same calls: every sequence through Q.forward_chunk with its own cache, on cell
      models of llama.cpp's KV cache (which cells hold which (sequence, position)), recurrent cache (which row and
      snapshot plane every sequence reads) and k-pool layout (which pools a graph re-pools, in which order).
Rows are in ubatch order (llama_batch_allocr::split_equal, replicated by Q.split_equal: sequence sets concatenated),
logits.bin in batch order. Cross-sequence leakage is checked where it would show: kq_soft_max of both attention layers
is exactly 0 on every cell that is not the row's sequence's visible cell (and equals (B)'s probabilities on those),
indexer_score is finite exactly on the row's sequence's visible pools, the QSA selection (kq_mask_qsa, cell indices)
maps to positions of the row's own sequence, and the recurrent state each sequence reads equals its own prefix's.

"Diverged": llama.cpp reads a recurrent row whose content is the state of another token prefix than the sequence's KV
history (a stale rollback snapshot, or the pre-rollback state through a seq_cp); (B) predicts it, (A) cannot. Every
run's expected divergences per pin are in RUNS[run]["diverge"] (none elsewhere); test_edge_behaviour asserts them,
that (A) is off there, and every other run- and pin-specific behaviour (refusals, sharing, cell moves).
G = src/models/qwen4exp.cpp; line numbers @ec7630a640789c393694fb194f1bbbf0369fc62d unless "@79e2e74".
Env: QWEN4EXP_MULTISEQ_DUMP (root with one directory per run), QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT (which pin's behaviour to expect; set by the driver), QWEN4EXP_RS_GUARD=1 (N20: the dumps come
from llama.cpp with patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch; expect RUNS[run]["guard_ok"] (else
the R1 pin's seq_rm results) and RUNS[run]["diverge"]["guard"] (else none); set by
tools/tinygen/run_parity_qwen4exp_rs_guard.sh).

Tolerances as N2/N6: normwise relative error < 1e-4 for f32 paths; exact for the embedding and PLE gathers, the zero
state of a new sequence, the masked (zero) attention probabilities; MoE ids equal as sets; QSA selections equal, or
another valid top-k by (B)'s own scores (Q.qsa_tie_variant), replayed like N6 and counted (TABLE).
"""
import importlib.util
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
from tests.test_qwen4exp_decode_parity import model  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen4exp_multiseq_plans",
                                               ROOT / "tools" / "tinygen" / "qwen4exp_multiseq_plans.py")
PLANS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PLANS)

DUMP = os.environ.get("QWEN4EXP_MULTISEQ_DUMP")
GGUF = os.environ.get("QWEN4EXP_TINY_GGUF")
GGUF_PY = os.environ.get("LLAMA_CPP_GGUF_PY")
LLAMA_COMMIT = os.environ.get("PARITY_LLAMA_COMMIT", "ec7630a640789c393694fb194f1bbbf0369fc62d")
PIN = {"ec7630a": "ec", "79e2e74": "r1"}.get(LLAMA_COMMIT[:7])
GUARD = os.environ.get("QWEN4EXP_RS_GUARD") == "1"
TOL = 1e-4


def expected_ok(plan):
    """The seq_rm result every "r" step of a plan must give: at the pin, or with the N20 guard patch (the same at both
    pins: the ec7630a patch adds 79e2e74's shared-cell refusal)."""
    rs = [st for st in plan["steps"] if st[0] == "r"]
    if GUARD:
        return list(plan.get("guard_ok", [st[4] for st in rs]))
    return [st[3] if PIN == "ec" else st[4] for st in rs]


def expected_diverge(plan):
    return plan.get("diverge", {}).get("guard" if GUARD else PIN, {})
SKIP = ("set QWEN4EXP_MULTISEQ_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY: "
        "run tools/tinygen/run_parity_qwen4exp_multiseq.sh")
TABLE = []        # (run, component, layer, graph, max abs, rel)


class Replay:
    """(B) over one dumped run, plus the bookkeeping (A) needs: which life of a sequence id every row belongs to, which
    rows a later rollback rejected, the final tokens of every life."""

    def __init__(self, run_dir, W, hp, pin, overrides=None):
        self.dir = pathlib.Path(run_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.fed = [int(x) for x in (self.dir / "tokens.txt").read_text().split(",")]
        man = self.man
        self.M = M = Q.MultiSeqContext(W, hp, man["n_seq_max"], man["n_ctx_seq"], man["kv_unified"], man["n_rs_seq"],
                                       pin, man["n_ubatch"], guard=GUARD)
        self.graphs, self.steps = [], []       # graph dicts with g, call; per manifest call: what B did
        life = {s: s for s in range(man["n_seq_max"])}
        self.life_tokens = {s: [] for s in range(man["n_seq_max"])}
        self.life_seq = {s: s for s in range(man["n_seq_max"])}
        next_life = man["n_seq_max"]
        lo_row = 0
        for ci, call in enumerate(man["calls"]):
            rec = dict(call=ci, kind=call["kind"])
            if call["kind"] == "seq_rm":
                ok = M.seq_rm(call["seq"], call["p0"], call["p1"])
                rec["ok"] = ok
                if ok and call["p0"] < 0:                # the whole sequence: a new life starts empty
                    s = call["seq"]
                    life[s] = next_life
                    self.life_tokens[next_life], self.life_seq[next_life] = [], s
                    next_life += 1
                elif ok:
                    s, keep = call["seq"], call["p0"]
                    lf = life[s]
                    for gr in self.graphs:              # rows at or after the cut were rejected
                        gr["kept"] = [k and not (lv == lf and p >= keep) for k, lv, p in
                                      zip(gr["kept"], gr["life"], [r[1] for r in gr["rows"]])]
                    self.life_tokens[lf] = self.life_tokens[lf][:keep]
            elif call["kind"] == "seq_cp":
                M.seq_cp(call["seq"], call["seq_dst"])
                d = call["seq_dst"]
                life[d] = next_life
                self.life_tokens[next_life] = list(self.life_tokens[life[call["seq"]]])
                self.life_seq[next_life] = d
                next_life += 1
            elif call["ret"] == 0:
                parts = [(p["seq"], self.fed[p["tok0"]:p["tok0"] + p["n"]]) for p in call["parts"]]
                ov = {(ui, s): v for (c, ui, s), v in (overrides or {}).items() if c == ci}
                gs = M.decode(parts, ov)
                g0, g1 = call["graphs"]
                rec["n_graphs"] = len(gs)
                for ui, (g, G) in enumerate(zip(range(g0, g1), gs)):
                    G.update(g=g, call=ci, ui=ui, life=[life[s] for s, _, _ in G["rows"]],
                             kept=[True] * len(G["rows"]), logit_rows=[lo_row + b for b in G["batch_idx"]])
                    self.graphs.append(G)
                for s, toks in parts:
                    self.life_tokens[life[s]] += toks
                lo_row += call["n"]
            rec["pos"] = {s: M.seq_pos(s) for s in range(man["n_seq_max"])}
            self.steps.append(rec)
        self.n_logit_rows = lo_row

    def dump(self, g, name, k=0):
        p = self.dir / f"g{g:03d}"
        j = json.loads((p / f"{name}__{k}.json").read_text())
        dt = np.float32 if j["type"] == "f32" else np.int32
        return np.fromfile(p / f"{name}__{k}.bin", dtype=dt).reshape(j["ne"][::-1]).astype(np.float64), j["ne"]

    def has(self, g, name, k=0):
        return (self.dir / f"g{g:03d}" / f"{name}__{k}.json").exists()


class _MultiSeqParity:
    RUN = None
    PLANS = None      # the plans module of RUN (None: tools/tinygen/qwen4exp_multiseq_plans.py)

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
            raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")
        if PIN is None:
            raise AssertionError(f"PARITY_LLAMA_COMMIT {LLAMA_COMMIT}: no expectations for this llama.cpp commit")
        cls.W, cls.hp = model(GGUF)
        hp = cls.hp
        cls.PLAN = (cls.PLANS or PLANS).RUNS[cls.RUN]
        cls.LIN = [i for i in range(hp["n_layer"]) if hp["recurrent"][i]]
        cls.ATT = [i for i in range(hp["n_layer"]) if not hp["recurrent"][i]]
        cls.QSA = [i for i in cls.ATT if hp["compress"][i] > 0]
        cls.ties, cls.not_ties, cls.ov = 0, [], {}
        cls._oracles()
        while cls._replay_first_tie():
            cls._oracles()

    @classmethod
    def _oracles(cls):
        cls.R = Replay(pathlib.Path(DUMP, cls.RUN), cls.W, cls.hp, PIN, cls.ov)
        ov_a = {}
        for (ci, ui, s), v in cls.ov.items():
            for gr in cls.R.graphs:
                if gr["call"] == ci and gr["ui"] == ui:
                    lf = gr["life"][[r[0] for r in gr["rows"]].index(s)]
                    for key, d in v.items():
                        ov_a.setdefault(lf, {}).setdefault(key, {}).update(d)
        cls.A = {lf: Q.forward(toks, cls.W, cls.hp, ov_a.get(lf)) for lf, toks in cls.R.life_tokens.items() if toks}

    @classmethod
    def _replay_first_tie(cls):
        """N6's replay of a QSA choice among tied pools: the first row whose dumped selection differs from (B)'s is
        replayed if both are valid top-k sets by (B)'s own scores (Q.qsa_tie_variant), else kept in not_ties."""
        hp = cls.hp
        for gr in cls.R.graphs:
            for il in cls.QSA:
                sel = cls.selected_positions(gr, il)
                for r, (s, p, _) in enumerate(gr["rows"]):
                    ob = cls.seq_rows(gr, s)["layers"][il]["attn"]
                    k = r - gr["seqs"].index(s) * gr["nst"]
                    ref, own = sel[r], ob["cells"][k]
                    if ref == own:
                        continue
                    args = (p, hp["idx_top_k"], hp["kpool"], 1e-6)
                    (ok, why), (ok2, why2) = Q.qsa_tie_variant(ob["scores"][k], ref, *args), \
                        Q.qsa_tie_variant(ob["scores"][k], own, *args)
                    if not (ok and ok2):
                        cls.not_ties.append(f"L{il} seq {s} pos {p} graph {gr['g']}: reference {why or 'valid'}, "
                                            f"oracle {why2 or 'valid'}; {ref} vs {own}")
                        return False
                    cls.ov.setdefault((gr["call"], gr["ui"], s), {}).setdefault(("qsa", il), {})[p] = ref
                    cls.ties += 1
                    return True
        return False

    # ---- helpers
    @classmethod
    def seq_rows(cls, gr, s):
        return gr["per_seq"][s]

    @classmethod
    def selected_positions(cls, gr, il):
        """kq_mask_qsa as positions of the row's own sequence (asserting every selected cell holds it)."""
        m, ne = cls.R.dump(gr["g"], f"kq_mask_qsa-{il}")
        m = m.reshape(len(gr["rows"]), -1)
        assert m.shape[1] == gr["n_kv"], (cls.RUN, gr["g"], m.shape, gr["n_kv"])
        out = []
        for r, (s, p, _) in enumerate(gr["rows"]):
            cells = gr["cells"][cls.R.M.kv.stream(s)]
            sel = np.where(np.isfinite(m[r]) & (m[r] > -1e30))[0]
            assert all(s in cells[c][1] for c in sel), (cls.RUN, gr["g"], r, "a selected cell of another sequence")
            out.append(sorted(cells[c][0] for c in sel))
        return out

    def a_ok(self, gr, r):
        """Row r of graph gr has an (A): kept, and its sequence has not diverged."""
        s = gr["rows"][r][0]
        return gr["kept"][r] and not gr["diverged"][s]

    def close(self, what, il, g, got, want, tol=TOL):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        self.assertEqual(got.size, want.size, f"{self.RUN} {what} L{il} g{g}: size {got.shape} vs {want.shape}")
        got = got.reshape(want.shape)
        d = np.abs(got - want)
        mx = float(d.max()) if d.size else 0.0
        rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
        TABLE.append((self.RUN, what, il, g, mx, rel))
        self.assertLess(rel, tol, f"{self.RUN} {what} L{il} graph {g}: rel err {rel:.2e} (abs {mx:.2e})")

    def rows(self, gr, getter):
        """(B) rows of graph gr in ubatch order: getter(per-sequence result) for every sequence, concatenated."""
        return np.concatenate([np.asarray(getter(self.seq_rows(gr, s))) for s in gr["seqs"]])

    def both(self, what, il, gr, got, getter_b, getter_a, tol=TOL):
        """Compare a per-row tensor with (B) and, on the rows that have one, with (A) (getter_a(A dict) indexed by
        position)."""
        want_b = self.rows(gr, getter_b)
        got = np.asarray(got, dtype=np.float64).reshape(want_b.shape)
        self.close(f"{what} [B]", il, gr["g"], got, want_b, tol)
        idx = [r for r in range(len(gr["rows"])) if self.a_ok(gr, r)]
        if idx:
            want_a = np.stack([np.asarray(getter_a(self.A[gr["life"][r]]))[gr["rows"][r][1]] for r in idx])
            self.close(f"{what} [A]", il, gr["g"], got[idx], want_a, tol)

    def prefix_state(self, lf, il, n):
        G = self.A[lf]["layers"][il]["gdn"]
        if n == 0:
            S = self.hp["ssm_state"]
            return np.zeros((self.hp["ssm_v_heads"], S, S))
        return gated_delta_rule(G["q"][:n], G["k"][:n], G["v"][:n], G["g"][:n], G["beta"][:n], normalize_qk=True,
                                eps=self.hp["eps"])[1]

    def first_rows(self, gr):
        """Per ubatch sequence k: (seq, its first position in the graph, row index of that position, life)."""
        return [(s, gr["rows"][k * gr["nst"]][1], k * gr["nst"], gr["life"][k * gr["nst"]])
                for k, s in enumerate(gr["seqs"])]

    # ---- tests
    def test_graphs_rows_and_memory_positions(self):
        """Graphs per call and the rows of every graph (split_equal replica; embedding rows exact), seq_rm results,
        and llama_memory_seq_pos_min / _max of every sequence after every step equal the cell models'."""
        R, hp = self.R, self.hp
        self.assertEqual(len(R.man["graphs"]), len(R.graphs))
        self.assertEqual([gr["g"] for gr in R.graphs], list(range(len(R.graphs))))
        for call, rec in zip(R.man["calls"], R.steps):
            if call["kind"] == "decode":
                self.assertEqual(call["ret"], 0, f"call {rec['call']} refused")
                self.assertEqual(call["graphs"][1] - call["graphs"][0], rec["n_graphs"], f"call {rec['call']}")
            if call["kind"] == "seq_rm":
                self.assertEqual(call["ok"], rec["ok"], f"seq_rm of call {rec['call']}: llama.cpp vs the model")
            want = [rec["pos"][s] for s in range(R.man["n_seq_max"])]
            got = list(zip(call["pos_min"], call["pos_max"]))
            self.assertEqual(got, [tuple(x) for x in want], f"seq_pos_min/max after call {rec['call']}")
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32)
        self.assertEqual(lo.size, R.n_logit_rows * self.W["output.weight"].shape[0])
        for gr in R.graphs:
            got, ne = R.dump(gr["g"], "hc_init")
            self.assertEqual(ne[2], len(gr["rows"]))
            want = Q.hc_init(self.W["token_embd.weight"][[t for _, _, t in gr["rows"]]], hp["hc"])
            self.close("hc_init (exact)", -1, gr["g"], got, want, tol=1e-300)

    def test_carried_state(self):
        """The recurrent state every sequence of a graph read (GDN conv history, delta-rule state, PLE conv history)
        equals (B)'s read and, unless the sequence diverged, its own prefix's state from (A); the state written back
        (new_state, n_rs_seq = 0) equals (B)'s."""
        R, hp = self.R, self.hp
        ple = hp["ple_layers"]
        n_ok = 0
        for gr in R.graphs:
            g, nseq = gr["g"], len(gr["seqs"])
            for il in self.LIN:
                a, ne = R.dump(g, f"conv_state_at-{il}", 1 if il in ple else 0)
                conv = a.reshape(nseq, ne[1], ne[0]).transpose(0, 2, 1)
                st, _ = R.dump(g, f"state_predelta-{il}")
                st = st.reshape(nseq, hp["ssm_v_heads"], hp["ssm_state"], hp["ssm_state"]).transpose(0, 1, 3, 2)
                for k, (s, p0, r0, lf) in enumerate(self.first_rows(gr)):
                    Lb = self.seq_rows(gr, s)["layers"][il]["gdn"]
                    self.close("conv_state_at (GDN conv history) [B]", il, g, conv[k], Lb["conv_in"])
                    self.close("state_predelta (delta-rule state) [B]", il, g, st[k], Lb["state_in"])
                    if gr["diverged"][s] or not gr["kept"][r0]:     # (A): a kept row's prefix is kept too
                        continue
                    La = self.A[lf]["layers"][il]["gdn"]
                    want = np.vstack([np.zeros((conv.shape[1], La["qkv"].shape[1])), La["qkv"][:p0]])[-conv.shape[1]:]
                    self.close("conv_state_at (GDN conv history) [A]", il, g, conv[k], want)
                    self.close("state_predelta (delta-rule state) [A]", il, g, st[k], self.prefix_state(lf, il, p0))
                    n_ok += p0 > 0
                if R.man["n_rs_seq"] == 0:
                    ns, _ = R.dump(g, f"new_state-{il}")
                    ns = ns.reshape(st.shape).transpose(0, 1, 3, 2)
                    for k, s in enumerate(gr["seqs"]):
                        self.close("new_state (written back) [B]", il, g, ns[k],
                                   self.seq_rows(gr, s)["layers"][il]["gdn"]["state"])
                else:
                    self.assertFalse(R.has(g, f"new_state-{il}"), "n_rs_seq > 0 writes snapshots, not new_state")
            for il in ple:
                a, ne = R.dump(g, f"conv_state_at-{il}", 0)
                hist = a.reshape(nseq, ne[1], ne[0]).transpose(0, 2, 1)
                w = Q.layer_weights(self.W, il)
                for k, (s, p0, r0, lf) in enumerate(self.first_rows(gr)):
                    Pb = self.seq_rows(gr, s)["layers"][il]["ple"]
                    self.close("conv_state_at (PLE conv history) [B]", il, g, hist[k], Pb["hist_in"])
                    if gr["diverged"][s] or not gr["kept"][r0]:
                        continue
                    Pa = self.A[lf]["layers"][il]["ple"]
                    normed = Q.grouped_rms_norm(Pa["gated"], w["ple_norm_conv.weight"], hp["eps"]).reshape(
                        Pa["gated"].shape[0], -1)
                    Kh = hist.shape[1]
                    want = np.vstack([np.zeros((Kh, normed.shape[1])), normed[:p0]])[-Kh:]
                    self.close("conv_state_at (PLE conv history) [A]", il, g, hist[k], want)
        if len(self.PLAN.get("diverge", {}).get(PIN, {})) < R.man["n_seq_max"]:   # some sequence never diverges
            self.assertGreater(n_ok, 0)

    def test_gdn_and_ple_rows(self):
        R = self.R
        for gr in R.graphs:
            g = gr["g"]
            for il in self.LIN:
                for name, key in (("conv_output_raw", "conv_raw"), ("conv_output_silu", "conv"),
                                  ("attn_output", "scan"), ("final_output", "final"), ("linear_attn_out", "out")):
                    self.both(name, il, gr, R.dump(g, f"{name}-{il}")[0],
                              lambda b, k=key, i=il: np.asarray(b["layers"][i]["gdn"][k]).reshape(
                                  len(b["pos"]), -1),
                              lambda a, k=key, i=il: np.asarray(a["layers"][i]["gdn"][k]).reshape(
                                  a["logits"].shape[0], -1))
            for il in self.hp["ple_layers"]:
                for name, key in (("ple_gate", "gate"), ("ple_gated_value", "gated"), ("ple_conv_out", "conv_out")):
                    self.both(name, il, gr, R.dump(g, f"{name}-{il}")[0],
                              lambda b, k=key, i=il: b["layers"][i]["ple"][k],
                              lambda a, k=key, i=il: a["layers"][i]["ple"][k])
            self.both("ple_embd = table[rows] (exact)", -1, gr, R.dump(g, "ple_embd")[0], lambda b: b["ple_embd"],
                      lambda a: a["ple_embd"], tol=1e-300)

    def test_attention_cells_and_no_leakage(self):
        """q/k/v at absolute positions, the attention output, and per row: kq_soft_max is exactly 0 on every cell but
        the visible cells of the row's own sequence (QSA: the selected ones; dense: every position <= the row's) and
        equals (B)'s softmax there; the QSA selection maps to (B)'s (and (A)'s) positions."""
        R, hp = self.R, self.hp
        H, Hkv, D = hp["n_head"], hp["n_head_kv"], hp["head_dim"]
        for gr in R.graphs:
            g, n = gr["g"], len(gr["rows"])
            for il in self.ATT:
                for name, key, k in (("Qcur", "q", 0), ("Kcur", "k", 1), ("Vcur", "v", 0), ("attn_pregate", "pregate", 0),
                                     ("attn_output", "out", 0)):
                    self.both(name, il, gr, R.dump(g, f"{name}-{il}", k)[0],
                              lambda b, kk=key, i=il: np.asarray(b["layers"][i]["attn"][kk]).reshape(len(b["pos"]), -1),
                              lambda a, kk=key, i=il: np.asarray(a["layers"][i]["attn"][kk]).reshape(
                                  a["logits"].shape[0], -1))
                sm, ne = R.dump(g, f"kq_soft_max-{il}")
                self.assertEqual(ne[0], gr["n_kv"], f"n_kv g{g}")
                n_str = ne[3]
                sm = sm.reshape(n_str, H, n // n_str, ne[0])            # [stream][head][row][cell]
                sel = self.selected_positions(gr, il) if il in self.QSA else None
                for r, (s, p, _) in enumerate(gr["rows"]):
                    k = gr["seqs"].index(s)
                    b = self.seq_rows(gr, s)
                    st, rr = (0, r) if n_str == 1 else (k, r - k * gr["nst"])
                    cells = gr["cells"][R.M.kv.stream(s)]
                    pos_cell = {c_pos: c for c, (c_pos, seqs) in enumerate(cells) if s in seqs}
                    vis = sel[r] if sel is not None else list(range(p + 1))
                    if sel is not None:
                        self.assertEqual(vis, b["layers"][il]["attn"]["cells"][r - k * gr["nst"]],
                                         f"QSA cells L{il} seq {s} pos {p} g{g}")
                        if self.a_ok(gr, r):
                            self.assertEqual(vis, self.A[gr["life"][r]]["layers"][il]["attn"]["cells"][p],
                                             f"QSA cells (A) L{il} seq {s} pos {p} g{g}")
                    vc = [pos_cell[q] for q in vis]
                    got = sm[st, :, rr, :]
                    mask = np.ones(ne[0], bool)
                    mask[vc] = False
                    self.assertFalse(got[:, mask].any(), f"L{il} seq {s} pos {p} g{g}: probability on a cell that "
                                                         f"is not its own visible cell")
                    q = np.asarray(b["layers"][il]["attn"]["q"])[r - k * gr["nst"]]
                    K = gr["kcache"][s][il]
                    want = np.empty((H, len(vis)))
                    for h in range(H):
                        sc = K[vis, h // (H // Hkv)] @ q[h] / np.sqrt(D)
                        e = np.exp(sc - sc.max())
                        want[h] = e / e.sum()
                    self.close("kq_soft_max (visible cells) [B]", il, g, got[:, vc], want)

    def test_qsa_pool_layout(self):
        """The k-pool lists of the shared indexer cache: the pools a graph pools (indexer_pool_k_new, padded to
        n_new_g with pin-dependent padding cells), every pool of every sequence in layout order (indexer_k), and the
        scores of every row over the pools of its own sequence only (indexer_score)."""
        R, hp = self.R, self.hp
        d, kp = hp["idx_dim"], hp["kpool"]
        self.assertEqual(self.not_ties, [], "QSA selections that are no valid top-k by the oracle's scores")
        for gr in R.graphs:
            g, n = gr["g"], len(gr["rows"])
            for il in self.QSA:
                w = Q.layer_weights(self.W, il)
                self.both("indexer_k_raw", il, gr, R.dump(g, f"indexer_k_raw-{il}")[0],
                          lambda b, i=il: b["layers"][i]["attn"]["k_raw"], lambda a, i=il: a["layers"][i]["attn"]["k_raw"])
                self.both("indexer_q", il, gr, R.dump(g, f"indexer_q-{il}")[0],
                          lambda b, i=il: b["layers"][i]["attn"]["iq"], lambda a, i=il: a["layers"][i]["attn"]["iq"])
                pools = gr["pools"]
                for s, k, mem in pools:
                    self.assertEqual([p for p, _ in mem], list(range(kp * k, kp * k + kp)), "pool positions")
                key = [gr["pooled"][il][s][k] for s, k, _ in pools]
                new, _ = R.dump(g, f"indexer_pool_k_new-{il}")
                new = new.reshape(-1, d)
                self.assertEqual(new.shape[0], gr["n_new_g"], f"n_new_g g{g}")
                if gr["new"]:
                    self.close("indexer_pool_k_new (pooled here) [B]", il, g, new[:len(gr["new"])],
                               np.stack([key[i] for i in gr["new"]]))
                raw = gr["idx_raw"][il].reshape(-1, d)
                for r, cell in zip(range(len(gr["new"]), new.shape[0]), gr["pads"]):
                    self.close("indexer_pool_k_new (padding rows)", il, g, new[r],
                               Q.rms_norm(raw[cell], hp["eps"], w["indexer.k_norm.weight"]))
                got_k, ne = R.dump(g, f"indexer_k-{il}")
                got_k = got_k.reshape(-1, d)
                self.assertEqual(got_k.shape[0], Q.kpool_pad(len(pools)), f"padded pool count g{g}")
                if pools:
                    self.close("indexer_k (every pool, layout order) [B]", il, g, got_k[:len(pools)], np.stack(key))
                sc = R.dump(g, f"indexer_score-{il}")[0].reshape(n, -1)
                self.assertFalse(np.isfinite(sc[:, len(pools):]).any(), f"padded pools visible g{g}")
                for r, (s, p, _) in enumerate(gr["rows"]):
                    k = gr["seqs"].index(s)
                    own = self.seq_rows(gr, s)["layers"][il]["attn"]["scores"][r - k * gr["nst"]]
                    vis = [i for i, (s2, kk, mem) in enumerate(pools) if s2 == s and mem[-1][0] <= p]
                    fin = list(np.where(np.isfinite(sc[r, :len(pools)]))[0])
                    self.assertEqual(fin, vis, f"pools visible to seq {s} pos {p} g{g}")
                    self.assertEqual(len(vis), int(np.isfinite(own).sum()))
                    if vis:
                        self.close("indexer_score (own visible pools) [B]", il, g, sc[r, vis], own[np.isfinite(own)])
        TABLE.append((self.RUN, f"qsa selection: tie replays = {self.ties}", 3, -1, 0.0, 0.0))

    def test_moe_head_and_logits(self):
        R = self.R
        k = self.hp["n_expert_used"]
        n_vocab = self.W["output.weight"].shape[0]
        lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_logit_rows, n_vocab).astype(np.float64)
        for gr in R.graphs:
            g, n = gr["g"], len(gr["rows"])
            for il in range(self.hp["n_layer"]):
                ids = np.sort(R.dump(g, f"ffn_moe_topk-{il}")[0].reshape(n, k).astype(int), 1)
                want = np.sort(self.rows(gr, lambda b, i=il: b["layers"][i]["moe"]["ids"]), 1)
                self.assertTrue(np.array_equal(ids, want), f"top-k L{il} g{g}")
                for r in range(n):
                    if self.a_ok(gr, r):
                        wa = np.sort(self.A[gr["life"][r]]["layers"][il]["moe"]["ids"][gr["rows"][r][1]])
                        self.assertTrue(np.array_equal(ids[r], wa), f"top-k (A) L{il} g{g} row {r}")
                for name, getter in (("ffn_out", lambda L: L["moe"]["out"]), ("hc_combine", lambda L: L["combine_attn"]),
                                     ("l_last", lambda L: L["combine_ffn"])):
                    self.both(name, il, gr, R.dump(g, f"{name}-{il}")[0],
                              lambda b, i=il, f=getter: f(b["layers"][i]), lambda a, i=il, f=getter: f(a["layers"][i]))
            self.both("result_norm", -1, gr, R.dump(g, "result_norm")[0], lambda b: b["result_norm"],
                      lambda a: a["result_norm"])
            self.both("result_output", -1, gr, R.dump(g, "result_output")[0], lambda b: b["logits"],
                      lambda a: a["logits"])
            rows = lo[gr["logit_rows"]]
            self.both("logits.bin (batch order)", -1, gr, rows, lambda b: b["logits"], lambda a: a["logits"])
            self.assertTrue(np.array_equal(rows.argmax(1), self.rows(gr, lambda b: b["logits"]).argmax(1)),
                            f"greedy tokens g{g}")

    def test_edge_behaviour(self):
        """What this run is for, at this pin (tools/tinygen/qwen4exp_multiseq_plans.py RUNS)."""
        R, run, plan = self.R, self.RUN, self.PLAN
        steps = plan["steps"]
        self.assertEqual(len(steps), len(R.man["calls"]))
        # every seq_rm result as the plan says for this pin (or for the N20 guard patch)
        want_ok = iter(expected_ok(plan))
        for i, (st, call) in enumerate(zip(steps, R.man["calls"])):
            if st[0] == "r":
                self.assertEqual(call["ok"], next(want_ok), f"{run} step {i} {st}: ok")
            if st[0] == "x":
                self.assertTrue(call["ok"])
        # divergence: exactly the expected sequences, from the expected call on, and (A) is off there
        first = {}
        for gr in R.graphs:
            for s in gr["seqs"]:
                if gr["diverged"][s] and s not in first:
                    first[s] = gr
        expect = expected_diverge(plan)
        self.assertEqual({s: gr["call"] for s, gr in first.items()}, expect, f"{run}: diverged sequences")
        for s, gr in first.items():
            k = gr["seqs"].index(s)
            stale = gr["rs_reads"][s]["stale"]
            self.assertTrue(stale, f"seq {s}: no stale row read in the first diverged graph")
            p0 = gr["rows"][k * gr["nst"]][1]
            lf = gr["life"][k * gr["nst"]]
            kept = self.A[lf]
            off = []
            for il, kind in stale:
                if kind != "state":
                    continue
                got = R.dump(gr["g"], f"state_predelta-{il}")[0].reshape(
                    len(gr["seqs"]), self.hp["ssm_v_heads"], self.hp["ssm_state"], self.hp["ssm_state"])[k]
                want = self.prefix_state(lf, il, p0)
                off.append(float(np.abs(got.transpose(0, 2, 1) - want).max() / np.abs(want).max()))
            self.assertTrue(off and min(off) > 100 * TOL, f"seq {s}: the stale state is not off: {off}")
            later = [(g2, r) for g2 in R.graphs for r, (s2, p, _) in enumerate(g2["rows"])
                     if s2 == s and g2["g"] >= gr["g"] and g2["kept"][r] and g2["life"][r] == lf]
            lo = np.fromfile(R.dir / "logits.bin", dtype=np.float32).reshape(R.n_logit_rows, -1)
            dev = max(float(np.abs(lo[g2["logit_rows"][r]] - kept["logits"][g2["rows"][r][1]]).max())
                      / float(np.abs(kept["logits"][g2["rows"][r][1]]).max()) for g2, r in later)
            TABLE.append((run, f"diverged seq {s}: state off by {min(off):.2e}, logits vs (A) up to", -1, gr["g"],
                          dev, dev))
            self.assertGreater(dev, 100 * TOL, f"seq {s}: llama.cpp's logits do not differ from (A)")
        if run.startswith("multi"):
            self.assertTrue(any(len(gr["seqs"]) == 3 for gr in R.graphs), "no ubatch with three sequences")
            orders = [gr["seqs"] for gr in R.graphs]
            if R.man["kv_unified"]:
                self.assertTrue(any(o != sorted(o) for o in orders), "no ubatch in non-ascending sequence order")
            else:
                self.assertTrue(all(o == list(range(o[0], o[0] + len(o))) for o in orders),
                                "one stream per sequence: every ubatch must hold consecutive ascending sequences")
            self.assertTrue(any(len(gr["rows"]) > 1 and len(set(gr["seqs"])) > 1 and gr["nst"] > 1
                                for gr in R.graphs))
        if run in ("cp_uni", "cp_str"):
            cps = [i for i, st in enumerate(steps) if st[0] == "c"]
            after_cp = [gr for gr in R.graphs if gr["call"] == cps[0] + 1]
            if PIN == "ec" and R.man["kv_unified"]:
                # shared cells: every pool re-pooled every ubatch while two sequences share cells (HI:728-733)
                sharing = [gr for gr in R.graphs if not gr["cache_safe"]]
                self.assertTrue(sharing)
                for gr in sharing:
                    self.assertEqual(gr["new"], list(range(len(gr["pools"]))))
                # the drop of the last sharing sequence stales every pool (HI:196-200): the next graph re-pools all
                x_last = [i for i, st in enumerate(steps) if st[0] == "x"][-1]
                gr = [gr for gr in R.graphs if gr["call"] > x_last][0]
                self.assertTrue(gr["cache_safe"])
                self.assertEqual(gr["new"], list(range(len(gr["pools"]))))
                nxt = [g2 for g2 in R.graphs if g2["call"] > gr["call"]][0]
                self.assertLess(len(nxt["new"]), len(nxt["pools"]), "pools read back from the cache afterwards")
            elif R.man["kv_unified"]:
                # 79e2e74: one pooled row per rep cell (HI:653-662@79e2e74); a copy's pools are marked under the
                # destination only (it alone is staled, HI:200-212@79e2e74), the source keeps its pooled rows
                self.assertTrue(all(gr["cache_safe"] for gr in R.graphs))
                for gr in R.graphs:
                    reps = [gr["pools"][i][2][-1][1] for i in gr["new"]]
                    self.assertEqual(len(reps), len(set(reps)), f"g{gr['g']}: a rep cell pooled twice")
                gr = after_cp[0]
                src_pools = [i for i, (s, _, _) in enumerate(gr["pools"]) if s == 0]
                dst_pools = [i for i, (s, _, _) in enumerate(gr["pools"]) if s == 1]
                shared = [i for i in src_pools if gr["pools"][i][2][-1][1] in
                          {gr["pools"][j][2][-1][1] for j in dst_pools}]
                self.assertTrue(shared)
                self.assertFalse(set(shared) & set(gr["new"]), "a source pool re-pooled after its copy")
                # two copies before a decode: both destinations stale, every pool of theirs shares its rep with the
                # other's, so only one of each pair is pooled
                gr = [gr for gr in R.graphs if gr["call"] == cps[-1] + 1][0]
                stale = [i for i, (s, _, _) in enumerate(gr["pools"]) if s in (1, 2)]
                self.assertEqual(len(gr["new"]), len(stale) // 2)
            else:
                self.assertTrue(all(gr["cache_safe"] for gr in R.graphs), "no shared cells across streams")
        if run.startswith("rb"):
            ex = [(gr["g"], e) for gr in R.graphs for e in gr["rs_extra"] if e[1][0] > 0]
            # one KV stream: call 5 (sequences 0 and 2) spans recurrent cells 0..2, so sequence 1's pending rollback
            # is applied by the extra-state copy (src/llama-graph.cpp:3546-3551, s_copy RC:1364-1381); one stream per
            # sequence splits that call into two ubatches of one sequence each, with no cell in between
            self.assertEqual(bool(ex), bool(R.man["kv_unified"]), f"pending rollbacks through the extra range: {ex}")
            multi_ub = [i for i, st in enumerate(steps) if st[0] == "d" and len(st[1]) == 1 and st[1][0][1] > 8]
            for i in multi_ub:
                gs = [gr for gr in R.graphs if gr["call"] == i]
                self.assertGreater(len(gs), 1)
                self.assertGreaterEqual(gs[-1]["nst"], R.man["n_rs_seq"] + 1, "the last ubatch keeps n_rs_seq + 1")
        if run == "cp_shared_rb":
            rb = [i for i, st in enumerate(steps) if st[0] == "r"][0]
            pos = R.man["calls"][rb]["pos_max"]
            # ec7630a: the shared recurrent cell moves back for both sequences (RC:207-213), so the other sequence's
            # seq_pos_max drops by 2 although its KV cells still hold 22, 23; 79e2e74 refuses (RC:202-205@79e2e74), and
            # so does ec7630a with the N20 guard patch
            self.assertEqual(pos, [21, 21] if PIN == "ec" and not GUARD else [23, 23])


def _class(run, plans=None, module=__name__):
    return unittest.skipUnless(DUMP and GGUF and GGUF_PY, SKIP)(
        type(f"Run_{run}", (_MultiSeqParity, unittest.TestCase), {"RUN": run, "PLANS": plans, "__module__": module}))


for _run in PLANS.RUNS:
    globals()[f"Run_{_run}"] = _class(_run)
del _run


def tearDownModule():
    if not TABLE:
        return
    worst = {}
    for run, what, il, g, mx, rel in TABLE:
        key = (run, what)
        if key not in worst or rel > worst[key][3]:
            worst[key] = (il, g, mx, rel)
    print("\nworst error per run and component over layers and graphs (rel = max|got-want| / max|want|):",
          file=sys.stderr)
    print(f"{'run':13s} {'component':62s} {'layer':>5s} {'graph':>5s} {'max abs':>10s} {'rel':>10s}", file=sys.stderr)
    for (run, what), (il, g, mx, rel) in worst.items():
        print(f"{run:13s} {what:62s} {il:5d} {g:5d} {mx:10.3e} {rel:10.3e}", file=sys.stderr)
    TABLE.clear()     # printed once: a module that reuses these classes (N20) prints its own runs


if __name__ == "__main__":
    unittest.main()
