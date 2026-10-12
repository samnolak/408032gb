"""N21: the mimo2 MTP-heads finding of N17 and its fix (patches/llama.cpp/0001-mtp-heads-slot-replay.patch), on the
dumps of tools/tinygen/run_parity_mimo2_n21.sh (env MIMO2_N21_DUMP = its <work>/n21, LLAMA_CPP_GGUF_PY).

The finding (docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md finding 2, docs/evidence/2026-10-11-n21-mimo-mtp-heads-
fix.md): common/speculative.cpp process() re-decodes every target call once per MTP head after
llama_memory_seq_rm(first position) (speculative.cpp:1597-1621 @79e2e74); the head layers share the MTP context's
cells and each head writes only its own layer. (a) A call that head 0 places across the end of the SWA ring is placed
by heads 1 .. H-1 from the lowest freed cell, so head 0's layer keeps K/V of other positions under those cells (mtp3,
N17's plan: rows 257..263 of head 0). (b) A call of several ubatches: head 0's later ubatches overwrite cells that the
first ubatch's window still needs, before heads 1 .. H-1 run that ubatch, so their first rows attend to an incomplete
window, and the final placement again leaves head 0 with foreign K/V (mtp3b). The patch places each ubatch once for all
heads: per chunk of n_ubatch tokens, head 0 after seq_rm from the chunk's first position, heads 1 .. H-1 in exactly head
0's cells (llama_memory_set_slot_replay).

Classes:
  MtpUpstream, MtpUpstreamB   the upstream order (replayed by tools/dump/dump_tensors.cpp DUMP_MTP=1, as N17) on mtp3 /
                              mtp3b against ref/mimo2.py (B: the cell model, which reproduces both manifestations; A:
                              the full-sequence block, on the rows that read only their own K/V and a full window);
  MtpFix, MtpFixB, MtpFixF16FA  the patched order (DUMP_MTP_FIX=1, needs the patched build; skipped otherwise) against
                              ref/mimo2.py mtp_process(fixed=True): no row reads foreign K/V or an incomplete window,
                              every row equals (A), and the rows the upstream run got wrong now match (A);
  SpecReplay                  llama.cpp's own common/speculative.cpp (DUMP_MTP_SPEC=1) against the replay of the same
                              order, bit for bit (target dumps, every MTP graph tensor before the output crop, the draft
                              tokens): at an unpatched pin upstream vs the N17 replay, on the patched build the fixed
                              code vs DUMP_MTP_FIX and LLAMA_SPEC_MTP_NO_SLOT_REPLAY=1 vs the upstream replay.
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
from ref.mtp_cells import MtpCells  # noqa: E402
from tests import test_mimo2_mtp_heads_parity as H  # noqa: E402

DUMP = os.environ.get("MIMO2_N21_DUMP")
GGUF_PY = H.GGUF_PY
SKIP = "set MIMO2_N21_DUMP, LLAMA_CPP_GGUF_PY: run tools/tinygen/run_parity_mimo2_n21.sh"
SKIP_UNPATCHED = "no patched build: the fixed runs need llama.cpp with patches/llama.cpp/0001-mtp-heads-slot-replay.patch"
# the tensors after the output crop of graph_mtp (src/models/mimo2.cpp:378 @79e2e74): the real process() marks no row
# as an output, the replay every row, so only these differ between the two in process graphs
POST_CROP = ("h_nextn", "mtp_shared_head_norm", "result_output")


def patched():
    f = pathlib.Path(DUMP or "", "n21.json")
    return bool(DUMP) and f.exists() and json.loads(f.read_text())["patched"]


def mtp_manifest(run):
    return json.loads(pathlib.Path(DUMP, run, "mtp", "manifest.json").read_text())


def cells_replay(man, H_, ub, size=256, n_swa=8, fixed=None):
    """ref/mtp_cells.py MtpCells driven by an MTP manifest (replay or spec records); returns the model, whose log has
    one entry per MTP graph: (kind, head, positions, foreign rows, cells, missing-window rows)."""
    fixed = bool(man.get("slot_replay")) if fixed is None else fixed
    m = MtpCells(size, n_swa, H_, ub, fixed)
    mc, i = man["calls"], 0
    while i < len(mc):
        c = mc[i]
        k = c["kind"]
        if k == "process":
            # H records with this position and length: a whole call (upstream) or one chunk of n_ubatch tokens (fixed;
            # the patched process() runs a call chunk by chunk, so a chunk is processed exactly like a call of its own)
            grp = mc[i:i + H_]
            assert [(r["kind"], r["head"], r["pos0"], r["n"]) for r in grp] == \
                [("process", h, c["pos0"], c["n"]) for h in range(H_)], grp
            m.process(c["pos0"], c["n"])
            i += H_
        elif k == "spec_process":
            m.process(c["pos0"], c["n"])
            i += 1
        elif k == "draft":
            j = i + 1
            while j < len(mc) and mc[j]["kind"] == "draft" and len(mc[j]["tokens"]) == j - i + 1:
                j += 1
            m.draft(c["pos0"], j - i)
            i = j
        elif k == "spec_draft":
            m.draft(c["pos0"], c["n"])
            i += 1
        elif k in ("rollback", "seq_rm"):
            m.accept(c["pos0"])
            i += 1
        else:
            raise AssertionError(k)
    return m


class _Root:
    ROOT = DUMP
    MODEL = "n17.gguf"


class _Upstream(_Root, H._Heads):
    def test_cell_model_agrees(self):
        """ref/mtp_cells.py (placement only) flags the same rows as the cell oracle (B) of ref/mimo2.py, graph by
        graph: its cells, foreign K/V and incomplete windows."""
        m = cells_replay(self.man, self.H, self.T.man["n_ubatch"])
        self.assertEqual(len(m.log), len(self.graphs))
        for e, gr in zip(m.log, self.graphs):
            kind, head, pos, f, cells, miss = e
            b = self.B[gr["g"]]
            self.assertEqual((kind, head, pos), (gr["kind"], gr["head"], gr["pos"]), gr["g"])
            self.assertEqual(cells, b["slots"]["swa"], gr["g"])
            self.assertEqual(list(f), list(b["attn"]["foreign"]), gr["g"])
            self.assertEqual(list(miss), list(b["attn"]["missing"]), gr["g"])


class MtpUpstream(_Upstream, unittest.TestCase):
    """N17's run mtp3 (same model, tokens and plan) on the build under test: N17's checks, the head-0 rows 257..263."""
    RUN = "mtp3"


class MtpUpstreamB(_Upstream, unittest.TestCase):
    """mtp3b: 40, 4 x 1, a call of 260 tokens (ubatches of 64 at 44..107, .., 236..299, 300..303), 3 x 1, d3, 4 (1
    wrong), r1, 2 x 1, a final 3-step draft. The 256-cell ring and the 8-position window: head 0's fourth ubatch
    (236..299) wraps the ring and takes the cells of positions 0..43 (masked for it); seq_rm then frees cell 0 first, so
    heads 1 and 2 place the whole call from cell 0 (44..299 in cells 0..255) and run their first ubatch without
    positions 37..43 (rows 44..50 attend to an incomplete window); the cells follow head 2, and head 0's layer holds K/V
    of other positions under almost every cell, which its rows 304..310 (process and draft) read (ref/mtp_cells.py)."""
    RUN = "mtp3b"
    MIN_DRAFTS = 2

    def test_head0_reads_foreign_kv(self):
        nl = self.hp["n_layer"]
        foreign = [(gr, r) for gr in self.graphs for r in np.nonzero(self.B[gr["g"]]["attn"]["foreign"])[0]]
        missing = [(gr, r) for gr in self.graphs for r in np.nonzero(self.B[gr["g"]]["attn"]["missing"])[0]]
        self.assertEqual({gr["head"] for gr, _ in foreign}, {0})
        self.assertEqual(sorted({gr["pos"][r] for gr, r in foreign}), list(range(304, 311)))
        self.assertEqual(len(foreign), 9)
        self.assertEqual({(gr["kind"], gr["head"]) for gr, _ in missing}, {("process", 1), ("process", 2)})
        self.assertEqual(sorted({gr["pos"][r] for gr, r in missing}), list(range(44, 51)))
        self.assertEqual(len(missing), 14)
        worst = 0.0
        for gr, r in foreign + missing:
            if gr["kind"] != "process" or not gr["kept"][r]:
                continue
            il = nl + gr["head"]
            got = self.dump(gr["g"], f"kqv_out-{il}")[0].reshape(gr["n"], -1)[r]
            want = self.A[gr["head"]]["attn"]["kqv"][gr["pos"][r]]
            worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
        self.assertGreater(worst, 1e-2)
        self.TABLE.append((self.RUN, "kqv_out vs (A), rows with foreign K/V or an incomplete window (expected to differ)",
                           -1, -1, 0.0, worst))

    def test_run_exercises_its_mechanisms(self):
        self.assertEqual(self.H, 3)
        self.assertEqual(self.mcache["swa"].size, 256)
        big = [gr for gr in self.graphs if gr["kind"] == "process" and gr.get("call_n") == 260]
        self.assertEqual(len(big), 15)                        # 5 ubatches x 3 heads
        self.assertEqual([(c["pos0"], c["n"]) for c in self.man["calls"] if c["kind"] == "rollback"], [(310, 1)])
        self.assertEqual([d["pos0"] for d in self.drafts], [307, 312])


class _FixedMixin(_Root):
    FIXED = True

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not patched():
            raise unittest.SkipTest(SKIP_UNPATCHED)
        super().setUpClass()

    def test_head0_reads_foreign_kv(self):
        """With the fix no row of any head reads foreign K/V or attends to an incomplete window (B's flags and the
        placement-only model agree), and the heads of a chunk write the same cells."""
        m = cells_replay(self.man, self.H, self.T.man["n_ubatch"])
        self.assertEqual(len(m.log), len(self.graphs))
        for e, gr in zip(m.log, self.graphs):
            b = self.B[gr["g"]]["attn"]
            self.assertFalse(np.asarray(b["foreign"]).any() or np.asarray(b["missing"]).any(), gr["g"])
            self.assertFalse(e[3].any() or e[5].any(), gr["g"])
            self.assertEqual(e[4], self.B[gr["g"]]["slots"]["swa"], gr["g"])
        per_chunk = {}
        for gr in self.graphs:
            if gr["kind"] == "process":
                per_chunk.setdefault((gr["rec"]["tok0"] - gr["head"] * gr["n"], gr["p0"]), {})[gr["head"]] = \
                    self.B[gr["g"]]["slots"]["swa"]
        for key, cells in per_chunk.items():
            self.assertEqual(sorted(cells), list(range(self.H)), key)
            self.assertEqual(len({json.dumps(c) for c in cells.values()}), 1, (key, cells))

    def test_run_exercises_its_mechanisms(self):
        self.assertEqual(self.H, 3)
        self.assertEqual(self.mcache["swa"].size, 256)
        self.assertTrue(any(gr["pos"] != self.B[gr["g"]]["slots"]["swa"] for gr in self.graphs), "no wrap")
        for head in range(self.H):
            self.assertEqual({gr["kind"] for gr in self.graphs if gr["head"] == head}, {"process", "draft"})

    def _upstream_rows_now_match_full(self, up_run, want_rows):
        """The rows the upstream run (same root, same plan) got wrong, by the cell model of its manifest: in this
        run's graph of the same kind, head and positions they equal (A) (kqv_out, and logits where the row is an
        output), while llama.cpp's upstream rows differ from (A) and from these."""
        up = mtp_manifest(up_run)
        mu = cells_replay(up, self.H, self.T.man["n_ubatch"])
        bad = [(g, k, h, pos, r) for g, (k, h, pos, f, c, mi) in enumerate(mu.log) for r in range(len(pos))
               if f[r] or mi[r]]
        self.assertEqual(sorted({(k, h, pos[r]) for _, k, h, pos, r in bad}), want_rows)
        mine = {(gr["kind"], gr["head"], tuple(gr["pos"])): gr for gr in self.graphs}
        up_dir = pathlib.Path(DUMP, up_run, "mtp")
        n_vocab = self.W["output.weight"].shape[0]
        up_lo = np.fromfile(up_dir / "logits.bin", dtype=np.float32).reshape(-1, n_vocab)
        lo, _, _, _ = self.files()
        ub = self.T.man["n_ubatch"]
        g2rec = {g: c for c in up["calls"] for g in range(*c["graphs"])}
        rel = lambda a, b: float(np.abs(np.asarray(a, np.float64) - b).max() / np.abs(b).max())   # noqa: E731
        n_rows = n_out = 0
        worst_fix = worst_up = worst_lo_fix = worst_lo_up = 0.0
        nl = self.hp["n_layer"]
        for g_up, kind, head, pos, r in bad:
            gr = mine[(kind, head, tuple(pos))]
            # (A) holds the final sequence and the target's h: kept process rows, and a draft's first step (head 0,
            # id_last at pos0 with the pending target row); a later step's rows carry draft tokens and MTP h rows
            self.assertTrue(kind == "process" or (head == 0 and len(pos) == 1), (kind, head, pos))
            if kind == "process" and not gr["kept"][r]:
                continue
            il = nl + head
            A = self.A[head]
            want = A["attn"]["kqv"][pos[r]]
            got = self.dump(gr["g"], f"kqv_out-{il}")[0].reshape(gr["n"], -1)[r]
            j = json.loads((up_dir / f"g{g_up:03d}" / f"kqv_out-{il}__0.json").read_text())
            upv = np.fromfile(up_dir / f"g{g_up:03d}" / f"kqv_out-{il}__0.bin", dtype=np.float32).reshape(
                j["ne"][::-1]).reshape(len(pos), -1)[r]
            worst_fix, worst_up = max(worst_fix, rel(got, want)), max(worst_up, rel(upv, want))
            n_rows += 1
            if r in gr["out_idx"]:
                rec = g2rec[g_up]
                b = (g_up - rec["graphs"][0]) * ub + r         # the row's batch index in the upstream call
                up_row = up_lo[rec["out0"] + list(rec["outputs"]).index(b)]
                my_row = lo[gr["out_rows"][list(gr["out_idx"]).index(r)]]
                worst_lo_fix = max(worst_lo_fix, rel(my_row, A["logits"][pos[r]]))
                worst_lo_up = max(worst_lo_up, rel(up_row, A["logits"][pos[r]]))
                n_out += 1
        self.assertGreater(n_rows, 0)
        self.assertGreater(n_out, 0)
        self.assertLess(worst_fix, self.TOL)
        self.assertLess(worst_lo_fix, self.TOL)
        self.assertGreater(worst_up, 1e-2)
        self.assertGreater(worst_lo_up, 1e-3)
        self.TABLE.append((self.RUN, f"{n_rows} rows upstream got wrong: kqv_out vs (A), fixed", -1, -1, 0.0, worst_fix))
        self.TABLE.append((self.RUN, f"{n_out} of them outputs: logits vs (A), fixed", -1, -1, 0.0, worst_lo_fix))
        self.TABLE.append((self.RUN, f"the same rows upstream ({up_run}): kqv_out vs (A) (expected to differ)", -1, -1,
                           0.0, worst_up))
        self.TABLE.append((self.RUN, f"the same rows upstream ({up_run}): logits vs (A) (expected to differ)", -1, -1,
                           0.0, worst_lo_up))
        print(f"\n{self.RUN}: {len(bad)} rows wrong upstream ({up_run}), {n_rows} compared with (A): fixed kqv_out "
              f"{worst_fix:.3e}, logits {worst_lo_fix:.3e} ({n_out} output rows); upstream kqv_out {worst_up:.3e}, "
              f"logits {worst_lo_up:.3e}", file=sys.stderr)


class MtpFix(_FixedMixin, H._Heads, unittest.TestCase):
    RUN = "fix/mtp3"

    def test_upstream_rows_now_match_full(self):
        self._upstream_rows_now_match_full("mtp3", self._n17_rows())

    @staticmethod
    def _n17_rows():
        # N17 finding 2: 11 rows of head 0 at positions 257..263 (process and draft graphs, mtp3)
        m = cells_replay(mtp_manifest("mtp3"), 3, 64)
        return sorted({(k, h, pos[r]) for k, h, pos, f, c, mi in m.log for r in range(len(pos)) if f[r] or mi[r]})

    def test_n17_rows(self):
        rows = self._n17_rows()
        self.assertEqual({h for _, h, _ in rows}, {0})
        self.assertEqual(sorted({p for _, _, p in rows}), list(range(257, 264)))
        m = cells_replay(mtp_manifest("mtp3"), 3, 64)
        self.assertEqual(sum(int(f.sum()) for _, _, _, f, _, _ in m.log), 11)


class MtpFixB(_FixedMixin, H._Heads, unittest.TestCase):
    RUN = "fix/mtp3b"
    MIN_DRAFTS = 2

    def test_upstream_rows_now_match_full(self):
        m = cells_replay(mtp_manifest("mtp3b"), 3, 64)
        want = sorted({(k, h, pos[r]) for k, h, pos, f, c, mi in m.log for r in range(len(pos)) if f[r] or mi[r]})
        self.assertEqual(sorted({p for _, _, p in want}), list(range(44, 51)) + list(range(304, 311)))
        self._upstream_rows_now_match_full("mtp3b", want)


class MtpFixF16FA(_FixedMixin, H._HeadsF16FA, unittest.TestCase):
    """The fixed order at the R1 default precision (-ctk f16 -ctv f16 -fa on)."""
    RUN = "fix/fa-mtp3"


class SpecReplay(unittest.TestCase):
    """llama.cpp's own common/speculative.cpp (DUMP_MTP_SPEC=1) computes bit for bit what the replay of the same order
    computes: the target's dumps, every MTP graph's tensors (the three after the output crop excepted in process
    graphs, where the real process() marks no output and the replay every row) and the draft tokens."""

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        cls.pairs = [("spec/mtp3", "mtp3"), ("spec/mtp3b", "mtp3b")]
        if patched():
            cls.pairs = [("spec/mtp3", "fix/mtp3"), ("spec/mtp3b", "fix/mtp3b"),
                         ("spec-off/mtp3", "mtp3"), ("spec-off/mtp3b", "mtp3b")]
        for a, b in cls.pairs:
            for r in (a, b):
                if not pathlib.Path(DUMP, r, "mtp", "manifest.json").exists():
                    raise unittest.SkipTest(f"no run {r} under {DUMP}")

    def test_target_identical(self):
        for a, b in self.pairs:
            ra, rb = pathlib.Path(DUMP, a), pathlib.Path(DUMP, b)
            fa = sorted(p.relative_to(ra) for p in ra.rglob("*") if p.is_file() and p.relative_to(ra).parts[0] != "mtp")
            fb = sorted(p.relative_to(rb) for p in rb.rglob("*") if p.is_file() and p.relative_to(rb).parts[0] != "mtp")
            self.assertEqual(fa, fb, (a, b))
            self.assertGreater(len(fa), 1000)
            for p in fa:
                if p.name == "tokens.txt":
                    continue
                self.assertEqual(pathlib.Path(DUMP, a, p).read_bytes(), pathlib.Path(DUMP, b, p).read_bytes(), (a, b, p))

    def test_mtp_graphs_identical(self):
        for a, b in self.pairs:
            ma, mb = mtp_manifest(a), mtp_manifest(b)
            self.assertTrue(ma.get("spec"))
            self.assertEqual(len(ma["graphs"]), len(mb["graphs"]), (a, b))
            kinds = {}
            for c in mb["calls"]:
                for g in range(*c["graphs"]):
                    kinds[g] = c["kind"]
            n_same = n_crop = 0
            for g in range(len(mb["graphs"])):
                da, db = pathlib.Path(DUMP, a, "mtp", f"g{g:03d}"), pathlib.Path(DUMP, b, "mtp", f"g{g:03d}")
                names = sorted(p.name for p in db.glob("*.bin"))
                self.assertEqual(names, sorted(p.name for p in da.glob("*.bin")), (a, b, g))
                self.assertGreater(len(names), 20)
                for n in names:
                    if (da / n).read_bytes() == (db / n).read_bytes():
                        n_same += 1
                        continue
                    self.assertEqual(kinds[g], "process", (a, b, g, n))
                    self.assertIn(n.rsplit("__", 1)[0], POST_CROP, (a, b, g, n))
                    # no output row in the real process(), every row in the replay
                    ne_a = json.loads((da / n.replace(".bin", ".json")).read_text())["ne"]
                    ne_b = json.loads((db / n.replace(".bin", ".json")).read_text())["ne"]
                    self.assertEqual((ne_a[0], ne_a[1]), (ne_b[0], 0), (a, b, g, n))
                    self.assertGreater(ne_b[1], 0, (a, b, g, n))
                    n_crop += 1
            self.assertEqual(n_crop, len(POST_CROP) * sum(1 for g in kinds.values() if g == "process"), (a, b))
            print(f"\n{a} vs {b}: {len(mb['graphs'])} MTP graphs, {n_same} tensors bit-identical, {n_crop} post-crop "
                  f"tensors of process graphs with 0 rows (real code) instead of every row (replay)", file=sys.stderr)

    def test_draft_tokens_identical(self):
        n_vocab = None
        for a, b in self.pairs:
            ma, mb = mtp_manifest(a), mtp_manifest(b)
            spec = [c["tokens"] for c in ma["calls"] if c["kind"] == "spec_draft"]
            lo = np.fromfile(pathlib.Path(DUMP, b, "mtp", "logits.bin"), dtype=np.float32)
            if n_vocab is None:
                n_vocab = lo.size // sum(len(c.get("outputs", [])) for c in mb["calls"])
            lo = lo.reshape(-1, n_vocab)
            rep, cur = [], None
            for c in mb["calls"]:
                if c["kind"] == "draft":
                    cur = c["tokens"] + [int(np.argmax(lo[c["out0"] + len(c["outputs"]) - 1]))]
                elif cur is not None:
                    rep.append(cur)
                    cur = None
            if cur is not None:
                rep.append(cur)
            self.assertEqual(spec, rep, (a, b))
            self.assertEqual(len(spec), {"mtp3": 5, "mtp3b": 2}[b.split("/")[-1]])

    def test_cell_model_counts(self):
        """The placement-only model of the spec run's calls yields as many graphs as llama.cpp computed."""
        for a, _ in self.pairs:
            ma = mtp_manifest(a)
            self.assertEqual(len(cells_replay(ma, 3, 64, fixed=a.startswith("spec/") and patched()).log),
                             len(ma["graphs"]), a)

    def test_target_independent_of_the_mtp_order(self):
        """On the patched build: the target's dumps (every graph, logits, h_nextn) of the upstream and the fixed replay
        of the same plan are identical: the order of the MTP calls never touches the target context."""
        if not patched():
            self.skipTest(SKIP_UNPATCHED)
        for plan in ("mtp3", "mtp3b"):
            ra, rb = pathlib.Path(DUMP, plan), pathlib.Path(DUMP, "fix", plan)
            fa = sorted(p.relative_to(ra) for p in ra.rglob("*") if p.is_file() and p.relative_to(ra).parts[0] != "mtp")
            fb = sorted(p.relative_to(rb) for p in rb.rglob("*") if p.is_file() and p.relative_to(rb).parts[0] != "mtp")
            self.assertEqual(fa, fb, plan)
            self.assertGreater(len(fa), 1000)
            for f in fa:
                self.assertEqual((ra / f).read_bytes(), (rb / f).read_bytes(), (plan, f))

    def test_fix_changes_drafts(self):
        """On the patched build: the fixed and the upstream code draft different tokens exactly where the upstream head
        0 read foreign K/V (the draft at 259 in mtp3, at 307 in mtp3b); every other draft is the same."""
        if not patched():
            self.skipTest(SKIP_UNPATCHED)
        for plan, where in (("mtp3", 259), ("mtp3b", 307)):
            on = {c["pos0"]: c["tokens"] for c in mtp_manifest(f"spec/{plan}")["calls"] if c["kind"] == "spec_draft"}
            off = {c["pos0"]: c["tokens"] for c in mtp_manifest(f"spec-off/{plan}")["calls"] if c["kind"] == "spec_draft"}
            self.assertEqual(sorted(on), sorted(off))
            self.assertEqual([p for p in on if on[p] != off[p]], [where], (plan, on, off))
            print(f"\n{plan}: draft at {where}: fixed {on[where]}, upstream {off[where]}", file=sys.stderr)


class ReplayRefusal(unittest.TestCase):
    """The fallback of patches/llama.cpp/0001 when a replay cannot be honoured (run fix/probe, DUMP_MTP_REPLAY_PROBE=1,
    tools/dump/dump_tensors.cpp header): after 40, 1 (and the final draft, removed), head 0 runs one token at P = 41
    (a) by a plain search, (b) by replay of (a)'s cell after seq_rm(41), (c) at 42 with replay still on, where (a)'s cell
    holds 41, so llama_kv_cache::prepare must refuse the replay and search, (d) at 42 by a plain search after seq_rm(42).
    A replay onto the taken cell would make apply_ubatch purge position 41 and every one before it."""

    @classmethod
    def setUpClass(cls):
        if not (DUMP and GGUF_PY):
            raise unittest.SkipTest(SKIP)
        if not patched():
            raise unittest.SkipTest(SKIP_UNPATCHED)
        cls.dir = pathlib.Path(DUMP, "fix", "probe", "mtp")
        cls.man = mtp_manifest("fix/probe")
        cls.probes = [c for c in cls.man["calls"] if c["kind"] == "probe"]
        cls.log = pathlib.Path(DUMP, "fix_probe.log").read_text()

    def graph(self, k):
        g0, g1 = self.probes[k]["graphs"]
        self.assertEqual(g1 - g0, 1)
        return self.dir / f"g{g0:03d}"

    def same(self, i, j):
        a, b = self.graph(i), self.graph(j)
        names = sorted(p.name for p in a.iterdir())
        self.assertEqual(names, sorted(p.name for p in b.iterdir()))
        self.assertGreater(len(names), 40)
        for n in names:
            self.assertEqual((a / n).read_bytes(), (b / n).read_bytes(), (i, j, n))
        return len(names)

    def test_the_probe_calls(self):
        self.assertEqual([(c["pos0"], c["n"], c["head"]) for c in self.probes], [(41, 1, 0), (41, 1, 0), (42, 1, 0),
                                                                                  (42, 1, 0)])

    def test_replay_of_a_freed_cell_equals_the_search(self):
        """(b) = (a) bit for bit (every dumped tensor of the graph)."""
        self.same(0, 1)

    def test_refused_replay_falls_back_to_the_search(self):
        """(c) = (d) bit for bit; llama.cpp warned once per cache of the iSWA pair (the base and the SWA cache both
        refuse); (c) attends to its whole window (8 positions, 35..42)."""
        n = self.same(2, 3)
        self.assertEqual(self.log.count("prepare: slot replay refused"), 2, self.log[-3000:])
        g = self.graph(2)
        j = json.loads((g / "kq_soft_max-6__0.json").read_text())
        ne = j["ne"]                                              # [n_kv, n_tokens, n_head, 1]
        sm = np.fromfile(g / "kq_soft_max-6__0.bin", dtype=np.float32).reshape(ne[2], ne[1], ne[0])
        self.assertEqual(sm.shape[:2], (4, 1))                    # 4 query heads, one query row
        self.assertEqual({int((sm[h] > 0).sum()) for h in range(sm.shape[0])}, {8})
        print(f"\nreplay probe: (a) = (b) and (c) = (d) bit for bit ({n} tensors each), refused in both caches, (c) "
              f"sees 8 cells",
              file=sys.stderr)


def tearDownModule():
    H.tearDownModule()


if __name__ == "__main__":
    unittest.main()
