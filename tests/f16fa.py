"""Shared part of tests/test_qwen4exp_f16fa_parity.py and tests/test_glm5next_f16fa_parity.py (N10, gap 3 of
docs/reassessment/CORRECTNESS.md): the N6 / N9 dump-parity test classes re-run on dumps made with the R1 default
precision (-ctk f16 -ctv f16 -fa on, tools/tinygen/run_parity_f16fa.sh), against the precision-aware oracles
(ref/fattn.py precision(...) passed to ref/qwen4exp.py and ref/glm5next.py), plus checks only an f16 / flash-attention
run has: the cache contents, the flash-attention op on llama.cpp's own inputs, the CPU path taken, and the count of
discrete choices (QSA/DSA selection, MoE experts, greedy token) that differ from the plain f32 oracle.

Not a test module (no test_ prefix): the two modules import it.
Env: PARITY_KV (f16 | f32, default f16) and PARITY_FA (on | off, default on): the flags the dumps were made with.
"""
import os
import sys

import numpy as np

from ref import fattn as F

KV = os.environ.get("PARITY_KV", "f16")
FA = os.environ.get("PARITY_FA", "on") == "on"
PREC = F.precision(KV, FA)
U16 = F.U16


def ulp_diff(got, want):
    """|got - want| in units of the binary16 spacing at want."""
    return np.abs(np.asarray(got, dtype=np.float64) - want) / F.ulp16(want)


def store_check(t, what, il, g, cache_rows, pre_store):
    """The cache store itself, on llama.cpp's own values: the rows the graph wrote equal its dumped pre-store f32
    values rounded to binary16, round to nearest even (f16 cache), or copied (f32 cache), element for element."""
    want = F.f16(pre_store) if KV == "f16" else np.asarray(pre_store, dtype=np.float64)
    got = np.asarray(cache_rows, dtype=np.float64).reshape(want.shape)
    n_bad = int((got != want).sum())
    t.assertEqual(n_bad, 0, f"{what} L{il} g{g}: {n_bad} of {want.size} cached elements differ from "
                            f"{'f16(' if KV == 'f16' else '('}the dumped pre-store value)")
    return want.size


def cache_check(t, what, il, g, got, stored, stats, tol_pre):
    """Cache rows llama.cpp holds (got) against the rows the oracle stored (stored: f16 values with an f16 cache).
    Every element within one binary16 spacing plus tol_pre * max|stored|, the tolerance the same quantity has before
    the store (the oracle's pre-store value may differ from llama.cpp's by that much, e.g. Kcur, kv_cmpr, and then
    round to the neighbouring binary16 value); the exactly equal elements are counted per quantity in stats
    ({what: [equal, total]}), cache_share() asserts their share. f32 cache: tol_pre normwise."""
    stored = np.asarray(stored, dtype=np.float64).reshape(np.shape(got))
    if KV == "f16":
        d = np.abs(got - stored)
        st = stats.setdefault(what, [0, 0])
        st[0] += int((d == 0).sum())
        st[1] += d.size
        bound = F.ulp16(stored) + tol_pre * float(np.abs(stored).max())
        t.assertTrue((d <= bound).all(), f"{what} L{il} g{g}: {float((d / bound).max()):.2f} x (1 spacing + "
                                         f"{tol_pre:.1e} max|stored|)")
        t.close(f"{what} (cache rows vs oracle)", il, g, got, stored)
    else:
        t.close(f"{what} (cache rows vs oracle)", il, g, got, stored, tol_pre)


def cache_share(t, stats):
    """Per cached quantity, the share of elements equal to what the oracle stored: >= CACHE_AGREE (f16 cache)."""
    if KV != "f16" or not stats:
        return
    eq, n = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
    low = min(stats, key=lambda w: stats[w][0] / stats[w][1])
    t.TABLE.append((t.RUN, f"cache rows equal to the oracle's stored rows: {eq}/{n}; lowest {low} "
                           f"{stats[low][0]}/{stats[low][1]}", -1, -1, 0.0, 1.0 - eq / n, ()))
    for what, (e, m) in stats.items():
        t.assertGreaterEqual(e / m, t.CACHE_AGREE, f"{what}: the oracle stores other values ({e} of {m} equal)")


class Recorder:
    """Collects (run, component, layer, graph, max abs, rel) rows of one oracle precision."""

    def __init__(self):
        self.rows = []

    def worst(self):
        w = {}
        for run, what, il, g, mx, rel in self.rows:
            if (run, what) not in w or rel > w[(run, what)][3]:
                w[(run, what)] = (il, g, mx, rel)
        return w


class F16Mixin:
    """Mixed into a subclass of the N6/N9 _RunParity. Class attributes set by the module: ROOT, GGUF_PATH, SKIP,
    TOL (f16 path tolerance, see the module), TABLE (the module's list), F32 (a Recorder for the plain f32 oracle),
    MISMATCH, BASE_TESTS (the names of the N6/N9 test bodies); the module implements attention_pairs() and
    discrete()."""
    PREC = PREC
    F32 = None
    MISMATCH = None          # {(run, kind): count}: discrete choices and recorded checks that differ (printed)
    BASE_TESTS = ()
    # share of the cached elements of each quantity the oracle stores exactly as llama.cpp does (f16 cache): an
    # element differs only when the two pre-store values straddle a rounding boundary; measured >= 0.85 overall
    # (docs/evidence/2026-10-10-quant-fa-parity.md); an oracle that does not round its store matches almost none
    CACHE_AGREE = 0.5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # the plain f32 oracle (prec None) on the same dumps: compared, recorded, never asserted
        prec_saved, A, B = cls.PREC, cls.A, cls.B
        cls.PREC = None
        try:
            cls._oracles({})
            cls.A32, cls.B32 = cls.A, cls.B
        finally:
            cls.PREC, cls.A, cls.B = prec_saved, A, B

    # ---- recording mode: the base test bodies run against the f32 oracle without asserting ----
    def _recording(self):
        return getattr(self, "_rec", False)

    def _miss(self, msg):
        """A check the plain oracle fails while recording: counted per kind (layer and graph cut off)."""
        kind = "assertion: " + (msg or "other").split(" L")[0].split(" g")[0].strip()
        key = (self.RUN, kind)
        self.MISMATCH[key] = self.MISMATCH.get(key, 0) + 1

    def assertEqual(self, first, second, msg=None):
        if self._recording():
            try:
                same = bool(first == second)
            except Exception:
                same = False
            if not same:
                self._miss(msg)
            return
        super().assertEqual(first, second, msg)

    def assertTrue(self, expr, msg=None):
        if self._recording():
            if not expr:
                self._miss(msg)
            return
        super().assertTrue(expr, msg)

    def assertFalse(self, expr, msg=None):
        if self._recording():
            if expr:
                self._miss(msg)
            return
        super().assertFalse(expr, msg)

    def close(self, what, il, g, got, want, tol=None):
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        if self._recording():
            if got.size != want.size:
                self._miss(f"size {what}")
                return
            d = np.abs(got.reshape(want.shape) - want)
            mx = float(d.max()) if d.size else 0.0
            rel = mx / (float(np.abs(want).max()) + 1e-30) if d.size else 0.0
            self.F32.rows.append((self.RUN, what, il, g, mx, rel))
            return
        if tol is None:                          # explicit tolerances (exact 1e-300, op-level ones) are kept
            tol = self.tol_for(what, il)
        super().close(what, il, g, got, want, tol)

    def tol_for(self, what, il):
        return self.TOL

    def attention_pairs(self, which):
        """(layer, graph, dumped attention output, oracle rows) of every graph; which = "aware" | "plain"."""
        raise NotImplementedError

    def discrete(self, B):
        """{kind: number of rows where the oracle's discrete choice differs from llama.cpp's} for the oracle B."""
        raise NotImplementedError

    def test_attention_output_vs_plain_oracle(self):
        """What the precision model buys, on the attention output (the first value the f16 cache and flash attention
        change): the precision-aware oracle must be at least twice as close to llama.cpp as the plain f32 oracle in
        the 2-norm. The plain oracle misses the deterministic roundings (cache store, f16 Q of one_chunk); the aware
        one is left with the noise of the f16 accumulator, whose rounding positions any f32-level input difference
        reshuffles (measured ratio 3.3 to 1e3, docs/evidence/2026-10-10-quant-fa-parity.md). Recorded: the share of
        elements within half a binary16 spacing of each oracle."""
        res = {}
        for which in ("aware", "plain"):
            d2 = n2 = 0.0
            n_ok = n = 0
            for _il, _g, got, want in self.attention_pairs(which):
                d2 += float(((got - want) ** 2).sum())
                n2 += float((want ** 2).sum())
                n_ok += int((ulp_diff(got, want) <= 0.5).sum())
                n += got.size
            res[which] = (np.sqrt(d2 / max(n2, 1e-300)), n_ok / max(n, 1), n)
        for which, rec in (("aware", None), ("plain", self.F32)):
            e2, f, n = res[which]
            row = (self.RUN, f"attention output: 2-norm rel {e2:.3e}, within 0.5 spacing {f:.4f} of {n} ({which})",
                   -1, -1, 0.0, e2)
            (self.TABLE.append(row + ((),)) if rec is None else rec.rows.append(row))
        # nothing is rounded with an f32 cache when no graph ran flash attention (the ec7630a glm5-next gather path)
        if KV == "f16" or any(self.graph_mode(gr) != "exact" for gr in self.R.graphs):
            self.assertLess(2 * res["aware"][0], res["plain"][0],
                            f"{self.RUN}: the plain f32 oracle is about as close: precision model untested")

    def test_plain_f32_oracle(self):
        """The N6/N9 checks against the plain f32 oracle on these dumps: recorded and printed at the end, never
        asserted. Asserted: the precision-aware oracle makes llama.cpp's discrete choices (selection, experts, greedy
        token) everywhere apart from its counted tie replays; the plain oracle's differences are counted."""
        self.run_against_f32(self.BASE_TESTS)
        for kind, n in self.discrete(self.B32).items():
            self.MISMATCH[(self.RUN, f"{kind} (plain f32 oracle)")] = n
        aware = self.discrete(self.B)
        for kind, n in aware.items():
            self.MISMATCH[(self.RUN, f"{kind} (precision-aware oracle)")] = n
        self.assertEqual(sum(aware.values()), 0, f"{self.RUN}: the precision-aware oracle chooses differently {aware}")

    def run_against_f32(self, names):
        """Run the given base test bodies with the plain f32 oracle (A32 / B32) in recording mode."""
        self._rec = True
        self.A, self.B = self.A32, self.B32
        self.TABLE = []          # rows the base bodies append directly (tie counts) belong to the aware run
        try:
            for name in names:
                try:
                    getattr(super(), name)()
                except Exception as e:          # a body that cannot even run against the f32 oracle
                    self._miss(f"{name} raised {type(e).__name__}")
        finally:
            self._rec = False
            del self.A, self.B, self.TABLE


def print_tables(title, table, rec, mismatch, commit, stream=sys.stderr):
    """The worst error per run and component against the precision-aware oracle (asserted) and the plain f32 oracle
    (recorded), side by side, then the worst per layer, then the discrete choices the f32 oracle makes differently."""
    summary = [r for r in table if r[3] == -1] + [r + ((),) for r in rec.rows if r[3] == -1]
    table = [r for r in table if r[3] != -1]
    rec32 = Recorder()
    rec32.rows = [r for r in rec.rows if r[3] != -1]
    worst = {}
    for run, what, il, g, mx, rel, _ in table:
        if (run, what) not in worst or rel > worst[(run, what)][3]:
            worst[(run, what)] = (il, g, mx, rel)
    w32 = rec32.worst()
    flags = f"-ctk {KV} -ctv {KV} -fa {'on' if FA else 'off'}"
    print(f"\n{title}, llama.cpp {commit[:7]}, {flags}: worst error per run and component (rel = max|got-want| / "
          f"max|want|) against the precision-aware oracle (asserted) and the plain f32 oracle (recorded only)",
          file=stream)
    print(f"{'run':8s} {'component':58s} {'layer':>5s} {'graph':>5s} {'rel aware':>10s} {'rel f32':>10s}", file=stream)
    for (run, what), (il, g, mx, rel) in worst.items():
        r32 = w32.get((run, what))
        print(f"{run:8s} {what:58s} {il:5d} {g:5d} {rel:10.3e} {(r32[3] if r32 else float('nan')):10.3e}",
              file=stream)
    per_layer, per_layer32 = {}, {}
    for run, what, il, g, mx, rel, _ in table:
        if il not in per_layer or rel > per_layer[il][4]:
            per_layer[il] = (run, what, g, mx, rel)
    for run, what, il, g, mx, rel in rec32.rows:
        if il not in per_layer32 or rel > per_layer32[il][4]:
            per_layer32[il] = (run, what, g, mx, rel)
    print(f"\n{title}: worst error per layer (layer -1 = embedding and head), precision-aware oracle | plain f32 "
          f"oracle", file=stream)
    print(f"{'layer':>5s} {'max abs':>10s} {'rel':>10s} {'component (aware)':50s} | {'max abs':>10s} {'rel':>10s} "
          f"component (f32)", file=stream)
    for il in sorted(set(per_layer) | set(per_layer32)):
        a = per_layer.get(il, ("", "-", -1, float("nan"), float("nan")))
        b = per_layer32.get(il, ("", "-", -1, float("nan"), float("nan")))
        print(f"{il:5d} {a[3]:10.3e} {a[4]:10.3e} {(a[0] + ' ' + a[1])[:50]:50s} | {b[3]:10.3e} {b[4]:10.3e} "
              f"{b[0]} {b[1]}", file=stream)
    print(f"\n{title}: summaries (cache store and cache rows, flash-attention op, attention output, tie replays)",
          file=stream)
    for run, what, il, g, mx, rel, _ in sorted(summary, key=lambda r: (r[0], r[1])):
        print(f"{run:8s} {what}", file=stream)
    print(f"\n{title}: rows where an oracle's discrete choice differs from llama.cpp (the precision-aware oracle after "
          f"its tie replays, listed in the first table), and N6/N9 checks the plain f32 oracle fails (assertion: ...)",
          file=stream)
    for (run, kind), n in sorted(mismatch.items()):
        print(f"{run:8s} {kind:58s} {n:6d}", file=stream)
