"""Cell-level model of a multi-head MTP context under llama-server's call sequence (N21): which position's K/V each
head layer of the MTP context holds under every cell, and which attention rows read a cell whose K/V of that layer
were computed for another position than the cell holds ("foreign" K/V, the head-0 finding of
docs/evidence/2026-10-11-n17-mimo-mtp-heads-fa.md). No weights: only placement, built on ref/mimo2.py Cells (the
find_slot / apply_ubatch / seq_rm model that N16 and N17 checked against llama.cpp's own cells, kq on every n_kv cell).

All MTP layers of mimo2 are sliding-window layers on the one SWA cache of the MTP context (ref/mimo2.py docstring); this
model holds that cache only. Calls, as common/speculative.cpp @79e2e74 and tools/server/server-context.cpp @79e2e74 make
them (docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md section 1):
  process(p0, n)  after every target decode of n tokens at p0..p0+n-1 (SC:4232): for head k = 0 .. H-1: seq_rm(p0),
                  offset k, decode the n tokens (split into ubatches of n_ubatch, one find_slot + apply each)
                  (SP:1597-1621). fixed=True: the patched order (patches/llama.cpp/0001-*.patch): the call is cut into
                  chunks of n_ubatch tokens; per chunk head 0 places it (after seq_rm of the chunk's first position) and
                  heads 1 .. H-1 re-use exactly head 0's cells (llama_set_kv_slot_replay), so every head layer is
                  written under the same cells for every position.
  draft(pos0, n)  (SP:1646-1800) step k = 0 .. n-1: seq_rm(pos0), offset k, decode [id_last, d_1 .. d_k] at
                  pos0 .. pos0+k; only the last row is an output (its token is the draft). Then the server's
                  seq_rm(pos0) (SC:3423).
  accept(keep)    seq_rm(keep) after verification (SC:4505).
A row "reads foreign K/V" when a cell it can see (same sequence, p - n_swa < pos_cell <= p) holds, in the row's head
layer, the K/V of a position other than the cell's. K/V of an MTP layer come from its own input projection (eh_proj of
the token and the h row), not from attention, so a foreign read changes that row's attention output and what follows
(logits, h_nextn) but never the K/V any head writes."""
import numpy as np

from ref.mimo2 import Cells, KV_PAD, pad


def ring_size(n_ctx, n_swa, n_ubatch):
    """llama_kv_cache_iswa: size_swa = pad(min(n_ctx, n_swa + n_ubatch), 256) with swa_full off (KVI:73)."""
    return pad(min(n_ctx, n_swa + n_ubatch), KV_PAD)


class MtpCells:
    def __init__(self, size, n_swa, n_heads, n_ubatch, fixed=False):
        self.c = Cells(size, n_swa)
        self.H, self.n_ubatch, self.fixed = int(n_heads), int(n_ubatch), bool(fixed)
        self.P = np.full((self.H, size), -1, dtype=np.int64)  # per head layer: position of the K/V under each cell
        self.log = []          # (kind, head, positions, foreign rows mask, cells, missing-window rows mask)

    # ---- primitives ----
    def seq_rm(self, p0):
        self.c.seq_rm(p0, -1)

    def _rows_foreign(self, head, pos):
        """For each row at position p: does a visible cell hold, in this head's layer, K/V of another position."""
        cp = self.c.pos
        out = []
        for p in pos:
            vis = (cp >= 0) & (cp <= p) & (p - cp < self.c.n_swa)
            out.append(bool((vis & (self.P[head] != cp)).any()))
        return np.array(out, dtype=bool)

    def _rows_missing(self, pos):
        """For each row at position p: is a position of its window, max(0, p - n_swa + 1) .. p, held by no cell (the
        row then attends to less than the full-sequence block would)."""
        held = set(int(x) for x in self.c.pos[self.c.pos >= 0])
        return np.array([any(q not in held for q in range(max(0, p - self.c.n_swa + 1), p + 1)) for p in pos],
                        dtype=bool)

    def _ubatch(self, kind, head, pos, idxs=None):
        """One ubatch under `head`: find_slot (or the replayed cells), apply, write the head's layer, then the rows
        attend. Returns (cells, foreign mask)."""
        pos = [int(p) for p in pos]
        if idxs is None:
            idxs = self.c.find_slot(len(pos))
        else:
            # replay: llama_kv_cache::prepare() re-uses the recorded cells only when every one is empty now
            assert all(self.c.pos[i] < 0 for i in idxs), "replayed cell not empty"
        self.c.apply(idxs, pos)
        self.P[head][idxs] = pos
        f = self._rows_foreign(head, pos)
        self.log.append((kind, head, pos, f, list(idxs), self._rows_missing(pos)))
        return list(idxs), f

    # ---- llama-server / common_speculative calls ----
    def process(self, p0, n):
        pos = list(range(p0, p0 + n))
        ub = self.n_ubatch
        if self.H == 1:
            for j in range(0, n, ub):
                self._ubatch("process", 0, pos[j:j + ub])
            return
        if not self.fixed:
            for head in range(self.H):
                self.seq_rm(p0)
                for j in range(0, n, ub):
                    self._ubatch("process", head, pos[j:j + ub])
            return
        for j in range(0, n, ub):
            chunk = pos[j:j + ub]
            idxs = None
            for head in range(self.H):
                self.seq_rm(chunk[0])
                idxs, _ = self._ubatch("process", head, chunk, idxs)

    def draft(self, pos0, n_steps):
        """Returns per step whether its output row (the last) read foreign K/V."""
        res = []
        if self.H == 1:
            for i in range(n_steps):
                _, f = self._ubatch("draft", 0, [pos0 + i])
                res.append(bool(f[-1]))
        else:
            for k in range(min(n_steps, self.H)):
                self.seq_rm(pos0)
                _, f = self._ubatch("draft", k, list(range(pos0, pos0 + k + 1)))
                res.append(bool(f[-1]))
        self.seq_rm(pos0)            # the server's removal of the draft positions
        return res

    def accept(self, keep):
        self.seq_rm(keep)


def serve(prompt, n_gen, accept_prob, size, n_swa, n_heads=3, n_ubatch=512, n_batch=2048, fixed=False, seed=0,
          n_draft=None):
    """llama-server with MTP drafting, one slot, greedy verification: the prompt in calls of n_batch, then
    draft / verify / accept until n_gen tokens are generated. accept_prob: i.i.d. chance that a draft token matches the
    target (a stand-in for the acceptance that the bug itself would change; the model counts exposure, not acceptance).
    Returns dict(drafts=[(pos0, [foreign per step])], m=MtpCells)."""
    rng = np.random.default_rng(seed)
    m = MtpCells(size, n_swa, n_heads, n_ubatch, fixed)
    n_draft = n_heads if n_draft is None else n_draft
    p = 0
    while p < prompt:
        n = min(n_batch, prompt - p)
        m.process(p, n)
        p += n
    drafts, gen = [], 0
    pos0 = prompt                    # id_last (the token sampled from the last prompt row) sits at pos0
    while gen < n_gen:
        f = m.draft(pos0, n_draft)
        drafts.append((pos0, f))
        n_verify = 1 + len(f)
        m.process(pos0, n_verify)   # the target's verify batch [id_last, d_1 .. d_n]
        n_acc = 0
        while n_acc < len(f) and rng.random() < accept_prob:
            n_acc += 1
        m.accept(pos0 + n_acc + 1)
        gen += n_acc + 1
        pos0 += n_acc + 1
    return dict(drafts=drafts, m=m)


def summary(r):
    """Counts over a serve() result: draft steps per head, how many read foreign K/V at their output row, process
    ubatches whose rows read foreign K/V, and the positions of the drafts affected."""
    drafts = r["drafts"]
    H = max([r["m"].H] + [len(f) for _, f in drafts])   # one head drafts up to n_max steps
    n_steps = [0] * H
    n_bad = [0] * H
    bad_pos = []
    for pos0, f in drafts:
        for k, x in enumerate(f):
            n_steps[k] += 1
            n_bad[k] += int(x)
        if any(f):
            bad_pos.append(pos0)
    n_proc = sum(1 for e in r["m"].log if e[0] == "process")
    n_proc_bad = sum(1 for e in r["m"].log if e[0] == "process" and e[3].any())
    n_proc_missing = sum(1 for e in r["m"].log if e[0] == "process" and e[5].any())
    return dict(n_drafts=len(drafts), steps=n_steps, bad=n_bad, bad_drafts=len(bad_pos), first_bad=bad_pos[:5],
                last_bad=bad_pos[-1] if bad_pos else None, n_proc=n_proc, n_proc_bad=n_proc_bad,
                n_proc_missing=n_proc_missing)
