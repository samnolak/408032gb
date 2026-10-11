#!/usr/bin/env python3
"""How often llama-server's multi-head MTP drafting reads foreign K/V (N21, docs/evidence/2026-10-11-n21-mimo-mtp-heads-
fix.md), estimated with the placement-only model ref/mtp_cells.py (the find_slot / apply_ubatch / seq_rm rules that N16
and N17 checked against llama.cpp's cells, and that tests/test_mimo2_mtp_fix_parity.py checks graph by graph against
llama.cpp's dumps of a tiny model). Nothing here runs llama.cpp or a model: the numbers are PROVISIONAL (a model of the
placement under an i.i.d. acceptance stand-in), to be replaced by the hardware experiment of the evidence file.

For each prompt length, acceptance rate and seed: the prompt in calls of --n-batch, then draft (H steps) / verify (1 + H)
/ accept until --gen tokens; counts the drafts whose head-0 (first) token was computed from foreign K/V, and the process
ubatches in which a head attends to an incomplete window. Modes: upstream (79e2e74), fixed (the patch), one head
(--override of the head count), and a ring as large as the context (--swa-full).

Usage: python3 tools/mtp_ring_exposure.py [--prompts 100,700,2000,5000] [--accept 0.3,0.6,0.85] [--seeds 1,2,3]
       [--gen 4000] [--n-swa 128] [--n-ubatch 512] [--n-batch 2048] [--heads 3] [--ctx 262144]
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ref.mtp_cells import ring_size, serve, summary  # noqa: E402


def ints(s):
    return [int(x) for x in s.split(",") if x]


def floats(s):
    return [float(x) for x in s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", type=ints, default=[100, 700, 2000, 5000])
    ap.add_argument("--accept", type=floats, default=[0.3, 0.6, 0.85])
    ap.add_argument("--seeds", type=ints, default=[1, 2, 3])
    ap.add_argument("--gen", type=int, default=4000)
    ap.add_argument("--n-swa", type=int, default=128)
    ap.add_argument("--n-ubatch", type=int, default=512)
    ap.add_argument("--n-batch", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--np", type=int, default=1, help="llama-server -np with a unified KV cache: the ring is "
                    "pad(min(ctx, n_swa * np + n_ubatch), 256) (KVI:73 @79e2e74); one request is active")
    ap.add_argument("--modes", default="upstream,fixed,one-head,swa-full")
    a = ap.parse_args(argv)
    ring = ring_size(a.ctx, a.n_swa * a.np, a.n_ubatch)
    print(f"SWA ring of the MTP context: {ring} cells (pad(min({a.ctx}, {a.n_swa} * {a.np} + {a.n_ubatch}), 256)); "
          f"{a.heads} heads, verify batches of {a.heads + 1}, prompt calls of {a.n_batch}, gen {a.gen}, "
          f"seeds {a.seeds}")
    print(f"{'mode':9s} {'prompt':>6s} {'accept':>6s} {'drafts':>7s} {'head-0 drafts on foreign K/V':>29s} "
          f"{'share':>7s} {'first':>6s} {'heads 1..':>9s} {'proc ub incompl. window':>23s}")
    for mode in a.modes.split(","):
        for prompt in a.prompts:
            for acc in a.accept:
                tot = dict(n=0, bad=0, other=0, miss=0, first=[])
                for seed in a.seeds:
                    kw = dict(n_ubatch=a.n_ubatch, n_batch=a.n_batch, seed=seed)
                    if mode == "upstream":
                        r = serve(prompt, a.gen, acc, ring, a.n_swa, a.heads, **kw)
                    elif mode == "fixed":
                        r = serve(prompt, a.gen, acc, ring, a.n_swa, a.heads, fixed=True, **kw)
                    elif mode == "one-head":
                        r = serve(prompt, a.gen, acc, ring, a.n_swa, 1, n_draft=a.heads, **kw)
                    elif mode == "swa-full":
                        r = serve(prompt, a.gen, acc, prompt + a.gen + 2 * a.heads + 8, a.n_swa, a.heads, **kw)
                    else:
                        raise SystemExit(f"unknown mode {mode}")
                    s = summary(r)
                    tot["n"] += s["n_drafts"]
                    tot["bad"] += s["bad"][0]
                    tot["other"] += sum(s["bad"][1:])
                    tot["miss"] += s["n_proc_missing"]
                    if s["first_bad"]:
                        tot["first"].append(s["first_bad"][0])
                first = min(tot["first"]) if tot["first"] else "-"
                print(f"{mode:9s} {prompt:6d} {acc:6.2f} {tot['n']:7d} {tot['bad']:29d} "
                      f"{tot['bad'] / max(tot['n'], 1):7.1%} {first!s:>6s} {tot['other']:9d} {tot['miss']:23d}")


if __name__ == "__main__":
    main()
