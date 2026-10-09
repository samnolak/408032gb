#!/usr/bin/env python3
"""Trace-driven simulation of per-layer expert caches (VRAM tier) on real routing traces.

Input: per-token routing traces written by tools/g0/expert_trace.cpp (`*.trace`, format below), i.e. for every
token (prompt and generated) and every MoE layer the top-k expert ids the router chose.
Question: with a VRAM budget of C experts per layer, what share of routed-expert reads hit VRAM under each policy?

Policies (all exact-routing: they only decide WHERE an expert is read from, never WHICH experts run):
  static   top-C experts per layer by frequency on TRAIN traces (held-out evaluation, like tools/g0/analyze.py)
  lru      least recently used
  lfu      least frequently used, counts halved every `--lfu-decay` accesses of the layer (aging)
  tinylfu  LRU main cache with frequency-based admission (a missed expert replaces the LRU victim only if it has
           been used more often recently, count-min-free exact counts with the same aging)
  hybrid   S = C/2 static slots + C - S LRU slots (the shape of Strata's static VRAM part + VRAM LRU)
  opt      Belady's MIN with bypass and a free preload of the experts used first: the offline optimum for ANY policy
           holding C experts per layer, warm or cold (an upper bound, not an implementable policy)
  lru_global   ONE LRU shared by all layers with the same total budget (C x layers): llama.cpp --moe-cache-mib
  opt_global   Belady-with-bypass bound for any policy sharing that total budget across layers
Dynamic policies start EMPTY (cold) unless --warm, which preloads them with the static set.

Trace format (little endian): b"G0TR", u32 version=1, u32 n_layers, u32 k, u32 n_tokens, u32 n_prompt,
u32 layer_ids[n_layers], then n_tokens x n_layers x k u16 expert ids (token-major). Tokens [0, n_prompt) are prompt
tokens, the rest were generated greedily.

Usage:
  python3 tools/g0/cachesim.py g0-trace/ --n-experts 288 --capacity 0.25,0.5,0.67,0.8 --holdout-every 4 \
      -o docs/evidence/G0-cachesim.json
"""
import argparse
import json
import pathlib
import struct
from collections import OrderedDict, defaultdict

MAGIC = b"G0TR"
LAYER_POLICIES = ("static", "lru", "lfu", "tinylfu", "hybrid", "opt")      # a budget of C experts per layer
GLOBAL_POLICIES = ("lru_global", "opt_global")                           # one budget of C x layers
POLICIES = LAYER_POLICIES + GLOBAL_POLICIES


# ------------------------------------------------------------------------------------------- trace I/O

def write_trace(path, layer_ids, k, tokens, n_prompt):
    """tokens: list over tokens of list over layers of list of k expert ids."""
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", 1, len(layer_ids), k, len(tokens), n_prompt))
        f.write(struct.pack(f"<{len(layer_ids)}I", *layer_ids))
        for tok in tokens:
            for row in tok:
                f.write(struct.pack(f"<{k}H", *row))


def read_trace(path):
    data = pathlib.Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path}: not a G0 trace")
    version, n_layers, k, n_tokens, n_prompt = struct.unpack_from("<5I", data, 4)
    if version != 1:
        raise ValueError(f"{path}: version {version}")
    off = 24
    layer_ids = list(struct.unpack_from(f"<{n_layers}I", data, off))
    off += 4 * n_layers
    need = off + 2 * n_tokens * n_layers * k
    if len(data) != need:
        raise ValueError(f"{path}: size {len(data)} != {need}")
    flat = struct.unpack_from(f"<{n_tokens * n_layers * k}H", data, off)
    # per layer: list of k-tuples in token order
    per_layer = {lid: [] for lid in layer_ids}
    i = 0
    for _ in range(n_tokens):
        for lid in layer_ids:
            per_layer[lid].append(flat[i:i + k])
            i += k
    return {"layer_ids": layer_ids, "k": k, "n_tokens": n_tokens, "n_prompt": n_prompt, "per_layer": per_layer}


# ------------------------------------------------------------------------------------------- policies

def _static_set(freq, n_experts, cap):
    return set(sorted(range(n_experts), key=lambda e: (-freq.get(e, 0), e))[:cap])


def simulate_layer(seq, policy, cap, n_experts, static=None, warm=False, lfu_decay=4096):
    """seq: list of k-tuples (one per token); static: experts ranked by train frequency (list).

    Returns per-token hit counts (list of ints).
    """
    hits = []
    if cap <= 0:
        return [0] * len(seq)
    if n_experts is not None and cap >= n_experts:
        if policy == "static" or warm:
            return [len(t) for t in seq]
    if policy == "static":
        s = _top(static, cap)
        return [sum(1 for e in t if e in s) for t in seq]
    if policy == "opt":
        # Belady's MIN with bypass on the serialized access sequence (token order, then expert order inside the
        # token, the same order the other policies use): on a miss, cache the expert only if its next use comes
        # before the farthest next use among cached experts (then evict that one). This is the offline optimum
        # for any policy holding at most `cap` experts, including policies that may refuse to cache (tinylfu).
        INF = float("inf")
        acc = [e for t in seq for e in t]
        nxt = [INF] * len(acc)
        last = {}
        for i in range(len(acc) - 1, -1, -1):
            nxt[i] = last.get(acc[i], INF)
            last[acc[i]] = i
        # free preload of the `cap` experts used first: the best possible starting content, so the bound also
        # covers policies that start warm (static, hybrid, --warm)
        cache = {}                     # expert -> index of its next use
        for e in sorted(last, key=lambda x: last[x])[:cap]:
            cache[e] = last[e]
        pos = 0
        for t in seq:
            h = 0
            for e in t:
                if e in cache:
                    h += 1
                    cache[e] = nxt[pos]
                elif len(cache) < cap:
                    cache[e] = nxt[pos]
                else:
                    victim = max(cache, key=lambda x: cache[x])
                    if cache[victim] > nxt[pos]:
                        del cache[victim]
                        cache[e] = nxt[pos]
                pos += 1
            hits.append(h)
        return hits
    if policy in ("lru", "hybrid", "tinylfu"):
        fixed = set()
        lru_cap = cap
        if policy == "hybrid":
            fixed = _top(static, cap // 2)
            lru_cap = cap - len(fixed)
        lru = OrderedDict()
        freq = defaultdict(int)
        accesses = 0
        if warm:
            for e in _top(static, cap):
                if e not in fixed and len(lru) < lru_cap:
                    lru[e] = True
        for t in seq:
            h = 0
            for e in t:
                accesses += 1
                freq[e] += 1
                if policy == "tinylfu" and accesses % lfu_decay == 0:
                    for x in list(freq):
                        freq[x] //= 2
                if e in fixed:
                    h += 1
                elif e in lru:
                    h += 1
                    lru.move_to_end(e)
                elif lru_cap > 0:
                    if len(lru) < lru_cap:
                        lru[e] = True
                    else:
                        victim = next(iter(lru))
                        if policy == "tinylfu" and freq[e] <= freq[victim]:
                            continue          # not admitted: read from the lower tier, cache unchanged
                        lru.popitem(last=False)
                        lru[e] = True
            hits.append(h)
        return hits
    if policy == "lfu":
        cache = set(_top(static, cap)) if warm else set()
        freq = defaultdict(int)
        accesses = 0
        for t in seq:
            h = 0
            for e in t:
                accesses += 1
                freq[e] += 1
                if accesses % lfu_decay == 0:
                    for x in list(freq):
                        freq[x] //= 2
                if e in cache:
                    h += 1
                else:
                    if len(cache) >= cap:
                        cand = [x for x in cache if x not in t]
                        if not cand:
                            continue
                        victim = min(cand, key=lambda x: (freq[x], x))
                        cache.discard(victim)
                    cache.add(e)
            hits.append(h)
        return hits
    raise ValueError(policy)


def _top(ranked, n):
    """The first n experts of a ranked list, as a set."""
    return set(ranked[:n]) if ranked is not None else set()


# ------------------------------------------------------------------------------------------- one cache for all layers

def global_sequence(tr):
    """Per token, the (layer, expert) keys in layer order: the order in which a decoder visits them."""
    lids = tr["layer_ids"]
    return [tuple((lid, e) for lid in lids for e in tr["per_layer"][lid][t]) for t in range(tr["n_tokens"])]


def simulate_global(tr, policy, cap_total):
    """'lru_global': ONE LRU over all layers' experts (how llama.cpp --moe-cache-mib works: layers of a device share
    the slots, src/llama-moe-cache.cpp:15-120 @10a60cf30). 'opt_global': Belady with bypass over the same keys and the
    same total capacity (offline bound for any global policy). Returns per-token hit counts."""
    seq = global_sequence(tr)
    if policy == "lru_global":
        from collections import OrderedDict as _OD
        lru = _OD()
        hits = []
        for t in seq:
            h = 0
            for key in t:
                if key in lru:
                    h += 1
                    lru.move_to_end(key)
                else:
                    if len(lru) >= cap_total:
                        lru.popitem(last=False)
                    lru[key] = True
            hits.append(h)
        return hits
    if policy == "opt_global":
        return simulate_layer(seq, "opt", cap_total, n_experts=None, static=None)
    raise ValueError(policy)


# ------------------------------------------------------------------------------------------- speculation windows

def window_distinct(traces, windows=(1, 2, 4, 8), part="gen"):
    """Mean number of DISTINCT experts per layer touched by k consecutive tokens (sliding windows), vs the
    independent-uniform expectation E * (1 - (1 - k_top/E)^k). This is the expert traffic multiplier of verifying a
    k-token speculation window at batch 1 (tools/roofline.py --spec-k assumes the independent case)."""
    out = {}
    for w in windows:
        tot, n = 0, 0
        for tr in traces:
            lo = tr["n_prompt"] if part == "gen" else 0
            hi = tr["n_tokens"] if part in ("gen", "all") else tr["n_prompt"]
            for seq in tr["per_layer"].values():
                for i in range(lo, hi - w + 1):
                    tot += len(set(e for t in seq[i:i + w] for e in t))
                    n += 1
        out[w] = tot / n if n else None
    return out


def independent_distinct(n_experts, k, w):
    return n_experts * (1 - (1 - k / n_experts) ** w)


# ------------------------------------------------------------------------------------------- driver

def layer_freq(traces):
    freq = defaultdict(lambda: defaultdict(int))
    for tr in traces:
        for lid, seq in tr["per_layer"].items():
            f = freq[lid]
            for t in seq:
                for e in t:
                    f[e] += 1
    return freq


def evaluate(train, test, n_experts, fractions, policies=POLICIES, warm=False, lfu_decay=4096):
    freq = layer_freq(train)
    out = {}
    for frac in fractions:
        cap = round(frac * n_experts)
        res = {}
        for pol in policies:
            tot = {"all": [0, 0], "prompt": [0, 0], "gen": [0, 0]}
            per_layer = {}
            if pol.endswith("_global"):
                for tr in test:
                    hits = simulate_global(tr, pol, cap * len(tr["layer_ids"]))
                    per_tok = tr["k"] * len(tr["layer_ids"])
                    for i, h in enumerate(hits):
                        part = "prompt" if i < tr["n_prompt"] else "gen"
                        for key in (part, "all"):
                            tot[key][0] += h
                            tot[key][1] += per_tok
                rate = {pp: (v[0] / v[1] if v[1] else None) for pp, v in tot.items()}
                rate["layer_min"] = rate["layer_max"] = None
                res[pol] = rate
                continue
            for tr in test:
                for lid, seq in tr["per_layer"].items():
                    ranked = sorted(range(n_experts), key=lambda e: (-freq[lid].get(e, 0), e))
                    hits = simulate_layer(seq, pol, cap, n_experts, static=ranked, warm=warm, lfu_decay=lfu_decay)
                    k = tr["k"]
                    lh = [0, 0]
                    for i, h in enumerate(hits):
                        part = "prompt" if i < tr["n_prompt"] else "gen"
                        tot[part][0] += h
                        tot[part][1] += k
                        tot["all"][0] += h
                        tot["all"][1] += k
                        lh[0] += h
                        lh[1] += k
                    pl = per_layer.setdefault(lid, [0, 0])
                    pl[0] += lh[0]
                    pl[1] += lh[1]
            rate = {p: (v[0] / v[1] if v[1] else None) for p, v in tot.items()}
            lr = [v[0] / v[1] for v in per_layer.values() if v[1]]
            rate["layer_min"] = min(lr) if lr else None
            rate["layer_max"] = max(lr) if lr else None
            res[pol] = rate
        out[str(frac)] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("--n-experts", type=int, required=True)
    ap.add_argument("--capacity", default="0.25,0.5,0.67,0.8", help="fractions of each layer's experts kept in VRAM")
    ap.add_argument("--holdout-every", type=int, default=4, help="every k-th trace file is held out for evaluation")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--warm", action="store_true", help="dynamic policies start with the static set loaded")
    ap.add_argument("--lfu-decay", type=int, default=4096)
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    files = sorted(pathlib.Path(a.trace_dir).glob("*.trace"))
    if not files:
        raise SystemExit(f"no *.trace files in {a.trace_dir}")
    traces = [read_trace(f) for f in files]
    if len(traces) == 1:
        train, test = traces, traces
        split = "in-sample (only one trace)"
    else:
        test = [t for i, t in enumerate(traces) if i % a.holdout_every == a.holdout_every - 1] or traces[-1:]
        train = [t for t in traces if t not in test]
        split = f"held-out every {a.holdout_every}th file: {len(train)} train / {len(test)} test"
    fr = [float(x) for x in a.capacity.split(",")]
    res = evaluate(train, test, a.n_experts, fr, a.policies.split(","), a.warm, a.lfu_decay)
    k = traces[0]["k"]
    spec = {}
    for part in ("gen", "prompt"):
        wd = window_distinct(traces, part=part)
        spec[part] = {str(w): {"distinct": d, "independent": independent_distinct(a.n_experts, k, w)}
                      for w, d in wd.items()}
    report = {"label": "CONFIRMED only for the traces given; routing of a quantized model may differ from the original",
              "files": [f.name for f in files], "split": split, "n_experts": a.n_experts, "warm": a.warm,
              "hit_rate_by_capacity": res, "distinct_experts_per_window": spec}
    txt = json.dumps(report, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(txt + "\n")
    print(f"# cachesim: {len(files)} traces, {split}; warm={a.warm}")
    print("capacity  " + "  ".join(f"{p:>8}" for p in a.policies.split(",")))
    for f in fr:
        row = res[str(f)]
        print(f"{f:8.2f}  " + "  ".join(f"{row[p]['all']:8.3f}" for p in a.policies.split(",")))
    for part in ("gen", "prompt"):
        cells = []
        for w, v in spec[part].items():
            if v["distinct"] is not None:
                cells.append(f"k={w}: {v['distinct']:.2f} (indep. {v['independent']:.2f})")
        print(f"distinct experts per layer per window [{part}]: " + "; ".join(cells))


if __name__ == "__main__":
    main()
