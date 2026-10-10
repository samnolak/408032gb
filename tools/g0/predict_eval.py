#!/usr/bin/env python3
"""How well can the experts of MoE layer l+d be predicted while layer l runs? (predictive expert prefetch, real models)

Input: *.probe files of tools/g0/expert_probe.cpp (router input x_l, router logits and selected experts per MoE layer for
a window of prompt-tail and generated tokens), the model's router weights W_l (and selection bias b_l) read from the GGUF
(or an .npz with arrays W<l>, b<l> for tests), and optionally *.trace files of the same model (tools/g0/expert_trace.cpp)
as training data for the frequency-based predictors.

Self-checks (a failed check makes the run fail - a predictor evaluated with the wrong weights or the wrong selection rule
would be meaningless):
  1. W_l x_l reproduces the recorded logits of layer l (x is stored as fp16, so a small relative error is expected);
  2. our selection rule (softmax / sigmoid, + bias, expert groups) applied to the recorded logits reproduces the recorded
     top-k of layer l;
  3. (--check-traces) the probe's selected experts equal those of the routing trace of the same conversation
     (same -b/-ub, greedy generation, so the two runs must route identically).

Predictors for target layer L = l + d, each returning m candidate experts (recall = |candidates & true top-k| / k):
  pregate     top-m of select(W_L x_l): layer l's router input pushed through layer L's router (cross-layer pre-gating)
  prev_token  the experts layer L selected for the previous token (padded with popular experts up to m)
  popularity  the m experts layer L selects most often in the training traces
  transition  sum over layer l's selected experts e of the training co-occurrence counts C_L[e, e'], top-m
  pregate+prev  pregate top-(m - k) united with prev_token (fills up to m)
Lead time: with d = 1 the prediction is available one layer (attention + MoE of layer l) before layer L needs it.

Usage:
  python3 tools/g0/predict_eval.py probe_dir/ --gguf model.gguf [--train-traces trace_dir/] [--d 1,2] [-o out.json]
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np


def read_probe(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != b"G0PB":
        raise ValueError(f"{path}: not a G0PB probe")
    (ver, n_layers, k, n_embd, n_expert, n_rec, first, n_prompt, n_total) = struct.unpack_from("<9I", buf, 4)
    if ver != 1:
        raise ValueError(f"{path}: version {ver}")
    off = 4 + 36
    layers = list(struct.unpack_from(f"<{n_layers}I", buf, off))
    off += 4 * n_layers
    rec = np.dtype([("topk", "<u2", (k,)), ("x", "<f2", (n_embd,)), ("logits", "<f4", (n_expert,))])
    data = np.frombuffer(buf, dtype=rec, count=n_rec * n_layers, offset=off)
    if data.size != n_rec * n_layers or off + data.nbytes != len(buf):
        raise ValueError(f"{path}: truncated or trailing bytes")
    data = data.reshape(n_rec, n_layers)
    return {"path": path, "layers": layers, "k": k, "n_embd": n_embd, "n_expert": n_expert, "first": first,
            "n_prompt": n_prompt, "n_total": n_total, "topk": np.array(data["topk"], dtype=np.int64),
            "x": np.array(data["x"], dtype=np.float32), "logits": np.array(data["logits"], dtype=np.float32)}


def load_router_gguf(path, layers):
    from gguf import GGUFReader  # gguf-py of the pinned llama.cpp (PYTHONPATH)
    from gguf.quants import dequantize
    r = GGUFReader(path)

    def field(key, default=None):
        f = r.get_field(key)
        return default if f is None else f.contents()

    arch = field("general.architecture")
    W, b = {}, {}
    want = {f"blk.{l}.ffn_gate_inp.weight": ("W", l) for l in layers}
    want.update({f"blk.{l}.exp_probs_b.bias": ("b", l) for l in layers})
    for t in r.tensors:
        if t.name in want:
            kind, l = want[t.name]
            arr = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
            if kind == "W":
                W[l] = arr.reshape(-1, int(t.shape[0]))      # ggml ne = [n_embd, n_expert] -> (n_expert, n_embd)
            else:
                b[l] = arr.reshape(-1)
    gating = field(f"{arch}.expert_gating_func", 0)
    if not gating:
        n_layer = field(f"{arch}.block_count", 0)
        n_vocab = len(r.get_field("tokenizer.ggml.tokens").data) if r.get_field("tokenizer.ggml.tokens") else 0
        # llama.cpp ec7630a src/models/deepseek2.cpp:22-31 (GLM 4.7 Lite rule); other archs here use softmax
        gating = 2 if arch == "deepseek2" and n_layer in (47, 48) and n_vocab == 154880 else 1
    groups = (int(field(f"{arch}.expert_group_count", 0) or 0), int(field(f"{arch}.expert_group_used_count", 0) or 0))
    return {"arch": arch, "W": W, "b": b, "gating": int(gating), "groups": groups}


def load_router_npz(path, layers):
    z = np.load(path)
    return {"arch": str(z["arch"]) if "arch" in z else "npz", "gating": int(z["gating"]),
            "groups": (int(z["n_group"]), int(z["n_group_used"])),
            "W": {l: z[f"W{l}"] for l in layers}, "b": {l: z[f"b{l}"] for l in layers if f"b{l}" in z}}


def selection_scores(logits, gating, bias=None, groups=(0, 0)):
    """Scores whose top-k llama.cpp selects (src/llama-graph.cpp:2062-2134 @ec7630a)."""
    z = np.asarray(logits, dtype=np.float64)
    if gating == 1:                                      # softmax (monotone: top-k of probs == top-k of logits)
        p = np.exp(z - z.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
    elif gating == 2:                                    # sigmoid
        p = 1.0 / (1.0 + np.exp(-z))
    elif gating == 3:                                    # softmax_weight: probs = logits
        p = z
    elif gating == 4:                                    # sqrt(softplus)
        p = np.sqrt(np.logaddexp(0.0, z))
    else:
        raise ValueError(f"gating {gating}")
    s = p + bias if bias is not None else p
    n_group, n_used = groups
    if n_group and n_group > 1:
        n_exp = s.shape[-1]
        g = s.reshape(*s.shape[:-1], n_group, n_exp // n_group)
        top2 = np.sort(g, axis=-1)[..., -2:].sum(axis=-1)            # sum of the 2 best per group
        keep = np.argsort(-top2, axis=-1, kind="stable")[..., :n_used]
        mask = np.full(top2.shape, -np.inf)
        np.put_along_axis(mask, keep, 0.0, axis=-1)
        s = (g + mask[..., None]).reshape(s.shape)
    return s


def topm(scores, m):
    return np.argsort(-scores, axis=-1, kind="stable")[..., :m]


def read_trace_counts(trace_dir, layer_ids):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cachesim
    pop, trans = {}, {}
    files = sorted(glob.glob(os.path.join(trace_dir, "*.trace")))
    for p in files:
        tr = cachesim.read_trace(p)
        rows = {l: np.array(tr["per_layer"][l], dtype=np.int64) for l in tr["layer_ids"]}
        for i, l in enumerate(tr["layer_ids"]):
            n_exp = int(rows[l].max()) + 1
            pop.setdefault(l, np.zeros(65536, dtype=np.int64))
            np.add.at(pop[l], rows[l].ravel(), 1)
            if i > 0:
                lp = tr["layer_ids"][i - 1]
                c = trans.setdefault((lp, l), np.zeros((512, 512), dtype=np.int64))
                for a_row, b_row in zip(rows[lp], rows[l]):
                    c[np.ix_(a_row, b_row)] += 1
    return pop, trans, len(files)


def check_against_traces(probes, trace_dir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cachesim
    same = total = 0
    for pr in probes:
        stem = os.path.splitext(os.path.basename(pr["path"]))[0]
        tr = cachesim.read_trace(os.path.join(trace_dir, stem + ".trace"))
        if tr["n_prompt"] != pr["n_prompt"] or tr["n_tokens"] != pr["n_total"]:
            raise SystemExit(f"{stem}: trace has {tr['n_prompt']}+{tr['n_tokens']} tokens, probe "
                             f"{pr['n_prompt']}+{pr['n_total']}")
        for i, l in enumerate(pr["layers"]):
            rows = np.array(tr["per_layer"][l], dtype=np.int64)[pr["first"]:pr["first"] + pr["topk"].shape[0]]
            same += int((np.sort(rows, axis=-1) == np.sort(pr["topk"][:, i, :], axis=-1)).all(axis=-1).sum())
            total += rows.shape[0]
    return same / max(1, total)


def recall(cands, true_k):
    """cands [T, m], true [T, k] -> mean |cands & true| / k."""
    hit = (cands[:, :, None] == true_k[:, None, :]).any(axis=1)
    return float(hit.mean())


def evaluate(probes, router, ds=(1, 2), ms=None, pop=None, trans=None, check_tol=0.02, check_sel=0.99):
    out = {"checks": {}, "results": [], "n_files": len(probes)}
    k, n_expert = probes[0]["k"], probes[0]["n_expert"]
    ms = ms or sorted({k, 2 * k, 4 * k} & set(range(1, n_expert + 1)))
    # 1. router weights reproduce the recorded logits; 2. selection rule reproduces the recorded top-k
    worst_rel, sel_ok, sel_n = 0.0, 0, 0
    for pr in probes:
        for i, l in enumerate(pr["layers"]):
            W = router["W"][l]
            pred = pr["x"][:, i, :] @ W.T
            ref = pr["logits"][:, i, :]
            worst_rel = max(worst_rel, float(np.abs(pred - ref).max() / max(1e-12, np.abs(ref).max())))
            s = selection_scores(ref, router["gating"], router["b"].get(l), router["groups"])
            mine = np.sort(topm(s, k), axis=-1)
            sel_ok += int((mine == np.sort(pr["topk"][:, i, :], axis=-1)).all(axis=-1).sum())
            sel_n += mine.shape[0]
    out["checks"] = {"router_logits_max_rel_err": worst_rel, "selection_rule_match": sel_ok / max(1, sel_n),
                     "gating": router["gating"], "groups": list(router["groups"])}
    if worst_rel > check_tol or sel_ok / max(1, sel_n) < check_sel:
        raise SystemExit(f"self-check failed: {out['checks']} (wrong router weights or selection rule)")
    for d in ds:
        for part in ("prompt", "gen"):
            acc = {m: {name: [] for name in ("pregate", "prev_token", "popularity", "transition", "pregate+prev")}
                   for m in ms}
            n_pairs = 0
            for pr in probes:
                tok = np.arange(pr["topk"].shape[0]) + pr["first"]
                sel = tok >= pr["n_prompt"] if part == "gen" else tok < pr["n_prompt"]
                idx = np.nonzero(sel)[0]
                idx_prev = idx[idx > 0]                              # prev_token needs the previous recorded token
                if idx.size == 0:
                    continue
                for i, l in enumerate(pr["layers"]):
                    if i + d >= len(pr["layers"]):
                        break
                    L = pr["layers"][i + d]
                    true = pr["topk"][idx, i + d, :]
                    s = selection_scores(pr["x"][idx, i, :] @ router["W"][L].T, router["gating"],
                                         router["b"].get(L), router["groups"])
                    n_pairs += idx.size
                    for m in ms:
                        pg = topm(s, m)
                        acc[m]["pregate"].append((recall(pg, true), idx.size))
                        popular = (np.argsort(-pop[L][:n_expert], kind="stable")[:m] if pop is not None
                                   else np.arange(m))
                        if pop is not None:
                            acc[m]["popularity"].append((recall(np.tile(popular, (idx.size, 1)), true), idx.size))
                        if idx_prev.size:
                            prev = pr["topk"][idx_prev - 1, i + d, :]
                            fill = np.array([np.concatenate([p, [e for e in popular if e not in p]])[:m]
                                             for p in prev])
                            acc[m]["prev_token"].append((recall(fill, pr["topk"][idx_prev, i + d, :]), idx_prev.size))
                            spg = selection_scores(pr["x"][idx_prev, i, :] @ router["W"][L].T, router["gating"],
                                                   router["b"].get(L), router["groups"])
                            pgk = topm(spg, n_expert)
                            uni = []
                            for p, row in zip(prev, pgk):
                                take = list(dict.fromkeys(list(row[:max(0, m - k)]) + list(p)))
                                take += [e for e in row if e not in take][:m - len(take)]
                                uni.append(take[:m])
                            acc[m]["pregate+prev"].append((recall(np.array(uni), pr["topk"][idx_prev, i + d, :]),
                                                           idx_prev.size))
                        if trans is not None and d == 1 and (l, L) in trans:
                            C = trans[(l, L)][:n_expert, :n_expert]
                            sc = C[pr["topk"][idx, i, :]].sum(axis=1).astype(np.float64)
                            acc[m]["transition"].append((recall(topm(sc, m), true), idx.size))
            for m in ms:
                row = {"d": d, "part": part, "m": m, "k": k, "n_expert": n_expert, "token_layer_pairs": n_pairs,
                       "random_baseline": m / n_expert}
                for name, vals in acc[m].items():
                    if vals:
                        row[name] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
                out["results"].append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe_dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gguf")
    g.add_argument("--npz")
    ap.add_argument("--train-traces")
    ap.add_argument("--check-traces", help="dir with the *.trace of the same conversations (self-check 3)")
    ap.add_argument("--d", default="1,2")
    ap.add_argument("--m", default="", help="candidate set sizes (default k, 2k, 4k)")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    probes = [read_probe(p) for p in sorted(glob.glob(os.path.join(a.probe_dir, "*.probe")))]
    if not probes:
        raise SystemExit("no *.probe files")
    layers = sorted({l for p in probes for l in p["layers"]})
    router = load_router_gguf(a.gguf, layers) if a.gguf else load_router_npz(a.npz, layers)
    pop = trans = None
    if a.train_traces:
        pop, trans, n = read_trace_counts(a.train_traces, layers)
    res = evaluate(probes, router, ds=[int(x) for x in a.d.split(",")],
                   ms=[int(x) for x in a.m.split(",")] if a.m else None, pop=pop, trans=trans)
    if a.check_traces:
        res["checks"]["probe_equals_trace_rows"] = check_against_traces(probes, a.check_traces)
        if res["checks"]["probe_equals_trace_rows"] < 0.999:
            raise SystemExit(f"self-check 3 failed: {res['checks']}")
    res["arch"] = router["arch"]
    res["probe_files"] = [os.path.basename(p["path"]) for p in probes]
    res["recorded_tokens"] = int(sum(p["topk"].shape[0] for p in probes))
    print(json.dumps(res["checks"]))
    print("pairs (token, layer l -> l+d) per row: see token_layer_pairs in the JSON; train traces: %s" % (
        "none" if pop is None else n))
    names = ("pregate", "prev_token", "popularity", "transition", "pregate+prev")
    print("%-3s %-6s %-4s %-7s " % ("d", "part", "m", "random") + " ".join("%-12s" % n for n in names))
    for r in res["results"]:
        print("%-3d %-6s %-4d %-7.3f " % (r["d"], r["part"], r["m"], r["random_baseline"]) +
              " ".join("%-12s" % ("%.3f" % r[n] if n in r else "-") for n in names))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
