#!/usr/bin/env python3
"""Compare next-token distributions of two OpenAI-compatible servers on the same prefixes (experiments Q1, R2.1).

Use cases (both servers are any engine that returns `logprobs` with `top_logprobs` on /v1/completions):
  * quantisation quality: candidate quant (e.g. UD-Q4_K_XL) against a reference (Q8_0, possibly mmap'ed from NVMe);
  * implementation parity: the same GGUF in two engines (e.g. Strata vs llama.cpp).
For each prefix (cut points spread over a text corpus) both servers are asked for ONE token at temperature 0 with the
top-K log-probabilities. Metrics per prefix and aggregated (median, p95, max):
  top1_agree   the two greedy tokens are the same string
  kl_topk      KL(ref || cand) over the union of both top-K sets; a token missing from one side gets that side's
               floor = min(its top-K logprob) - log(2) (a conservative stand-in); both renormalised on the union
  ref_in_cand  the reference greedy token is in the candidate's top-K
This is an approximation of full-vocabulary KLD (true KLD needs all logits); it is comparable between runs with the same K
and prefixes, and it is what an engine-agnostic API allows. Stdlib only.

Usage:
  python3 tools/quality/logprob_compare.py --ref http://127.0.0.1:8081 --cand http://127.0.0.1:8080 \
      --corpus corpus/*.txt --prefixes 200 --top-k 20 --out runs/q1-qwen-q4-vs-q8.json
"""
import argparse
import glob
import json
import math
import random
import statistics
import sys
import time
import urllib.request


def complete_top(url, prompt, top_k, model=None, timeout=600, api_key=None):
    body = {"prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": top_k, "stream": False}
    if model:
        body["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    tops = lp.get("top_logprobs")
    if not tops or not isinstance(tops, list) or not tops[0]:
        raise ValueError(f"{url}: response has no top_logprobs (engine does not support logprobs on /v1/completions?)")
    top = {str(k): float(v) for k, v in tops[0].items()}
    text = (lp.get("tokens") or [ch.get("text", "")])[0]
    return {"token": text, "top": top, "seconds": dt}


def compare(ref, cand):
    """ref, cand: dict token -> logprob (top-K each). Returns metrics for one prefix."""
    union = set(ref) | set(cand)
    fr = min(ref.values()) - math.log(2)
    fc = min(cand.values()) - math.log(2)
    p = {t: math.exp(ref.get(t, fr)) for t in union}
    q = {t: math.exp(cand.get(t, fc)) for t in union}
    zp, zq = sum(p.values()), sum(q.values())
    kl = sum((p[t] / zp) * math.log((p[t] / zp) / (q[t] / zq)) for t in union)
    r1 = max(ref, key=ref.get)
    c1 = max(cand, key=cand.get)
    # Near-tie data for the V6 tie tolerance (tools/verify/pack.py, criteria logprob_tie_tolerance_nats):
    #   ref_gap  = lp_ref(r1) - lp_ref(c1)   how far the candidate's choice is behind in the reference (>= 0)
    #   cand_gap = lp_cand(c1) - lp_cand(r1) how far the reference's choice is behind in the candidate (>= 0)
    # None when the other token is not in that side's top-K (the gap is then unknown, at least the top-K span).
    # ref_gap + cand_gap is the change of the logit difference l(r1) - l(c1) between the engines: a flip caused by
    # numerical noise alone has ref_gap + cand_gap <= 2 * max_t |lp_ref(t) - lp_cand(t)|.
    # max_abs_dlogprob = max |lp_ref(t) - lp_cand(t)| over the tokens in both top-K lists (the measured noise).
    common = set(ref) & set(cand)
    return {"top1_agree": r1 == c1, "kl_topk": max(0.0, kl), "ref_in_cand": r1 in cand, "ref_top1": r1, "cand_top1": c1,
            "ref_gap": (ref[r1] - ref[c1]) if c1 in ref else None,
            "cand_gap": (cand[c1] - cand[r1]) if r1 in cand else None,
            "max_abs_dlogprob": max(abs(ref[t] - cand[t]) for t in common) if common else None,
            "n_common": len(common)}


def make_prefixes(texts, n, seed=1, min_chars=200, max_chars=20000):
    rng = random.Random(seed)
    out = []
    usable = [t for t in texts if len(t) > min_chars]
    if not usable:
        raise ValueError("corpus too small")
    for _ in range(n):
        t = rng.choice(usable)
        cut = rng.randint(min_chars, min(len(t), max_chars))
        out.append(t[:cut])
    return out


def summarize(rows):
    kls = sorted(r["kl_topk"] for r in rows)

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))]
    out = {"n": len(rows), "top1_agree": sum(r["top1_agree"] for r in rows) / len(rows),
           "ref_in_cand_topk": sum(r["ref_in_cand"] for r in rows) / len(rows),
           "kl_topk_mean": statistics.fmean(kls), "kl_topk_median": statistics.median(kls),
           "kl_topk_p95": pct(kls, 0.95), "kl_topk_max": kls[-1]}
    # the measured noise between the two engines (V6 tie tolerance calibration); absent in files written before N24
    d = sorted(r["max_abs_dlogprob"] for r in rows if r.get("max_abs_dlogprob") is not None)
    if d:
        out.update({"max_abs_dlogprob_n": len(d), "max_abs_dlogprob_median": statistics.median(d),
                    "max_abs_dlogprob_p95": pct(d, 0.95), "max_abs_dlogprob_p99": pct(d, 0.99),
                    "max_abs_dlogprob_max": d[-1]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref-model")
    ap.add_argument("--cand-model")
    ap.add_argument("--corpus", nargs="+", required=True, help="text files (globs allowed)")
    ap.add_argument("--prefixes", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-chars", type=int, default=20000)
    ap.add_argument("--api-key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = sorted({f for g in a.corpus for f in glob.glob(g)})
    texts = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            texts.append(fh.read())
    prefixes = make_prefixes(texts, a.prefixes, a.seed, max_chars=a.max_chars)
    rows, errors = [], []
    for i, pfx in enumerate(prefixes):
        try:
            r = complete_top(a.ref, pfx, a.top_k, a.ref_model, api_key=a.api_key)
            c = complete_top(a.cand, pfx, a.top_k, a.cand_model, api_key=a.api_key)
        except Exception as e:  # recorded, never invented
            errors.append({"prefix": i, "error": repr(e)})
            continue
        m = compare(r["top"], c["top"])
        m.update({"prefix": i, "chars": len(pfx), "ref_s": r["seconds"], "cand_s": c["seconds"]})
        rows.append(m)
    res = {"ref": a.ref, "cand": a.cand, "ref_model": a.ref_model, "cand_model": a.cand_model, "files": files,
           "top_k": a.top_k, "seed": a.seed, "prefixes": len(prefixes), "errors": errors,
           "summary": summarize(rows) if rows else None, "rows": rows,
           "_label": "CONFIRMED only as measured between these two endpoints; KL over top-K union, not full-vocabulary KLD"}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    if s:
        print(f"{s['n']} prefixes ({len(errors)} errors): top-1 agree {s['top1_agree']:.4f}, ref top-1 in cand top-{a.top_k} "
              f"{s['ref_in_cand_topk']:.4f}, KL(top-K) mean {s['kl_topk_mean']:.4f} median {s['kl_topk_median']:.4f} "
              f"p95 {s['kl_topk_p95']:.4f} max {s['kl_topk_max']:.4f} -> {a.out}")
    else:
        print(f"no successful prefixes ({len(errors)} errors) -> {a.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
