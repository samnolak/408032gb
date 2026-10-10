#!/usr/bin/env python3
"""Plain-text views of hybrid_server_checks.py result files, for evidence (values printed as stored, nothing derived
except the comparisons named below).

  lines <result.json> ...            one line per check
  attr <name>=<result.json> ...      precision attribution: warm-vs-cold |delta log-prob| per check for every run,
                                     and each run's cold answer against the FIRST run (the reference) compared only
                                     up to the first differing token - later steps have different histories
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def lines(path):
    r = load(path)
    out = [f"## {r['model'].rsplit('/', 1)[-1]}: server_args {' '.join(r['server_args'][:-2])}",
           "verdict " + json.dumps(r["verdict"])]
    if "error" in r:
        return out + ["error " + r["error"]]
    d, pe, pd = r["determinism"]["cmp"], r["prefix_extend"], r["prefix_diverge"]
    cr, sp, iso = r["cache_reuse"], r["slot_persist"], r["isolation"]
    out.append(f"determinism      max|d| {d['max_abs_dlogprob']} same_tokens {d['same_tokens']}")
    out.append(f"prefix_extend    n_next {pe['n_next']} reusable {pe['reusable_tokens']} warm prompt_n "
               f"{pe['warm']['prompt_n']} cache_n {pe['warm']['cache_n']} reused {pe['reused_tokens']} | max|d| "
               f"{pe['cmp']['max_abs_dlogprob']} same_tokens {pe['cmp']['same_tokens']}")
    out.append(f"prefix_diverge   n_next {pd['n_next']} shared {pd['shared_prefix']} warm prompt_n {pd['warm']['prompt_n']} "
               f"cache_n {pd['warm']['cache_n']} reused {pd['reused_tokens']} | max|d| {pd['cmp']['max_abs_dlogprob']} "
               f"same_tokens {pd['cmp']['same_tokens']}")
    out.append(f"cache_reuse      n_next {cr['n_next']} deleted {cr['deleted']} warm prompt_n {cr['warm']['prompt_n']} "
               f"reused {cr['reused_tokens']} flag_disabled {cr['flag_disabled']} chunk_shifted {cr['chunk_shifted']} "
               f"forced_full {cr['forced_full_reprocess']} | max|d| {cr['cmp']['max_abs_dlogprob']}")
    out.append(f"slot_persist     same-process identical {sp['run_once']['same_continuation']} warm prompt_n "
               f"{sp['run_once']['warm']['prompt_n']} | restart: same_tokens {sp['cmp_restart']['same_tokens']} max|d| "
               f"{sp['cmp_restart']['max_abs_dlogprob']} warm prompt_n {sp['after_restart']['prompt_n']} | file "
               f"{sp['file_bytes']} B | exits {sp['server_exit']}")
    out.append(f"isolation        server max_busy_slots {iso.get('max_busy_slots')} client overlap {iso['overlap_s']:.3f} s "
               f"same_tokens {[c.get('same_tokens') for c in iso['cmp']]} "
               f"max|d| {[c.get('max_abs_dlogprob') for c in iso['cmp']]}")
    if "mtp" in r:
        m = r["mtp"]
        if "cmp" in m:
            out.append(f"mtp              {' '.join(m['mtp_args'])} on_exit {m['on_exit']} off_exit {m['off_exit']} | "
                       f"draft_n {m['on']['draft_n']} accepted {m['on']['draft_n_accepted']} | same_tokens "
                       f"{m['cmp']['same_tokens']} max|d| {m['cmp']['max_abs_dlogprob']} | repeat same_tokens "
                       f"{m['cmp_repeat']['same_tokens']}")
        else:
            out.append(f"mtp              {' '.join(m['mtp_args'])} on_exit {m.get('on_exit')} error {m.get('error')}")
        out += ["mtp server log   " + ln.split(" I ", 1)[-1].strip() for ln in m["log_mentions_mtp"]
                if "creating MTP" in ln or "acceptance" in ln]
    return out


def _upto_first_diff(a, b):
    ta, tb = a["tokens"], b["tokens"]
    fd = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), len(ta))
    ds = [max([abs(sa[k] - sb[k]) for k in set(sa) & set(sb)] or [0.0]) for sa, sb in list(zip(a["top"], b["top"]))[:fd + 1]]
    return fd, ds[0], max(ds)


def attr(runs):
    ref = load(runs[0][1])
    out = ["%-22s %-11s %-11s %-11s %-11s | %s" % ("config", "extend", "isolation", "determin.", "mtp",
           "cold 256-token prompt vs the first run: first differing token, |d| at step 0, max |d| up to it")]
    for name, path in runs:
        r = load(path)
        iso = max(c["max_abs_dlogprob"] for c in r["isolation"]["cmp"])
        m = r.get("mtp", {}).get("cmp", {}).get("max_abs_dlogprob")
        fd, s0, mx = _upto_first_diff(r["prefix_extend"]["cold"], ref["prefix_extend"]["cold"])
        out.append("%-22s %-11.3g %-11.3g %-11.3g %-11s | %d, %.3g, %.3g" % (
            name, r["prefix_extend"]["cmp"]["max_abs_dlogprob"], iso, r["determinism"]["cmp"]["max_abs_dlogprob"],
            "-" if m is None else "%.3g" % m, fd, s0, mx))
    return out


def main(argv):
    if len(argv) < 2 or argv[0] not in ("lines", "attr"):
        print(__doc__)
        return 2
    if argv[0] == "lines":
        for p in argv[1:]:
            print("\n".join(lines(p)) + "\n")
    else:
        print("\n".join(attr([a.split("=", 1) for a in argv[1:]])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
