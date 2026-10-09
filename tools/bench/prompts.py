#!/usr/bin/env python3
"""Deterministic long-prompt builders that hit an exact token count (stdlib only).

Filler kinds (all generated from embedded word lists and a seed, nothing downloaded):
  prose  natural-ish English sentences and paragraphs
  code   synthetic Python "files" (header line per file, small functions and classes)
  logs   JSON-ish log lines

Token counting, in order of preference:
  server  POST /tokenize on the server under test (exact for that server's tokenizer)
  ratio   chars-per-token ratio calibrated from usage.prompt_tokens of two real requests;
          the fit is then an estimate, optionally refined with one request at full size
A document is a monotone function of the number of filler units n plus a few pad words,
so fitting is an interpolation/bisection search over n followed by single-word padding.

CLI: report the achieved count for each target, e.g.
  python3 tools/bench/prompts.py --url http://127.0.0.1:8080 --kind prose,code,logs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oa_client import OAClient  # noqa: E402

DEFAULT_TARGETS = [2048, 8192, 32768, 65536, 131072, 204800, 262144]   # 2K..262K, K = 1024
PAD_PIECE = " the"          # one token in the BPE vocabularies we know of; the fit verifies anyway

WORDS = (
    "about above across after again against almost along already also although always among "
    "amount analysis ancient another answer anyone anything appear approach area argue around "
    "article attention autumn available balance basic beauty because become before begin behind "
    "believe below beside between beyond border bridge bright broad build business calm careful "
    "carry center certain chance change chapter choice circle city clear climate close coast cold "
    "collect color common community complete concern consider continue control country course "
    "cover create culture current daily danger dark decide deep degree describe design detail "
    "develop different direction discover distance divide doctor during early earth easy economy "
    "edge effect effort either energy enough entire environment equal evening event every exactly "
    "example expect experience explain fact family famous farm feeling field figure final finally "
    "follow forest forward freedom friend future garden general gentle glass global government "
    "great green ground group growth habit half harbor heavy height history hollow horizon hour "
    "human idea image important include increase industry inside instead island journey kitchen "
    "knowledge language large later lead learn least letter level light likely limit listen local "
    "machine market material matter measure memory method middle million minute modern moment "
    "morning mountain movement music narrow nation natural nearly network never night normal north "
    "notice number object ocean offer office often order other outside paper particular partner "
    "pattern people perhaps period person picture piece place plain planet plant pleasure policy "
    "popular position possible power practice present pressure probably problem process produce "
    "program project public purpose quality question quiet quite rather reach reason record region "
    "remember report research result return river road rule safe sample science season second "
    "section series service several shape share short shoulder signal silent simple single slowly "
    "small social soft source south space special spring square stable station steady stone story "
    "strange stream street strong structure student study subject summer surface system table "
    "teacher theory thought through together toward tower trade travel trouble truth typical "
    "under unit until usual valley value various village visit voice water weather window winter "
    "within without wonder wooden world writer yellow young"
).split()

NOUNS = ("river falcon ember harbor lantern meadow orbit pebble quarry raven saddle thicket "
         "willow anchor beacon cinder delta fjord glacier hollow ingot juniper kestrel lagoon "
         "mesa nectar onyx prism quill ridge spruce tundra umber vertex walnut zephyr").split()
ADJS = ("amber bold calm dusty eager faint gentle hidden icy jolly keen lucid mellow noble "
        "olive proud quiet rapid silent tidy upper vivid warm young zesty").split()
VERBS = ("parse load merge scale shift build check fetch flush index pack route sort split "
         "store sync trace wrap apply clamp emit fold join map probe reduce scan seal tally").split()
SERVICES = ("auth billing cache gateway ingest ledger mailer notifier planner queue "
            "render scheduler search storage telemetry").split()
LOG_EVENTS = ("request completed", "cache miss", "retrying upstream call", "connection reset by peer",
              "slow query detected", "token refreshed", "job enqueued", "job finished",
              "rate limit applied", "health check ok", "config reloaded", "lease renewed")


class Filler:
    """Deterministic stream of text units; text(n) is the concatenation of the first n units."""

    def __init__(self, kind, seed):
        if kind not in ("prose", "code", "logs"):
            raise ValueError("unknown filler kind %r" % kind)
        self.kind = kind
        self.seed = seed
        self.rng = random.Random("filler:%s:%s" % (kind, seed))
        self.units = []
        self.boundary = []          # True where an insert may go before this unit
        self._file_no = 0
        self._fn_no = 0
        self._ts = 1791504000.0     # 2026-10-09T00:00:00Z, log timestamps start here

    # -- generators -----------------------------------------------------------
    def _prose_batch(self):
        r = self.rng
        words = [r.choice(WORDS) for _ in range(r.randint(6, 18))]
        words[0] = words[0].capitalize()
        out = []
        for i, w in enumerate(words):
            if i == len(words) - 1:
                tail = "." + ("\n\n" if r.random() < 0.18 else " ")
            elif r.random() < 0.08:
                tail = ", "
            else:
                tail = " "
            out.append(w + tail)
        return out, [True] + [False] * (len(out) - 1)

    def _code_batch(self):
        r = self.rng
        self._file_no += 1
        pkg = r.choice(NOUNS)
        path = "src/%s/%s_%s_%d.py" % (pkg, r.choice(ADJS), r.choice(NOUNS), self._file_no)
        lines = ["# ===== file: %s =====\n" % path, "import math\n", "\n"]
        for _ in range(r.randint(3, 7)):
            self._fn_no += 1
            if r.random() < 0.7:
                a, b, v = r.sample(["x", "y", "n", "acc", "lim", "rate", "size", "step"], 3)
                lines += [
                    "def %s_%s_%d(%s, %s):\n" % (r.choice(VERBS), r.choice(NOUNS), self._fn_no, a, b),
                    '    """%s the %s using the %s %s."""\n' % (r.choice(VERBS).capitalize(), r.choice(NOUNS),
                                                           r.choice(ADJS), r.choice(NOUNS)),
                    "    %s = %s * %d + %s\n" % (v, a, r.randint(2, 97), b),
                    "    if %s > %d:\n" % (v, r.randint(100, 9999)),
                    "        return %s - %d\n" % (v, r.randint(1, 500)),
                    "    return %s + math.floor(%s / %d)\n" % (v, b, r.randint(2, 31)),
                    "\n",
                ]
            else:
                attr = r.choice(["width", "depth", "limit", "count", "weight"])
                lines += [
                    "class %s%s%d:\n" % (r.choice(ADJS).capitalize(), r.choice(NOUNS).capitalize(), self._fn_no),
                    "    def __init__(self, %s):\n" % attr,
                    "        self.%s = %s\n" % (attr, attr),
                    "\n",
                    "    def %s(self):\n" % r.choice(VERBS),
                    "        return self.%s * %d\n" % (attr, r.randint(2, 64)),
                    "\n",
                ]
        bnd = [True] + [l == "\n" for l in lines[1:]]
        return lines, bnd

    def _logs_batch(self):
        r = self.rng
        self._ts += r.uniform(0.001, 2.5)
        t = time.gmtime(self._ts)
        ms = int((self._ts % 1) * 1000)
        line = ('{"ts": "%s.%03dZ", "level": "%s", "svc": "%s-%d", "event": "%s", "latency_ms": %d, '
                '"req": "%08x"}\n') % (time.strftime("%Y-%m-%dT%H:%M:%S", t), ms,
                                      r.choice(["INFO", "INFO", "INFO", "WARN", "DEBUG", "ERROR"]),
                                      r.choice(SERVICES), r.randint(1, 9), r.choice(LOG_EVENTS),
                                      r.randint(1, 4000), r.getrandbits(32))
        parts = re.findall(r"\S+\s*", line)          # word units: fine-grained fit, inserts only at line starts
        return parts, [True] + [False] * (len(parts) - 1)

    def ensure(self, n):
        gen = {"prose": self._prose_batch, "code": self._code_batch, "logs": self._logs_batch}[self.kind]
        while len(self.units) < n:
            u, b = gen()
            self.units.extend(u)
            self.boundary.extend(b)

    def text(self, n):
        self.ensure(n)
        return "".join(self.units[:n])

    def mean_unit_chars(self):
        self.ensure(200)
        return sum(len(u) for u in self.units[:200]) / 200.0

    def snap(self, i, n):
        """First boundary index >= i (within 400 units), clipped to [1, n]."""
        self.ensure(n + 1)
        j = i
        while j < n and j < i + 400 and not self.boundary[j]:
            j += 1
        if j >= n or not self.boundary[j]:
            j = i
        return min(max(j, 1), n) if n >= 1 else 0


def assemble(filler, n, header="", inserts=(), pad=0, suffix=""):
    """header + first n filler units with `inserts` [(depth in [0,1], text), ...] + pad words + suffix.

    Inserts keep their list order on ties and go on their own lines at unit boundaries."""
    filler.ensure(n + 1)
    units = filler.units
    if not inserts:
        body = "".join(units[:n])
    else:
        pos = []
        for i, (d, t) in enumerate(inserts):
            p = filler.snap(int(round(d * n)), n)
            pos.append((p, i, t))
        pos.sort()
        parts, prev = [], 0
        for p, _, t in pos:
            parts.append("".join(units[prev:p]))
            parts.append("\n" + t + "\n")
            prev = p
        parts.append("".join(units[prev:n]))
        body = "".join(parts)
    return header + body + PAD_PIECE * pad + suffix


# ---------------------------------------------------------------------------- counters
class CounterError(RuntimeError):
    pass


class ServerCounter:
    method = "server-tokenize"
    exact = True

    def __init__(self, client):
        self.client = client
        self.calls = 0

    def count(self, text):
        self.calls += 1
        n = self.client.tokenize_count(text)
        if n is None:
            raise CounterError("server /tokenize unavailable")
        return n


class RatioCounter:
    method = "chars-per-token-ratio"
    exact = False

    def __init__(self, chars_per_token):
        self.cpt = float(chars_per_token)
        self.calls = 0

    def count(self, text):
        self.calls += 1
        return int(round(len(text) / self.cpt))


class FnCounter:
    method = "local-function"
    exact = True

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def count(self, text):
        self.calls += 1
        return self.fn(text)


class FitResult(dict):
    """dict with: text, target, tokens, n_units, pad, exact, method, calls, seconds."""

    @property
    def text(self):
        return self["text"]

    def public(self):
        return {k: v for k, v in self.items() if k != "text"}


def fit(make, target, counter, n_hint=None, unit_chars=8.0, chars_per_token=4.0, max_calls=60):
    """Find (n, pad) with counter.count(make(n, pad)) == target when possible (<= target always).

    make(n, pad) -> str must be (close to) non-decreasing in n and pad."""
    t0 = time.perf_counter()
    memo = {}

    def c(n, pad=0):
        k = (n, pad)
        if k not in memo:
            memo[k] = counter.count(make(n, pad))
        return memo[k]

    base = c(0)
    res = FitResult(target=target, exact=counter.exact, method=counter.method)
    if base > target:
        res.update(text=make(0, 0), tokens=base, n_units=0, pad=0, calls=len(memo),
                   seconds=time.perf_counter() - t0, note="header+suffix alone exceed the target")
        return res
    lo, clo, hi, chi = 0, base, None, None
    n = max(1, int(n_hint) if n_hint else int((target - base) * chars_per_token / max(unit_chars, 1.0)))
    while hi is None and len(memo) < max_calls:
        cn = c(n)
        if cn <= target:
            lo, clo = n, cn
            grow = (target - base) / max(cn - base, 1)
            n = max(n + 1, int(n * grow * 1.02) + 2)
        else:
            hi, chi = n, cn
    same_side, last_side = 0, None
    while hi is not None and hi - lo > 1 and len(memo) < max_calls:
        if same_side >= 2 or chi == clo:
            g = (lo + hi) // 2
        else:
            g = lo + int((target - clo) * (hi - lo) / float(chi - clo))
        g = min(max(g, lo + 1), hi - 1)
        cg = c(g)
        side = "lo" if cg <= target else "hi"
        if side == "lo":
            lo, clo = g, cg
        else:
            hi, chi = g, cg
        same_side = same_side + 1 if side == last_side else 0
        last_side = side
    pad, cur = 0, clo
    while cur < target and len(memo) < max_calls:
        d = target - cur
        cp = c(lo, pad + d)
        if cp == target:
            pad, cur = pad + d, cp
            break
        if cp < target:
            if cp <= cur:
                break                                   # pad words add no tokens; give up
            pad, cur = pad + d, cp
            continue
        lo_k, hi_k = 0, d                               # overshoot: largest k with count <= target
        while hi_k - lo_k > 1 and len(memo) < max_calls:
            mid = (lo_k + hi_k) // 2
            if c(lo, pad + mid) <= target:
                lo_k = mid
            else:
                hi_k = mid
        pad += lo_k
        cur = c(lo, pad)
        break
    res.update(text=make(lo, pad), tokens=cur, n_units=lo, pad=pad, calls=len(memo),
               seconds=time.perf_counter() - t0)
    return res


def fit_filler(kind, seed, target, counter, header="", suffix="", inserts=(), n_hint=None):
    """Fit header + filler(kind, seed) [+ inserts] + suffix to `target` tokens."""
    f = Filler(kind, seed)

    def make(n, pad):
        return assemble(f, n, header=header, inserts=inserts, pad=pad, suffix=suffix)

    cpt = getattr(counter, "cpt", 4.0)
    res = fit(make, target, counter, n_hint=n_hint, unit_chars=f.mean_unit_chars(), chars_per_token=cpt)
    res["kind"] = kind
    res["seed"] = seed
    res["chars"] = len(res["text"])
    return res


# ---------------------------------------------------------------------------- server-side helpers
def usage_prompt_tokens(client, content, endpoint="chat", extra=None):
    """One real request with max_tokens=1; returns (usage.prompt_tokens or None, StreamResult)."""
    if endpoint == "chat":
        r = client.chat_stream([{"role": "user", "content": content}], max_tokens=1, extra=extra)
    else:
        r = client.completion_stream(content, max_tokens=1, extra=extra)
    pt = (r.usage or {}).get("prompt_tokens") if r.ok else None
    return (pt if isinstance(pt, int) else None), r


def make_counter(client, mode="auto", kind="prose", seed=0, endpoint="chat", extra=None, log=None):
    """Return (counter, overhead_tokens or None, info dict).

    overhead = usage.prompt_tokens - tokens(content) for a chat request (template tokens), measured."""
    info = {"requested": mode}
    log = log or (lambda row: None)
    if mode in ("auto", "server"):
        n = client.tokenize_count("Hello world.")
        info["tokenize_probe"] = client.tokenize_probe_log
        if n:
            counter = ServerCounter(client)
            content = "Reply with OK."
            pt, r = usage_prompt_tokens(client, content, endpoint, extra)
            log({"phase": "overhead_probe", "result": r.to_dict(keep_text=False)})
            overhead = None if pt is None else pt - counter.count(content)
            info.update(method=counter.method, tokenize_style=client.tokenize_style, overhead=overhead)
            if overhead is None:
                info["note"] = "usage.prompt_tokens missing; template overhead unknown, assumed 0 for targets"
            return counter, overhead, info
        if mode == "server":
            raise CounterError("--counter server requested but /tokenize is unusable")
    # ratio: two real requests, linear fit tokens = overhead + chars / cpt
    short = "Hi."
    long_text = Filler(kind, "calib:%s" % seed).text(2500)
    p1, r1 = usage_prompt_tokens(client, short, endpoint, extra)
    log({"phase": "calibration", "chars": len(short), "result": r1.to_dict(keep_text=False)})
    p2, r2 = usage_prompt_tokens(client, long_text, endpoint, extra)
    log({"phase": "calibration", "chars": len(long_text), "result": r2.to_dict(keep_text=False)})
    if p1 is None or p2 is None or p2 <= p1:
        info.update(method="chars-per-token-ratio", calibrated=False, overhead=None,
                    note="calibration failed (usage.prompt_tokens missing); default 4.0 chars/token")
        return RatioCounter(4.0), None, info
    cpt = (len(long_text) - len(short)) / float(p2 - p1)
    overhead = int(round(p1 - len(short) / cpt))
    info.update(method="chars-per-token-ratio", calibrated=True, chars_per_token=cpt, overhead=overhead,
                calibration_kind=kind)
    return RatioCounter(cpt), overhead, info


def refine_ratio(client, counter, fit_res, overhead, endpoint="chat", extra=None, log=None):
    """One full-size request (max_tokens=1) to re-calibrate a RatioCounter at this length.

    Returns the measured usage.prompt_tokens (or None)."""
    pt, r = usage_prompt_tokens(client, fit_res["text"], endpoint, extra)
    if log:
        log({"phase": "ratio_refine", "target": fit_res["target"], "result": r.to_dict(keep_text=False)})
    if pt is not None and pt - (overhead or 0) > 0:
        counter.cpt = len(fit_res["text"]) / float(pt - (overhead or 0))
    return pt


# ---------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key (optional)")
    ap.add_argument("--model", default=None, help="model id (default: first id from /v1/models)")
    ap.add_argument("--kind", default="prose", help="comma list of prose,code,logs")
    ap.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)), help="comma list of token targets")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--counter", choices=["auto", "server", "ratio"], default="auto")
    ap.add_argument("--verify", action="store_true",
                    help="also send each prompt once (max_tokens=1) and report usage.prompt_tokens")
    ap.add_argument("--save-text", default=None, help="directory to write the fitted texts to")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    client = OAClient(a.url, api_key=os.environ.get(a.api_key_env) or None, timeout=a.timeout, model=a.model)
    client.resolve_model()
    kinds = [k.strip() for k in a.kind.split(",") if k.strip()]
    targets = [int(t) for t in a.targets.split(",") if t.strip()]
    counter, overhead, info = make_counter(client, a.counter, kind=kinds[0], seed=a.seed)
    report = {"server": client.url, "model": client.model, "counter": info, "results": []}
    for kind in kinds:
        for t in targets:
            r = fit_filler(kind, a.seed, t, counter, header="Document %s-%s.\n" % (kind, a.seed))
            row = r.public()
            if a.verify:
                pt, _ = usage_prompt_tokens(client, r["text"])
                row["verify_usage_prompt_tokens"] = pt
                row["verify_note"] = "includes chat template tokens (measured overhead %s)" % overhead
            if a.save_text:
                os.makedirs(a.save_text, exist_ok=True)
                with open(os.path.join(a.save_text, "%s-%d.txt" % (kind, t)), "w", encoding="utf-8") as f:
                    f.write(r["text"])
            report["results"].append(row)
            line = {k: row[k] for k in ("kind", "target", "tokens", "exact", "method", "chars", "n_units", "pad",
                                        "calls")}
            if a.verify:
                line["verify_usage_prompt_tokens"] = row["verify_usage_prompt_tokens"]
            print(json.dumps(line), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    exact_miss = [r for r in report["results"] if r["exact"] and r["tokens"] != r["target"]]
    return 1 if exact_miss else 0


if __name__ == "__main__":
    sys.exit(main())
