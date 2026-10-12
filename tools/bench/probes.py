#!/usr/bin/env python3
"""Long-context and agent-quality probes against any OpenAI-compatible server (stdlib only).

Every probe is deterministic for a given --seed and produces score + pass/fail:
  needle       N needles (key -> random 6-digit value) at depths 5%..95% of a filler of the
               target length; ask for all of them; score = fraction exactly right
  multihop     RULER-style variable tracking: chains VAR_A = 12345, VAR_B = VAR_A, ...
               scattered through the context (plus distractor chains); ask final values
  codeqa       synthetic code base; specific functions return specific constants; ask which
               constant each returns and which file defines it (0.5 + 0.5 per function)
  json         JSON object for a small schema, validated with json.loads + key/type/value
               checks; tried without and with response_format={"type":"json_object"}
               (whether the server accepted response_format is recorded)
  tools        tools=[3 function schemas]; one call required; name + JSON arguments checked;
               the tool result is sent back and the final answer must use it
  determinism  the same prompt twice at temperature 0; identical output expected; reported,
               never a hard failure
needle/multihop/codeqa run at every --contexts length (prompt tokens fitted with the
server's /tokenize when available; the measured usage.prompt_tokens is recorded).
Output: run directory with meta.json, raw.jsonl, summary.json, summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402
from oa_client import OAClient, StreamResult  # noqa: E402
from prompts import ADJS, NOUNS, CounterError, fit_filler, make_counter  # noqa: E402

ALL_PROBES = ["needle", "multihop", "codeqa", "json", "tools", "determinism"]
LONG_PROBES = {"needle", "multihop", "codeqa"}
DEFAULT_THRESHOLDS = {"needle": 0.8, "multihop": 0.75, "codeqa": 0.75, "json": 1.0, "tools": 1.0}  # needle = K1
FIRST = "Alice Maria Olga Irene Nadia Clara Sofia Elena Vera Lidia".split()
LAST = "Smith Novak Larsen Moreau Okafor Tanaka Ivanova Silva Becker Haddad".split()
CITIES = ["Reykjavik", "Tromso", "Ushuaia", "Hobart", "Valparaiso", "Kirkenes", "Nuuk", "Yakutsk"]
CONDITIONS = ["freezing drizzle", "light snow", "volcanic haze", "gusty sleet", "clear and still", "dense fog"]

WEATHER_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_stock_price", "description": "Get the latest stock price for a ticker symbol.",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "search_docs", "description": "Search the internal documentation.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model", default=None)
    ap.add_argument("--probes", default=",".join(ALL_PROBES), help="comma list of " + ",".join(ALL_PROBES))
    ap.add_argument("--contexts", default="32768", help="token lengths for needle/multihop/codeqa (comma list)")
    ap.add_argument("--length-mode", choices=["prompt", "total"], default="total",
                    help="total (default): prompt = context - --answer-tokens, so prompt + answer fit the window")
    ap.add_argument("--filler", choices=["prose", "code", "logs"], default="prose", help="filler for needle/multihop")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--send-seed", type=int, default=None, help="also send this value as the request 'seed'")
    ap.add_argument("--answer-tokens", type=int, default=1024, help="max_tokens for answers (reasoning models need room)")
    ap.add_argument("--needles", type=int, default=10)
    ap.add_argument("--chains", type=int, default=4, help="multihop: asked chains")
    ap.add_argument("--hops", type=int, default=4, help="multihop: assignments after the literal")
    ap.add_argument("--distractors", type=int, default=4, help="multihop: chains not asked about")
    ap.add_argument("--code-questions", type=int, default=4)
    ap.add_argument("--det-tokens", type=int, default=1024, help="determinism: prompt tokens")
    ap.add_argument("--det-max-tokens", type=int, default=128)
    ap.add_argument("--threshold", default="", help="override pass thresholds, e.g. needle=1.0,codeqa=0.5 "
                    "(defaults %s)" % ",".join("%s=%s" % kv for kv in sorted(DEFAULT_THRESHOLDS.items())))
    ap.add_argument("--counter", choices=["auto", "server", "ratio"], default="auto")
    ap.add_argument("--extra-body", default="", help="JSON object (or @file) merged into every request body")
    ap.add_argument("--stream-options", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--timeout", type=float, default=3600.0, help="idle socket timeout per request, seconds")
    ap.add_argument("--total-timeout", type=float, default=None)
    ap.add_argument("--out-dir", default=None, help="run directory (default ./bench-runs/<UTC>-probes)")
    ap.add_argument("--engine-desc", default=None)
    ap.add_argument("--model-desc", default=None)
    ap.add_argument("--no-nvidia-smi", action="store_true")
    return ap


def _uniq(rng, make, n):
    out, seen = [], set()
    while len(out) < n:
        v = make(rng)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _depths(n, lo=0.05, hi=0.95):
    return [0.5] if n == 1 else [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def validate_person_json(content, expected, strict):
    """Returns dict(valid, score, parse, checks)."""
    text = (content or "").strip()
    obj, parse = None, None
    try:
        obj, parse = json.loads(text), "strict"
    except ValueError:
        if not strict:
            m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
            if m:
                try:
                    obj, parse = json.loads(m.group(1)), "after_cleanup"
                except ValueError:
                    pass
    checks = {"parsed": obj is not None, "is_object": isinstance(obj, dict)}
    if isinstance(obj, dict):
        checks["exact_keys"] = set(obj) == set(expected)
        types = {"name": str, "city": str, "tags": list}
        for k, t in types.items():
            checks["type_" + k] = isinstance(obj.get(k), t)
        checks["type_age"] = isinstance(obj.get("age"), int) and not isinstance(obj.get("age"), bool)
        checks["type_active"] = isinstance(obj.get("active"), bool)
        checks["tags_are_strings"] = isinstance(obj.get("tags"), list) and all(isinstance(x, str) for x in obj["tags"])
        for k in expected:
            v = obj.get(k)
            checks["value_" + k] = (v.strip() == expected[k]) if isinstance(v, str) else (v == expected[k])
    else:
        for k in ["exact_keys", "type_name", "type_city", "type_tags", "type_age", "type_active",
                  "tags_are_strings"] + ["value_" + k for k in expected]:
            checks[k] = False
    score = sum(1 for v in checks.values() if v) / float(len(checks))
    return {"valid": all(checks.values()), "score": score, "parse": parse, "checks": checks}


class ProbeRunner:
    def __init__(self, args, argv):
        self.a = args
        self.extra_body = runlog.parse_json_arg(args.extra_body)
        self.client = OAClient(args.url, api_key=os.environ.get(args.api_key_env) or None, timeout=args.timeout,
                               total_timeout=args.total_timeout, model=args.model,
                               stream_options=args.stream_options)
        out = args.out_dir or os.path.join("bench-runs", runlog.utc_stamp() + "-probes")
        self.rd = runlog.RunDir(out, "probes", argv, vars(args))
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        for part in [p for p in args.threshold.split(",") if p.strip()]:
            k, v = part.split("=")
            self.thresholds[k.strip()] = float(v)
        self.results = []
        self.seq = 0
        self.counter = None
        self.overhead = 0

    # ------------------------------------------------------------------ plumbing
    def log(self, row):
        self.seq += 1
        row = dict(row, seq=self.seq, utc=runlog.utc_now())
        self.rd.raw(row)
        return row

    def ask(self, messages, max_tokens=None, **kw):
        try:
            return self.client.chat_stream(messages, max_tokens or self.a.answer_tokens,
                                           temperature=self.a.temperature, extra=self.extra_body,
                                           seed=self.a.send_seed, **kw)
        except Exception as e:  # noqa: BLE001
            r = StreamResult("internal")
            r.error = "harness exception: %s: %s" % (type(e).__name__, e)
            r.notes.append(traceback.format_exc()[-2000:])
            return r

    def record(self, probe, res, step, context=None, **fields):
        d = res.to_dict(keep_text=True, max_text=20000)
        m = runlog.request_metrics(d)
        return self.log(dict(fields, scenario="probe", probe=probe, step=step, context=context, result=d, metrics=m))

    def finish_probe(self, probe, context, score, hard, details, rows, prompt_tokens=None, error=None):
        thr = self.thresholds.get(probe)
        passed = (error is None and score is not None and (thr is None or score >= thr))
        res = {"probe": probe, "context": context, "score": score, "threshold": thr, "pass": passed,
               "hard": hard, "error": error, "prompt_tokens": prompt_tokens, "details": details,
               "raw_seqs": [r["seq"] for r in rows],
               "ttft_ms": rows[0]["metrics"]["ttft_ms"] if rows else None,
               "decode_tok_s": rows[0]["metrics"]["decode_tok_s"] if rows else None,
               "finish_reason": rows[0]["metrics"]["finish_reason"] if rows else None}
        self.results.append(res)
        self.log({"scenario": "probe", "phase": "probe_result", **{k: v for k, v in res.items()}})
        return res

    def target(self, ctx):
        t = ctx - (self.a.answer_tokens if self.a.length_mode == "total" else 0) - self.overhead
        return max(t, 64)

    def long_doc(self, probe, kind, ctx, inserts, question):
        header = "Document %s.\n" % random.Random("nonce:%s:%s:%d" % (probe, self.a.seed, ctx)).getrandbits(64)
        fr = fit_filler(kind, "%s:%s:%d" % (probe, self.a.seed, ctx), self.target(ctx), self.counter,
                        header=header, suffix=question, inserts=inserts)
        text = fr["text"]
        missing = [t for _, t in inserts if text.count("\n" + t + "\n") != 1]
        return text, fr.public(), missing

    def _single(self, probe, ctx, text, fit, missing, grade):
        if missing:
            return self.finish_probe(probe, ctx, None, True, {"missing_inserts": missing[:5], "fit": fit}, [],
                                     error="harness: %d inserts missing from the fitted document" % len(missing))
        r = self.ask([{"role": "user", "content": text}])
        row = self.record(probe, r, "answer", ctx, fit=fit)
        if not r.ok:
            return self.finish_probe(probe, ctx, None, True, {"fit": fit}, [row],
                                     row["metrics"]["prompt_tokens"], error=r.error)
        score, details = grade(r.content)
        details["fit"] = fit
        if not r.content.strip():
            details["note"] = "empty content (reasoning chars %d, finish %s)" % (len(r.reasoning), r.finish_reason)
        return self.finish_probe(probe, ctx, score, True, details, [row], row["metrics"]["prompt_tokens"])

    # ------------------------------------------------------------------ probes
    def probe_needle(self, ctx):
        rng = random.Random("needle:%s:%d" % (self.a.seed, ctx))
        n = self.a.needles
        keys = _uniq(rng, lambda r: "%s-%s-%02d" % (r.choice(ADJS), r.choice(NOUNS), r.randint(10, 99)), n)
        vals = _uniq(rng, lambda r: "%06d" % r.randint(100000, 999999), n)
        depths = _depths(n)
        inserts = [(d, "The secret code for %s is %s." % (k, v)) for d, k, v in zip(depths, keys, vals)]
        asked = keys[:]
        rng.shuffle(asked)
        q = ("\n\n---\nQuestion: The document above contains secret codes for several identifiers. Report the "
             "secret code of each of these identifiers: %s. Answer with exactly one line per identifier in the form "
             "<identifier>: <code>, and nothing else." % ", ".join("`%s`" % k for k in asked))
        text, fit, missing = self.long_doc("needle", self.a.filler, ctx, inserts, q)
        truth = dict(zip(keys, vals))

        def grade(content):
            items = []
            for k, d in zip(keys, depths):
                m = re.search(re.escape(k) + r"\D{0,40}?(\d{6})", content)
                got = m.group(1) if m else None
                items.append({"key": k, "depth": round(d, 3), "expected": truth[k], "got": got,
                              "correct": got == truth[k]})
            return sum(i["correct"] for i in items) / float(n), {"items": items}

        return self._single("needle", ctx, text, fit, missing, grade)

    def probe_multihop(self, ctx):
        rng = random.Random("multihop:%s:%d" % (self.a.seed, ctx))
        n_chains = self.a.chains + self.a.distractors
        names = _uniq(rng, lambda r: "VAR_" + "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)),
                      n_chains * (self.a.hops + 1))
        values = _uniq(rng, lambda r: str(r.randint(10000, 99999)), n_chains)
        inserts, chains = [], []
        for c in range(n_chains):
            vs = names[c * (self.a.hops + 1):(c + 1) * (self.a.hops + 1)]
            ds = sorted(rng.uniform(0.03, 0.97) for _ in vs)
            inserts.append((ds[0], "%s = %s" % (vs[0], values[c])))
            for i in range(1, len(vs)):
                inserts.append((ds[i], "%s = %s" % (vs[i], vs[i - 1])))
            chains.append({"vars": vs, "value": values[c], "asked": c < self.a.chains})
        inserts.sort(key=lambda x: x[0])
        asked = [ch["vars"][-1] for ch in chains if ch["asked"]]
        truth = {ch["vars"][-1]: ch["value"] for ch in chains if ch["asked"]}
        q = ("\n\n---\nQuestion: Some lines in the document above assign variables, either a number "
             "(like VAR_P = 123) or another variable (like VAR_Q = VAR_P, which copies the value of VAR_P). "
             "What is the final numeric value of each of these variables: %s? Answer with exactly one line per "
             "variable in the form <variable>: <number>, and nothing else." % ", ".join("`%s`" % v for v in asked))
        text, fit, missing = self.long_doc("multihop", self.a.filler, ctx, inserts, q)

        def grade(content):
            items = []
            for v in asked:
                m = re.search(re.escape(v) + r"\D{0,40}?(\d{5})(?!\d)", content)
                got = m.group(1) if m else None
                items.append({"var": v, "expected": truth[v], "got": got, "correct": got == truth[v]})
            return sum(i["correct"] for i in items) / float(len(items)), {"items": items, "chains": chains}

        return self._single("multihop", ctx, text, fit, missing, grade)

    def probe_codeqa(self, ctx):
        rng = random.Random("codeqa:%s:%d" % (self.a.seed, ctx))
        n = self.a.code_questions
        names = _uniq(rng, lambda r: "resolve_%s_%s_constant" % (r.choice(ADJS), r.choice(NOUNS)), n)
        consts = _uniq(rng, lambda r: r.randint(1000000, 9999999), n)
        depths = _depths(n, 0.1, 0.9)
        rng.shuffle(depths)
        inserts = sorted(((d, "def %s():\n    return %d" % (nm, c)) for d, nm, c in zip(depths, names, consts)),
                         key=lambda x: x[0])
        asked = names[:]
        rng.shuffle(asked)
        q = ("\n\n# ---\nQuestion: The code base above defines functions that return constants. Which constant does "
             "each of these functions return, and which file defines it: %s? Answer with exactly one line per "
             "function in the form <function>: <constant> in <file path>, and nothing else."
             % ", ".join("`%s`" % x for x in asked))
        text, fit, missing = self.long_doc("codeqa", "code", ctx, inserts, q)
        truth = {}
        for nm, c in zip(names, consts):
            pos = text.find("def %s():" % nm)
            head = text.rfind("# ===== file: ", 0, pos) if pos >= 0 else -1
            path = re.match(r"# ===== file: (\S+) =====", text[head:]).group(1) if head >= 0 else None
            truth[nm] = {"constant": c, "file": path}

        def grade(content):
            items, total = [], 0.0
            lines = content.splitlines()
            for nm in names:
                line = next((l for l in lines if nm in l), "")
                c_ok = str(truth[nm]["constant"]) in re.findall(r"\d+", line)
                f_ok = bool(truth[nm]["file"]) and truth[nm]["file"] in line
                total += 0.5 * c_ok + 0.5 * f_ok
                items.append({"function": nm, "expected": truth[nm], "line": line[:300], "constant_ok": c_ok,
                              "file_ok": f_ok})
            return total / float(n), {"items": items}

        return self._single("codeqa", ctx, text, fit, missing, grade)

    def probe_json(self):
        rng = random.Random("json:%s" % self.a.seed)
        exp = {"name": "%s %s" % (rng.choice(FIRST), rng.choice(LAST)), "age": rng.randint(19, 88),
               "city": rng.choice(CITIES), "tags": rng.sample(NOUNS, 3), "active": rng.random() < 0.5}
        facts = ("The customer's name is %s. She is %d years old and lives in %s. Her account tags are %s, %s and %s. "
                 "Her account is %s." % (exp["name"], exp["age"], exp["city"], exp["tags"][0], exp["tags"][1],
                                         exp["tags"][2], "active" if exp["active"] else "inactive"))
        prompt = ('Extract the facts below into a JSON object with exactly these keys: "name" (string), "age" '
                  '(integer), "city" (string), "tags" (array of strings, in the order given), "active" (boolean). '
                  "Output only the JSON object, with no prose and no code fences.\n\nFacts: %s\n\n"
                  "Question: produce the JSON object." % facts)
        msgs = [{"role": "user", "content": prompt}]
        rows, attempts = [], {}
        r1 = self.ask(msgs)
        rows.append(self.record("json", r1, "plain"))
        attempts["plain"] = ({"ok": True, **validate_person_json(r1.content, exp, strict=False)} if r1.ok
                             else {"ok": False, "error": r1.error})
        r2 = self.ask(msgs, response_format={"type": "json_object"})
        rows.append(self.record("json", r2, "response_format_json_object"))
        rf_accepted = r2.ok or (r2.http_status == 200)
        if r2.http_status in (400, 422):
            rf_accepted = False
        attempts["response_format"] = {"accepted": rf_accepted, "http_status": r2.http_status}
        if r2.ok:
            attempts["response_format"].update(validate_person_json(r2.content, exp, strict=True))
        else:
            attempts["response_format"]["error"] = r2.error
        error = None if r1.ok else r1.error
        if r2.http_status not in (200, 400, 422) and not r2.ok:
            error = error or ("response_format attempt failed: %s" % r2.error)
        scores = [attempts["plain"]["score"]] if r1.ok else []
        if r2.ok:
            scores.append(attempts["response_format"]["score"])
        score = (sum(scores) / len(scores)) if scores else None
        valid_all = r1.ok and attempts["plain"]["valid"] and (not r2.ok or attempts["response_format"]["valid"])
        details = {"expected": exp, "attempts": attempts, "response_format_accepted": rf_accepted,
                   "all_attempted_valid": valid_all}
        res = self.finish_probe("json", None, score, True, details, rows, rows[0]["metrics"]["prompt_tokens"], error)
        if res["pass"] and not valid_all:
            res["pass"] = False
        return res

    def probe_tools(self):
        rng = random.Random("tools:%s" % self.a.seed)
        city, temp, cond = rng.choice(CITIES), round(rng.uniform(-20, 35), 1), rng.choice(CONDITIONS)
        msgs = [{"role": "user", "content": "What is the weather like in %s right now? Use the available tools "
                                            "and report the temperature in Celsius." % city}]
        rows = []
        r1 = self.ask(msgs, tools=WEATHER_TOOLS, tool_choice="auto")
        rows.append(self.record("tools", r1, "call"))
        if not r1.ok:
            return self.finish_probe("tools", None, None, True, {}, rows, rows[0]["metrics"]["prompt_tokens"], r1.error)
        call = {"n_tool_calls": len(r1.tool_calls), "name_ok": False, "args_json_ok": False, "city_ok": False}
        tc = r1.tool_calls[0] if r1.tool_calls else None
        args = None
        if tc:
            call["name"] = tc["function"]["name"]
            call["arguments"] = tc["function"]["arguments"]
            call["name_ok"] = tc["function"]["name"] == "get_weather"
            try:
                args = json.loads(tc["function"]["arguments"] or "")
                call["args_json_ok"] = isinstance(args, dict)
            except ValueError:
                pass
            if isinstance(args, dict):
                call["city_ok"] = city.lower() in str(args.get("city", "")).lower()
        call_ok = call["name_ok"] and call["args_json_ok"] and call["city_ok"]
        synthetic = not call_ok
        tc_id = (tc or {}).get("id") or "call_0"
        arguments = tc["function"]["arguments"] if call_ok else json.dumps({"city": city, "unit": "celsius"})
        msgs2 = msgs + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": tc_id, "type": "function",
                                                                 "function": {"name": "get_weather",
                                                                              "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": tc_id,
             "content": json.dumps({"city": city, "temperature_c": temp, "condition": cond})}]
        r2 = self.ask(msgs2, tools=WEATHER_TOOLS)
        rows.append(self.record("tools", r2, "final", synthetic_call=synthetic))
        uses = False
        if r2.ok:
            forms = {"%.1f" % temp}
            if float(temp).is_integer():
                forms.add("%d" % int(temp))
            uses = any(re.search(r"(?<![\d.])%s(?![\d])" % re.escape(f), r2.content) for f in forms)
        score = 0.5 * call_ok + 0.5 * uses
        details = {"expected": {"city": city, "temperature_c": temp, "condition": cond}, "call": call,
                   "call_ok": call_ok, "second_turn_used_synthetic_call": synthetic, "final_uses_result": uses,
                   "final_content": r2.content[:500]}
        return self.finish_probe("tools", None, score, True, details, rows, rows[0]["metrics"]["prompt_tokens"],
                                 None if r2.ok else r2.error)

    def probe_determinism(self):
        fr = fit_filler("prose", "det:%s" % self.a.seed, max(self.a.det_tokens - self.overhead, 64), self.counter,
                        header="Story %s.\n" % self.a.seed,
                        suffix="\n\nQuestion: Continue the text above for a few sentences.")
        msgs = [{"role": "user", "content": fr["text"]}]
        r1 = self.ask(msgs, max_tokens=self.a.det_max_tokens)
        r2 = self.ask(msgs, max_tokens=self.a.det_max_tokens)
        rows = [self.record("determinism", r1, "first"), self.record("determinism", r2, "second")]
        if not (r1.ok and r2.ok):
            return self.finish_probe("determinism", None, None, False, {}, rows,
                                     rows[0]["metrics"]["prompt_tokens"], r1.error or r2.error)
        a, b = r1.reasoning + "\x00" + r1.content, r2.reasoning + "\x00" + r2.content
        same = a == b
        diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if same else min(len(a), len(b)))
        details = {"identical": same, "first_diff_char": diff, "len_first": len(a), "len_second": len(b),
                   "sha256_first": runlog.text_digest(a), "sha256_second": runlog.text_digest(b),
                   "cached_tokens_second": rows[1]["metrics"]["cached_tokens"]}
        res = self.finish_probe("determinism", None, 1.0 if same else 0.0, False, details, rows,
                                rows[0]["metrics"]["prompt_tokens"])
        res["pass"] = same
        return res

    # ------------------------------------------------------------------ main flow
    def setup(self):
        self.rd.collect_env(self.client, self.a.engine_desc, self.a.model_desc, smi=not self.a.no_nvidia_smi)
        model = self.client.resolve_model()
        self.rd.meta["model"] = model
        self.rd.meta["thresholds"] = self.thresholds
        if not model:
            self.rd.meta["fatal"] = "cannot resolve a model id from /v1/models; pass --model"
            return False
        try:
            self.counter, overhead, info = make_counter(self.client, self.a.counter, kind=self.a.filler,
                                                        seed=self.a.seed, extra=self.extra_body,
                                                        log=lambda row: self.log(dict(row, scenario="setup")))
        except CounterError as e:
            self.rd.meta["fatal"] = str(e)
            return False
        self.overhead = overhead or 0
        self.rd.meta["token_counter"] = info
        self.rd.write_meta()
        return True

    def run(self):
        probes = [p.strip() for p in self.a.probes.split(",") if p.strip()]
        for p in probes:
            if p not in ALL_PROBES:
                raise SystemExit("unknown probe %r" % p)
        for p in probes:
            if p in LONG_PROBES:
                for ctx in runlog.parse_int_list(self.a.contexts):
                    getattr(self, "probe_" + p)(ctx)
            else:
                getattr(self, "probe_" + p)()
        self.rd.meta["stream_options_supported"] = self.client.stream_options_supported

    def summarize(self):
        hard = [r for r in self.results if r["hard"]]
        summary = {"schema": runlog.SCHEMA, "results": self.results, "thresholds": self.thresholds,
                   "all_hard_pass": all(r["pass"] for r in hard),
                   "n_hard": len(hard), "n_errors": sum(1 for r in self.results if r["error"]),
                   "is_mock_server": self.rd.meta.get("is_mock_server"),
                   "token_counter": self.rd.meta.get("token_counter")}
        self.rd.write_json("summary.json", summary)
        tc = summary.get("token_counter") or {}
        lines = runlog.md_header(self.rd.meta, "Probe run", [
            "- token counting: `%s`; length mode `%s`; answer max_tokens %d; temperature %s" % (
                tc.get("method"), self.a.length_mode, self.a.answer_tokens, self.a.temperature),
            "- hard probes: %d, all pass: %s; probes with errors: %d" % (
                summary["n_hard"], summary["all_hard_pass"], summary["n_errors"])])
        rows = []
        for r in self.results:
            note = r["error"] or ""
            if r["probe"] == "json":
                note = note or "response_format accepted: %s" % r["details"].get("response_format_accepted")
            if r["probe"] == "determinism" and r["details"]:
                note = note or "identical: %s, first diff char: %s" % (r["details"].get("identical"),
                                                                     r["details"].get("first_diff_char"))
            if r["details"].get("note"):
                note = (note + " " + r["details"]["note"]).strip()
            rows.append([r["probe"], runlog.fmt(r["context"]), runlog.fmt(r["prompt_tokens"]),
                         runlog.fmt(r["score"], 3), runlog.fmt(r["threshold"], 2),
                         ("PASS" if r["pass"] else "FAIL") + ("" if r["hard"] else " (info)"),
                         runlog.fmt(r["ttft_ms"]), runlog.fmt(r["decode_tok_s"]), r["finish_reason"] or "n/a",
                         note.replace("|", "/").replace("\n", " ")[:200]])
        lines.append(runlog.md_table(["probe", "context target", "prompt tok (measured)", "score", "threshold",
                                      "result", "TTFT ms", "decode tok/s", "finish", "notes"], rows))
        lines += ["", "Scores are computed from the literal answers stored in raw.jsonl (field result.content).",
                  "Context target is the window (prompt + answer budget) when length mode is `total`."]
        self.rd.write_text("summary.md", "\n".join(lines) + "\n")
        return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    run = ProbeRunner(args, argv)
    status, code = "ok", 0
    try:
        if not run.setup():
            status, code = "fatal", 2
        else:
            run.run()
    except KeyboardInterrupt:
        status, code = "interrupted", 130
    finally:
        run.rd.finish(status, run.client)
        s = run.summarize()
        run.rd.close()
    if code == 0 and (not s["all_hard_pass"] or s["n_errors"] or not s["results"]):
        code = 1
    print("run dir: %s (status %s, all hard probes pass: %s)" % (run.rd.path, status, s["all_hard_pass"]))
    return code


if __name__ == "__main__":
    sys.exit(main())
