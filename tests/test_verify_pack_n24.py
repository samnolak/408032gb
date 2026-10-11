"""N24 rules of tools/verify/pack.py: declared synthetic runs and the V6 near-tie tolerance; logprob_compare gap fields.

EVERY FIXTURE HERE IS SYNTHETIC TEST DATA, NOT EVIDENCE. The bench runs are the ones tests/test_verify_pack.py generates
with the real tools against tools/bench/mock_server.py (its setUpModule is reused), edited here only where a test needs
a probe score, a probe request error or a model description changed (summary.json and raw.jsonl edited together, so the
pack's summary-vs-raw check still passes). The logprob artifacts are written by tools/quality/logprob_compare.py against
stdlib stub servers in this file, or built directly where a test needs exact gaps.
Run: python3 -m unittest tests.test_verify_pack_n24 -v   (from the repo root)
"""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_verify_pack as T  # noqa: E402  (fixtures; its module setup is called below)

vp, logprob_compare = T.vp, T.logprob_compare
FIX = None


def setUpModule():
    global FIX
    T.setUpModule()
    FIX = tempfile.mkdtemp(prefix="n24-SYNTHETIC-")


def tearDownModule():
    T.tearDownModule()
    if FIX:
        shutil.rmtree(FIX, ignore_errors=True)


def edit_probes(src, dst, scores=None, error=None, model_desc=None):
    """Copy a probes run; set probe scores / a request error in summary.json and raw.jsonl alike; optionally replace
    meta.model_desc. SYNTHETIC."""
    shutil.copytree(src, dst)
    sp, rp, mp = (os.path.join(dst, n) for n in ("summary.json", "raw.jsonl", "meta.json"))
    with open(sp) as f:
        summ = json.load(f)
    with open(rp) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    def fix(x):
        p = x.get("probe")
        if scores and p in scores:
            x["score"] = scores[p]
            x["pass"] = False
        if error and p == error[0]:
            x["error"] = error[1]
    for x in summ["results"]:
        fix(x)
    for r in rows:
        if r.get("phase") == "probe_result":
            fix(r)
    with open(sp, "w") as f:
        json.dump(summ, f, indent=2, sort_keys=True)
    with open(rp, "w") as f:
        f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    if model_desc is not None:
        with open(mp) as f:
            meta = json.load(f)
        meta["model_desc"] = model_desc
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    return dst


def criteria_file(name, **kw):
    p = os.path.join(FIX, "criteria-SYNTHETIC-%s.json" % name)
    with open(p, "w") as f:
        json.dump(dict(T.SMALL, **kw), f)
    return p


# The N11 Run B2 probe scores (docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md section 3.5): random weights.
N11_SCORES = {"tools": 0.0, "json": 0.071}


class Case(T.PackCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="case-SYNTHETIC-", dir=FIX)

    def build_with(self, runs, crit_path, name="pack"):
        out = os.path.join(self.tmp, name)
        code, text = T.quiet(vp.main, ["--runs", runs, "--out", out, "--criteria", crit_path])
        with open(os.path.join(out, "verdict.json")) as f:
            v = json.load(f)
        with open(os.path.join(out, "VERDICT.md")) as f:
            md = f.read()
        return code, v, md

    def probes_variant(self, key, **kw):
        dst = os.path.join(self.tmp, "src", "probes-%s-SYNTHETIC" % key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        T.G["n24_" + key] = edit_probes(T.G["probes"], dst, **kw)
        return "n24_" + key


class CriteriaValidation(unittest.TestCase):
    def test_defaults(self):
        crit, dev = vp.load_criteria()
        self.assertIs(crit["synthetic_run"], False)
        self.assertEqual(set(crit["synthetic_not_applicable"]), set(vp.BEHAVIOUR_CRITERIA))
        self.assertEqual(crit["logprob_tie_tolerance_nats"], 0.025)
        self.assertEqual(dev, {})

    def test_only_behaviour_criteria_may_be_not_applicable(self):
        for bad in (["weights_integrity"], ["agent_api_probes", "decode_filled_ctx_floor"], ["exact_routing_logprobs"],
                    "agent_api_probes"):
            with self.assertRaises(ValueError, msg=bad):
                vp.load_criteria(criteria_file("bad-na", synthetic_not_applicable=bad))

    def test_synthetic_flag_and_tolerance_types(self):
        for kw in ({"synthetic_run": "yes"}, {"synthetic_run": 1}, {"logprob_tie_tolerance_nats": -0.001},
                   {"logprob_tie_tolerance_nats": True}, {"logprob_tie_tolerance_nats": "0.02"},
                   {"logprob_tie_tolerance_nats": 2.0}):
            with self.assertRaises(ValueError, msg=kw):
                vp.load_criteria(criteria_file("bad-type", **kw))


class SyntheticRun(Case):
    def test_n11_cause_reproduced_random_weights_fail_agent_probes(self):
        """Without a declaration the pack says what N11 Run B2 said: agent_api_probes FAIL, overall FAIL."""
        k = self.probes_variant("n11", scores=N11_SCORES)
        runs = self.make_runs(**dict(T.PASSING, probes=k))
        code, v, md = self.build_with(runs, criteria_file("plain"))
        r = self.by_id(v)["agent_api_probes"]
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertIn("tools 0.000", r["measured"])
        self.assertEqual(v["overall"], "FAIL")
        self.assertEqual(code, 1)
        self.assertEqual(v["synthetic_run"], {"declared": False})

    def test_declared_synthetic_makes_behaviour_fail_not_applicable_and_lists_it(self):
        k = self.probes_variant("n11", scores=N11_SCORES)
        runs = self.make_runs(**dict(T.PASSING, probes=k))
        code, v, md = self.build_with(runs, criteria_file("synth", synthetic_run=True))
        res = self.by_id(v)
        r = res["agent_api_probes"]
        self.assertEqual(r["status"], "NOT RUN")
        self.assertEqual(r["reason_code"], "not_applicable_synthetic")
        self.assertEqual(r["details"]["synthetic_raw"]["raw_status"], "FAIL")
        self.assertIn("raw verdict FAIL", r["reason"])
        sr = v["synthetic_run"]
        self.assertTrue(sr["declared"] and sr["applied"], sr)
        listed = {x["id"]: x for x in sr["not_applicable"]}
        self.assertEqual(listed["agent_api_probes"]["raw_status"], "FAIL")
        # the long probes passed on the mock fixture: PASS is not applicable either, and listed with its raw verdict
        self.assertEqual(res["context_probes_full_length"]["reason_code"], "not_applicable_synthetic")
        self.assertEqual(listed["context_probes_full_length"]["raw_status"], "PASS")
        # engine/API criteria are judged as usual
        for cid in ("openai_api", "decode_filled_ctx_floor", "exact_routing_logprobs", "kv_slot_persist"):
            self.assertEqual(res[cid]["status"], "PASS", cid)
        self.assertEqual(v["overall"], "NOT RUN")
        self.assertIn("declared synthetic run", v["overall_reason"])
        self.assertEqual(code, 3)
        self.assertIn("synthetic_run", v["criteria_deviations"])
        self.assertIn("объявлен СИНТЕТИЧЕСКИМ", md)
        self.assertIn("`agent_api_probes`: не применимо к синтетической модели; исходный вердикт **FAIL**", md)

    def test_declared_synthetic_is_never_pass_overall(self):
        runs = self.make_runs(**T.PASSING)
        code, v, md = self.build_with(runs, criteria_file("synth-pass", synthetic_run=True))
        self.assertEqual(v["overall"], "NOT RUN", v["overall_reason"])
        self.assertEqual(code, 3)
        # even with nothing declared not applicable (every mandatory criterion PASS) the overall verdict is NOT RUN
        code, v, md = self.build_with(runs, criteria_file("synth-pass-none", synthetic_run=True,
                                                          synthetic_not_applicable=[]), name="pack-none")
        self.assertTrue(all(r["status"] == "PASS" for r in v["criteria_results"] if r["mandatory"]))
        self.assertEqual(v["overall"], "NOT RUN", v["overall_reason"])
        self.assertIn("never acceptance evidence", v["overall_reason"])
        self.assertEqual(code, 3)
        self.assertNotIn("**Итог: PASS**", md)

    def test_declaration_must_match_the_runs(self):
        """A real-model run (no SYNTHETIC in its model description) cannot hide a behaviour FAIL behind the flag."""
        k = self.probes_variant("real", scores=N11_SCORES, model_desc="Qwen3.8-Flash-Next UD-Q4_K_XL (real weights)")
        runs = self.make_runs(probes=k)
        code, v, md = self.build_with(runs, criteria_file("synth-mismatch", synthetic_run=True))
        r = self.by_id(v)["agent_api_probes"]
        self.assertEqual(r["status"], "FAIL", r["reason"])
        sr = v["synthetic_run"]
        self.assertTrue(sr["declared"])
        self.assertFalse(sr["applied"])
        self.assertTrue(sr["unmarked_runs"])
        self.assertEqual(sr["not_applicable"], [])
        self.assertEqual(v["overall"], "FAIL")
        self.assertIn("Правило НЕ применено", md)

    def test_probe_request_error_stays_fail(self):
        k = self.probes_variant("err", error=("tools", "HTTPError 500: SYNTHETIC server error"))
        runs = self.make_runs(**dict(T.PASSING, probes=k))
        code, v, md = self.build_with(runs, criteria_file("synth-err", synthetic_run=True))
        r = self.by_id(v)["agent_api_probes"]
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertTrue(r["details"]["request_errors"])
        self.assertEqual(v["overall"], "FAIL")

    def test_engine_criterion_fail_stays_fail(self):
        runs = self.make_runs(**dict(T.PASSING, matrix="slow", roofline="roofline_slow"))
        code, v, md = self.build_with(runs, criteria_file("synth-slow", synthetic_run=True))
        self.assertEqual(self.by_id(v)["decode_filled_ctx_floor"]["status"], "FAIL")
        self.assertEqual(v["overall"], "FAIL")

    def test_subset_of_behaviour_criteria(self):
        k = self.probes_variant("n11", scores=N11_SCORES)
        runs = self.make_runs(**dict(T.PASSING, probes=k))
        code, v, md = self.build_with(runs, criteria_file("synth-ctx-only", synthetic_run=True,
                                                          synthetic_not_applicable=["context_probes_full_length"]))
        self.assertEqual(self.by_id(v)["agent_api_probes"]["status"], "FAIL")
        self.assertEqual(v["overall"], "FAIL")


# ---------------------------------------------------------------------------- V6 tie tolerance
def lp_artifact(path, agree, flips, ref="http://127.0.0.1:1", cand="http://127.0.0.1:2"):
    """A logprob_compare.py --out file with `agree` agreeing rows and `flips` = list of (ref_gap, cand_gap) rows."""
    rows = []
    for i in range(agree):
        rows.append({"prefix": i, "top1_agree": True, "kl_topk": 1e-6, "ref_in_cand": True, "ref_top1": "a",
                     "cand_top1": "a", "ref_gap": 0.0, "cand_gap": 0.0, "max_abs_dlogprob": 1e-3, "n_common": 20})
    for j, (g, h) in enumerate(flips):
        r = {"prefix": agree + j, "top1_agree": False, "kl_topk": 1e-4, "ref_in_cand": True, "ref_top1": "a",
             "cand_top1": "b", "max_abs_dlogprob": 5e-3, "n_common": 20}
        if g != "absent":
            r.update({"ref_gap": g, "cand_gap": h})
        rows.append(r)
    data = {"ref": ref, "cand": cand, "ref_model": None, "cand_model": None, "files": ["SYNTHETIC"], "top_k": 20,
            "seed": 1, "prefixes": len(rows), "errors": [], "rows": rows,
            "summary": logprob_compare.summarize(rows), "_label": "SYNTHETIC TEST FIXTURE"}
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TieTolerance(Case):
    def judge(self, name, agree, flips, **crit):
        runs = os.path.join(self.tmp, "runs-" + name)
        os.makedirs(runs)
        lp_artifact(os.path.join(runs, "logprob_compare.%s.SYNTHETIC.json" % name), agree, flips)
        code, v, md = self.build_with(runs, criteria_file("tie-" + name, **crit), name="pack-" + name)
        return self.by_id(v)["exact_routing_logprobs"]

    def test_near_tie_flips_count_as_agreement(self):
        r = self.judge("ties", 190, [(0.004, 0.006)] * 10)
        self.assertEqual(r["status"], "PASS", r["reason"])
        self.assertEqual(r["value"], 1.0)
        self.assertEqual(r["details"]["comparisons"][0]["near_tie_flips"], 10)
        self.assertEqual(r["details"]["comparisons"][0]["top1_agree"], 0.95)

    def test_strict_when_tolerance_is_zero(self):
        r = self.judge("strict", 190, [(0.004, 0.006)] * 10, logprob_tie_tolerance_nats=0.0)
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertAlmostEqual(r["value"], 0.95)

    def test_gap_above_tolerance_is_a_real_flip(self):
        r = self.judge("wide", 190, [(0.010, 0.016)] * 10)        # 0.026 > 0.025
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertIn("0 near-tie flip", r["reason"])

    def test_rows_without_gaps_or_with_unknown_or_negative_gaps_are_never_excused(self):
        for name, flips in (("old-file", ["absent"] * 10), ("outside-topk", [(None, 0.001)] * 10),
                            ("negative", [(-0.001, 0.002)] * 10)):
            flips = [(f, None) if f == "absent" else f for f in flips]
            r = self.judge(name, 190, flips)
            self.assertEqual(r["status"], "FAIL", "%s: %s" % (name, r["reason"]))

    def test_tolerance_does_not_rescue_many_large_flips(self):
        r = self.judge("mixed", 180, [(0.004, 0.006)] * 10 + [(0.5, 0.4)] * 10)
        self.assertEqual(r["status"], "FAIL", r["reason"])
        self.assertAlmostEqual(r["value"], 0.95)

    def test_pack_judges_real_logprob_compare_output_against_near_tie_stubs(self):
        """End to end: logprob_compare.py writes the gaps; the pack excuses only the noise-sized flips."""
        ref_srv, ref_url = T._serve(near_tie_handler(0.0))
        noise_srv, noise_url = T._serve(near_tie_handler(0.004))     # moves l(b) - l(a) by 2 x 0.004 on near ties
        big_srv, big_url = T._serve(near_tie_handler(0.2))
        try:
            outs = {}
            for name, url in (("noise", noise_url), ("big", big_url)):
                outs[name] = os.path.join(self.tmp, "logprob_compare.%s.SYNTHETIC.json" % name)
                code, _ = T.quiet(logprob_compare.main, ["--ref", ref_url, "--cand", url, "--corpus", T.G["corpus"],
                                                        "--prefixes", "200", "--out", outs[name]])
                self.assertEqual(code, 0)
        finally:
            for s in (ref_srv, noise_srv, big_srv):
                s.shutdown()
                s.server_close()
        with open(outs["noise"]) as f:
            d = json.load(f)
        flips = [r for r in d["rows"] if not r["top1_agree"]]
        self.assertTrue(flips, "the stub must produce near-tie flips")
        for r in flips:
            self.assertAlmostEqual(r["ref_gap"] + r["cand_gap"], 0.008, places=9)
        self.assertAlmostEqual(d["summary"]["max_abs_dlogprob_max"], 0.004, places=4)   # each token moves ~shift
        for name, status in (("noise", "PASS"), ("big", "FAIL")):
            runs = os.path.join(self.tmp, "runs-e2e-" + name)
            os.makedirs(runs)
            shutil.copy(outs[name], runs)
            code, v, md = self.build_with(runs, criteria_file("e2e"), name="pack-e2e-" + name)
            r = self.by_id(v)["exact_routing_logprobs"]
            self.assertEqual(r["status"], status, "%s: %s" % (name, r["reason"]))


def near_tie_handler(shift):
    """/v1/completions with top_logprobs. Every 4th prompt (by hash) has a near tie: the best two tokens 0.003 apart;
    `shift` lowers the first and raises the second by `shift` nats (before renormalisation). SYNTHETIC."""
    import hashlib
    import http.server
    vocab = ["v%d" % i for i in range(30)]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            h = hashlib.sha256(body["prompt"].encode()).digest()
            logits = [-(i * 0.37) for i in range(len(vocab))]
            if h[0] % 4 == 0:
                logits[0], logits[1] = 0.0015, -0.0015          # near tie between v0 and v1
                logits[0] -= shift
                logits[1] += shift
            m = max(logits)
            lse = m + math.log(sum(math.exp(x - m) for x in logits))
            lp = sorted(((vocab[i], logits[i] - lse) for i in range(len(vocab))), key=lambda x: -x[1])
            top = dict(lp[:int(body.get("logprobs") or 5)])
            best = lp[0][0]
            T._reply(self, 200, {"choices": [{"index": 0, "text": best, "finish_reason": "length", "logprobs": {
                "tokens": [best], "token_logprobs": [top[best]], "top_logprobs": [top]}}]})
    return H


class CompareFields(unittest.TestCase):
    def test_gaps_and_noise(self):
        ref = {"a": -0.60, "b": -0.61, "c": -3.0}
        cand = {"a": -0.62, "b": -0.59, "d": -3.1}
        m = logprob_compare.compare(ref, cand)
        self.assertFalse(m["top1_agree"])
        self.assertAlmostEqual(m["ref_gap"], 0.01)
        self.assertAlmostEqual(m["cand_gap"], 0.03)
        self.assertAlmostEqual(m["max_abs_dlogprob"], 0.02)
        self.assertEqual(m["n_common"], 2)

    def test_unknown_gap_when_token_outside_top_k(self):
        m = logprob_compare.compare({"a": -0.1, "c": -3.0}, {"b": -0.1, "a": -2.0})
        self.assertIsNone(m["ref_gap"])
        self.assertAlmostEqual(m["cand_gap"], 1.9)
        m = logprob_compare.compare({"a": -0.1}, {"b": -0.1})
        self.assertIsNone(m["max_abs_dlogprob"])
        self.assertEqual(m["n_common"], 0)

    def test_agreement_has_zero_gaps(self):
        m = logprob_compare.compare({"a": -0.1, "b": -2.0}, {"a": -0.2, "b": -1.9})
        self.assertTrue(m["top1_agree"])
        self.assertEqual((m["ref_gap"], m["cand_gap"]), (0.0, 0.0))

    def test_summary_noise_percentiles(self):
        rows = [{"kl_topk": 0.0, "top1_agree": True, "ref_in_cand": True, "max_abs_dlogprob": i / 100.0}
                for i in range(100)]
        s = logprob_compare.summarize(rows)
        self.assertEqual(s["max_abs_dlogprob_n"], 100)
        self.assertAlmostEqual(s["max_abs_dlogprob_max"], 0.99)
        self.assertAlmostEqual(s["max_abs_dlogprob_p99"], 0.99)
        self.assertAlmostEqual(s["max_abs_dlogprob_p95"], 0.95)
        old = logprob_compare.summarize([{"kl_topk": 0.0, "top1_agree": True, "ref_in_cand": True}])
        self.assertNotIn("max_abs_dlogprob_max", old)


if __name__ == "__main__":
    unittest.main()
