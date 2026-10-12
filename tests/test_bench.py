"""tools/bench: OpenAI-compatible bench harness tested end-to-end against the stdlib mock server.

No GPU and no network: the mock runs on 127.0.0.1 with an ephemeral port in a thread.
The mock's numbers say nothing about any engine; these tests only check the harness logic:
exact prompt fitting, timing arithmetic against configured delays, prefix-cache detection,
probe grading (including a wrong-answer mutation), and that faults become recorded failures.
"""
import contextlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "tools", "bench")
sys.path.insert(0, BENCH)
import mock_server  # noqa: E402
import oa_client  # noqa: E402
import probes  # noqa: E402
import prompts  # noqa: E402
import run_matrix  # noqa: E402
import runlog  # noqa: E402

for _m in (mock_server, oa_client, probes, prompts, run_matrix, runlog):
    assert os.path.dirname(os.path.abspath(_m.__file__)) == BENCH, _m.__file__

PREFILL = 0.02      # mock seconds per 1000 uncached prompt tokens
DECODE = 0.001      # mock seconds per output token
SRV = None


def setUpModule():
    global SRV
    SRV = mock_server.MockServer(prefill_s_per_1k=PREFILL, decode_s_per_token=DECODE).start()


def tearDownModule():
    SRV.stop()


def reset(**kw):
    kw.setdefault("prefill_s_per_1k", PREFILL)
    kw.setdefault("decode_s_per_token", DECODE)
    SRV.reset(**kw)


def quiet(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench-test-")
        reset()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load_run(self, d):
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(d, "summary.json")) as f:
            summary = json.load(f)
        raw = runlog.load_raw(os.path.join(d, "raw.jsonl"))
        with open(os.path.join(d, "summary.md")) as f:
            md = f.read()
        return meta, raw, summary, md


# ---------------------------------------------------------------------------- statistics
class Stats(unittest.TestCase):
    def test_percentile_linear(self):
        self.assertEqual(runlog.percentile([5], 95), 5.0)
        self.assertEqual(runlog.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(runlog.percentile([10, 20, 30], 95), 29.0)
        self.assertEqual(runlog.percentile([3, 1, 2], 0), 1.0)
        self.assertEqual(runlog.percentile([3, 1, 2], 100), 3.0)

    def test_describe_skips_missing_and_fmt_prints_na(self):
        d = runlog.describe([None, 2.0, None, 4.0])
        self.assertEqual(d["n"], 2)
        self.assertEqual(d["p50"], 3.0)
        self.assertEqual(runlog.describe([None]), {"n": 0})
        self.assertEqual(runlog.fmt(None), "n/a")
        self.assertEqual(runlog.fstat({"n": 0}, "p50"), "n/a")

    def test_metrics_of_failed_request_are_none(self):
        m = runlog.request_metrics({"ok": False, "events": [[1000, 3, 0, 0, 0], [2000, 3, 0, 0, 0]],
                                    "usage": None, "t_end_us": 5000})
        for k in ("ttft_ms", "decode_tok_s", "prefill_tok_s", "itl_ms_p50", "total_ms"):
            self.assertIsNone(m[k], k)

    def test_decode_formula_one_and_multi_token_chunks(self):
        ev = [[1000 * (10 + i), 1, 0, 0, 0] for i in range(11)]          # 11 chunks, 1 ms apart
        m = runlog.request_metrics({"ok": True, "events": ev, "usage": {"prompt_tokens": 100,
                                    "completion_tokens": 11}, "t_end_us": 30000})
        self.assertAlmostEqual(m["decode_tok_s"], 10 / 0.010)            # (n-1)/(t_last-t_first)
        self.assertAlmostEqual(m["ttft_ms"], 10.0)
        self.assertAlmostEqual(m["prefill_tok_s"], 100 / 0.010)
        m4 = runlog.request_metrics({"ok": True, "events": ev, "usage": {"completion_tokens": 44},
                                     "t_end_us": 30000})
        self.assertAlmostEqual(m4["decode_tok_s"], (44 - 4) / 0.010)     # 4 tokens per chunk
        self.assertIsNone(m4["prefill_tok_s"])                            # no prompt_tokens -> not invented

    def test_uncached_prefill_subtracts_server_cached_tokens(self):
        # prefix-cache accounting (mutation B10 of docs/evidence/2026-10-09-tool-mutations.md survived without this)
        ev = [[1000 * (10 + i), 1, 0, 0, 0] for i in range(3)]           # TTFT 10 ms
        base = {"ok": True, "events": ev, "t_end_us": 20000}
        m = runlog.request_metrics(dict(base, usage={"prompt_tokens": 1000, "completion_tokens": 3,
                                                     "prompt_tokens_details": {"cached_tokens": 900}}))
        self.assertEqual(m["cached_tokens"], 900)
        self.assertAlmostEqual(m["prefill_tok_s"], 1000 / 0.010)         # includes cached tokens by definition
        self.assertAlmostEqual(m["uncached_prefill_tok_s"], 100 / 0.010)
        m2 = runlog.request_metrics(dict(base, usage={"prompt_tokens": 1000, "completion_tokens": 3},
                                         server_timings={"cache_n": 250}))   # llama.cpp timings fallback
        self.assertEqual(m2["cached_tokens"], 250)
        self.assertAlmostEqual(m2["uncached_prefill_tok_s"], 750 / 0.010)
        m3 = runlog.request_metrics(dict(base, usage={"prompt_tokens": 1000, "completion_tokens": 3}))
        self.assertIsNone(m3["cached_tokens"])
        self.assertIsNone(m3["uncached_prefill_tok_s"])                  # not reported -> not invented


# ---------------------------------------------------------------------------- client
class Client(TmpDirCase):
    def test_stream_timestamps_usage_and_role_chunk_not_counted(self):
        c = oa_client.OAClient(SRV.url + "/v1")
        self.assertEqual(c.resolve_model(), "mock-model")
        r = c.chat_stream([{"role": "user", "content": "Hello there"}], max_tokens=20, extra={"ignore_eos": True})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.finish_reason, "length")
        self.assertTrue(r.done_seen)
        self.assertEqual(r.usage["completion_tokens"], 20)
        self.assertEqual(len(r.events), 20)                               # role-only first chunk excluded
        ts = [e[0] for e in r.events]
        self.assertEqual(ts, sorted(ts))
        self.assertTrue(c.stream_options_supported)
        self.assertIsNotNone(r.server_timings)

    def test_non_streaming_chat(self):
        c = oa_client.OAClient(SRV.url, model="mock-model")
        h = c.chat([{"role": "user", "content": "Hello"}], max_tokens=8)
        self.assertEqual(h.status, 200)
        self.assertEqual(h.json()["choices"][0]["message"]["content"], "This is a short toy answer.")

    def test_tokenize_dialects(self):
        text = "Hello, world: three more words."
        for style in ("llamacpp", "vllm"):
            reset(tokenize_style=style)
            c = oa_client.OAClient(SRV.url, model="mock-model")
            self.assertEqual(c.tokenize_count(text), mock_server.toy_count(text))
            self.assertEqual(c.tokenize_style, style)
        reset(tokenize_style="none")
        c = oa_client.OAClient(SRV.url, model="mock-model")
        self.assertIsNone(c.tokenize_count(text))
        self.assertIsNone(c.tokenize_style)

    def test_stream_options_rejected_is_retried_and_recorded(self):
        reset(accept_stream_options=False)
        c = oa_client.OAClient(SRV.url, model="mock-model")
        r = c.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=4)
        self.assertTrue(r.ok, r.error)
        self.assertIs(c.stream_options_supported, False)
        self.assertEqual(r.attempts, 2)
        self.assertIsNone(r.usage)                                        # server sent none: stays None
        self.assertIsNone(runlog.request_metrics(r.to_dict())["prefill_tok_s"])

    def test_faults_become_errors(self):
        c = oa_client.OAClient(SRV.url, model="mock-model", timeout=0.3)
        for fault, needle in (("http500", "HTTP 500"), ("truncate", "finish_reason"), ("slow_first", "timeout")):
            reset(fault=fault, slow_first_s=1.0)
            r = c.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=10, extra={"ignore_eos": True})
            self.assertFalse(r.ok, fault)
            self.assertIn(needle, r.error, fault)
            m = runlog.request_metrics(r.to_dict())
            self.assertIsNone(m["ttft_ms"], fault)
            self.assertIsNone(m["decode_tok_s"], fault)

    def test_multi_token_chunks_and_reasoning(self):
        reset(tokens_per_chunk=4, reasoning_tokens=8, decode_s_per_token=0.002)
        c = oa_client.OAClient(SRV.url, model="mock-model")
        r = c.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=80, extra={"ignore_eos": True})
        m = runlog.request_metrics(r.to_dict())
        self.assertEqual(m["tokens_per_chunk"], 4.0)
        self.assertGreater(m["decode_tok_s"], 0.6 * 500)
        self.assertLess(m["decode_tok_s"], 1.05 * 500)
        self.assertGreater(m["ttfc_ms"], m["ttft_ms"])                   # reasoning chunks come first
        self.assertEqual(r.reasoning.count("think"), 8)

    def test_completions_endpoint(self):
        c = oa_client.OAClient(SRV.url, model="mock-model")
        r = c.completion_stream("Once upon a time", max_tokens=12, extra={"ignore_eos": True})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.usage["completion_tokens"], 12)
        self.assertEqual(len(r.events), 12)


# ---------------------------------------------------------------------------- prompts
class Prompts(TmpDirCase):
    def test_exact_counts_all_targets_via_server_tokenize(self):
        c = oa_client.OAClient(SRV.url, model="mock-model")
        counter = prompts.ServerCounter(c)
        for t in prompts.DEFAULT_TARGETS:                                 # 2K .. 262K
            r = prompts.fit_filler("prose", 7, t, counter, header="Doc.\n", suffix="\nQ?")
            self.assertEqual(r["tokens"], t)
            self.assertEqual(mock_server.toy_count(r["text"]), t)        # independent recount
            self.assertTrue(r["exact"])
        for kind in ("code", "logs"):
            for t in (2048, 8192, 32768):
                r = prompts.fit_filler(kind, 7, t, counter)
                self.assertEqual(mock_server.toy_count(r["text"]), t, (kind, t))

    def test_vllm_dialect_exact(self):
        reset(tokenize_style="vllm")
        c = oa_client.OAClient(SRV.url, model="mock-model")
        r = prompts.fit_filler("code", 1, 4096, prompts.ServerCounter(c))
        self.assertEqual(mock_server.toy_count(r["text"]), 4096)

    def test_deterministic_and_seed_sensitive(self):
        f = prompts.FnCounter(mock_server.toy_count)
        a = prompts.fit_filler("prose", 3, 3000, f)["text"]
        b = prompts.fit_filler("prose", 3, 3000, f)["text"]
        c = prompts.fit_filler("prose", 4, 3000, f)["text"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        for kind in ("prose", "code", "logs"):
            self.assertEqual(prompts.Filler(kind, 9).text(500), prompts.Filler(kind, 9).text(500))

    def test_inserts_keep_depth_order(self):
        f = prompts.FnCounter(mock_server.toy_count)
        ins = [(0.9, "MARK_C"), (0.1, "MARK_A"), (0.5, "MARK_B")]
        ins.sort()
        t = prompts.fit_filler("prose", 1, 5000, f, inserts=ins)["text"]
        pos = [t.index("\n%s\n" % m) for m in ("MARK_A", "MARK_B", "MARK_C")]
        self.assertEqual(pos, sorted(pos))
        self.assertLess(pos[0], len(t) * 0.2)
        self.assertGreater(pos[2], len(t) * 0.8)

    def test_ratio_fallback_without_tokenize(self):
        reset(tokenize_style="none", prefill_s_per_1k=0.0)
        c = oa_client.OAClient(SRV.url, model="mock-model")
        counter, overhead, info = prompts.make_counter(c, "auto")
        self.assertEqual(info["method"], "chars-per-token-ratio")
        self.assertTrue(info["calibrated"])
        self.assertFalse(counter.exact)
        fr = prompts.fit_filler("prose", 2, 16384, counter)
        prompts.refine_ratio(c, counter, fr, overhead)
        fr = prompts.fit_filler("prose", 2, 16384, counter)
        pt, _ = prompts.usage_prompt_tokens(c, fr["text"])
        self.assertLess(abs(pt - (16384 + overhead)) / 16384.0, 0.01)

    def test_cli_reports_achieved_counts(self):
        out = os.path.join(self.tmp, "fit.json")
        code, stdout = quiet(prompts.main, ["--url", SRV.url, "--kind", "prose,logs", "--targets", "2048,8192",
                                            "--out", out, "--verify"])
        self.assertEqual(code, 0)
        with open(out) as f:
            rep = json.load(f)
        self.assertEqual([r["tokens"] for r in rep["results"]], [2048, 8192, 2048, 8192])
        ov = rep["counter"]["overhead"]
        self.assertEqual([r["verify_usage_prompt_tokens"] for r in rep["results"]],
                         [2048 + ov, 8192 + ov, 2048 + ov, 8192 + ov])


# ---------------------------------------------------------------------------- matrix
class Matrix(TmpDirCase):
    def run_matrix(self, extra, name="m"):
        d = os.path.join(self.tmp, name)
        code, out = quiet(run_matrix.main, ["--url", SRV.url, "--out-dir", d, "--no-nvidia-smi"] + extra)
        return code, d

    def test_small_matrix_files_timings_cache(self):
        code, d = self.run_matrix(["--inputs", "300,1500", "--outputs", "16,40", "--concurrency", "1,2",
                                   "--repeats", "2", "--scenarios", "matrix", "--engine-desc", "mock engine",
                                   "--model-desc", "toy"])
        self.assertEqual(code, 0)
        meta, raw, s, md = self.load_run(d)
        for k in ("utc_start", "utc_end", "argv", "args", "server_url", "models_response", "git", "host",
                  "nvidia_smi", "eos", "token_counter"):
            self.assertIn(k, meta)
        self.assertTrue(meta["is_mock_server"])
        self.assertEqual(meta["engine_desc"], "mock engine")
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["eos"]["mode"], "vllm")
        self.assertTrue(meta["eos"]["verified"])
        self.assertTrue(os.path.exists(os.path.join(d, "metrics_before.txt")))
        self.assertIn("MOCK SERVER RUN", md)
        self.assertEqual(len(s["cells"]), 2 * 2 * 2 * 2)
        self.assertEqual(meta["raw_rows"], len(raw))
        measure = [r for r in raw if r.get("phase") == "measure"]
        self.assertEqual(len(measure), 2 * 2 * 2 * (2 * 1 + 2 * 2))       # in x out x states x reps*(1+2)
        for r in measure:
            m = r["metrics"]
            self.assertTrue(m["ok"], r["result"]["error"])
            self.assertEqual(m["completion_tokens"], r["output"])
            self.assertLessEqual(abs(m["prompt_tokens"] - r["input"]), 1)  # fitted with measured template overhead
            self.assertGreater(m["ttft_ms"], 0)
            if r["state"] == "cold" and r["concurrency"] == 1:
                # cold: only the chat-template head ("<|user|>\nDocument") can match an earlier prompt
                self.assertLess(m["cached_tokens"], 8)
                expect = (m["prompt_tokens"] - m["cached_tokens"]) / 1000.0 * PREFILL * 1000.0   # ms
                self.assertGreater(m["ttft_ms"], 0.9 * expect)
                self.assertLess(m["ttft_ms"], expect + 60.0)
                self.assertGreater(m["decode_tok_s"], 0.6 / DECODE)
                self.assertLess(m["decode_tok_s"], 1.05 / DECODE)
        single = {(c["input"], c["output"], c["state"]): c for c in s["cells"] if c["concurrency"] == 1}
        for inp in (300, 1500):
            for out in (16, 40):
                cold = single[(inp, out, "cold")]["metrics"]["ttft_ms"]["p50"]
                warm = single[(inp, out, "warm")]["metrics"]["ttft_ms"]["p50"]
                self.assertLess(warm, cold, (inp, out))
                self.assertGreater(single[(inp, out, "warm")]["metrics"]["cached_tokens"]["p50"], 0.9 * inp)
        multi = [c for c in s["cells"] if c["concurrency"] == 2]
        self.assertTrue(all("waves" in c and c["waves"]["aggregate_output_tok_s"]["n"] == 2 for c in multi))
        self.assertTrue(all("waves" not in c for c in s["cells"] if c["concurrency"] == 1))
        self.assertIn("NOT comparable with single-stream", md)
        waves = [r for r in raw if r.get("phase") == "wave"]
        self.assertEqual(len(waves), 2 * 2 * 2 * 2)
        self.assertTrue(all(w["aggregate_output_tok_s"] > 0 for w in waves))

    def test_eos_detection_llamacpp_dialect(self):
        reset(honor_keys="llamacpp", reject_unknown_keys=True)
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "24", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "1", "--scenarios", "matrix"])
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(code, 0)
        self.assertEqual(meta["eos"]["mode"], "llamacpp")
        self.assertTrue(meta["eos"]["verified"])
        vllm_try = [a for a in meta["eos"]["attempts"] if a["mode"] == "vllm"][0]
        self.assertEqual(vllm_try["http_status"], 400)                    # min_tokens rejected, recorded
        self.assertEqual(s["cells"][0]["n_short_output"], 0)

    def test_no_length_forcing_is_flagged_not_hidden(self):
        reset(honor_keys="none")
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "40", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "2", "--scenarios", "matrix"])
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(meta["eos"]["mode"], "none")
        self.assertEqual(s["cells"][0]["n_short_output"], 2)
        self.assertTrue(all(r["short_output"] for r in raw if r.get("phase") == "measure"))

    def test_agent_loop_prefix_cache(self):
        code, d = self.run_matrix(["--scenarios", "agent", "--agent-turns", "8", "--agent-tool-tokens", "300,600",
                                   "--agent-target-tokens", "100000"])
        self.assertEqual(code, 0)
        meta, raw, s, md = self.load_run(d)
        turns = s["agent_loop"]["turns"]
        self.assertEqual(len(turns), 8)
        self.assertTrue(all(t["ok"] for t in turns))
        pts = [t["prompt_tokens"] for t in turns]
        self.assertEqual(pts, sorted(pts))
        for t in turns[1:]:
            self.assertGreaterEqual(t["new_tokens"], 300)
            self.assertLess(t["new_tokens"], 700)
            self.assertGreaterEqual(t["cached_tokens"], t["prompt_tokens"] - t["new_tokens"] - 5)
        ctrl = s["agent_loop"]["cold_control"]
        self.assertTrue(ctrl["ok"])
        self.assertLessEqual(abs(ctrl["prompt_tokens"] - pts[-1]), 2)
        self.assertLess(turns[-1]["ttft_ms"], ctrl["ttft_ms"])            # cache reuse visible in TTFT
        self.assertIn("Agent loop", md)
        roles = [r for r in raw if r.get("phase") == "agent_turn"]
        self.assertIn("tools", roles[0]["result"]["request_keys"])

    COLD_ARGS = ["--inputs", "400", "--outputs", "8", "--states", "cold", "--concurrency", "1", "--repeats", "2",
                 "--scenarios", "matrix,agent", "--agent-turns", "2", "--agent-tool-tokens", "300,400",
                 "--agent-max-tokens", "4"]

    def two_runs(self, extra=()):
        out = []
        for name in ("first", "second"):
            code, d = self.run_matrix(self.COLD_ARGS + list(extra), name)
            self.assertEqual(code, 0)
            meta, raw, s, md = self.load_run(d)
            out.append((meta, [r["metrics"] for r in raw if r.get("phase") == "measure"],
                        s["agent_loop"]["cold_control"]))
        return out

    def test_second_run_against_the_same_server_is_still_cold(self):
        # N11 (docs/evidence/2026-10-11-n11-r1-tools-cpu-server.md): the cold nonce depended only on --seed and the cell,
        # so a second run against the same server (a re-run, or ctx-2k after smoke-2k in scripts/r1/configs/
        # llamacpp-server.json: same seed, same key ("cold", 2048, 128, 1, 0, 0)) re-sent byte-identical "cold" prompts
        # and the real llama-server answered them from its prompt cache (cached 1019 of 1023 tokens)
        (m1, cold1, ctrl1), (m2, cold2, ctrl2) = self.two_runs()
        self.assertTrue(m1["nonce_salt"] and m2["nonce_salt"] and m1["nonce_salt"] != m2["nonce_salt"])
        self.assertEqual(len(cold2), 2)
        for m in cold1 + cold2:
            self.assertLess(m["cached_tokens"], 8, m)          # only "<|user|>\nDocument" can match an earlier prompt
        # the agent loop's cold control shares only the tool schemas and the template head with earlier prompts
        self.assertLess(ctrl2["cached_tokens"], 0.5 * ctrl2["prompt_tokens"])
        self.assertLessEqual(ctrl2["cached_tokens"], ctrl1["cached_tokens"] + 2)

    def test_nonce_salt_reproduces_a_run(self):
        # the same --nonce-salt (recorded in meta.json) re-sends byte-identical prompts: the server then reuses them
        (m1, cold1, ctrl1), (m2, cold2, ctrl2) = self.two_runs(["--nonce-salt", "n11"])
        self.assertEqual((m1["nonce_salt"], m2["nonce_salt"]), ("n11", "n11"))
        for m in cold2:
            self.assertGreaterEqual(m["cached_tokens"], m["prompt_tokens"] - 1, m)
        self.assertGreaterEqual(ctrl2["cached_tokens"], ctrl2["prompt_tokens"] - 1)

    def test_http500_everywhere_records_failures_without_numbers(self):
        reset(fault="http500")
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "16", "--concurrency", "1,2", "--repeats", "1",
                                   "--agent-turns", "2", "--agent-tool-tokens", "100,200"])
        self.assertEqual(code, 1)
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(meta["status"], "completed_with_failures")
        for c in s["cells"]:
            self.assertEqual(c["n_ok"], 0)
            for k, v in c["metrics"].items():
                self.assertEqual(v, {"n": 0}, k)
            if c["concurrency"] > 1:
                self.assertEqual(c["waves"]["aggregate_output_tok_s"], {"n": 0})
        for r in raw:
            if "metrics" in r:
                self.assertFalse(r["metrics"]["ok"])
                self.assertIsNone(r["metrics"]["ttft_ms"])
        self.assertTrue(s["failures"])
        self.assertTrue(all("HTTP 500" in f["error"] for f in s["failures"]))
        self.assertNotIn("| ok |", md)
        self.assertIn("n/a", md)

    def test_partial_faults_aggregate_only_successes(self):
        reset(fault="http500", fault_every=3)
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "16", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "6", "--scenarios", "matrix"])
        meta, raw, s, md = self.load_run(d)
        c = s["cells"][0]
        self.assertGreater(c["n_failed"], 0)
        self.assertGreater(c["n_ok"], 0)
        self.assertEqual(c["n_ok"] + c["n_failed"], 6)
        self.assertEqual(c["metrics"]["ttft_ms"]["n"], c["n_ok"])
        self.assertEqual(code, 1)

    def test_truncated_stream_and_idle_timeout(self):
        reset(fault="truncate", fault_after_chunks=3)
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "16", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "1", "--scenarios", "matrix"], name="trunc")
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(code, 1)
        self.assertEqual(s["cells"][0]["n_ok"], 0)
        self.assertIn("finish_reason", s["failures"][-1]["error"])
        reset(fault="slow_first", slow_first_s=1.0)
        code, d = self.run_matrix(["--inputs", "200", "--outputs", "16", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "1", "--scenarios", "matrix", "--timeout", "0.4"], name="slow")
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(code, 1)
        self.assertIn("timeout", s["failures"][-1]["error"])

    def test_unreachable_server_is_fatal_not_a_crash(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        d = os.path.join(self.tmp, "dead")
        code, _ = quiet(run_matrix.main, ["--url", "http://127.0.0.1:%d" % port, "--out-dir", d, "--no-nvidia-smi",
                                          "--timeout", "2"])
        self.assertEqual(code, 2)
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(meta["status"], "fatal")
        self.assertIn("FATAL", md)
        self.assertEqual(raw, [])

    def test_variants_completions_ratio_user_format_options(self):
        reset(tokenize_style="none")
        eb = os.path.join(self.tmp, "extra.json")
        with open(eb, "w") as f:
            json.dump({"chat_template_kwargs": {"enable_thinking": False}}, f)
        code, d = self.run_matrix(["--endpoint", "completions", "--counter", "ratio", "--inputs", "3000",
                                   "--outputs", "16", "--states", "cold,warm", "--concurrency", "1", "--repeats", "1",
                                   "--agent-format", "user", "--agent-turns", "3", "--agent-tool-tokens", "200,300",
                                   "--save-text", "--send-seed", "7", "--extra-body", "@" + eb])
        self.assertEqual(code, 0)
        meta, raw, s, md = self.load_run(d)
        self.assertEqual(meta["token_counter"]["method"], "chars-per-token-ratio")
        self.assertTrue(any(r.get("phase") == "ratio_refine" for r in raw))
        measure = [r for r in raw if r.get("phase") == "measure"]
        self.assertTrue(all(r["result"]["endpoint"] == "/v1/completions" for r in measure))
        self.assertTrue(all(abs(r["metrics"]["prompt_tokens"] - 3000) / 3000.0 < 0.02 for r in measure))
        self.assertTrue(all("content" in r["result"] and "seed" in r["result"]["request_keys"] for r in measure))
        self.assertTrue(all("chat_template_kwargs" in r["result"]["request_keys"] for r in measure))
        agent = [r for r in raw if r.get("phase") == "agent_turn"]
        self.assertEqual(len(agent), 3)
        self.assertNotIn("tools", agent[0]["result"]["request_keys"])
        self.assertTrue(all(t["ok"] for t in s["agent_loop"]["turns"]))

    def test_refuses_to_overwrite_a_run(self):
        code, d = self.run_matrix(["--inputs", "100", "--outputs", "8", "--states", "cold", "--concurrency", "1",
                                   "--repeats", "1", "--scenarios", "matrix"])
        self.assertEqual(code, 0)
        with self.assertRaises(SystemExit):
            self.run_matrix(["--inputs", "100", "--outputs", "8", "--scenarios", "matrix"])


# ---------------------------------------------------------------------------- probes
class Probes(TmpDirCase):
    def run_probes(self, extra, name="p"):
        d = os.path.join(self.tmp, name)
        code, _ = quiet(probes.main, ["--url", SRV.url, "--out-dir", d, "--no-nvidia-smi"] + extra)
        meta, raw, s, md = self.load_run(d)
        return code, s, raw, md

    def by(self, s, probe):
        return [r for r in s["results"] if r["probe"] == probe]

    def test_honest_mock_passes_everything(self):
        code, s, raw, md = self.run_probes(["--contexts", "2048,6144", "--answer-tokens", "256"])
        self.assertEqual(code, 0)
        self.assertTrue(s["all_hard_pass"])
        for p in ("needle", "multihop", "codeqa"):
            rs = self.by(s, p)
            self.assertEqual([r["context"] for r in rs], [2048, 6144])
            for r in rs:
                self.assertEqual(r["score"], 1.0, p)
                self.assertTrue(r["pass"])
                self.assertEqual(r["prompt_tokens"], r["context"] - 256)  # length mode total, exact counter
        j = self.by(s, "json")[0]
        self.assertTrue(j["details"]["response_format_accepted"])
        t = self.by(s, "tools")[0]
        self.assertTrue(t["details"]["call_ok"] and t["details"]["final_uses_result"])
        det = self.by(s, "determinism")[0]
        self.assertTrue(det["details"]["identical"])
        needle_items = self.by(s, "needle")[0]["details"]["items"]
        self.assertEqual(len(needle_items), 10)
        self.assertAlmostEqual(needle_items[0]["depth"], 0.05)
        self.assertAlmostEqual(needle_items[-1]["depth"], 0.95)
        self.assertIn("MOCK SERVER RUN", md)

    def test_wrong_answer_mutation_scores_zero(self):
        reset(answer_mode="wrong")
        code, s, raw, md = self.run_probes(["--contexts", "2048", "--answer-tokens", "256"])
        self.assertEqual(code, 1)
        self.assertFalse(s["all_hard_pass"])
        self.assertEqual(self.by(s, "needle")[0]["score"], 0.0)
        self.assertEqual(self.by(s, "multihop")[0]["score"], 0.0)
        self.assertEqual(self.by(s, "codeqa")[0]["score"], 0.0)
        for p in ("needle", "multihop", "codeqa", "json", "tools"):
            self.assertFalse(self.by(s, p)[0]["pass"], p)
        self.assertEqual(self.by(s, "tools")[0]["score"], 0.0)

    def test_response_format_rejected_is_recorded(self):
        reset(accept_response_format=False)
        code, s, raw, md = self.run_probes(["--probes", "json"])
        j = self.by(s, "json")[0]
        self.assertFalse(j["details"]["response_format_accepted"])
        self.assertEqual(j["details"]["attempts"]["response_format"]["http_status"], 400)
        self.assertTrue(j["pass"])                                        # plain attempt was valid
        self.assertEqual(code, 0)

    def test_nondeterminism_is_reported_not_fatal(self):
        reset(nondeterministic=True)
        code, s, raw, md = self.run_probes(["--probes", "determinism,needle", "--contexts", "2048"])
        det = self.by(s, "determinism")[0]
        self.assertFalse(det["pass"])
        self.assertFalse(det["hard"])
        self.assertFalse(det["details"]["identical"])
        self.assertIsNotNone(det["details"]["first_diff_char"])
        self.assertEqual(code, 0)

    def test_ratio_counter_probe(self):
        reset(tokenize_style="none")
        code, s, raw, md = self.run_probes(["--probes", "needle", "--contexts", "4096", "--answer-tokens", "256"])
        n = self.by(s, "needle")[0]
        self.assertEqual(n["score"], 1.0)
        self.assertLess(abs(n["prompt_tokens"] - (4096 - 256)) / 3840.0, 0.03)
        self.assertEqual(s["token_counter"]["method"], "chars-per-token-ratio")

    def test_faults_fail_probes_without_scores(self):
        reset(fault="http500")
        code, s, raw, md = self.run_probes(["--probes", "needle,tools", "--contexts", "2048"])
        self.assertEqual(code, 1)
        for r in s["results"]:
            self.assertFalse(r["pass"])
            self.assertIsNone(r["score"])
            self.assertIn("HTTP 500", r["error"])


class Graders(unittest.TestCase):
    """Grader checks the wrong-answer mock cannot isolate (mutations B22-B24 of
    docs/evidence/2026-10-09-tool-mutations.md survived without these)."""
    EXP = {"name": "Alice Smith", "age": 33, "city": "Nuuk", "tags": ["river", "orbit", "prism"], "active": True}

    def test_json_validator_rejects_valid_json_with_wrong_values(self):
        ok = probes.validate_person_json(json.dumps(self.EXP), self.EXP, strict=True)
        self.assertTrue(ok["valid"])
        self.assertEqual(ok["score"], 1.0)
        for k, v in (("name", "Alice Novak"), ("age", 34), ("city", "Hobart"),
                     ("tags", ["river", "prism", "orbit"]), ("active", False)):
            r = probes.validate_person_json(json.dumps(dict(self.EXP, **{k: v})), self.EXP, strict=True)
            self.assertFalse(r["valid"], k)
            self.assertFalse(r["checks"]["value_" + k], k)
            self.assertLess(r["score"], 1.0, k)


class ToolGrading(TmpDirCase):
    def run_tools_with_call(self, name, args_fn):
        """Run the tools probe while the mock's first-turn tool call is rewritten to (name, args_fn(args))."""
        orig = mock_server.toy_answer

        def patched(prompt, messages, tools, cfg):
            kind, ans = orig(prompt, messages, tools, cfg)
            if kind == "tool":
                return "tool", (name, args_fn(ans[1]))
            return kind, ans
        mock_server.toy_answer = patched
        try:
            d = os.path.join(self.tmp, "p")
            code, _ = quiet(probes.main, ["--url", SRV.url, "--out-dir", d, "--no-nvidia-smi", "--probes", "tools"])
        finally:
            mock_server.toy_answer = orig
        with open(os.path.join(d, "summary.json")) as f:
            return code, json.load(f)["results"][0]

    def test_wrong_tool_name_with_right_city_fails(self):
        code, r = self.run_tools_with_call("get_stock_price", lambda a: a)
        self.assertFalse(r["details"]["call"]["name_ok"])
        self.assertTrue(r["details"]["call"]["city_ok"])
        self.assertFalse(r["details"]["call_ok"])
        self.assertTrue(r["details"]["final_uses_result"])            # second turn used the synthetic call
        self.assertEqual(r["score"], 0.5)
        self.assertFalse(r["pass"])
        self.assertEqual(code, 1)

    def test_right_tool_name_with_wrong_city_fails(self):
        code, r = self.run_tools_with_call("get_weather", lambda a: dict(a, city="Atlantis"))
        self.assertTrue(r["details"]["call"]["name_ok"])
        self.assertFalse(r["details"]["call"]["city_ok"])
        self.assertFalse(r["details"]["call_ok"])
        self.assertEqual(r["score"], 0.5)
        self.assertFalse(r["pass"])
        self.assertEqual(code, 1)


class CLIHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        for fn in (run_matrix.main, probes.main, prompts.main, mock_server.main):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                fn(["--help"])
            self.assertEqual(cm.exception.code, 0)
            self.assertIn("--", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
