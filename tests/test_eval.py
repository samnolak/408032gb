"""tools/eval: agentic evaluation stand, tested against a scripted mock endpoint (no GPU, no network).

The mock (tools/eval/mock_agent_server.py) answers from scripts chosen by the first user message: the
reference answers of the SYNTHETIC task set (must pass), deliberately wrong answers (must fail), malformed
answers, tool-calling sequences (right, wrong arguments, wrong order, forbidden call, endless loop) and
transport faults (must be request errors, never model failures). Nothing here says anything about a model.
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "tools", "eval")
BENCH = os.path.join(ROOT, "tools", "bench")
sys.path.insert(0, BENCH)
sys.path.insert(0, EVAL)
import compare  # noqa: E402
import ev_answers  # noqa: E402
import ev_patch  # noqa: E402
import ev_sandbox  # noqa: E402
import ev_schema  # noqa: E402
import ev_scoring  # noqa: E402
import ev_stats  # noqa: E402
import ev_tasks  # noqa: E402
import ev_toolsim  # noqa: E402
import mock_agent_server  # noqa: E402
import mock_server as bench_mock  # noqa: E402
import run_eval  # noqa: E402
import runlog  # noqa: E402

for _m in (compare, ev_answers, ev_patch, ev_sandbox, ev_schema, ev_scoring, ev_stats, ev_tasks, ev_toolsim,
           mock_agent_server, run_eval):
    assert os.path.dirname(os.path.abspath(_m.__file__)) == EVAL, _m.__file__
assert os.path.dirname(os.path.abspath(runlog.__file__)) == BENCH

SYN = os.path.join(EVAL, "tasks-synthetic")
SRV = None
TASKS = None
CAPS = None


def setUpModule():
    global SRV, TASKS, CAPS
    SRV = mock_agent_server.MockAgentServer().start()
    tasks, errs = ev_tasks.load_dir(SYN)
    assert not errs, errs
    TASKS = {t.id: t for t in tasks}
    CAPS = ev_sandbox.capabilities()


def tearDownModule():
    SRV.stop()


def first_user_sha(task):
    return mock_agent_server.sha256_text(next(m["content"] for m in task.messages if m["role"] == "user"))


def call(name, args):
    return {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}


CORRECT_TOOL_TURNS = {
    "SYNTHETIC-tool_use-001": [
        {"tool_calls": [call("get_order", {"order_id": "A-1027"})]},
        {"tool_calls": [call("issue_refund", {"order_id": "A-1027", "amount": 12.5, "reason": "damaged"})]},
        {"content": "The blue mug was refunded (12.50 EUR). Refund id: RF-55102."}],
    "SYNTHETIC-tool_use-002": [
        {"tool_calls": [call("calculator", {"expression": "17*23.40*1.08"})]},
        {"content": "17 x 23.40 = 397.80; with 8% tax 429.624.\nANSWER: 429.62"}],
    "SYNTHETIC-tool_use-003": [
        {"tool_calls": [call("get_stock", {"sku": "KB-310"}), call("get_stock", {"sku": "KB-200"})]},
        {"content": "KB-310 has 215 units, KB-200 has 140.\nANSWER: KB-310"}],
}
WRONG_TOOL_TURNS = {
    "SYNTHETIC-tool_use-001": [                                      # right calls, wrong order
        {"tool_calls": [call("issue_refund", {"order_id": "A-1027", "amount": 12.5})]},
        {"tool_calls": [call("get_order", {"order_id": "A-1027"})]},
        {"content": "Refund RF-55102 issued."}],
    "SYNTHETIC-tool_use-002": [
        {"tool_calls": [call("calculator", {"expression": "17*23.40"})]},
        {"content": "ANSWER: 397.80"}],
    "SYNTHETIC-tool_use-003": [                                      # forbidden call
        {"tool_calls": [call("get_stock", {"sku": "KB-200"}), call("get_stock", {"sku": "KB-310"})]},
        {"tool_calls": [call("place_order", {"sku": "KB-200", "qty": 100})]},
        {"content": "ANSWER: KB-310"}],
}
WRONG_TEXT = {
    "SYNTHETIC-code_fix-001": "```diff\n--- a/stats.py\n+++ b/stats.py\n@@ -8 +8 @@\n-    return sum(xs) / (len(xs) - 1)\n"
                              "+    return sum(xs) / (len(xs) + 1)\n```",
    "SYNTHETIC-code_fix-002": "FILE: durations.py\n```python\ndef parse_duration(text):\n    return 0\n```",
    "SYNTHETIC-code_fix-003": "```diff\n--- a/tests/test_stock.py\n+++ b/tests/test_stock.py\n@@ -1,1 +1,1 @@\n"
                              "-import unittest\n+import unittest  # tests edited\n```",
    "SYNTHETIC-code_write-001": "FILE: slug.py\n```python\ndef slugify(text):\n    return text.lower().replace(' ', '-')\n```",
    "SYNTHETIC-code_write-002": "Sort the intervals and merge neighbours; that is all.",       # malformed: no code
    "SYNTHETIC-long_doc_qa-001": "ANSWER: 30",
    "SYNTHETIC-long_doc_qa-002": "ANSWER: X-Kestrel-Span",
    "SYNTHETIC-long_doc_qa-003": "ANSWER: the travel desk manager",
    "SYNTHETIC-structured_output-001": "Here is the JSON: {\"invoice_id\": \"INV-2026-0042\"}",
    "SYNTHETIC-structured_output-002": "[{\"id\": \"T1\", \"label\": \"bug\", \"priority\": \"high\"}, "
                                       "{\"id\": \"T2\", \"label\": \"billing\", \"priority\": \"high\"}, "
                                       "{\"id\": \"T3\", \"label\": \"bug\", \"priority\": \"low\"}, "
                                       "{\"id\": \"T4\", \"label\": \"other\", \"priority\": \"low\"}]",
    "SYNTHETIC-reasoning-001": "ANSWER: 12:25",
    "SYNTHETIC-reasoning-002": "ANSWER: 46",
}


def scripts(mode, override=None):
    out = []
    for tid, t in sorted(TASKS.items()):
        if override and tid in override:
            turns = override[tid]
        elif t.category == "tool_use":
            turns = (CORRECT_TOOL_TURNS if mode == "correct" else WRONG_TOOL_TURNS)[tid]
        else:
            turns = [{"content": t.reference_answer if mode == "correct" else WRONG_TEXT[tid]}]
        s = {"first_user_sha256": first_user_sha(t), "turns": turns}
        if isinstance(turns, dict):                   # {"loop": response}
            s = {"first_user_sha256": first_user_sha(t), "turns": [], "loop": turns["loop"]}
        out.append(s)
    return out


class RunCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eval-test-")
        self.n = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_eval(self, extra=(), scripts_=None, owner=runlog.MOCK_OWNER, url=None, tasks=SYN):
        if scripts_ is not None:
            SRV.set_scripts(scripts_, owner=owner)
        self.n += 1
        out = os.path.join(self.tmp, "run%d" % self.n)
        argv = ["--url", url or SRV.url, "--tasks", tasks, "--out", out, "--no-nvidia-smi"] + list(extra)
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = run_eval.main(argv)
        r = {"code": code, "out": out, "stdout": buf.getvalue(), "stderr": err.getvalue()}
        if os.path.exists(os.path.join(out, "meta.json")):
            with open(os.path.join(out, "meta.json")) as f:
                r["meta"] = json.load(f)
            with open(os.path.join(out, "summary.json")) as f:
                r["summary"] = json.load(f)
            with open(os.path.join(out, "summary.md")) as f:
                r["md"] = f.read()
            r["raw"] = runlog.load_raw(os.path.join(out, "raw.jsonl"))
            r["results"] = runlog.load_raw(os.path.join(out, "results.jsonl"))
            r["by_id"] = {x["task_id"]: x for x in r["results"]}
        return r


# ---------------------------------------------------------------------------- statistics
class Stats(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = ev_stats.wilson(8, 10)                  # textbook: 8/10 -> [0.4902, 0.9433]
        self.assertAlmostEqual(lo, 0.49016, places=4)
        self.assertAlmostEqual(hi, 0.94332, places=4)
        lo, hi = ev_stats.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.27753, places=4)
        lo, hi = ev_stats.wilson(10, 10)
        self.assertAlmostEqual(lo, 0.72247, places=4)
        self.assertEqual(hi, 1.0)
        self.assertEqual(ev_stats.wilson(0, 0), (None, None))

    def test_mcnemar_exact_uses_only_discordant_pairs(self):
        self.assertAlmostEqual(ev_stats.mcnemar_exact(8, 1), 20 / 512)     # 2 * (C(9,0)+C(9,1)) / 2^9
        self.assertAlmostEqual(ev_stats.mcnemar_exact(1, 8), 20 / 512)
        self.assertEqual(ev_stats.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(ev_stats.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(ev_stats.mcnemar_exact(0, 6), 2 / 64)
        self.assertAlmostEqual(ev_stats.mcnemar_exact(0, 5), 2 / 32)          # 0.0625: 5 pairs never reach 0.05

    def test_bootstrap_is_paired(self):
        a = [1] * 50 + [0] * 50
        b = list(a)
        for i in range(5):                                # 5 tasks: A pass, B fail; nothing the other way
            b[i] = 0
        r = ev_stats.bootstrap_paired(a, b, n_boot=4000, seed=1)
        self.assertAlmostEqual(r["diff"], 0.05)
        self.assertGreaterEqual(r["lo"], 0.0)             # every paired resample has diff >= 0
        self.assertLess(r["hi"] - r["lo"], 0.12)          # resampling A and B independently is much wider
        self.assertEqual(ev_stats.bootstrap_paired([1, 1], [0, 0], n_boot=100)["lo"], 1.0)
        self.assertEqual(r, ev_stats.bootstrap_paired(a, b, n_boot=4000, seed=1))   # seeded

    def test_task_outcomes_majority_and_errors(self):
        rows = [{"task_id": "t1", "category": "reasoning", "repeat": 0, "status": "pass"},
                {"task_id": "t1", "category": "reasoning", "repeat": 1, "status": "fail"},
                {"task_id": "t2", "category": "reasoning", "repeat": 0, "status": "request_error"},
                {"task_id": "t2", "category": "reasoning", "repeat": 1, "status": "pass"},
                {"task_id": "t3", "category": "reasoning", "repeat": 0, "status": "harness_error"}]
        o = ev_stats.task_outcomes(rows, "majority")
        self.assertFalse(o["t1"]["pass"])                 # tie is a fail
        self.assertTrue(o["t2"]["pass"])                  # the error repeat is not a fail
        self.assertEqual(o["t2"]["n_request_errors"], 1)
        self.assertIsNone(o["t3"]["pass"])
        self.assertTrue(ev_stats.task_outcomes(rows, "any")["t1"]["pass"])


# ---------------------------------------------------------------------------- JSON schema subset
class Schema(unittest.TestCase):
    S = {"type": "object", "required": ["n", "tags"], "additionalProperties": False,
         "properties": {"n": {"type": "integer", "minimum": 1}, "tags": {"type": "array", "items": {"enum": ["a", "b"]},
                                                                          "uniqueItems": True},
                        "id": {"$ref": "#/$defs/id"}},
         "$defs": {"id": {"type": "string", "pattern": "^X-\\d+$"}}}

    def test_valid_and_invalid(self):
        self.assertEqual(ev_schema.validate({"n": 2, "tags": ["a"], "id": "X-1"}, self.S), [])
        self.assertEqual(ev_schema.validate({"n": 2.0, "tags": []}, self.S), [])        # 2.0 is an integer
        bad = ev_schema.validate({"n": True, "tags": ["a", "a", "c"], "id": "Y", "z": 1}, self.S)
        text = "\n".join(bad)
        for frag in ("/n: expected type integer, got boolean", "additional property 'z'", "not in enum",
                     "not unique", "does not match pattern"):
            self.assertIn(frag, text)
        self.assertIn("missing required property 'n'", "\n".join(ev_schema.validate({"tags": []}, self.S)))
        self.assertTrue(ev_schema.validate(1, {"oneOf": [{"type": "integer"}, {"type": "number"}]}))
        self.assertEqual(ev_schema.validate(1.5, {"oneOf": [{"type": "integer"}, {"type": "number"}]}), [])

    def test_unsupported_keywords_are_reported(self):
        self.assertEqual(ev_schema.schema_problems(self.S), [])
        p = ev_schema.schema_problems({"type": "object", "patternProperties": {}, "properties": {
            "x": {"items": [{"type": "string"}]}}})
        self.assertTrue(any("patternProperties" in x for x in p))
        self.assertTrue(any("tuple-form" in x for x in p))


# ---------------------------------------------------------------------------- answer matching
class Answers(unittest.TestCase):
    def test_extract_and_normalize(self):
        self.assertEqual(ev_answers.extract("work...\n**Answer:** 42", "answer_line")[0], "42")
        self.assertEqual(ev_answers.extract("<think>ANSWER: 1</think>\nANSWER: 2", "answer_line")[0], "2")
        self.assertIsNone(ev_answers.extract("no marker here", "answer_line")[0])
        self.assertEqual(ev_answers.normalize('  **"Paris".** '), "paris")
        self.assertEqual(ev_answers.parse_number("1 234,5 EUR"), 1234.5)
        self.assertEqual(ev_answers.parse_number("3, 4"), 3.0)
        self.assertEqual(ev_answers.parse_number("1,234.5"), 1234.5)

    def test_matchers(self):
        g = ev_answers.grade_text
        self.assertTrue(g("ANSWER: Regional Finance Director.", {"type": "normalized", "answers": ["regional finance director"],
                                                                 "extract": "answer_line"})["ok"])
        self.assertFalse(g("ANSWER: Regional Finance Director.", {"type": "exact", "answers": ["regional finance director"],
                                                                  "extract": "answer_line"})["ok"])
        self.assertTrue(g("ANSWER: 429,62", {"type": "number", "value": 429.62, "abs_tol": 0.005, "extract": "answer_line"})["ok"])
        self.assertFalse(g("ANSWER: 429.7", {"type": "number", "value": 429.62, "abs_tol": 0.005, "extract": "answer_line"})["ok"])
        self.assertTrue(g("id x-kestrel-trace", {"type": "regex", "pattern": "X-Kestrel-Trace", "flags": "i"})["ok"])
        self.assertFalse(g("refund issued", {"type": "contains_all", "values": ["RF-55102"]})["ok"])


# ---------------------------------------------------------------------------- applying answers
class Patch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="eval-patch-")
        self.repo = os.path.join(self.d, "repo")
        os.makedirs(os.path.join(self.repo, "pkg"))
        with open(os.path.join(self.repo, "pkg", "m.py"), "w") as f:
            f.write("".join("line %d\n" % i for i in range(1, 21)))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def read(self, rel):
        with open(os.path.join(self.repo, rel)) as f:
            return f.read()

    def test_diff_with_wrong_line_number_and_counts(self):
        ans = "```diff\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -2,99 +2,99 @@\n line 14\n-line 15\n+LINE FIFTEEN\n line 16\n```"
        r = ev_patch.apply_answer(ans, self.repo)
        self.assertTrue(r["ok"], r)
        self.assertIn("line 14\nLINE FIFTEEN\nline 16\n", self.read("pkg/m.py"))

    def test_bare_hunk_headers_without_numbers(self):
        ans = "```diff\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@\n line 3\n-line 4\n+four\n@@ @@\n-line 18\n+eighteen\n```"
        r = ev_patch.apply_answer(ans, self.repo)
        self.assertTrue(r["ok"], r)
        self.assertIn("line 3\nfour\nline 5\n", self.read("pkg/m.py"))
        self.assertIn("line 17\neighteen\nline 19\n", self.read("pkg/m.py"))

    def test_new_file_delete_and_file_block(self):
        ans = ("```diff\n--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n```\n"
               "```diff\n--- a/pkg/m.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-line 1\n```")
        r = ev_patch.apply_answer(ans, self.repo)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.read("pkg/new.py"), "a = 1\nb = 2\n")
        self.assertFalse(os.path.exists(os.path.join(self.repo, "pkg", "m.py")))
        r = ev_patch.apply_answer("### FILE: `pkg/x.py`\n\n```python\nprint(1)\n```\n```diff\n--- a/q\n+++ b/q\n@@ -1 +1 @@\n-a\n+b\n```",
                                  self.repo)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.read("pkg/x.py"), "print(1)\n")
        self.assertIn("diff blocks ignored", r["notes"][0])

    def test_malformed_and_context_mismatch(self):
        r = ev_patch.apply_answer("I fixed it, trust me.", self.repo)
        self.assertFalse(r["ok"])
        self.assertIn("no FILE: block or unified diff", r["error"])
        r = ev_patch.apply_answer("```diff\n--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-not there\n+x\n```", self.repo)
        self.assertFalse(r["ok"])
        self.assertIn("does not apply", r["error"])

    def test_paths_cannot_escape_or_touch_protected_files(self):
        outside = os.path.join(self.d, "escaped.py")
        os.symlink(self.d, os.path.join(self.repo, "link"))
        for ans in ("FILE: ../escaped.py\n```\nx\n```", "FILE: /tmp/escaped.py\n```\nx\n```",
                    "FILE: link/escaped.py\n```\nx\n```",
                    "```diff\n--- a/../escaped.py\n+++ b/../escaped.py\n@@ -0,0 +1 @@\n+x\n```",
                    "FILE: tests/test_a.py\n```\nx\n```", "FILE: pkg/test_m.py\n```\nx\n```"):
            r = ev_patch.apply_answer(ans, self.repo)
            self.assertFalse(r["ok"], ans)
            self.assertFalse(os.path.exists(outside), ans)
        r = ev_patch.apply_answer("FILE: other.py\n```\nx\n```", self.repo, allowed=["pkg/*"])
        self.assertIn("outside the allowed set", r["error"])


# ---------------------------------------------------------------------------- sandbox
NET_PROBE = """import os, socket, sys
try:
    socket.create_connection(("127.0.0.1", int(os.environ["PORT"])), timeout=2).close()
except OSError as e:
    print("blocked:", e)
    sys.exit(3)
print("CONNECTED")
sys.exit(0)
"""
FILE_PROBE = """import os, sys
try:
    with open(os.environ["ESCAPE"], "w") as f:
        f.write("escaped")
except OSError as e:
    print("blocked:", e)
    sys.exit(3)
sys.exit(0)
"""


def make_task_dir(root, test_source, timeout_s=10, env=None, tid="T-sbx"):
    os.makedirs(os.path.join(root, "fx"), exist_ok=True)
    with open(os.path.join(root, "fx", "test_x.py"), "w") as f:
        f.write(test_source)
    chk = {"type": "code_tests", "command": ["{python}", "test_x.py"], "timeout_s": timeout_s}
    if env:
        chk["env"] = env
    with open(os.path.join(root, tid + ".task.json"), "w") as f:
        json.dump({"id": tid, "category": "code_write", "messages": [{"role": "user", "content": "x"}],
                   "fixture": "fx", "check": chk}, f)
    tasks, errs = ev_tasks.load_dir(root)
    assert not errs, errs
    return tasks[0]


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="eval-sbx-test-")
        self.sb = ev_scoring.CodeSandbox(self.d)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_reference_patch_passes_wrong_patch_fails(self):
        t = TASKS["SYNTHETIC-code_fix-001"]
        good = ev_scoring.grade(t, t.reference_answer, self.sb)
        self.assertEqual(good["status"], "pass", good["reason"])
        bad = ev_scoring.grade(t, WRONG_TEXT["SYNTHETIC-code_fix-001"], self.sb)
        self.assertEqual(bad["status"], "fail")
        self.assertIn("tests failed (exit code 1)", bad["reason"])
        self.assertTrue(bad["details"]["apply"]["ok"])                       # applied, but the tests fail
        hidden = ev_scoring.grade(TASKS["SYNTHETIC-code_write-001"], WRONG_TEXT["SYNTHETIC-code_write-001"], self.sb)
        self.assertEqual(hidden["status"], "fail")                          # hidden overlay tests ran

    def test_timeout_is_a_fail_with_reason(self):
        t = make_task_dir(os.path.join(self.d, "tt"), "import time\ntime.sleep(60)\n", timeout_s=1.5)
        g = self.sb.run_fixture(t, None, apply_answer=False)
        self.assertEqual(g["status"], "fail")
        self.assertIn("timeout", g["reason"])
        self.assertTrue(g["details"]["run"]["timed_out"])
        self.assertLess(g["details"]["run"]["elapsed_ms"], 10000)

    @unittest.skipIf(ev_sandbox.missing_isolation(ev_sandbox.capabilities()),
                     "strict sandbox not available on this host: %s" % ev_sandbox.missing_isolation(ev_sandbox.capabilities()))
    def test_no_network_and_no_file_escape(self):
        escape = os.path.join(self.d, "outside.txt")
        t = make_task_dir(os.path.join(self.d, "net"), NET_PROBE, env={"PORT": str(SRV.port)}, tid="T-net")
        g = self.sb.run_fixture(t, None, apply_answer=False)
        self.assertEqual(g["details"]["run"]["rc"], 3, g["details"]["run"])  # connect to a LISTENING port failed
        self.assertIn("blocked", g["details"]["run"]["stdout_tail"])
        iso = g["details"]["run"]["isolation"]
        self.assertTrue(iso["landlock_fs"] and (iso["netns"] or iso["landlock_net"]), iso)
        t2 = make_task_dir(os.path.join(self.d, "fs"), FILE_PROBE, env={"ESCAPE": escape}, tid="T-fs")
        g2 = self.sb.run_fixture(t2, None, apply_answer=False)
        self.assertEqual(g2["details"]["run"]["rc"], 3, g2["details"]["run"])
        self.assertFalse(os.path.exists(escape))

    def test_each_network_layer_blocks_alone(self):
        work = os.path.join(self.d, "layers")
        os.makedirs(os.path.join(work, "repo"))
        with open(os.path.join(work, "repo", "probe.py"), "w") as f:
            f.write(NET_PROBE)
        layers = [("python guard only", dict(use_netns=False, use_landlock=False, py_guard=True))]
        if CAPS["netns"]:
            layers.append(("netns only", dict(use_netns=True, use_landlock=False, py_guard=False)))
        if CAPS["landlock_net"]:
            layers.append(("landlock only", dict(use_netns=False, use_landlock=True, py_guard=False)))
        for name, kw in layers:
            r = ev_sandbox.run_command(work, ["{python}", "probe.py"], {"timeout_s": 10}, CAPS,
                                       env_extra={"PORT": str(SRV.port)}, **kw)
            self.assertEqual(r["rc"], 3, (name, r))
        r = ev_sandbox.run_command(work, ["{python}", "probe.py"], {"timeout_s": 10}, CAPS,
                                   env_extra={"PORT": str(SRV.port)}, use_netns=False, use_landlock=False, py_guard=False)
        self.assertEqual(r["rc"], 0, r)                                     # control: the port is reachable

    def test_missing_isolation_is_a_harness_error_not_a_fail(self):
        weak_caps = {"netns": False, "netns_prefix": None, "landlock_abi": 0, "landlock_fs": False, "landlock_net": False}
        sb = ev_scoring.CodeSandbox(self.d, caps=weak_caps)
        t = TASKS["SYNTHETIC-code_fix-001"]
        g = ev_scoring.grade(t, t.reference_answer, sb)
        self.assertEqual(g["status"], "harness_error")
        self.assertIn("sandbox unavailable", g["reason"])


# ---------------------------------------------------------------------------- tool simulator
class ToolSim(unittest.TestCase):
    def test_canned_builtin_and_errors(self):
        t = TASKS["SYNTHETIC-tool_use-001"]
        sim = ev_toolsim.ToolSimulator(t.sim_spec, t.tools)
        out, rec = sim.call("get_order", '{"order_id": "A-1027"}')
        self.assertEqual(rec["source"], "canned")
        self.assertEqual(json.loads(out)["customer_id"], "C-88")
        out, rec = sim.call("issue_refund", '{"order_id": "A-1027", "amount": 12.50, "reason": "damaged"}')
        self.assertEqual(json.loads(out)["refund_id"], "RF-55102")         # subset match, 12.50 == 12.5
        out, rec = sim.call("issue_refund", '{"order_id": "A-1027", "amount": "12.5"}')
        self.assertEqual(rec["source"], "default")                          # a string is not a number
        out, rec = sim.call("get_order", "{order_id: A-1027")
        self.assertFalse(rec["args_valid"])
        out, rec = sim.call("delete_everything", "{}")
        self.assertEqual(rec["source"], "unknown_tool")

    def test_calculator_is_safe(self):
        self.assertEqual(ev_toolsim.safe_arith("17*23.40*1.08"), 429.624)
        self.assertEqual(ev_toolsim.safe_arith("2^10"), 1024)
        for bad in ("__import__('os').system('true')", "(1).__class__", "9**9**9", "1/0", "x + 1"):
            with self.assertRaises(ev_toolsim.ToolError, msg=bad):
                ev_toolsim.safe_arith(bad)


class ToolScoring(unittest.TestCase):
    T = None

    @classmethod
    def setUpClass(cls):
        cls.T = TASKS["SYNTHETIC-tool_use-001"]

    def calls(self, *pairs, valid=True):
        return [{"name": n, "args": a, "args_valid": valid} for n, a in pairs]

    def test_order_args_forbidden_and_limit(self):
        get = ("get_order", {"order_id": "A-1027"})
        ref = ("issue_refund", {"order_id": "A-1027", "amount": 12.5})
        ok = ev_scoring.grade_tool_use(self.T, self.calls(get, ref), "Refund RF-55102 done", False)
        self.assertEqual(ok["status"], "pass", ok["reason"])
        wrong_order = ev_scoring.grade_tool_use(self.T, self.calls(ref, get), "Refund RF-55102 done", False)
        self.assertEqual(wrong_order["status"], "fail")
        self.assertEqual(wrong_order["reason"], "failed: order")
        wrong_args = ev_scoring.grade_tool_use(self.T, self.calls(get, ("issue_refund", {"order_id": "A-1027", "amount": 19.5})),
                                               "Refund RF-55102 done", False)
        self.assertIn("required_call[1] issue_refund", wrong_args["reason"])
        forbidden = ev_scoring.grade_tool_use(self.T, self.calls(get, ref, ("cancel_account", {"customer_id": "C-88"})),
                                              "Refund RF-55102 done", False)
        self.assertIn("forbidden cancel_account", forbidden["reason"])
        limit = ev_scoring.grade_tool_use(self.T, self.calls(get, get, get, get, get), None, True)
        self.assertIn("turn limit", limit["reason"])
        self.assertEqual(limit["status"], "fail")

    def test_turn_limit_alone_fails(self):
        # mutation M8 (turn limit counted as OK) survived until this test: every other turn-limit case
        # also failed another sub-check (final answer, max_calls)
        t = copy.copy(self.T)
        t.check = {k: v for k, v in self.T.check.items() if k not in ("final_answer", "max_calls")}
        calls = self.calls(("get_order", {"order_id": "A-1027"}), ("issue_refund", {"order_id": "A-1027", "amount": 12.5}))
        done = ev_scoring.grade_tool_use(t, calls, "anything", False)
        self.assertEqual(done["status"], "pass", done["reason"])
        looped = ev_scoring.grade_tool_use(t, calls, None, True)
        self.assertEqual(looped["status"], "fail")
        self.assertEqual(looped["reason"], "turn limit 6 reached; failed: turn_limit")


# ---------------------------------------------------------------------------- task files
class TaskFiles(unittest.TestCase):
    def test_synthetic_set_shape(self):
        self.assertTrue(10 <= len(TASKS) <= 20)
        self.assertEqual({t.category for t in TASKS.values()}, set(ev_tasks.CATEGORIES))
        for tid, t in TASKS.items():
            self.assertIn("SYNTHETIC", tid)
            self.assertTrue(t.raw.get("synthetic"))
            self.assertIsNotNone(t.reference_answer, tid)
            self.assertNotIn("{{", "".join(m["content"] for m in t.messages), tid)

    def test_validate_only_on_synthetic_set(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_eval.main(["--tasks", SYN, "--validate-only"])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("validated 15 tasks: all OK", buf.getvalue())

    def test_invalid_tasks_are_rejected(self):
        d = tempfile.mkdtemp(prefix="eval-bad-")
        try:
            bad = [{"id": "A", "category": "reasoning", "messages": [{"role": "user", "content": "q"}],
                    "check": {"type": "exact", "answers": ["1"]}, "typo_field": 1},
                   {"id": "B", "category": "poetry", "messages": [], "check": {}},
                   {"id": "C", "category": "code_fix", "messages": [{"role": "user", "content": "q"}],
                    "fixture": "../../etc", "check": {"type": "code_tests", "command": ["true"]}},
                   {"id": "D", "category": "structured_output", "messages": [{"role": "user", "content": "q"}],
                    "check": {"type": "json_schema", "schema": {"type": "object", "patternProperties": {}}}},
                   {"id": "E", "category": "reasoning", "messages": [{"role": "user", "content": "q"}],
                    "check": {"type": "exact", "answers": ["1"]}},
                   {"id": "E", "category": "reasoning", "messages": [{"role": "user", "content": "q"}],
                    "check": {"type": "exact", "answers": ["1"]}}]
            for i, b in enumerate(bad):
                with open(os.path.join(d, "t%d.task.json" % i), "w") as f:
                    json.dump(b, f)
            tasks, errs = ev_tasks.load_dir(d)
            text = "\n".join(errs)
            for frag in ("typo_field", "'category' must be one of", "outside the task directory",
                         "patternProperties", "duplicate id 'E'"):
                self.assertIn(frag, text)
            self.assertEqual([t.id for t in tasks], ["E"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(run_eval.main(["--tasks", d, "--validate-only"]), 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------- end to end against the mock
class EndToEnd(RunCase):
    def test_all_correct_answers_pass_every_category(self):
        r = self.run_eval(["--repeats", "1"], scripts("correct"))
        self.assertEqual(r["code"], 0, r["stdout"] + r["stderr"])
        bad = {k: v["reason"] for k, v in r["by_id"].items() if v["status"] != "pass"}
        self.assertEqual(bad, {})
        s = r["summary"]
        self.assertEqual(set(s["categories"]), set(ev_tasks.CATEGORIES))
        for c, b in s["categories"].items():
            self.assertEqual(b["task_pass_rate"], 1.0, c)
        o = s["overall"]
        self.assertEqual((o["n_tasks"], o["task_passes"]), (15, 15))
        self.assertEqual(o["task_ci95"], list(ev_stats.wilson(15, 15)))
        self.assertTrue(r["meta"]["is_mock_server"])
        self.assertTrue(s["is_mock_server"])
        self.assertIn("MOCK SERVER RUN", r["md"])
        self.assertIn("SYNTHETIC TASK SET", r["md"])
        self.assertEqual(r["meta"]["raw_rows"], len(r["raw"]))
        kinds = {x["kind"] for x in r["raw"]}
        self.assertEqual(kinds, {"request", "tool_exec", "sandbox", "result"})
        req = [x for x in r["raw"] if x["kind"] == "request"]
        self.assertTrue(all(x["metrics"]["ttft_ms"] is not None for x in req))        # stream timings recorded
        self.assertTrue(all(x["response"]["usage"] for x in req))
        self.assertEqual(r["meta"]["eval"]["task_set_sha256"], ev_tasks.task_set_sha256(list(TASKS.values())))
        self.assertEqual(r["by_id"]["SYNTHETIC-tool_use-001"]["n_requests"], 3)
        # the tool result went back to the model with the matching tool_call_id
        turn1 = [b for b in SRV.requests if len(b["messages"]) == 4 and b.get("tools")
                 and b["messages"][1]["content"].startswith("The customer says")][0]
        self.assertEqual(turn1["messages"][2]["tool_calls"][0]["id"], turn1["messages"][3]["tool_call_id"])
        self.assertEqual(json.loads(turn1["messages"][3]["content"])["customer_id"], "C-88")

    def test_wrong_answers_fail_every_task_with_a_reason(self):
        r = self.run_eval([], scripts("wrong"))
        self.assertEqual(r["code"], 0)
        st = {k: v["status"] for k, v in r["by_id"].items()}
        self.assertEqual(set(st.values()), {"fail"}, st)
        reasons = {k: v["reason"] for k, v in r["by_id"].items()}
        self.assertIn("protected path", reasons["SYNTHETIC-code_fix-003"])
        self.assertIn("no FILE: block or unified diff", reasons["SYNTHETIC-code_write-002"])
        self.assertIn("order", reasons["SYNTHETIC-tool_use-001"])
        self.assertIn("forbidden place_order", reasons["SYNTHETIC-tool_use-003"])
        self.assertIn("not valid JSON", reasons["SYNTHETIC-structured_output-001"])
        self.assertIn("value /2/label", reasons["SYNTHETIC-structured_output-002"])
        o = r["summary"]["overall"]
        self.assertEqual((o["task_passes"], o["n_tasks_scored"]), (0, 15))
        self.assertEqual(o["task_ci95"], list(ev_stats.wilson(0, 15)))
        self.assertGreater(o["mean_score"], 0.0)                               # partial credit is informational

    def test_tool_loop_variants(self):
        tid = "SYNTHETIC-tool_use-001"
        variants = {
            "wrong_args": [{"tool_calls": [call("get_order", {"order_id": "A-1072"})]},
                           {"tool_calls": [call("issue_refund", {"order_id": "A-1027", "amount": 19.5})]},
                           {"content": "Refund RF-55102."}],
            "malformed_args": [{"tool_calls": [call("get_order", "{order_id: A-1027")]},
                               {"tool_calls": [call("get_order", {"order_id": "A-1027"})]},
                               {"tool_calls": [call("issue_refund", {"order_id": "A-1027", "amount": 12.5})]},
                               {"content": "RF-55102"}],
            "endless": {"loop": {"tool_calls": [call("get_order", {"order_id": "A-1027"})]}},
        }
        for name, turns in variants.items():
            r = self.run_eval(["--only", tid], scripts("correct", {tid: turns}))
            row = r["by_id"][tid]
            self.assertEqual(row["status"], "fail", (name, row["reason"]))
            if name == "wrong_args":
                self.assertIn("required_call[0] get_order", row["reason"])
                self.assertIn("required_call[1] issue_refund", row["reason"])
            if name == "malformed_args":
                self.assertEqual(row["reason"], "failed: valid_arguments")
                tool_rows = [x for x in r["raw"] if x["kind"] == "tool_exec"]
                self.assertFalse(tool_rows[0]["args_valid"])
            if name == "endless":
                self.assertIn("turn limit 6 reached", row["reason"])
                self.assertIn("turn_limit", row["reason"].split("failed: ", 1)[1])
                self.assertEqual(row["n_requests"], 6)

    def test_request_errors_are_recorded_separately(self):
        faults = {"SYNTHETIC-reasoning-001": [{"http_error": 500}],
                  "SYNTHETIC-reasoning-002": [{"truncate": True, "content": "partial answer that is cut off here"}],
                  "SYNTHETIC-long_doc_qa-001": [{"malformed": True}],
                  "SYNTHETIC-tool_use-002": [{"tool_calls": [call("calculator", {"expression": "1+1"})]},
                                             {"http_error": 503}]}
        r = self.run_eval([], scripts("correct", faults))
        self.assertEqual(r["code"], 1)                                          # completed with errors
        st = {k: v["status"] for k, v in r["by_id"].items()}
        for tid in faults:
            self.assertEqual(st[tid], "request_error", (tid, r["by_id"][tid]["reason"]))
            self.assertIsNone(r["by_id"][tid]["pass"])
        self.assertIn("HTTP 500", r["by_id"]["SYNTHETIC-reasoning-001"]["reason"])
        self.assertIn("turn 1: HTTP 503", r["by_id"]["SYNTHETIC-tool_use-002"]["reason"])
        s = r["summary"]
        self.assertEqual(s["n_request_errors"], 4)
        o = s["overall"]
        self.assertEqual((o["n_valid_attempts"], o["attempt_passes"]), (11, 11))   # errors are not failures
        self.assertEqual(o["task_pass_rate"], 1.0)
        self.assertEqual(o["task_ci95"], list(ev_stats.wilson(11, 11)))
        self.assertEqual(s["categories"]["reasoning"]["n_tasks_scored"], 0)
        self.assertIsNone(s["categories"]["reasoning"]["task_pass_rate"])
        self.assertIn("Tasks without a valid attempt", r["md"])

    def test_non_streaming_mode_and_repeats(self):
        faults = {"SYNTHETIC-long_doc_qa-002": [{"malformed": True}]}
        r = self.run_eval(["--stream", "off", "--repeats", "2", "--only", "long_doc_qa,tool_use,SYNTHETIC-code_fix-00[12]"],
                          scripts("correct", faults))
        self.assertEqual(len(r["results"]), 2 * 8)
        st = {(x["task_id"], x["repeat"]): x["status"] for x in r["results"]}
        self.assertEqual(st[("SYNTHETIC-long_doc_qa-002", 1)], "request_error")
        self.assertEqual(sum(1 for v in st.values() if v == "pass"), 14)
        self.assertTrue(all(b.get("stream") is False for b in SRV.requests))
        self.assertEqual(r["summary"]["categories"]["long_doc_qa"]["n_attempts"], 6)

    def test_dead_server_gives_request_errors_only(self):
        r = self.run_eval(["--model", "m", "--only", "reasoning", "--timeout", "2"], url="http://127.0.0.1:9")
        self.assertEqual(r["code"], 1)
        self.assertEqual({x["status"] for x in r["results"]}, {"request_error"})
        self.assertIsNone(r["summary"]["overall"]["task_pass_rate"])
        self.assertFalse(r["meta"]["is_mock_server"])

    def test_mock_detection_matches_bench(self):
        r = self.run_eval(["--only", "reasoning"], scripts("correct"), owner="someone-else")
        self.assertFalse(r["meta"]["is_mock_server"])
        self.assertNotIn("MOCK SERVER RUN", r["md"])
        bench = bench_mock.MockServer(prefill_s_per_1k=0.0, decode_s_per_token=0.0).start()
        try:
            r = self.run_eval(["--only", "reasoning"], url=bench.url)
        finally:
            bench.stop()
        self.assertTrue(r["meta"]["is_mock_server"])                           # same marker as tools/bench
        self.assertEqual({x["status"] for x in r["results"]}, {"fail"})        # the toy model cannot answer


# ---------------------------------------------------------------------------- comparator
def write_run(d, outcomes, extra_rows=(), mock=False, sha=None):
    """outcomes: {task_id: [status per repeat]}"""
    os.makedirs(d)
    rows = []
    for tid, sts in outcomes.items():
        for rep, s in enumerate(sts):
            rows.append({"task_id": tid, "category": "reasoning" if tid < "t10" else "code_fix", "repeat": rep,
                         "status": s, "task_sha256": (sha or {}).get(tid, "h-" + tid)})
    rows.extend(extra_rows)
    with open(os.path.join(d, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"status": "ok", "is_mock_server": mock, "engine_desc": "E", "model_desc": "M",
            "eval": {"task_set_sha256": "S", "temperature": 0.0, "repeats": 1}}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)


class Compare(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="eval-cmp-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_known_example(self):
        # 20 paired tasks: b = 8 (A only), c = 1 (B only), 4 both pass, 7 both fail
        ids = ["t%02d" % i for i in range(20)]
        pattern = [("pass", "fail")] * 8 + [("fail", "pass")] + [("pass", "pass")] * 4 + [("fail", "fail")] * 7
        a = {tid: [p[0]] for tid, p in zip(ids, pattern)}
        b = {tid: [p[1]] for tid, p in zip(ids, pattern)}
        a["x-tie"] = ["pass", "fail"]                    # majority tie -> fail in A
        b["x-tie"] = ["pass"]
        a["x-err"] = ["pass"]
        b["x-err"] = ["request_error"]                    # no valid outcome in B -> excluded
        a["x-hash"], b["x-hash"] = ["pass"], ["fail"]     # content differs -> excluded
        a["x-only-a"] = ["pass"]
        write_run(os.path.join(self.d, "A"), a)
        write_run(os.path.join(self.d, "B"), b, mock=True, sha={"x-hash": "other"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = compare.main([os.path.join(self.d, "A"), os.path.join(self.d, "B"), "--n-boot", "3000",
                                 "--json", os.path.join(self.d, "cmp.json")])
        self.assertEqual(code, 0)
        with open(os.path.join(self.d, "cmp.json")) as f:
            res = json.load(f)
        o = res["overall"]
        self.assertEqual(o["n_paired"], 21)               # 20 + x-tie
        self.assertEqual((o["b_a_only"], o["c_b_only"]), (8, 2))       # x-tie: A fail (tie), B pass
        self.assertAlmostEqual(o["mcnemar_p"], ev_stats.mcnemar_exact(8, 2))
        self.assertAlmostEqual(o["mcnemar_p"], 2 * (1 + 10 + 45) / 1024)
        self.assertAlmostEqual(o["diff"], (8 - 2) / 21)
        self.assertLess(o["diff_ci95"][0], o["diff"])
        self.assertGreater(o["diff_ci95"][1], o["diff"])
        self.assertEqual(res["excluded"]["no_valid_outcome"], ["x-err"])
        self.assertEqual(res["excluded"]["task_hash_differs"], ["x-hash"])
        self.assertEqual(res["excluded"]["only_in_a"], ["x-only-a"])
        r = res["categories"]["reasoning"]
        self.assertEqual((r["n_paired"], r["b_a_only"], r["c_b_only"]), (10, 8, 1))
        self.assertAlmostEqual(r["mcnemar_p"], 20 / 512)
        self.assertGreater(r["diff_ci95"][0], 0.0)
        text = out.getvalue()
        self.assertIn("MOCK server", text)
        self.assertIn("McNemar p", text)

    def test_identical_runs(self):
        a = {"t%02d" % i: ["pass" if i % 3 else "fail"] for i in range(12)}
        write_run(os.path.join(self.d, "A"), a)
        write_run(os.path.join(self.d, "B"), a)
        res = compare.compare(*compare.load_run(os.path.join(self.d, "A")), *compare.load_run(os.path.join(self.d, "B")),
                              n_boot=500)
        o = res["overall"]
        self.assertEqual((o["b_a_only"], o["c_b_only"], o["mcnemar_p"], o["diff"]), (0, 0, 1.0, 0.0))
        self.assertEqual(o["diff_ci95"], [0.0, 0.0])
        self.assertTrue(o["low_power"])


if __name__ == "__main__":
    unittest.main()
