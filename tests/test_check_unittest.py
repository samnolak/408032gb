"""tools/ci/check_unittest.py and tools/ci/parity_expectations.json (gap 11 (a) of docs/reassessment/CORRECTNESS.md).

No dumps needed. Checked here:
  * the parser reads the verbose unittest output of the running Python (a generated module run in a subprocess), and
    synthetic logs: one summary, the skip lines with their test ids (decorator, setUpClass, docstring, subtest) and
    reasons;
  * the comparison accepts exactly the expected run and rejects every kind of difference (count, extra / missing /
    wrong-reason skip, unattributed skip, failure, unknown commit);
  * end to end through the command wrapper on a generated module shaped like the dump-parity tests: complete dumps
    pass, the legitimate pin-dependent skip passes at its pin only, a missing environment, an empty dump directory and
    a partial dump fail, a failing command keeps its status;
  * the command is bound to the step's commit: a driver pointed at a checkout of another commit, or an inline command
    whose LLAMA_CPP_GGUF_PY lies in one, fails before it runs (the drivers' own PARITY_LLAMA_COMMIT is not visible to
    the checker);
  * the committed expectations file: well formed, its entry exists for the pin every workflow step runs at
    (ec7630a in ci.yml, tools/engines/pins.json in parity-r1-pin.yml), every parity step of jobs cpu-parity, parity and
    ik-glm5next goes through the checker with a known step id, the modules listed per step are the ones the step
    runs, and the step's real test modules run without their environment are rejected at every pin.
"""
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci"))
import check_unittest as C  # noqa: E402

CHECKER = ROOT / "tools" / "ci" / "check_unittest.py"
SHA_A, SHA_B = "a" * 40, "b" * 40
# variables through which the drivers hand dumps, models and pins to the parity tests
PARITY_ENV = re.compile(r".*_(DUMP|GGUF)$|LLAMA_CPP_GGUF_PY$|GGML_GDN_HARNESS$|PARITY_\w+$")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not PARITY_ENV.match(k)}
    env.update(extra)
    return env


# a module shaped like the dump-parity tests: a decorator-skipped prefill class, run classes skipped in setUpClass
# when their run directory is missing; "mtp" is the run that exists at pin B only
FAKE = textwrap.dedent('''
    import os, pathlib, unittest
    DUMP = os.environ.get("FAKE_DUMP")
    SKIP = "set FAKE_DUMP (run the fake driver)"

    @unittest.skipUnless(DUMP, SKIP)
    class Prefill(unittest.TestCase):
        def test_a(self):
            self.assertTrue(pathlib.Path(DUMP, "pf", "manifest.json").exists())

        def test_b(self):
            """Docstring of test_b."""
            print("a line the test prints")
            self.assertTrue(pathlib.Path(DUMP, "pf", "manifest.json").exists())

    class _Run:
        RUN = None

        @classmethod
        def setUpClass(cls):
            if not DUMP:
                raise unittest.SkipTest(SKIP)
            if not pathlib.Path(DUMP, cls.RUN, "manifest.json").exists():
                raise unittest.SkipTest(f"no run {cls.RUN} under {DUMP}")

        def test_x(self):
            pass

        def test_y(self):
            pass

    @unittest.skipUnless(DUMP, SKIP)
    class Decode(_Run, unittest.TestCase):
        RUN = "decode"

    @unittest.skipUnless(DUMP, SKIP)
    class Mtp(_Run, unittest.TestCase):
        RUN = "mtp"
''')
FAKE_EXPECT = {"steps": {"fake": {
    "pin": "llama.cpp", "driver": "fake", "modules": ["fake_parity"],
    "expect": {
        SHA_A: {"ran": 4, "skipped": [{"test": "fake_parity.Mtp.setUpClass", "reason": r"no run mtp under .+"}],
                "source": "unit test"},
        SHA_B: {"ran": 6, "skipped": [], "source": "unit test"}}}}}


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)
        self.addCleanup(self._td.cleanup)


class Parse(_Tmp):
    def test_real_verbose_output_of_this_python(self):
        (self.tmp / "fmt_probe.py").write_text(textwrap.dedent('''
            import unittest
            @unittest.skipUnless(False, "decorator")
            class A(unittest.TestCase):
                def test_a1(self): pass
            class B(unittest.TestCase):
                @classmethod
                def setUpClass(cls): raise unittest.SkipTest("it's in setUpClass")
                def test_b1(self): pass
            class D(unittest.TestCase):
                def test_d1(self):
                    """Docstring."""
                    self.skipTest("method")
                def test_d2(self):
                    for i in range(2):
                        with self.subTest(i=i):
                            if i: self.skipTest("subtest")
                def test_d3(self): print("printed")
        '''))
        out = subprocess.run([sys.executable, "-m", "unittest", "fmt_probe", "-v"], cwd=self.tmp, env=clean_env(),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        r = C.parse(out.stdout + out.stderr)
        self.assertEqual((r["ran"], r["status"], r["counts"]), (4, "OK", {"skipped": 4}))
        self.assertEqual(sorted(r["skips"]), sorted([
            ("fmt_probe.A.test_a1", "decorator"), ("fmt_probe.B.setUpClass", "it's in setUpClass"),
            ("fmt_probe.D.test_d1", "method"), ("fmt_probe.D.test_d2 (i=1)", "subtest")]))

    def test_synthetic_failed_summary(self):
        r = C.parse("x\n------\nRan 12 tests in 1.5s\n\nFAILED (failures=2, errors=1, skipped=3)\n")
        self.assertEqual((r["ran"], r["status"]), (12, "FAILED"))
        self.assertEqual(r["counts"], {"failures": 2, "errors": 1, "skipped": 3})
        self.assertEqual(r["skips"], [])

    def test_one_summary_required(self):
        with self.assertRaises(C.CheckError):
            C.parse("artifacts: /tmp/x\n")
        with self.assertRaises(C.CheckError):
            C.parse("Ran 1 test in 0.1s\n\nOK\nRan 2 tests in 0.1s\n\nOK\n")

    def test_skip_without_description_is_kept_unattributed(self):
        r = C.parse("test_a (m.C.test_a) ... printed\nskipped 'late'\n\nRan 1 test in 0.0s\n\nOK (skipped=1)\n")
        self.assertEqual(r["skips"], [(None, "late")])


class Check(unittest.TestCase):
    EXP = {"ran": 52, "skipped": [{"test": "t.M.setUpClass", "reason": r"no MTP run under .+"}], "source": "x"}

    def result(self, ran=52, status="OK", counts=None, skips=None):
        skips = [("t.M.setUpClass", "no MTP run under /tmp/w")] if skips is None else skips
        counts = {"skipped": len(skips)} if counts is None else counts
        return dict(ran=ran, status=status, counts={k: v for k, v in counts.items() if v}, skips=skips)

    def test_exact_match_passes(self):
        self.assertIn("Ran 52, skipped 1 as expected", C.check(self.result(), self.EXP, "s"))

    def assertRejects(self, result, *fragments):
        with self.assertRaises(C.CheckError) as cm:
            C.check(result, self.EXP, "s")
        for f in fragments:
            self.assertTrue(any(f in p for p in cm.exception.problems), (f, cm.exception.problems))

    def test_count_mismatch(self):
        self.assertRejects(self.result(ran=46), "Ran 46 tests, expected 52")

    def test_extra_skip(self):
        self.assertRejects(self.result(skips=[("t.M.setUpClass", "no MTP run under /w"), ("t.D.setUpClass", "no run ub16 under /w")]),
                           "skipped=2, expected skipped=1", "skip not allowed here: t.D.setUpClass")

    def test_right_reason_on_another_test(self):
        """The allowed reason does not license the skip of another class (both id and reason must match)."""
        self.assertRejects(self.result(skips=[("t.D.setUpClass", "no MTP run under /tmp/w")]),
                           "skip not allowed here: t.D.setUpClass", "expected skip did not happen: t.M.setUpClass")

    def test_wrong_reason(self):
        self.assertRejects(self.result(skips=[("t.M.setUpClass", "set GLM5NEXT_DECODE_DUMP")]),
                           "skip not allowed here", "expected skip did not happen")

    def test_missing_expected_skip(self):
        self.assertRejects(self.result(ran=58, skips=[]), "skipped=0, expected skipped=1", "expected skip did not happen")

    def test_unattributed_skip(self):
        self.assertRejects(self.result(counts={"skipped": 1}, skips=[]), "0 skip lines in the log for skipped=1")

    def test_failures_and_other_outcomes(self):
        self.assertRejects(self.result(status="FAILED", counts={"failures": 1, "skipped": 1}), "status FAILED")
        self.assertRejects(self.result(counts={"expected failures": 1, "skipped": 1}), "unexpected outcomes")

    def test_unknown_commit(self):
        data = {"steps": {"s": {"pin": "llama.cpp", "expect": {SHA_A: self.EXP}}}}
        with self.assertRaises(C.CheckError) as cm:
            C.expectation(data, "s", SHA_B)
        self.assertIn("no expectations at llama.cpp " + SHA_B, str(cm.exception))


class ResolveCommit(_Tmp):
    STEP = {"pin": "llama.cpp"}

    def test_checkout_head_and_pin_variable(self):
        g = ["git", "-C", str(self.tmp), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", str(self.tmp)], check=True)
        subprocess.run(g + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
        head = subprocess.run(["git", "-C", str(self.tmp), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        self.assertEqual(C.resolve_commit(self.STEP, checkout=self.tmp, env={}), head)
        self.assertEqual(C.resolve_commit(self.STEP, checkout=self.tmp, env={"PARITY_LLAMA_COMMIT": head}), head)
        with self.assertRaises(C.ConfigError):      # the dumps' commit and the checkout disagree
            C.resolve_commit(self.STEP, checkout=self.tmp, env={"PARITY_LLAMA_COMMIT": SHA_A})
        self.assertEqual(C.resolve_commit(self.STEP, env={"PARITY_LLAMA_COMMIT": SHA_A}), SHA_A)
        self.assertEqual(C.resolve_commit({"pin": "ik_llama.cpp"}, env={"PARITY_IK_COMMIT": SHA_B}), SHA_B)

    def test_bad_commits(self):
        for kw in (dict(commit="ec7630a"), dict(), dict(commit=SHA_A, checkout=self.tmp)):
            with self.assertRaises(C.ConfigError, msg=str(kw)):
                C.resolve_commit(self.STEP, env={}, **kw)
        with self.assertRaises(C.ConfigError):
            C.resolve_commit(self.STEP, checkout=self.tmp / "missing", env={})


class Wrapper(_Tmp):
    """End to end: tools/ci/check_unittest.py -- python3 -m unittest fake_parity -v, as a CI step would run it."""

    def setUp(self):
        super().setUp()
        (self.tmp / "fake_parity.py").write_text(FAKE)
        (self.tmp / "expect.json").write_text(json.dumps(FAKE_EXPECT))
        self.dump = self.tmp / "dump"

    def make_runs(self, *runs):
        for r in runs:
            (self.dump / r).mkdir(parents=True, exist_ok=True)
            (self.dump / r / "manifest.json").write_text("{}")

    def step(self, sha, dump=None, cmd=None):
        env = clean_env(**({"FAKE_DUMP": str(dump)} if dump is not None else {}))
        cmd = cmd or [sys.executable, "-m", "unittest", "fake_parity", "-v"]
        out = subprocess.run([sys.executable, str(CHECKER), "--step", "fake", "--commit", sha, "--expect",
                              str(self.tmp / "expect.json"), "--"] + cmd,
                             cwd=self.tmp, env=env, capture_output=True, text=True)
        return out.returncode, out.stdout + out.stderr

    def test_complete_dumps_pass_at_the_pin_that_runs_everything(self):
        self.make_runs("pf", "decode", "mtp")
        rc, out = self.step(SHA_B, self.dump)
        self.assertEqual(rc, 0, out)
        self.assertIn("::notice::fake @ llama.cpp bbbbbbb: Ran 6, skipped 0 as expected", out)
        self.assertIn("a line the test prints", out)        # the command's output is passed through

    def test_legitimate_skip_passes_at_its_pin_only(self):
        self.make_runs("pf", "decode")
        rc, out = self.step(SHA_A, self.dump)
        self.assertEqual(rc, 0, out)
        self.assertIn("skipped 1 as expected (fake_parity.Mtp.setUpClass)", out)
        rc, out = self.step(SHA_B, self.dump)
        self.assertEqual(rc, 1, out)
        self.assertIn("Ran 4 tests, expected 6", out)
        self.assertIn("skip not allowed here: fake_parity.Mtp.setUpClass: 'no run mtp under", out)

    def test_missing_environment_fails_at_every_pin(self):
        for sha in (SHA_A, SHA_B):
            rc, out = self.step(sha)
            self.assertEqual(rc, 1, out)
            self.assertIn("::error::fake @ llama.cpp", out)
            self.assertIn("skip not allowed here: fake_parity.Prefill.test_a: 'set FAKE_DUMP (run the fake driver)'", out)

    def test_partial_or_empty_dumps_fail(self):
        self.make_runs("pf")                                     # decode and mtp missing: both classes skip
        for sha in (SHA_A, SHA_B):
            rc, out = self.step(sha, self.dump)
            self.assertEqual(rc, 1, out)
            self.assertIn("skip not allowed here: fake_parity.Decode.setUpClass: 'no run decode under", out)
        empty = self.tmp / "empty"
        empty.mkdir()
        rc, out = self.step(SHA_A, empty)                        # the prefill tests fail, the run classes skip
        self.assertNotEqual(rc, 0, out)
        self.assertIn("the command exited with status 1", out)

    def test_unknown_commit_fails(self):
        self.make_runs("pf", "decode", "mtp")
        rc, out = self.step("c" * 40, self.dump)
        self.assertEqual(rc, 1, out)
        self.assertIn("no expectations at llama.cpp " + "c" * 40, out)

    def test_command_status_is_kept(self):
        rc, out = self.step(SHA_B, cmd=[sys.executable, "-c", "import sys; print('Ran 6 tests in 0.1s'); sys.exit(3)"])
        self.assertEqual(rc, 3, out)
        self.assertIn("the command exited with status 3", out)

    def test_saved_log_mode(self):
        self.make_runs("pf", "decode")
        out = subprocess.run([sys.executable, "-m", "unittest", "fake_parity", "-v"], cwd=self.tmp,
                             env=clean_env(FAKE_DUMP=str(self.dump)), capture_output=True, text=True)
        (self.tmp / "step.log").write_text(out.stdout + out.stderr)
        argv = ["--step", "fake", "--expect", str(self.tmp / "expect.json"), "--log", str(self.tmp / "step.log")]
        with mock.patch.dict(os.environ), redirect_stdout(io.StringIO()) as printed:
            os.environ.pop("PARITY_LLAMA_COMMIT", None)
            self.assertEqual(C.main(argv + ["--commit", SHA_A]), 0)
            self.assertEqual(C.main(argv + ["--commit", SHA_B]), 1)
            self.assertEqual(C.main(argv), 2)                           # no commit at all
            os.environ["PARITY_LLAMA_COMMIT"] = SHA_A                   # the pin variable alone
            self.assertEqual(C.main(argv), 0)
            self.assertEqual(C.main(argv + ["--commit", SHA_B]), 2)     # disagrees with the pin variable
        self.assertIn("::error::check_unittest: no commit", printed.getvalue())


def make_repo(path, msg):
    """A git repository with one empty commit; returns its full sha."""
    g = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(g + ["commit", "-q", "--allow-empty", "-m", msg], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


class CommandCommit(_Tmp):
    """The command must run on the step's commit: the drivers export PARITY_LLAMA_COMMIT only to their own unittest
    process, so --checkout of one commit around a driver pointed at another checkout must not pass (audit of 19d09d2,
    FAIL 1). The driver here is a shell script taking the checkout as its first argument, like run_parity*.sh."""

    def setUp(self):
        super().setUp()
        self.a, self.b = self.tmp / "llama-a", self.tmp / "llama-b"
        self.sha_a, self.sha_b = make_repo(self.a, "a"), make_repo(self.b, "b")
        (self.a / "gguf-py").mkdir()
        (self.b / "gguf-py").mkdir()
        (self.tmp / "fake_parity.py").write_text(FAKE)
        exp = {"ran": 6, "skipped": [], "source": "unit test"}
        (self.tmp / "expect.json").write_text(json.dumps({"steps": {
            "fake": {"pin": "llama.cpp", "driver": "fake", "modules": ["fake_parity"],
                     "expect": {self.sha_a: exp, self.sha_b: exp}},
            "fake-ik": {"pin": "ik_llama.cpp", "driver": "fake", "modules": ["fake_parity"],
                        "expect": {self.sha_a: exp}}}}))
        for r in ("pf", "decode", "mtp"):
            (self.tmp / "dump" / r).mkdir(parents=True)
            (self.tmp / "dump" / r / "manifest.json").write_text("{}")
        self.driver = self.tmp / "run_parity_fake.sh"
        self.driver.write_text('#!/bin/sh\n# <checkout> <build>: the checkout is not read, like a driver whose tests '
                               'ignore it\nexec "$PY" -m unittest fake_parity -v\n')
        self.driver.chmod(0o755)
        self.notgit = self.tmp / "plain-dir"
        self.notgit.mkdir()

    def step(self, *args, step="fake", **env):
        env = clean_env(FAKE_DUMP=str(self.tmp / "dump"), PY=sys.executable, **env)
        out = subprocess.run([sys.executable, str(CHECKER), "--step", step, "--expect", str(self.tmp / "expect.json")]
                             + list(args), cwd=self.tmp, env=env, capture_output=True, text=True)
        return out.returncode, out.stdout + out.stderr

    def test_command_target(self):
        d = str(self.a)
        self.assertEqual(C.command_target(["tools/tinygen/run_parity.sh", d, "../build-tools"]), d)
        self.assertEqual(C.command_target(["env", "X=1", "Y=2", "run_parity.sh", d, "b"]), d)
        self.assertIsNone(C.command_target([sys.executable, "-m", "unittest", "tests.x", "-v"]))
        self.assertIsNone(C.command_target(["run_parity.sh", str(self.tmp / "missing"), "b"]))
        self.assertIsNone(C.command_target(["run_parity.sh"]))

    def test_driver_on_the_checkout_of_the_step_passes(self):
        rc, out = self.step("--checkout", str(self.a), "--", str(self.driver), str(self.a), "build")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"Ran 6, skipped 0 as expected; the driver's checkout {self.a} is at {self.sha_a[:7]}", out)

    def test_driver_on_another_checkout_fails_before_running(self):
        for args in (("--checkout", str(self.a)), ("--commit", self.sha_a)):
            rc, out = self.step(*args, "--", str(self.driver), str(self.b), "build")
            self.assertEqual(rc, 2, out)
            self.assertIn(f"the driver's checkout {self.b} is at {self.sha_b}, but the step's commit", out)
            self.assertNotIn("Ran ", out)                       # nothing ran
        rc, out = self.step("--checkout", str(self.a), "--", str(self.driver), str(self.notgit), "build")
        self.assertEqual(rc, 2, out)
        self.assertIn(f"cannot read the commit of {self.notgit}", out)

    def test_inline_command_is_bound_through_llama_cpp_gguf_py(self):
        cmd = ["--", sys.executable, "-m", "unittest", "fake_parity", "-v"]
        rc, out = self.step("--checkout", str(self.a), *cmd, LLAMA_CPP_GGUF_PY=str(self.a / "gguf-py"))
        self.assertEqual(rc, 0, out)
        self.assertIn(f"; LLAMA_CPP_GGUF_PY {self.a / 'gguf-py'} is at {self.sha_a[:7]}", out)
        rc, out = self.step("--checkout", str(self.a), *cmd, LLAMA_CPP_GGUF_PY=str(self.b / "gguf-py"))
        self.assertEqual(rc, 2, out)
        self.assertIn(f"LLAMA_CPP_GGUF_PY {self.b / 'gguf-py'} is at {self.sha_b}", out)
        # an ik step's gguf-py comes from llama.cpp, not from the ik checkout: not compared
        rc, out = self.step("--checkout", str(self.a), *cmd, step="fake-ik", LLAMA_CPP_GGUF_PY=str(self.b / "gguf-py"))
        self.assertEqual(rc, 0, out)

    def test_pin_variable_of_the_checker_environment_must_agree(self):
        rc, out = self.step("--checkout", str(self.a), "--", str(self.driver), str(self.a), "build",
                            PARITY_LLAMA_COMMIT=self.sha_b)
        self.assertEqual(rc, 2, out)
        self.assertIn(f"PARITY_LLAMA_COMMIT={self.sha_b} disagrees", out)


def workflow_steps(path):
    """(job, step text) for every step of a workflow file; plain text, no YAML library needed."""
    job, cur, out, in_jobs = None, None, [], False
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        m = re.match(r"^  ([\w-]+):\s*$", line)
        if m:
            if cur is not None:
                out.append((job, "\n".join(cur)))
            job, cur = m[1], None
            continue
        if re.match(r"^      - ", line):
            if cur is not None:
                out.append((job, "\n".join(cur)))
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        out.append((job, "\n".join(cur)))
    return out


PARITY_JOBS = {".github/workflows/ci.yml": ["cpu-parity"],
               ".github/workflows/parity-r1-pin.yml": ["parity", "ik-glm5next"]}


def parity_steps():
    """(workflow, job, step id, step text) of every step that runs a parity driver or named test modules."""
    out = []
    for wf, jobs in PARITY_JOBS.items():
        for job, text in workflow_steps(ROOT / wf):
            if job not in jobs:
                continue
            if "tools/tinygen/run_parity" in text or re.search(r"-m unittest (?!discover)", text):
                m = re.search(r"tools/ci/check_unittest\.py --step ([\w-]+)", text)
                out.append((wf, job, m[1] if m else None, text))
    return out


def driver_modules(text):
    """The unittest modules a step runs: inline, or on the `python3 -m unittest` line of its driver."""
    m = re.search(r"(tools/tinygen/run_parity\w*\.sh)", text)
    src = (ROOT / m[1]).read_text() if m else text
    lines = re.findall(r"-m unittest ((?:tests\.\w+ ?)+)", src)
    assert len(lines) == 1, (m and m[1], lines)
    return lines[0].split()


class ExpectationsFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = C.load_expectations()
        cls.steps = parity_steps()
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        cls.ci_pin = re.search(r"git -C \.\./llama\.cpp checkout ([0-9a-f]{40})", ci)[1]
        engines = json.loads((ROOT / "tools/engines/pins.json").read_text())["engines"]
        cls.r1 = {"llama.cpp": engines["llama.cpp"]["sha"], "ik_llama.cpp": engines["ik_llama.cpp"]["sha"]}

    def test_ci_pin_is_the_agents_md_pin(self):
        self.assertIn(f"| ggml-org/llama.cpp | {self.ci_pin} |", (ROOT / "AGENTS.md").read_text())

    def test_every_parity_step_goes_through_the_checker(self):
        self.assertGreaterEqual(len(self.steps), 17)
        for wf, job, step, text in self.steps:
            with self.subTest(wf=wf, job=job, step=step):
                self.assertIsNotNone(step, f"{wf} {job}: a parity step without tools/ci/check_unittest.py:\n{text}")
                self.assertIn(step, self.data["steps"])
                st = self.data["steps"][step]
                self.assertRegex(text, rf"--checkout \.\./{re.escape(st['pin'])} ")
                if st["driver"].startswith("tools/"):
                    self.assertIn(st["driver"], text)
                self.assertEqual(driver_modules(text), st["modules"])

    def test_expectations_exist_for_the_pin_each_step_runs_at(self):
        used = set()
        for wf, job, step, _ in self.steps:
            st = self.data["steps"][step]
            pin = self.ci_pin if wf.endswith("ci.yml") else self.r1[st["pin"]]
            used.add(step)
            with self.subTest(wf=wf, step=step):
                self.assertIn(pin, st["expect"], f"{step}: no expectations at {pin}")
        self.assertEqual(used, set(self.data["steps"]), "expectations for steps no workflow runs")

    def test_the_legitimate_mtp_skip_reason_is_the_one_the_test_gives(self):
        """At ec7630a the N9 driver produces no MTP run; tests/test_glm5next_mtp_parity.py then skips with a reason
        the expectations must recognize (and nothing else may skip)."""
        exp = self.data["steps"]["n9-glm5next-decode"]["expect"][self.ci_pin]
        with tempfile.TemporaryDirectory() as empty:
            env = clean_env(GLM5NEXT_DECODE_DUMP=empty, GLM5NEXT_MTP_GGUF=os.path.join(empty, "none.gguf"),
                            LLAMA_CPP_GGUF_PY=empty, PARITY_LLAMA_COMMIT=self.ci_pin)
            out = subprocess.run([sys.executable, "-m", "unittest", "tests.test_glm5next_mtp_parity", "-v"],
                                 cwd=ROOT, env=env, capture_output=True, text=True)
        r = C.parse(out.stdout + out.stderr)
        self.assertEqual((r["ran"], r["counts"]), (0, {"skipped": 1}), out.stderr[-2000:])
        C.check(dict(r, ran=exp["ran"]), exp, "n9 skip only")

    def test_real_modules_without_their_environment_fail_at_every_pin(self):
        """What a broken driver would leave: the step's test modules with no dump / model / gguf-py variables."""
        for step, st in self.data["steps"].items():
            out = subprocess.run([sys.executable, "-m", "unittest"] + st["modules"] + ["-v"], cwd=ROOT,
                                 env=clean_env(), capture_output=True, text=True)
            with self.subTest(step=step):
                self.assertEqual(out.returncode, 0, out.stderr[-2000:])   # green on its own: the gap
                r = C.parse(out.stdout + out.stderr)
                self.assertGreater(r["counts"].get("skipped", 0), 0)
                for sha, exp in st["expect"].items():
                    with self.assertRaises(C.CheckError, msg=f"{step} @ {sha[:7]}"):
                        C.check(r, exp, step)


if __name__ == "__main__":
    unittest.main()
