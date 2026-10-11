"""scripts/r1/run_r1.py: one-command R1 orchestration, tested with tools/bench/mock_server.py as the "engine".

EVERYTHING THESE TESTS PRODUCE IS MOCK / SYNTHETIC TEST DATA, NOT EVIDENCE about any engine or model:
  * the engine is tools/bench/mock_server.py (or a tiny python -c process for the failure paths);
  * the model is the SYNTHETIC header-only GGUF of scripts/r1/make_mock_model.py;
  * the verification pack of a mock run must be NOT RUN (reason mock), never PASS.
Covered: every config template plans with every variable substituted and the preregistered roofline prediction before
the engine start; --plan runs nothing; a full mock run gives a complete run directory, journal and pack; failing steps
are recorded and the rest still runs; the engine is always stopped (step failure, health timeout, early engine exit,
exception, KeyboardInterrupt, real SIGINT/SIGTERM to the orchestrator process); SIGKILL after the stop timeout; a
signal that arrives inside a shielded section (journal write, engine start/stop) is deferred, not lost.
Run: python3 -m unittest tests.test_r1_orchestration -v   (from the repo root)
"""
import contextlib
import glob
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "r1"))
import make_mock_model  # noqa: E402
import run_r1  # noqa: E402

CONFIGS = os.path.join(ROOT, "scripts", "r1", "configs")
MOCK = os.path.join(CONFIGS, "mock.json")
TEMPLATES = sorted(glob.glob(os.path.join(CONFIGS, "*.json")))
BENCH_CRITERIA = {"single_configuration", "openai_api", "context_probes_full_length", "output_after_full_prompt",
                  "agent_api_probes", "decode_filled_ctx_floor", "prefix_cache", "agent_loop", "decode_vs_roofline"}
PY = sys.executable


# ---------------------------------------------------------------------------- helpers
def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"     # a zombie is dead
    except OSError:
        return False


def pids_with(text):
    """PIDs whose command line contains text (Linux /proc)."""
    out = []
    for d in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(d, "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        pid = int(d.split("/")[2])
        if text in cmd and pid != os.getpid() and alive(pid):
            out.append(pid)
    return out


def run_main(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = run_r1.main(args)
        except KeyboardInterrupt as e:            # would abort the whole unittest run instead of failing this test
            raise AssertionError("KeyboardInterrupt escaped run_r1.main: %r" % (e,)) from None
    return code, out.getvalue() + err.getvalue()


def journal(run_dir):
    with open(os.path.join(run_dir, "steps.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ends(run_dir):
    out = {}
    for r in journal(run_dir):
        if r["event"] == "end":
            key = r["step"] + ("[%s]" % r["sub"] if r.get("sub") else "")
            assert key not in out, "two end records for %s" % key
            out[key] = r
    return out


def only_run_dir(root):
    dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and os.path.basename(d) != "mock-model"]
    assert len(dirs) == 1, dirs
    return dirs[0]


def write_config(tmp, name, edit):
    with open(MOCK, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["name"] = name
    edit(cfg)
    path = os.path.join(tmp, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return path


def kill_leftovers(run_dir):
    """Never leak a process into the next test, even when a mutation breaks the stop."""
    try:
        recs = journal(run_dir)
    except OSError:
        return
    for r in recs:
        pid = r.get("pid")
        if pid and alive(pid):
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)
    for pid in pids_with(run_dir):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def flag_value(argv, name):
    return argv[argv.index(name) + 1]


# ---------------------------------------------------------------------------- plans
class Plans(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1-plan-")
        self.root = os.path.join(self.tmp, "runs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, config, *extra):
        code, out = run_main(["--config", config, "--out", self.root, "--plan", "--json"] + list(extra))
        self.assertEqual(code, 0, out)
        return json.loads(out)

    def test_every_template_substitutes_every_variable_and_preregisters_first(self):
        names = {os.path.basename(t) for t in TEMPLATES}
        self.assertTrue({"mock.json", "llamacpp-server.json", "llamacpp-server-moecache.json", "ik-llamacpp-server.json",
                         "strata-serve-qwen38fn.json"} <= names, names)
        for path in TEMPLATES:
            with self.subTest(config=os.path.basename(path)):
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                if raw.get("mock"):
                    self.assertTrue(raw["_label"].startswith("MOCK"), raw.get("_label"))
                else:
                    self.assertTrue(raw["_label"].startswith("PROVISIONAL"), raw.get("_label"))
                p = self.plan(path)
                body = json.dumps({k: p[k] for k in ("items", "vars", "engine_desc", "model_desc", "mode", "roofline")})
                self.assertNotIn("${", body)
                self.assertEqual(p["run_dir"], os.path.join(self.root, os.path.basename(p["run_dir"])))
                self.assertRegex(os.path.basename(p["run_dir"]), r"^\d{8}T\d{6}Z-%s$" % re.escape(p["name"]))
                steps = [it["step"] for it in p["items"]]
                self.assertEqual(sorted(set(steps), key=steps.index), run_r1.STEPS)     # every step, in order
                live = [it for it in p["items"] if not it["skip"]]
                for it in live:
                    if it["kind"] in ("cmd", "engine"):
                        self.assertTrue(it["argv"] and all(isinstance(a, str) and a for a in it["argv"]), it)
                idx = {it["step"] + (it["sub"] or ""): it["index"] for it in live}
                pred = [it["index"] for it in live if it["step"] == "roofline_prediction"]
                start = idx["engine_start"]
                self.assertTrue(pred and max(pred) < start, "prediction must be planned before the engine start")
                for s in ("health_wait", "matrix", "probes", "vs_roofline"):
                    self.assertTrue(all(it["index"] > start for it in live if it["step"] == s), s)
                self.assertEqual(max(it["index"] for it in live if it["step"] != "pack"),
                                 max(it["index"] for it in live if it["step"] in ("engine_stop", "nvidia_smi_after")))
                # the same engine mode goes to the prediction and to vs_roofline
                rf = [it["argv"] for it in live if it["step"] == "roofline_prediction" and it["kind"] == "cmd"]
                vs = [it["argv"] for it in live if it["step"] == "vs_roofline"][0]
                for a in rf:
                    for fl in ("--overlap", "--spec-k", "--spec-accept", "--miss", "--hw", "--expert-bpw", "--dense-bpw",
                               "--kv-bytes"):
                        self.assertEqual(flag_value(a, fl), flag_value(vs, fl), fl)
                    self.assertEqual(flag_value(a, "--strategies"), flag_value(vs, "--strategy"))
                    self.assertEqual(flag_value(a, "--sweep-hit"), flag_value(vs, "--hit"))
                    self.assertEqual(a[2], flag_value(vs, "--spec"))
                eng = [it for it in live if it["step"] == "engine_start"][0]
                self.assertIn(p["vars"]["PORT"], eng["argv"])
                pk = [it for it in live if it["step"] == "pack" and it["sub"] == "build"][0]["argv"]
                self.assertEqual(flag_value(pk, "--runs"), os.path.join(p["run_dir"], "artifacts"))
                self.assertEqual(flag_value(pk, "--out"), os.path.join(p["run_dir"], "pack"))
        self.assertFalse(os.path.exists(self.root), "--plan must not create anything")

    def test_plan_runs_nothing(self):
        mark = os.path.join(self.tmp, "MARK")
        touch = [PY, "-c", "import sys; open(sys.argv[1], 'w').close()"]

        def edit(c):
            c["engine"]["argv"] = touch + [mark + "-engine", "${PORT}"]
            c["engine"]["version_argv"] = touch + [mark + "-version"]
            c["engine"]["help_argv"] = touch + [mark + "-help"]
            c["engine"]["after_ready"] = [touch + [mark + "-ready"]]
            c["engine"]["mkdirs"] = [os.path.join(self.tmp, "MKDIR")]
        cfg = write_config(self.tmp, "plan-only", edit)
        code, out = run_main(["--config", cfg, "--out", self.root, "--plan", "--set",
                              "MOCK_MODEL_DIR=" + os.path.join(self.tmp, "model")])
        self.assertEqual(code, 0, out)
        for n, step in enumerate(run_r1.STEPS):
            self.assertIn(step, out)
        self.assertIn(mark + "-engine", out)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["plan-only.json"], "the plan created or ran something")
        self.assertFalse(os.path.exists(self.root))

    def test_config_errors_exit_2(self):
        bad = [("unknown-var", lambda c: c["engine"]["argv"].append("${NO_SUCH_VAR}"), "NO_SUCH_VAR"),
               ("owned-flag", lambda c: c["steps"]["matrix"]["runs"][0]["args"].extend(["--url", "x"]), "--url"),
               ("stop-disabled", lambda c: c["steps"].update(engine_stop={"enabled": False}), "engine_stop"),
               ("bad-mode", lambda c: c["mode"].update(overlap="maybe"), "mode.overlap"),
               ("typo-key", lambda c: c.update(stpes={}), "stpes")]
        for name, edit, needle in bad:
            with self.subTest(case=name):
                code, out = run_main(["--config", write_config(self.tmp, name, edit), "--out", self.root, "--plan"])
                self.assertEqual(code, 2, out)
                self.assertIn(needle, out)
        code, out = run_main(["--config", MOCK, "--out", self.root, "--plan", "--only", "matrix,bogus"])
        self.assertEqual(code, 2, out)
        self.assertIn("bogus", out)
        self.assertFalse(os.path.exists(self.root))


# ---------------------------------------------------------------------------- the full mock run
class FullMockRun(unittest.TestCase):
    """scripts/r1/configs/mock.json exactly as committed; only MOCK_MODEL_DIR comes from --set."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="r1-mock-")
        cls.root = os.path.join(cls.tmp, "runs")
        cls.model_dir = os.path.join(cls.tmp, "model-SYNTHETIC")
        make_mock_model.write(cls.model_dir)
        cls.code, cls.out = run_main(["--config", MOCK, "--out", cls.root, "--set", "MOCK_MODEL_DIR=" + cls.model_dir])
        cls.run_dir = only_run_dir(cls.root)
        cls.ends = ends(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        kill_leftovers(cls.run_dir)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exit_code_and_layout(self):
        self.assertEqual(self.code, 0, self.out)
        self.assertRegex(os.path.basename(self.run_dir), r"^\d{8}T\d{6}Z-mock$")
        for rel in ("steps.jsonl", "run.json", "artifacts/orchestrator/config.json", "artifacts/orchestrator/plan.txt",
                    "artifacts/orchestrator/resolved.json", "artifacts/orchestrator/steps.jsonl",
                    "artifacts/weights/weights_verify.txt", "artifacts/weights/SYNTHETIC-mock-model.manifest.json",
                    "artifacts/quant/quant_audit.json", "artifacts/prereg/prereg.json",
                    "artifacts/engine/argv.json", "artifacts/engine/engine.log", "artifacts/engine/exit.json",
                    "artifacts/engine/version.txt", "artifacts/engine/help.txt",
                    "artifacts/matrix/filled-4k/summary.json", "artifacts/matrix/smoke-1k/raw.jsonl",
                    "artifacts/probes/4k/summary.json", "artifacts/roofline/vs_roofline.json",
                    "artifacts/roofline/vs_roofline.cmd.txt", "pack/verdict.json", "pack/VERDICT.md",
                    "pack/MANIFEST.sha256"):
            self.assertTrue(os.path.isfile(os.path.join(self.run_dir, rel)), rel)
        with open(os.path.join(self.run_dir, "artifacts", "orchestrator", "config.json"), "rb") as f, \
                open(MOCK, "rb") as g:
            self.assertEqual(f.read(), g.read())
        with open(os.path.join(self.run_dir, "run.json")) as f:
            rj = json.load(f)
        self.assertEqual((rj["status"], rj["exit_code"], rj["mock"]), ("completed", 0, True))

    def test_journal_has_every_item_in_order_with_outputs(self):
        recs = journal(self.run_dir)
        self.assertEqual([r["seq"] for r in recs], list(range(1, len(recs) + 1)))
        plan = run_r1.Cell(MOCK, self.root, ["MOCK_MODEL_DIR=" + self.model_dir], stamp="20000101T000000Z").plan
        self.assertEqual(list(self.ends), [it.key for it in plan])           # one end record per item, plan order
        smi = shutil.which("nvidia-smi")
        for key, r in self.ends.items():
            want = "skipped" if key in ("slot_persist", "logprob_compare", "engine_build_check") or \
                (key.startswith("nvidia_smi") and not smi) else "ok"     # the mock has no engine.pin
            self.assertEqual(r["status"], want, (key, r.get("reason")))
            if r["status"] == "ok":
                self.assertLessEqual(r["utc_start"], r["utc_end"])
                for rel, digest in r["sha256"].items():
                    self.assertIn(rel, r["outputs"])
            if r["kind"] == "cmd" and r["status"] == "ok":
                self.assertEqual(r["exit_code"], 3 if key == "pack[build]" else 0, key)
                self.assertTrue(r["argv"] and r["outputs"], key)
                for rel in r["outputs"]:
                    self.assertTrue(os.path.exists(os.path.join(self.run_dir, rel)), (key, rel))
        self.assertEqual(self.ends["quant_audit"]["verdict"], "PASS")
        self.assertEqual(self.ends["pack[build]"]["verdict"], "NOT RUN")

    def test_prediction_is_written_before_the_engine_starts(self):
        recs = journal(self.run_dir)
        seq = {(r["step"], r.get("sub"), r["event"]): r["seq"] for r in recs}
        start = [r for r in recs if r["step"] == "engine_start" and r["event"] == "start"][0]
        preds = [r for r in recs if r["step"] == "roofline_prediction" and r["event"] == "end"]
        self.assertEqual(len(preds), 3)                      # ctx=1024, ctx=4096 (from --inputs), record
        for r in preds:
            self.assertEqual(r["status"], "ok")
            self.assertLess(r["seq"], start["seq"])
            self.assertLessEqual(r["utc_end"], start["utc_start"])
        self.assertLess(seq[("roofline_prediction", "record", "end")], seq[("engine_start", None, "start")])
        with open(os.path.join(self.run_dir, "artifacts", "prereg", "prereg.json")) as f:
            pre = json.load(f)
        self.assertEqual([p["ctx"] for p in pre["predictions"]], [1024, 4096])
        for p in pre["predictions"]:
            self.assertEqual(run_r1.sha256_file(os.path.join(self.run_dir, p["file"])), p["sha256"])
            self.assertTrue(p["decode"] and p["decode"][0]["tok_s"] > 0)
        self.assertEqual(pre["engine"]["argv"], self.ends["engine_start"]["argv"])
        self.assertEqual(pre["mode"]["overlap"], "no")

    def test_pack_is_not_run_because_of_the_mock(self):
        with open(os.path.join(self.run_dir, "pack", "verdict.json")) as f:
            v = json.load(f)
        self.assertEqual(v["overall"], "NOT RUN")
        self.assertEqual(os.path.realpath(v["runs_dir"]), os.path.realpath(os.path.join(self.run_dir, "artifacts")))
        res = {c["id"]: c for c in v["criteria_results"]}
        for cid in BENCH_CRITERIA:
            self.assertEqual((res[cid]["status"], res[cid]["reason_code"]), ("NOT RUN", "mock"), cid)
        kinds = {a["kind"] for a in v["artifacts"]}
        self.assertTrue({"matrix", "probes", "quant_audit", "roofline", "weights_manifest", "weights_verify"} <= kinds,
                        kinds)
        self.assertEqual(v["unparseable"], [])
        self.assertTrue(v["synthetic_markers"])
        with open(os.path.join(self.run_dir, "run.json")) as f:
            pk = json.load(f)["pack"]
        self.assertEqual(pk["overall"], "NOT RUN")
        self.assertTrue(BENCH_CRITERIA <= set(pk["mock_criteria"]), pk["mock_criteria"])
        chk = subprocess.run([PY, os.path.join(ROOT, "tools", "verify", "pack.py"), "--check",
                              os.path.join(self.run_dir, "pack")], capture_output=True, text=True, timeout=60)
        self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)
        self.assertIn("criteria NOT RUN because the server was the mock", self.out)

    def test_engine_recorded_and_stopped(self):
        start, stop = self.ends["engine_start"], self.ends["engine_stop"]
        pid = start["pid"]
        self.assertFalse(alive(pid), "the mock engine is still running")
        self.assertEqual(stop["status"], "ok")
        self.assertEqual(stop["signals_sent"], ["SIGTERM"])
        self.assertIsNotNone(stop["exit_code"])
        with open(os.path.join(self.run_dir, "artifacts", "engine", "argv.json")) as f:
            argv = json.load(f)
        self.assertEqual(argv["argv"], start["argv"])
        self.assertEqual(argv["env_overrides"], {"PYTHONUNBUFFERED": "1"})
        with open(os.path.join(self.run_dir, "artifacts", "engine", "exit.json")) as f:
            self.assertEqual(json.load(f)["exit_code"], stop["exit_code"])
        with open(os.path.join(self.run_dir, "artifacts", "engine", "engine.log")) as f:
            self.assertIn("mock server on", f.read())
        with open(os.path.join(self.run_dir, "artifacts", "engine", "version.txt")) as f:
            self.assertIn("Python", f.read())
        self.assertEqual(self.ends["engine_version[help]"]["flags_missing_from_help"], [])
        with open(os.path.join(self.run_dir, "artifacts", "matrix", "filled-4k", "meta.json")) as f:
            meta = json.load(f)
        self.assertTrue(meta["is_mock_server"])
        self.assertIn(" ".join(start["argv"][-4:]), meta["engine_desc"])
        self.assertIn("SYNTHETIC-mock-model-Q4_K.gguf", meta["model_desc"])


# ---------------------------------------------------------------------------- failure paths
class FailurePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1-fail-")
        self.root = os.path.join(self.tmp, "runs")
        self.run_dir = None

    def tearDown(self):
        if self.run_dir:
            kill_leftovers(self.run_dir)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cfg(self, cfg, *extra):
        code, out = run_main(["--config", cfg, "--out", self.root] + list(extra))
        self.run_dir = only_run_dir(self.root)
        return code, out, ends(self.run_dir)

    def test_failing_steps_are_recorded_and_the_engine_is_still_stopped(self):
        good = os.path.join(self.tmp, "model-SYNTHETIC")
        make_mock_model.write(good)
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        corpus = os.path.join(self.tmp, "corpus-SYNTHETIC.txt")
        with open(corpus, "w") as f:
            f.write(" ".join("SYNTHETIC corpus sentence %d." % i for i in range(300)))

        def edit(c):
            c["model"]["files"] = [os.path.join(empty, "SYNTHETIC-mock-model-Q4_K.gguf")]     # missing model file
            c["model"]["weights_manifest"] = os.path.join(good, make_mock_model.MANIFEST_NAME)
            c["model"]["weights_dest"] = empty                                               # -> MISSING, exit 1
            c["reference"] = {"url": "http://${HOST}:${PORT}"}       # the mock has no logprobs -> the tool fails
            c["steps"]["logprob_compare"] = {"corpus": [corpus], "args": ["--prefixes", "2", "--top-k", "3"]}
            c["engine"]["after_ready"] = [[PY, "-c", "import sys; sys.exit(5)"]]           # soft: must not block
            c["steps"]["matrix"]["runs"] = c["steps"]["matrix"]["runs"][:1]
            c["steps"]["probes"]["runs"][0]["args"] = ["--probes", "json", "--seed", "0", "--timeout", "30"]
        code, out, e = self.run_cfg(write_config(self.tmp, "failing", edit))
        self.assertEqual(code, 1, out)
        self.assertEqual((e["weights_verify"]["status"], e["weights_verify"]["exit_code"]), ("failed", 1))
        with open(os.path.join(self.run_dir, "artifacts", "weights", "weights_verify.txt")) as f:
            self.assertIn("MISSING", f.read())
        self.assertEqual((e["quant_audit"]["status"], e["quant_audit"]["exit_code"]), ("failed", 1))
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "artifacts", "quant", "quant_audit.json")))
        self.assertEqual((e["logprob_compare"]["status"], e["logprob_compare"]["exit_code"]), ("failed", 1))
        # N11: the tool must fail on the mock's missing logprobs, not on a missing artifacts/quality/ directory (on a
        # real llama-server it computed 200 prefixes and then died writing --out); its errors are kept as an artifact
        with open(os.path.join(self.run_dir, "artifacts", "quality", "logprob_compare.json")) as f:
            lp = json.load(f)
        self.assertEqual((lp["summary"], len(lp["errors"])), (None, 2))
        self.assertIn("artifacts/quality/logprob_compare.json", e["logprob_compare"]["outputs"])
        self.assertEqual(e["health_wait[after-ready-1]"]["status"], "failed")
        for key in ("roofline_prediction[record]", "engine_start", "health_wait", "matrix[smoke-1k]", "probes[4k]",
                    "vs_roofline", "engine_stop", "pack[build]", "pack[check]"):
            self.assertEqual(e[key]["status"], "ok", (key, e[key].get("reason")))
        self.assertFalse(alive(e["engine_start"]["pid"]), "the engine survived a failing step")
        self.assertIsNotNone(e["engine_stop"]["exit_code"])
        with open(os.path.join(self.run_dir, "run.json")) as f:
            rj = json.load(f)
        self.assertEqual((rj["status"], rj["exit_code"]), ("completed_with_failures", 1))
        self.assertEqual(rj["step_states"]["weights_verify"], "failed")
        self.assertNotEqual(rj["pack"]["overall"], "PASS")

    def test_health_timeout_fails_cleanly_and_kills_a_stubborn_engine(self):
        def edit(c):
            c["engine"]["argv"] = [PY, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                   "print('stubborn', flush=True); time.sleep(120)", "${PORT}"]
            c["engine"]["health"]["timeout_s"] = 1.5
            c["engine"]["stop"]["timeout_s"] = 1
        t0 = time.monotonic()
        code, out, e = self.run_cfg(write_config(self.tmp, "timeout", edit),
                                    "--only", "engine_start,health_wait,matrix,probes")
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(code, 1, out)
        h = e["health_wait"]
        self.assertEqual(h["status"], "failed")
        self.assertIn("timeout after 1.5 s", h["reason"])
        for key in ("matrix[smoke-1k]", "matrix[filled-4k]", "probes[4k]"):
            self.assertEqual((e[key]["status"], e[key].get("blocked_by")), ("skipped", "health_wait"), key)
        st = e["engine_stop"]
        self.assertEqual(st["signals_sent"], ["SIGTERM", "SIGKILL"])
        self.assertEqual(st["exit_code"], -signal.SIGKILL)
        self.assertFalse(alive(e["engine_start"]["pid"]))
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, "artifacts", "matrix")))

    def test_engine_that_exits_early_fails_health_at_once(self):
        def edit(c):
            c["engine"]["argv"] = [PY, "-c", "import sys; print('boom', flush=True); sys.exit(3)", "${PORT}"]
            c["engine"]["health"]["timeout_s"] = 20
        t0 = time.monotonic()
        code, out, e = self.run_cfg(write_config(self.tmp, "early-exit", edit), "--only", "engine_start,health_wait,matrix")
        self.assertLess(time.monotonic() - t0, 10, "health_wait did not notice that the engine exited")
        self.assertEqual(code, 1, out)
        self.assertIn("exited with code 3", e["health_wait"]["reason"])
        self.assertEqual((e["engine_stop"]["status"], e["engine_stop"]["exit_code"]), ("failed", 3))
        self.assertEqual(e["matrix[smoke-1k]"]["status"], "skipped")
        with open(os.path.join(self.run_dir, "artifacts", "engine", "engine.log")) as f:
            self.assertIn("boom", f.read())

    def test_exception_inside_a_step_is_recorded_and_the_run_continues(self):
        orig = run_r1.Run.write_prereg

        def broken(self_, it):
            raise RuntimeError("SYNTHETIC failure for the test")
        run_r1.Run.write_prereg = broken
        try:
            code, out, e = self.run_cfg(MOCK, "--set", "MOCK_MODEL_DIR=" + self.tmp,
                                        "--only", "roofline_prediction,engine_start,health_wait")
        finally:
            run_r1.Run.write_prereg = orig
        self.assertEqual(code, 1, out)
        r = e["roofline_prediction[record]"]
        self.assertEqual(r["status"], "failed")
        self.assertIn("SYNTHETIC failure", r["reason"])
        self.assertEqual(e["health_wait"]["status"], "ok")
        self.assertFalse(alive(e["engine_start"]["pid"]))
        self.assertEqual(e["engine_stop"]["status"], "ok")

    def test_engine_is_killed_even_when_the_regular_stop_fails(self):
        # lead re-check 2026-10-10: disabling the last-resort kill (Run.emergency_kill) survived every other test,
        # because the regular engine_stop step always succeeded before it
        orig = run_r1.Run.stop_engine

        def broken(self_, it):
            raise RuntimeError("SYNTHETIC failure of the regular stop")
        run_r1.Run.stop_engine = broken
        pid = None
        try:
            code, out, e = self.run_cfg(MOCK, "--set", "MOCK_MODEL_DIR=" + self.tmp,
                                        "--only", "roofline_prediction,engine_start,health_wait")
            pid = e["engine_start"]["pid"]
            self.assertNotEqual(code, 0, out)
            self.assertFalse(alive(pid), "the engine survived a failing regular stop")
        finally:
            run_r1.Run.stop_engine = orig
            if pid and alive(pid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)

    def test_keyboard_interrupt_stops_the_engine_and_skips_the_rest(self):
        orig = run_r1.Run.wait_health

        def then_interrupt(self_, it):
            rec = orig(self_, it)
            raise KeyboardInterrupt()
        run_r1.Run.wait_health = then_interrupt
        try:
            code, out, e = self.run_cfg(MOCK, "--set", "MOCK_MODEL_DIR=" + self.tmp,
                                        "--only", "engine_start,health_wait,matrix,probes,pack")
        finally:
            run_r1.Run.wait_health = orig
        self.assertEqual(code, 130, out)
        self.assertEqual(e["health_wait"]["status"], "ok")
        for key in ("matrix[smoke-1k]", "matrix[filled-4k]", "probes[4k]", "pack[build]", "pack[check]"):
            self.assertEqual(e[key]["status"], "skipped", key)
            self.assertIn("interrupted", e[key]["reason"], key)
        self.assertFalse(alive(e["engine_start"]["pid"]))
        self.assertEqual(e["engine_stop"]["status"], "ok")
        with open(os.path.join(self.run_dir, "run.json")) as f:
            rj = json.load(f)
        self.assertEqual((rj["status"], rj["exit_code"], rj["interrupted_by"]), ("interrupted", 130, "SIGINT"))

    def test_signal_in_a_shielded_section_is_deferred_not_lost(self):
        orig = run_r1.Run.write_prereg

        def signal_while_shielded(self_, it):
            with self_.shielded():
                os.kill(os.getpid(), signal.SIGTERM)      # arrives while the journal/engine would be in a shielded write
            return orig(self_, it)
        run_r1.Run.write_prereg = signal_while_shielded
        try:
            code, out, e = self.run_cfg(MOCK, "--set", "MOCK_MODEL_DIR=" + self.tmp,
                                        "--only", "roofline_prediction,engine_start,health_wait")
        finally:
            run_r1.Run.write_prereg = orig
        self.assertEqual(code, 128 + signal.SIGTERM, out)
        # the deferred signal fires right after the next complete journal record (this item's "start")
        self.assertEqual(e["roofline_prediction[record]"]["status"], "interrupted")
        self.assertEqual(e["engine_start"]["status"], "skipped")              # and nothing after it starts
        self.assertIn("interrupted by SIGTERM", e["engine_start"]["reason"])
        self.assertNotIn("pid", e["engine_start"])

    def test_signal_to_the_orchestrator_process_stops_engine_and_tool(self):
        model = os.path.join(self.tmp, "model-SYNTHETIC")
        make_mock_model.write(model)
        for sig, want in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig.name):
                root = os.path.join(self.tmp, "runs-" + sig.name)
                proc = subprocess.Popen([PY, os.path.join(ROOT, "scripts", "r1", "run_r1.py"), "--config", MOCK,
                                         "--out", root, "--set", "MOCK_MODEL_DIR=" + model,
                                         "--set", "DECODE_S_PER_TOKEN=0.05",
                                         "--only", "engine_start,health_wait,matrix,probes,pack"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
                run_dir = None
                try:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        dirs = glob.glob(os.path.join(root, "*"))
                        if dirs and os.path.exists(os.path.join(dirs[0], "steps.jsonl")):
                            run_dir = dirs[0]
                            if any(r["step"] == "matrix" and r["event"] == "start" for r in journal(run_dir)):
                                break
                        time.sleep(0.05)
                    else:
                        self.fail("the matrix step never started")
                    time.sleep(0.5)
                    self.assertTrue(pids_with(os.path.join(run_dir, "artifacts", "matrix")), "no tool running")
                    proc.send_signal(sig)
                    out, _ = proc.communicate(timeout=60)
                    out = out.decode("utf-8", "replace")
                    self.assertEqual(proc.returncode, want, out)
                    e = ends(run_dir)
                    self.assertEqual(e["matrix[smoke-1k]"]["status"], "interrupted")
                    # the tool got SIGINT from the orchestrator and exited (run_matrix: 130; uncaught: -SIGINT)
                    self.assertIn(e["matrix[smoke-1k]"]["exit_code"], (130, -signal.SIGINT))
                    self.assertEqual(e["matrix[filled-4k]"]["status"], "skipped")
                    self.assertFalse(alive(e["engine_start"]["pid"]), "the engine survived %s" % sig.name)
                    self.assertEqual(e["engine_stop"]["status"], "ok")
                    self.assertEqual(pids_with(run_dir), [], "a tool survived %s" % sig.name)
                    with open(os.path.join(run_dir, "run.json")) as f:
                        self.assertEqual(json.load(f)["interrupted_by"], sig.name)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    if run_dir:
                        kill_leftovers(run_dir)


# ---------------------------------------------------------------------------- pinned builds (engine_build_check)
class PinnedBuild(unittest.TestCase):
    """engine.pin ties a config to tools/engines/pins.json; engine_build_check compares the build-info.json of
    scripts/engines/build_engine.py with the pin before the engine starts. The build directory here is SYNTHETIC (files
    with fake contents, no CUDA); the engine that starts is still the mock server."""
    PIN = "llama.cpp"
    FILES = ("bin/llama-server", "bin/llama-bench", "bin/libggml-cuda.so")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1-pin-")
        self.pin = run_r1.load_pins()[self.PIN]
        self.build = os.path.join(self.tmp, "build", self.PIN)
        files = {}
        for rel in self.FILES:
            path = os.path.join(self.build, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(("SYNTHETIC %s\n" % rel).encode())
            files[rel] = run_r1.sha256_file(path)
        self.info = {"engine": self.PIN, "repo": self.pin["repo"], "sha": self.pin["sha"], "deps": {}, "arch": 89,
                     "checked_out": self.pin["sha"], "status": "ok", "bins": {b: True for b in self.pin["bins"]},
                     "cubin_archs": {"bin/libggml-cuda.so": ["sm_89"]}, "files_sha256": files}
        self.run_dirs = []

    def tearDown(self):
        for d in self.run_dirs:
            kill_leftovers(d)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_info(self, info):
        with open(os.path.join(self.build, "build-info.json"), "w") as f:
            json.dump(info, f)

    def config(self, name, **engine):
        def edit(c):
            del c["engine"]["commit"]
            c["engine"].update(pin=self.PIN, build_info=os.path.join(self.build, "build-info.json"),
                               binary=os.path.join(self.build, "bin", "llama-server"))
            c["engine"].update(engine)
        return write_config(self.tmp, name, edit)

    def run_cfg(self, cfg, name):
        root = os.path.join(self.tmp, "runs-" + name)
        code, out = run_main(["--config", cfg, "--out", root, "--only", "engine_start,health_wait"])
        run_dir = only_run_dir(root)
        self.run_dirs.append(run_dir)
        return code, out, ends(run_dir), run_dir

    def assert_blocked(self, code, out, e, check):
        self.assertEqual(code, 1, out)
        bc = e["engine_build_check"]
        self.assertEqual(bc["status"], "failed", bc)
        if bc.get("checks_failed"):
            self.assertIn(check, bc["checks_failed"], bc)
        else:
            self.assertIn(check, bc["reason"], bc)
        self.assertEqual((e["engine_start"]["status"], e["engine_start"].get("blocked_by")),
                         ("skipped", "engine_build_check"), "the engine must not start on a mismatching build")
        self.assertNotIn("pid", e["engine_start"])
        self.assertEqual(e["health_wait"]["status"], "skipped")

    def test_matching_build_is_checked_and_the_engine_starts(self):
        self.write_info(self.info)
        code, out, e, run_dir = self.run_cfg(self.config("pinned-ok"), "ok")
        self.assertEqual(code, 0, out)
        self.assertEqual(e["engine_build_check"]["status"], "ok", e["engine_build_check"])
        self.assertEqual(e["engine_start"]["status"], "ok")
        self.assertFalse(alive(e["engine_start"]["pid"]))
        eng = os.path.join(run_dir, "artifacts", "engine")
        with open(os.path.join(eng, "build_check.json")) as f:
            bc = json.load(f)
        self.assertTrue(bc["ok"])
        self.assertEqual({c["check"] for c in bc["checks"]},
                         {"engine", "repo", "sha", "checked_out", "deps", "status", "arch", "cubins", "binaries",
                          "files_sha256", "binary"})
        self.assertTrue(all(c["ok"] for c in bc["checks"]))
        with open(os.path.join(eng, "build-info.json")) as f:
            self.assertEqual(json.load(f), self.info)
        with open(os.path.join(eng, "argv.json")) as f:
            argv = json.load(f)
        self.assertEqual((argv["commit"], argv["pin"]), (self.pin["sha"], self.PIN))
        self.assertIn(self.pin["sha"], argv["engine_desc"])
        recs = journal(run_dir)
        order = [r["step"] for r in recs if r["event"] == "end"]
        self.assertLess(order.index("engine_build_check"), order.index("engine_start"))

    def test_every_mismatch_fails_the_check_and_the_engine_is_not_started(self):
        cases = [("sha", lambda i: i.update(sha="0" * 40)),
                 ("checked_out", lambda i: i.update(checked_out="0" * 40)),
                 ("engine", lambda i: i.update(engine="ik_llama.cpp")),
                 ("repo", lambda i: i.update(repo="https://github.com/example/llama.cpp")),
                 ("deps", lambda i: i.update(deps={"ggml": {"sha": "1" * 40}})),
                 ("status", lambda i: i.update(status="failed at: build")),
                 ("arch", lambda i: i.update(arch=86)),
                 ("cubins", lambda i: i.update(cubin_archs={"bin/libggml-cuda.so": ["sm_86"]})),
                 ("cubins", lambda i: i.update(cubin_archs="cuobjdump not found",
                                               status="ok (cubins not checked: no cuobjdump)")),
                 ("binaries", lambda i: i["bins"].update({"bin/llama-bench": False})),
                 ("files_sha256", lambda i: i.pop("files_sha256"))]
        for n, (check, mutate) in enumerate(cases):
            with self.subTest(check=check, case=n):
                info = json.loads(json.dumps(self.info))
                mutate(info)
                self.write_info(info)
                self.assert_blocked(*self.run_cfg(self.config("pinned-bad-%d" % n), "bad-%d" % n)[:3], check)

    def test_binary_changed_after_the_build_is_caught(self):
        self.write_info(self.info)
        with open(os.path.join(self.build, "bin", "llama-server"), "ab") as f:
            f.write(b"patched\n")
        code, out, e, run_dir = self.run_cfg(self.config("pinned-tampered"), "tampered")
        self.assert_blocked(code, out, e, "files_sha256")
        with open(os.path.join(run_dir, "artifacts", "engine", "build_check.json")) as f:
            bad = [c for c in json.load(f)["checks"] if c["check"] == "files_sha256"][0]
        self.assertEqual([x["file"] for x in bad["got"]], ["bin/llama-server"])

    def test_binary_outside_the_build_and_missing_build_info_are_caught(self):
        self.write_info(self.info)
        self.assert_blocked(*self.run_cfg(self.config("pinned-outside", binary=PY), "outside")[:3], "binary")
        cfg = self.config("pinned-missing", build_info=os.path.join(self.tmp, "nowhere", "build-info.json"))
        code, out, e, _ = self.run_cfg(cfg, "missing")
        self.assert_blocked(code, out, e, "cannot read")

    def test_source_checkout_must_be_at_the_pin(self):
        self.write_info(self.info)
        src = os.path.join(self.tmp, "src")
        os.makedirs(src)
        git = ["git", "-C", src, "-c", "user.name=SYNTHETIC", "-c", "user.email=synthetic@example.invalid"]
        subprocess.run(["git", "init", "-q", src], check=True)
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "SYNTHETIC"], check=True)
        self.assert_blocked(*self.run_cfg(self.config("pinned-src-bad", source_dir=src), "src-bad")[:3], "source_head")
        fake = os.path.join(self.tmp, "fakebin")
        os.makedirs(fake)
        with open(os.path.join(fake, "git"), "w") as f:
            f.write("#!/bin/sh\necho %s\n" % self.pin["sha"])
        os.chmod(os.path.join(fake, "git"), 0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = fake + os.pathsep + old
        try:
            code, out, e, _ = self.run_cfg(self.config("pinned-src-ok", source_dir=src), "src-ok")
        finally:
            os.environ["PATH"] = old
        self.assertEqual(code, 0, out)
        self.assertEqual(e["engine_build_check"]["status"], "ok", e["engine_build_check"])

    def test_pin_config_errors_exit_2(self):
        sha = self.pin["sha"]
        bad = [("unknown-pin", dict(pin="llama.cpp-fork"), "engine.pin"),
               ("wrong-commit", dict(commit="0" * 40), "contradicts"),
               ("short-garbage", dict(commit="UNKNOWN"), "contradicts")]
        for name, eng, needle in bad:
            with self.subTest(case=name):
                code, out = run_main(["--config", self.config(name, **eng), "--out", self.tmp, "--plan"])
                self.assertEqual(code, 2, out)
                self.assertIn(needle, out)

        def no_pin(c):
            c["engine"]["build_info"] = "/x/build-info.json"
        code, out = run_main(["--config", write_config(self.tmp, "no-pin", no_pin), "--out", self.tmp, "--plan"])
        self.assertEqual(code, 2, out)
        self.assertIn("needs engine.pin", out)
        code, out = run_main(["--config", self.config("prefix", commit=sha[:9]), "--out", self.tmp, "--plan", "--json"])
        self.assertEqual(code, 0, out)
        self.assertIn(sha[:9], json.loads(out)["engine_desc"])

    def test_engine_templates_are_pinned_and_checked(self):
        pins = run_r1.load_pins()
        used = set()
        for path in TEMPLATES:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("mock"):
                continue
            with self.subTest(config=os.path.basename(path)):
                e = raw["engine"]
                self.assertIn(e.get("pin"), pins)
                self.assertNotIn("commit", e, "the commit of a pinned template comes from tools/engines/pins.json")
                self.assertTrue(e.get("build_info"))
                used.add(e["pin"])
                code, out = run_main(["--config", path, "--out", self.tmp, "--plan", "--json"])
                self.assertEqual(code, 0, out)
                p = json.loads(out)
                bc = [it for it in p["items"] if it["step"] == "engine_build_check"]
                self.assertEqual(len(bc), 1)
                self.assertFalse(bc[0]["skip"], bc[0])
                self.assertIn(pins[e["pin"]]["sha"], bc[0]["text"])
                self.assertIn(pins[e["pin"]]["sha"], p["engine_desc"])
        # every template has a pinned engine; every pinned engine has a template or says why not ("r1_template")
        noted = {k for k, p in pins.items() if p.get("r1_template")}
        self.assertEqual(used | noted, set(pins))
        self.assertFalse(used & noted, "an engine with an R1 template must not say it has none")


if __name__ == "__main__":
    unittest.main()
