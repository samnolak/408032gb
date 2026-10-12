"""N26a: scripts/r1/safety_rules.py, the R1 safety rules of docs/reassessment/PHASE-CLOSE-2026-10-11.md section (v)
(accepted by the owner), and their wiring into scripts/r1/run_r1.py (refusal before any step; the weights gate of rule
8). Stdlib only, no GPU, no network; nothing here skips.

Every committed config passes; mutated copies of the committed configs fail with the expected rule id; run_r1 refuses
a violating config (exit 2, nothing created) and starts the engine of a non-mock config only after weights_verify
passed or on a config that marks its runs unverified. The engine in the run tests is tools/bench/mock_server.py and the
model the SYNTHETIC header-only GGUF of scripts/r1/make_mock_model.py: MOCK / SYNTHETIC TEST DATA, NOT EVIDENCE.
Run: python3 -m unittest tests.test_r1_safety_rules -v   (from the repo root)
"""
import contextlib
import copy
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "r1"))
import make_mock_model  # noqa: E402
import run_r1  # noqa: E402

S = run_r1.safety_rules                      # the module run_r1 itself uses
CONFIGS = os.path.join(ROOT, "scripts", "r1", "configs")
PATHS = sorted(glob.glob(os.path.join(CONFIGS, "*.json")))
TEMPLATE = os.path.join(ROOT, "templates", "glm-5.3-flash", "chat_template.jinja")
TEMPLATE_SHA = "5d47ac8e48510b5c6543db4000ef145201823bf46bae9cf51ac6a9553a16ef5a"   # sha256sum, N26a evidence
QWEN = "llamacpp-server.json"
GLM = "llamacpp-server-moecache.json"
IK = "ik-llamacpp-server.json"
STRATA = "strata-serve-qwen38fn.json"
MOCK = "mock.json"
PY = sys.executable


def load(name):
    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return json.load(f)


def rules(cfg, repo=ROOT):
    return sorted({v.rule for v in S.check(cfg, repo)})


def messages(cfg, repo=ROOT):
    return " | ".join("%s: %s" % v for v in S.check(cfg, repo))


def argv(cfg):
    return cfg["engine"]["argv"]


def drop_flag(cfg, flag):
    a = argv(cfg)
    i = a.index(flag)
    del a[i:i + 2]
    return cfg


def set_flag(cfg, flag, value):
    a = argv(cfg)
    a[a.index(flag) + 1] = value
    return cfg


def run_main(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = run_r1.main(args)
    return code, out.getvalue()


def journal_ends(run_dir):
    with open(os.path.join(run_dir, "steps.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    return {r["step"] + ("[%s]" % r["sub"] if r.get("sub") else ""): r for r in recs if r["event"] == "end"}


# ---------------------------------------------------------------------------- committed configs
class CommittedConfigs(unittest.TestCase):
    def test_every_committed_config_passes(self):
        self.assertEqual({os.path.basename(p) for p in PATHS} >= {QWEN, GLM, IK, STRATA, MOCK}, True)
        for p in PATHS:
            with self.subTest(config=os.path.basename(p)):
                with open(p, encoding="utf-8") as f:
                    self.assertEqual(S.check(json.load(f)), [])

    def test_resolved_by_run_r1_they_pass_too_and_the_template_sha_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            for p in PATHS:
                with self.subTest(config=os.path.basename(p)):
                    cell = run_r1.Cell(p, tmp, [])
                    self.assertEqual(cell.safety["violations"], [])
                    self.assertEqual(S.check(cell.cfg), [])
                    tpl = cell.safety["chat_templates"]
                    if os.path.basename(p) in (GLM, IK):
                        self.assertEqual(tpl, [{"path": TEMPLATE, "sha256": TEMPLATE_SHA,
                                                "expected_sha256": TEMPLATE_SHA}])
                    else:
                        self.assertEqual(tpl, [])
            self.assertEqual(os.listdir(tmp), [], "loading a config must not create anything")

    def test_template_sha256_is_the_file_on_disk(self):
        with open(TEMPLATE, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), TEMPLATE_SHA)
        for name in (GLM, IK):
            c = load(name)
            self.assertEqual(c["model"]["chat_template_sha256"], TEMPLATE_SHA, name)
            a = argv(c)
            self.assertEqual(a[a.index("--chat-template-file") + 1], "${REPO}/templates/glm-5.3-flash/chat_template.jinja")
            self.assertLess(a.index("--jinja"), a.index("--chat-template-file"), name)

    def test_every_llama_server_argv_gives_np_1(self):
        n = 0
        for p in PATHS:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            for a in S.server_argvs(c):
                n += 1
                self.assertEqual(S.rs._values(a, ["-np", "--parallel"]), ["1"], p)
        self.assertEqual(n, 3, "llama.cpp Qwen, llama.cpp GLM moe-cache, ik GLM")

    def test_only_the_mock_declares_a_synthetic_model(self):
        """Rule 10: the SYNTHETIC marker of tools/verify/pack.py, "mock": true or a synthetic_run criteria file."""
        declaring = []
        for p in PATHS:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            crit = ((c.get("steps") or {}).get("pack") or {}).get("criteria")
            synth_crit = False
            if crit:
                with open(os.path.join(ROOT, crit), encoding="utf-8") as f:
                    synth_crit = json.load(f).get("synthetic_run") is True
            if S.synthetic_marked(c["model"]["desc"]) or S.is_mock(c) or synth_crit:
                declaring.append(os.path.basename(p))
        self.assertEqual(declaring, [MOCK])

    def test_non_mock_configs_have_a_manifest_or_an_unverified_reason(self):
        for p in PATHS:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            if S.is_mock(c):
                continue
            with self.subTest(config=os.path.basename(p)):
                m = c["model"]
                self.assertNotEqual(bool(m.get("weights_manifest")), bool(m.get("weights_unverified_reason")))

    def test_operational_rules_are_written_into_the_configs(self):
        want = {QWEN: ("Rule 1", "Rule 3", "Rule 4", "Rule 6", "Rule 8", "Rule 11"),
                GLM: ("Rule 1", "Rule 2", "Rule 3", "Rule 6", "Rule 8"),
                IK: ("Rule 1", "Rule 2", "Rule 3", "Rule 6", "Rule 8", "Rule 9"),
                STRATA: ("Rule 3", "Rule 7", "Rule 8", "Rule 11"), MOCK: ("Rule 10",)}
        for name, rs in want.items():
            text = load(name)["_notes"]["safety_rules"]
            for r in rs:
                self.assertRegex(text, r"%s\b" % r, name)
        with open(os.path.join(ROOT, "scripts", "r1", "README.md"), encoding="utf-8") as f:
            readme = f.read()
        self.assertIn("safety_rules.py", readme)
        for i in range(1, 12):
            self.assertIn("| %d |" % i, readme, "rule %d missing from the README table" % i)

    def test_cli(self):
        cli = [PY, os.path.join(ROOT, "scripts", "r1", "safety_rules.py")]
        r = subprocess.run(cli + PATHS, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count(": ok ("), len(PATHS))
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as f:
                json.dump(drop_flag(load(QWEN), "-np"), f)
            r = subprocess.run(cli + [PATHS[0], bad], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("bad.json: VIOLATES R-NP", r.stdout)
            r = subprocess.run(cli + [os.path.join(tmp, "missing.json")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stdout)
        self.assertEqual(subprocess.run(cli, capture_output=True).returncode, 2)


# ---------------------------------------------------------------------------- mutated configs
MTP_OK = "${WEIGHTS_DIR}/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"


def with_args(name, *extra, env=None, **model):
    c = load(name)
    argv(c).extend(extra)
    if env:
        c["engine"]["env"].update(env)
    c["model"].update(model)
    return c


class Mutations(unittest.TestCase):
    def assertRules(self, cfg, want, repo=ROOT):
        got = rules(cfg, repo)
        self.assertEqual(got, want, "\n".join("%s: %s" % v for v in S.check(cfg, repo)))

    def test_np(self):
        self.assertRules(drop_flag(load(QWEN), "-np"), ["R-NP"])
        self.assertRules(drop_flag(load(GLM), "-np"), ["R-NP"])
        self.assertRules(drop_flag(load(IK), "--parallel"), ["R-NP"])
        self.assertRules(set_flag(load(QWEN), "-np", "x"), ["R-NP"])
        # a draft model without --spec-type (type inferred, rs_rollback_rule) and the default/explicit slot counts
        self.assertRules(set_flag(with_args(QWEN, "-md", MTP_OK, "--spec-type", "draft-mtp"), "-np", "4"), ["R-NP"])
        self.assertRules(set_flag(with_args(GLM, "-md", "/m/glm-mtp.gguf"), "-np", "2"), ["R-NP"])
        self.assertRules(with_args(GLM, "--spec-type", "draft-mtp", "-np", "2"), ["R-NP"])  # the last -np wins
        self.assertRules(set_flag(load(GLM), "-np", "2"), [])                                 # no speculation: -np free
        self.assertRules(set_flag(with_args(GLM, "--spec-type", "ngram-mod"), "-np", "2"), ["R-NP"])  # any speculation
        self.assertRules(set_flag(with_args(GLM, env={"LLAMA_ARG_SPEC_DRAFT_MODEL": "/m/d.gguf"}), "-np", "3"),
                         ["R-NP"])
        self.assertRules(set_flag(with_args(GLM, "--fim-qwen-7b-spec"), "-np", "2"), ["R-NP"])
        self.assertRules(set_flag(with_args(IK, "--spec-type", "mtp:n_max=1,p_min=0.0"), "--parallel", "2"),
                         ["R-NP"])                                                          # ik stage syntax
        self.assertRules(with_args(IK, "--spec-type", "mtp:n_max=1,p_min=0.0"), [])         # with --parallel 1
        self.assertRules(with_args(GLM, "--spec-type", "none"), [])
        # the recurrent-rollback part is rs_rollback_rule.check() itself (its message), not only the broader rule
        m = messages(with_args(GLM, "--spec-type", "draft-mtp", "-np", "2"))
        self.assertIn("n_rs_seq > 0", m)
        self.assertIn("(scripts/r1/rs_rollback_rule.py)", m)
        self.assertIn("any speculation", messages(set_flag(with_args(GLM, "--spec-type", "ngram-mod"), "-np", "2")))

    def test_qwen_mtp_files(self):
        self.assertRules(with_args(QWEN, "-md", MTP_OK, "--spec-type", "draft-mtp"), [])
        for q in ("BF16", "Q4_K_M"):
            self.assertRules(with_args(QWEN, "--model-draft", "/w/mtp-Qwen3.8-Flash-Next-%s.gguf" % q,
                                       "--spec-type", "draft-mtp"), [])
        shared = with_args(QWEN, "-md", "/w/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", "--spec-type", "draft-mtp")
        self.assertRules(shared, ["R-MTP-FILES"])
        self.assertIn("is a -shared- file", messages(shared))           # the N18 reason, not only the name pattern
        self.assertRules(with_args(QWEN, "-md", "/w/mtp-Qwen3.8-Flash-Next-Q2_K.gguf", "--spec-type", "draft-mtp"),
                         ["R-MTP-FILES"])
        self.assertRules(with_args(QWEN, "-md", MTP_OK), ["R-MTP-FILES"])                   # type not explicit
        self.assertRules(with_args(QWEN, "--spec-draft-model=" + MTP_OK, "--spec-type", "draft-eagle3"),
                         ["R-MTP-FILES"])
        self.assertRules(with_args(QWEN, "-hfd", "unsloth/x-GGUF", "--spec-type", "draft-mtp"), ["R-MTP-FILES"])
        self.assertRules(with_args(QWEN, "--mtp"), ["R-MTP-FILES"])
        self.assertRules(with_args(QWEN, "--spec-type", "draft-mtp",
                                   env={"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "unsloth/x-GGUF"}), ["R-MTP-FILES"])
        self.assertRules(with_args(QWEN, "--spec-type", "draft-mtp",
                                   env={"LLAMA_ARG_SPEC_DRAFT_MODEL": "/w/mtp-x-shared-BF16.gguf"}), ["R-MTP-FILES"])
        c = with_args(QWEN, "-md", "/w/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
        set_flag(c, "-np", "4")
        self.assertRules(c, ["R-MTP-FILES", "R-NP"])
        g = with_args(GLM, "-md", "/w/mtp-GLM-Q8_0.gguf", "--spec-type", "draft-mtp")
        self.assertRules(g, [])                                         # rule 4 is about the Qwen files

    def test_glm_template(self):
        self.assertRules(drop_flag(load(GLM), "--chat-template-file"), ["R-GLM-TEMPLATE"])
        self.assertRules(drop_flag(load(IK), "--chat-template-file"), ["R-GLM-TEMPLATE"])
        self.assertRules(set_flag(load(GLM), "--chat-template-file", "${REPO}/templates/glm-5.3-flash/other.jinja"),
                         ["R-GLM-TEMPLATE"])
        self.assertRules(set_flag(load(IK), "--chat-template-file", "/srv/llm/chat_template.jinja"),
                         ["R-GLM-TEMPLATE"])
        c = load(GLM)
        c["model"]["chat_template_sha256"] = "0" * 64
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        with tempfile.TemporaryDirectory() as tmp:                    # same bytes, other path: still refused
            copy_ = os.path.join(tmp, "chat_template.jinja")
            shutil.copyfile(TEMPLATE, copy_)
            c = set_flag(load(GLM), "--chat-template-file", copy_)
            self.assertRules(c, ["R-GLM-TEMPLATE"])
            self.assertIn("not ${REPO}/templates/glm-5.3-flash/chat_template.jinja", messages(c))
        c = load(IK)
        del c["model"]["chat_template_sha256"]
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        c = load(GLM)
        c["model"]["chat_template_sha256"] = TEMPLATE_SHA.upper()
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        c = drop_flag(load(GLM), "--chat-template-file")
        argv(c)[argv(c).index("--jinja"):argv(c).index("--jinja")] = [
            "--chat-template-file", "${REPO}/templates/glm-5.3-flash/chat_template.jinja"]
        self.assertRules(c, ["R-GLM-TEMPLATE"])                                            # before --jinja
        self.assertRules(with_args(GLM, "--chat-template", "chatml"), ["R-GLM-TEMPLATE"])
        c = load(QWEN)
        c["model"]["desc"] = c["model"]["desc"].replace("Qwen3.8-Flash-Next", "GLM-5.3-Flash")
        c["name"] = "llamacpp-glm"
        self.assertRules(c, ["R-GLM-TEMPLATE"])                                            # GLM by desc and name
        self.assertRules(with_args(QWEN, "--chat-template-file", TEMPLATE), ["R-GLM-TEMPLATE"])  # sha missing
        self.assertRules(with_args(QWEN, "--chat-template-file", TEMPLATE, chat_template_sha256=TEMPLATE_SHA), [])
        c = load(GLM)
        argv(c).remove("--jinja")
        self.assertRules(c, ["R-GLM-TEMPLATE"])       # jinja is on by default at the R1 pin (N26 audit F1)
        c = load(GLM)
        i = argv(c).index("--jinja")
        del argv(c)[i:i + 3]                          # --jinja and the template file both gone, sha kept
        c["model"].pop("chat_template_sha256", None)
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        c = load(IK)
        i = argv(c).index("--jinja")
        del argv(c)[i:i + 3]
        c["model"].pop("chat_template_sha256", None)
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        c = load(GLM)
        argv(c).append("--no-jinja")
        self.assertRules(c, ["R-GLM-TEMPLATE"])
        c = load(GLM)
        c["engine"].setdefault("env", {})["LLAMA_ARG_CHAT_TEMPLATE_FILE"] = "/tmp/other.jinja"
        self.assertRules(c, ["R-GLM-TEMPLATE"])

    def test_glm_template_changed_on_disk(self):
        with tempfile.TemporaryDirectory() as repo:
            dst = os.path.join(repo, "templates", "glm-5.3-flash", "chat_template.jinja")
            os.makedirs(os.path.dirname(dst))
            shutil.copyfile(TEMPLATE, dst)
            self.assertRules(load(GLM), [], repo)
            with open(dst, "ab") as f:
                f.write(b"{# changed #}")
            self.assertRules(load(GLM), ["R-GLM-TEMPLATE"], repo)
            self.assertRules(load(IK), ["R-GLM-TEMPLATE"], repo)
            os.remove(dst)
            self.assertRules(load(GLM), ["R-GLM-TEMPLATE"], repo)

    def test_mimo_speculation(self):
        def mimo(c):
            c["name"] = "llamacpp-mimo-v26f"
            c["model"]["desc"] = "MiMo-V2.6-Flash (test mutation)"
            return c
        self.assertRules(mimo(load(QWEN)), [])
        self.assertRules(mimo(with_args(QWEN, "-md", "/w/mimo-mtp.gguf", "--spec-type", "draft-mtp")), ["R-MIMO-MTP"])
        self.assertRules(mimo(with_args(QWEN, "--spec-type", "ngram-mod")), ["R-MIMO-MTP"])
        c = mimo(load(QWEN))
        c["mode"]["spec_k"] = 2
        self.assertRules(c, ["R-MIMO-MTP"])
        c = mimo(load(STRATA))                              # Strata: --spec ${SPEC_K} / --mtp in config_files, spec_k
        self.assertRules(c, ["R-MIMO-MTP"])
        self.assertIn("--spec 4 in engine.config_files[strata-ud-q4_k_xl.json].args", S.speculation(S.resolve(c, ROOT)))
        c["vars"]["SPEC_K"] = "0"
        c["engine"]["config_files"]["strata-ud-q4_k_xl.json"]["args"].remove("--mtp")
        self.assertRules(c, [])

    def test_weights(self):
        c = load(IK)
        del c["model"]["weights_unverified_reason"]
        self.assertRules(c, ["R-WEIGHTS"])
        c["model"]["weights_unverified_reason"] = "  "
        self.assertRules(c, ["R-WEIGHTS"])
        c["model"]["weights_unverified_reason"] = 1
        self.assertRules(c, ["R-WEIGHTS"])
        c = load(QWEN)
        c["model"]["weights_unverified_reason"] = "no time"
        self.assertRules(c, ["R-WEIGHTS"])                                                  # manifest AND reason
        c = load(QWEN)
        c["steps"]["weights_verify"] = {"enabled": False}
        self.assertRules(c, ["R-WEIGHTS"])
        c["steps"]["weights_verify"] = {"enabled": "off"}
        self.assertRules(c, ["R-WEIGHTS"])
        c = load(MOCK)
        c["model"]["weights_manifest"] = None
        self.assertRules(c, [])                                                             # the mock is exempt

    def test_synthetic(self):
        c = load(MOCK)
        c["model"]["desc"] = "mock model"
        self.assertRules(c, ["R-SYNTH"])
        c = load(MOCK)
        c["engine"]["argv"] = ["/opt/llama/bin/llama-server", "-m", "${MODEL}", "-np", "1"]
        self.assertRules(c, ["R-SYNTH"])                                                    # mock exemption borrowed
        with tempfile.TemporaryDirectory() as tmp:
            crit = os.path.join(tmp, "criteria.json")
            with open(crit, "w") as f:
                json.dump({"synthetic_run": True}, f)
            c = load(QWEN)
            c["steps"]["pack"] = {"criteria": crit}
            self.assertRules(c, ["R-SYNTH"])
            c["model"]["desc"] = "SYNTHETIC tiny model (test)"
            self.assertRules(c, [])
            with open(crit, "w") as f:
                json.dump({"synthetic_run": False}, f)
            c["model"]["desc"] = load(QWEN)["model"]["desc"]
            self.assertRules(c, [])
            c["steps"]["pack"] = {"criteria": os.path.join(tmp, "missing.json")}
            self.assertRules(c, ["R-SYNTH"])


# ---------------------------------------------------------------------------- run_r1 wiring
class RunR1Refuses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1-safety-")
        self.root = os.path.join(self.tmp, "runs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, cfg):
        path = os.path.join(self.tmp, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return path

    def test_violating_configs_are_refused_before_anything(self):
        c = load(IK)
        del c["model"]["weights_unverified_reason"]
        cases = [("np", drop_flag(load(QWEN), "-np"), "[R-NP]"),
                 ("tpl", drop_flag(load(GLM), "--chat-template-file"), "[R-GLM-TEMPLATE]"),
                 ("mtp", with_args(QWEN, "-md", "/w/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", "--spec-type",
                                   "draft-mtp"), "[R-MTP-FILES]"),
                 ("weights", c, "[R-WEIGHTS]")]
        for name, cfg, needle in cases:
            for extra in (["--plan"], []):
                with self.subTest(case=name, extra=extra):
                    code, out = run_main(["--config", self.write(name, cfg), "--out", self.root] + extra)
                    self.assertEqual(code, 2, out)
                    self.assertIn("R1 safety rules", out)
                    self.assertIn(needle, out)
                    self.assertFalse(os.path.exists(self.root), "a refused config must not create anything")

    def test_the_resolved_config_is_checked(self):
        """--set values are part of what is checked: a variable that turns speculation on with -np 2 is refused."""
        c = with_args(GLM, "--spec-type", "${SPEC}")
        set_flag(c, "-np", "${NP}")
        c["vars"].update(SPEC="none", NP="2")
        path = self.write("vars", c)
        code, out = run_main(["--config", path, "--out", self.root, "--plan"])
        self.assertEqual(code, 0, out)
        code, out = run_main(["--config", path, "--out", self.root, "--plan", "--set", "SPEC=draft-mtp"])
        self.assertEqual(code, 2, out)
        self.assertIn("[R-NP]", out)
        code, out = run_main(["--config", path, "--out", self.root, "--plan", "--set", "SPEC=draft-mtp",
                              "--set", "NP=1"])
        self.assertEqual(code, 0, out)


class WeightsGate(unittest.TestCase):
    """Rule 8 at run time: a non-mock config ("mock": false, still the mock server as its engine)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r1-wgate-")
        self.model = os.path.join(self.tmp, "model-SYNTHETIC")
        make_mock_model.write(self.model)
        self.run_dirs = []

    def tearDown(self):
        for d in self.run_dirs:
            for r in journal_ends(d).values():
                if r.get("pid"):
                    with contextlib.suppress(OSError):
                        os.killpg(r["pid"], 9)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, name, **model):
        c = load(MOCK)
        c["name"] = name
        c["mock"] = False
        c["model"]["desc"] = "mock model of the weights-gate test (not marked)"
        c["model"].update(model)
        path = os.path.join(self.tmp, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(c, f)
        return path

    def run_cfg(self, path, only="engine_start,health_wait"):
        root = os.path.join(self.tmp, "runs-" + os.path.basename(path))
        code, out = run_main(["--config", path, "--out", root, "--set", "MOCK_MODEL_DIR=" + self.model,
                              "--only", only])
        dirs = glob.glob(os.path.join(root, "*"))
        self.assertEqual(len(dirs), 1, out)
        self.run_dirs.append(dirs[0])
        return code, out, journal_ends(dirs[0]), dirs[0]

    def test_failed_verification_blocks_the_engine(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        code, out, e, _ = self.run_cfg(self.cfg("wgate-bad", weights_dest=empty))
        self.assertEqual(code, 1, out)
        self.assertEqual(e["weights_verify"]["status"], "failed", "--only engine_start must add weights_verify")
        st = e["engine_start"]
        self.assertEqual((st["status"], st.get("blocked_by")), ("skipped", "weights_verify"))
        self.assertIn("R1 safety rule 8", st["reason"])
        self.assertNotIn("pid", st)
        self.assertEqual(e["health_wait"]["status"], "skipped")

    def test_verified_weights_start_the_engine(self):
        code, out, e, run_dir = self.run_cfg(self.cfg("wgate-ok"))
        self.assertEqual(code, 0, out)
        self.assertEqual(e["weights_verify"]["status"], "ok")
        self.assertEqual(e["engine_start"]["status"], "ok")
        self.assertEqual(e["engine_start"]["weights"], "verified by weights_verify in this run")
        with open(os.path.join(run_dir, "artifacts", "orchestrator", "safety_rules.json")) as f:
            self.assertEqual(json.load(f)["violations"], [])

    def test_disabled_verification_is_refused_and_unselected_is_added(self):
        c = load(MOCK)
        c.update(mock=False, name="wgate-disabled")
        c["model"]["desc"] = "mock model (not marked)"
        c["steps"]["weights_verify"] = {"enabled": False}
        path = os.path.join(self.tmp, "disabled.json")
        with open(path, "w") as f:
            json.dump(c, f)
        code, out = run_main(["--config", path, "--out", os.path.join(self.tmp, "r"), "--plan"])
        self.assertEqual(code, 2, out)
        self.assertIn("[R-WEIGHTS]", out)
        code, out = run_main(["--config", self.cfg("wgate-plan"), "--out", os.path.join(self.tmp, "r"), "--plan",
                              "--json", "--only", "engine_start"])
        self.assertEqual(code, 0, out)
        self.assertIn("weights_verify", json.loads(out)["selected"])

    def test_unverified_reason_lets_the_engine_start_and_is_recorded(self):
        code, out, e, run_dir = self.run_cfg(self.cfg("wgate-unverified", weights_manifest=None,
                                                      weights_unverified_reason="TEST: no manifest for this model"))
        self.assertEqual(code, 0, out)
        self.assertEqual(e["weights_verify"]["status"], "skipped")
        self.assertIn("UNVERIFIED (TEST: no manifest for this model)", e["weights_verify"]["reason"])
        self.assertIn("V2", e["weights_verify"]["reason"])
        self.assertEqual(e["engine_start"]["status"], "ok")
        self.assertEqual(e["engine_start"]["weights"], "UNVERIFIED: TEST: no manifest for this model")
        with open(os.path.join(run_dir, "run.json")) as f:
            self.assertEqual(json.load(f)["safety_rules"]["weights_unverified_reason"],
                             "TEST: no manifest for this model")

    def test_the_mock_is_exempt(self):
        c = load(MOCK)
        c["name"] = "wgate-mock"
        path = os.path.join(self.tmp, "m.json")
        with open(path, "w") as f:
            json.dump(c, f)
        root = os.path.join(self.tmp, "runs-mock")
        code, out = run_main(["--config", path, "--out", root, "--set", "MOCK_MODEL_DIR=" + self.tmp,
                              "--only", "engine_start,health_wait"])
        d = glob.glob(os.path.join(root, "*"))[0]
        self.run_dirs.append(d)
        e = journal_ends(d)
        self.assertEqual(code, 0, out)
        self.assertIn("not selected", e["weights_verify"]["reason"])
        self.assertEqual(e["engine_start"]["weights"], "MOCK (the weights gate does not apply)")


if __name__ == "__main__":
    unittest.main()
