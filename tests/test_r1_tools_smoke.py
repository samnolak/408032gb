"""tools/srvcheck/r1_tools_smoke.py without a server: the CPU config it derives from scripts/r1/configs/llamacpp-server.json.

The real-server part runs in .github/workflows/server-cpu.yml (tools/srvcheck/run_n11.sh); here only what keeps the
derived config honest: run_r1 accepts it, the engine argv stays the template's, and what differs is what the docstring
of derive_r1_config says (no build-info, no weights manifest, a SYNTHETIC model, shorter lengths)."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "srvcheck"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "r1"))
import r1_tools_smoke as smoke  # noqa: E402
import run_r1  # noqa: E402


def plan(cfg_path, root, *sets):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = run_r1.main(["--config", cfg_path, "--out", root, "--plan", "--json"] + [x for s in sets for x in ("--set", s)])
    return code, (json.loads(out.getvalue()) if code == 0 else None)


class Derive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.tmp.name, "SYNTHETIC-tiny.gguf")
        self.prompt = os.path.join(self.tmp.name, "prompt.txt")
        for p in (self.model, self.prompt):
            with open(p, "w") as f:
                f.write("x")
        with open(smoke.TEMPLATE) as f:
            self.template = json.load(f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_lengths_plan_with_the_template_engine(self):
        sets = ("LLAMA_BIN=/opt/llama/bin", "REFERENCE_URL=http://127.0.0.1:1", "CORPUS=/tmp/c/*.txt", "PORT=8080")
        _, tpl = plan(smoke.TEMPLATE, self.tmp.name, *sets)
        for lengths in ("smoke", "evidence"):
            cfg = smoke.derive_r1_config(self.model, self.prompt, lengths)
            path = os.path.join(self.tmp.name, "%s.json" % lengths)
            with open(path, "w") as f:
                json.dump(cfg, f)
            code, p = plan(path, self.tmp.name, *sets)
            self.assertEqual(code, 0, lengths)
            items = {(i["step"], i["sub"]): i for i in p["items"]}
            self.assertEqual(items[("engine_start", None)]["argv"][3:],
                             [a for i in tpl["items"] if i["step"] == "engine_start" for a in i["argv"]][3:],
                             "the engine flags must stay the template's (only the -m file differs)")
            self.assertEqual(cfg["engine"]["argv"], self.template["engine"]["argv"])
            self.assertIn("no engine.build_info", items[("engine_build_check", None)]["skip"])
            self.assertIn("V2", items[("weights_verify", None)]["skip"])
            self.assertIn("SYNTHETIC", p["model_desc"])
            self.assertIsNone(items[("logprob_compare", None)]["skip"])
            self.assertIn("mkdir", json.dumps(items[("logprob_compare", None)]["pre"]))
            inputs = [int(a[a.index("--inputs") + 1]) for (s, _), i in items.items() if s == "matrix"
                      for a in [i["argv"]]]
            self.assertTrue(inputs and max(inputs) <= 16384, inputs)

    def test_cli(self):
        out = os.path.join(self.tmp.name, "c.json")
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "srvcheck", "r1_tools_smoke.py"),
                             "derive-config", "--model", self.model, "--prompt-file", self.prompt, "--out", out],
                            capture_output=True).returncode
        self.assertEqual(rc, 0)
        with open(out) as f:
            self.assertEqual(json.load(f), json.loads(json.dumps(smoke.derive_r1_config(self.model, self.prompt))))
        h = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "srvcheck", "r1_tools_smoke.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(h.returncode, 0)
        self.assertIn("--server", h.stdout)


if __name__ == "__main__":
    unittest.main()
