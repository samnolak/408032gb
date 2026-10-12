"""tools/srvcheck/n7_report.py on result files the harness writes against the stand-in llama-server."""
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "srvcheck"))
import hybrid_server_checks as hc  # noqa: E402
import n7_report  # noqa: E402

STUB = os.path.join(ROOT, "tests", "fixtures", "srvcheck", "stub_llama_server.py")


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        binary = os.path.join(cls.tmp.name, "llama-server")
        with open(binary, "w") as f:
            f.write(f"#!/bin/sh\nexec {sys.executable} {STUB} \"$@\"\n")
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)
        cls.paths = {}
        for name, fault in (("ok", ""), ("noisy", "noisy_warm")):
            os.environ["STUB_FAULT"] = fault
            try:
                res = hc.run_model(binary, "tiny.gguf", "tiny_mtp.gguf", ["--spec-type", "draft-mtp"], ["-c", "512"],
                                   tempfile.mkdtemp(dir=cls.tmp.name), gen=8, tol=1e-3, seed=7)
            finally:
                os.environ.pop("STUB_FAULT", None)
            cls.paths[name] = os.path.join(cls.tmp.name, name + ".json")
            with open(cls.paths[name], "w") as f:
                json.dump(res, f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_lines_print_stored_values(self):
        out = n7_report.lines(self.paths["ok"])
        self.assertTrue(out[1].startswith("verdict ") and '"all_pass": true' in out[1])
        ext = next(ln for ln in out if ln.startswith("prefix_extend"))
        # prompt 200 + answer 8 can be reused; the stand-in re-evaluates the 24-token tail and the answer's last token
        self.assertIn("reusable 208 warm prompt_n 25 cache_n None reused 207", ext)
        self.assertIn("max|d| 0.0", ext)

    def test_attr_compares_against_the_first_run(self):
        out = n7_report.attr([("ok", self.paths["ok"]), ("noisy", self.paths["noisy"])])
        ok, noisy = out[1].split(), out[2].split()
        self.assertEqual(ok[1], "0")                             # warm == cold on the healthy stand-in
        self.assertEqual(noisy[1], "0.01")                       # the injected 1e-2 shift of warm log-probs
        self.assertEqual(out[2].split("|")[1].split(","), [" 8", " 0", " 0"])   # cold answers identical: no diff

    def test_cli_usage(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(n7_report.main([]), 2)


if __name__ == "__main__":
    unittest.main()
