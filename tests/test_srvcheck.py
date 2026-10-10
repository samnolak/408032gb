"""tools/srvcheck/hybrid_server_checks.py: comparison logic, and the whole harness against a stand-in llama-server.

The stand-in (tests/fixtures/srvcheck/stub_llama_server.py) is started as a process with the same command line as the
real binary; each fault it can inject must fail exactly the check meant to catch it, and the healthy stand-in must
pass everything. The run against the real llama-server is tests/test_srvcheck_llama.py (needs a build).
"""
import contextlib
import io
import os
import stat
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "srvcheck"))
import hybrid_server_checks as hc  # noqa: E402

STUB = os.path.join(ROOT, "tests", "fixtures", "srvcheck", "stub_llama_server.py")


def run(tokens, top):
    return {"tokens": tokens, "top": top}


class Compare(unittest.TestCase):
    def test_identical_passes(self):
        a = run([5, 6], [{"5": -0.1, "7": -2.0}, {"6": -0.2}])
        self.assertTrue(hc.compare(a, run([5, 6], [{"5": -0.1, "7": -2.0}, {"6": -0.2}]), 1e-3)["pass"])

    def test_token_flip_fails_and_reports_margin(self):
        a = run([5, 6], [{"5": -0.1, "7": -2.0}, {"6": -0.2, "9": -0.3}])
        c = hc.compare(a, run([5, 9], [{"5": -0.1, "7": -2.0}, {"6": -0.2, "9": -0.3}]), 1e-3)
        self.assertFalse(c["pass"])
        self.assertEqual(c["first_diff"], 1)
        self.assertAlmostEqual(c["top1_margin_at_first_diff"], 0.1)

    def test_logprob_beyond_tol_fails_with_same_tokens(self):
        c = hc.compare(run([5], [{"5": -0.1}]), run([5], [{"5": -0.1 + 2e-3}]), 1e-3)
        self.assertTrue(c["same_tokens"])
        self.assertFalse(c["pass"])

    def test_nothing_to_compare_is_not_a_pass(self):
        c = hc.compare(run([5], []), run([5], []), 1e-3)
        self.assertIsNone(c["max_abs_dlogprob"])
        self.assertFalse(c["pass"])

    def test_null_logprob_is_an_infinite_difference(self):
        # llama-server sends null for a NaN log-prob (seen after restoring a corrupted slot file)
        c = hc.compare(run([5], [{"5": None, "7": -2.0}]), run([5], [{"5": -0.1, "7": -2.0}]), 1e-3)
        self.assertEqual(c["max_abs_dlogprob"], float("inf"))
        self.assertFalse(c["pass"])
        c = hc.compare(run([5, 6], [{"5": None, "6": -1.0}, {"6": -0.2}]), run([5, 9], [{"5": -0.1}, {"9": -0.3}]), 1e-3)
        self.assertEqual(c["first_diff"], 1)

    def test_step_count_mismatch_is_not_a_pass(self):
        self.assertFalse(hc.compare(run([5], [{"5": -0.1}]), run([5], [{"5": -0.1}, {"5": -0.1}]), 1e-3)["pass"])


class Harness(unittest.TestCase):
    """End to end on the stand-in: model files are only names, the stand-in ignores them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = os.path.join(cls.tmp.name, "llama-server")
        with open(cls.binary, "w") as f:                     # a launcher with the binary's name
            f.write(f"#!/bin/sh\nexec {sys.executable} {STUB} \"$@\"\n")
        os.chmod(cls.binary, os.stat(cls.binary).st_mode | stat.S_IEXEC)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def harness(self, fault=""):
        os.environ["STUB_FAULT"] = fault
        try:
            work = tempfile.mkdtemp(dir=self.tmp.name)
            return hc.run_model(self.binary, "tiny.gguf", "tiny_mtp.gguf", ["--spec-type", "draft-mtp"],
                                ["-c", "512"], work, gen=12, tol=1e-3, seed=7)
        finally:
            os.environ.pop("STUB_FAULT", None)

    def failed(self, res):
        return sorted(k for k, v in res["verdict"].items() if not v and k != "all_pass")

    def test_healthy_server_passes_and_reports_reuse(self):
        res = self.harness()
        self.assertTrue(res["verdict"]["all_pass"], res["verdict"])
        ext = res["prefix_extend"]
        self.assertEqual(ext["reused_tokens"], ext["reusable_tokens"] - 1)       # prompt + answer minus its last token
        self.assertEqual(res["prefix_diverge"]["reused_tokens"], 0)              # checkpoint is past the divergence
        self.assertEqual(res["slot_persist"]["after_restart"]["prompt_n"], 4)     # restored up to the checkpoint

    def test_stale_reuse_is_caught(self):
        self.assertEqual(self.failed(self.harness("stale_reuse")),
                         ["prefix_extend_correct", "slot_persist_after_restart", "slot_persist_same_process"])

    def test_noisy_warm_logprobs_are_caught(self):
        self.assertIn("prefix_extend_correct", self.failed(self.harness("noisy_warm")))

    def test_restore_noop_is_caught(self):
        self.assertEqual(self.failed(self.harness("restore_noop")),
                         ["slot_persist_after_restart", "slot_persist_same_process"])

    def test_restore_wrong_state_is_caught(self):
        self.assertEqual(self.failed(self.harness("restore_wrong")),
                         ["slot_persist_after_restart", "slot_persist_same_process"])

    def test_mtp_divergence_is_caught(self):
        res = self.harness("mtp_differs")
        self.assertEqual(self.failed(res), ["mtp_equals_plain_decode"])
        self.assertEqual(res["mtp"]["cmp"]["first_diff"], 4)

    def test_mtp_crash_is_caught(self):
        res = self.harness("mtp_crash")
        self.assertEqual(self.failed(res), ["mtp_equals_plain_decode"])
        self.assertIn("error", res["mtp"])
        self.assertEqual(res["mtp"]["on_exit"], 139)

    def test_mtp_that_never_drafts_is_not_a_pass(self):
        self.assertEqual(self.failed(self.harness("mtp_nodraft")), ["mtp_equals_plain_decode"])

    def test_stale_state_after_kv_shift_is_caught(self):
        self.assertEqual(self.failed(self.harness("shift_stale")), ["cache_reuse_correct"])

    def test_slot_crosstalk_is_caught(self):
        res = self.harness("crosstalk")
        self.assertEqual(self.failed(res), ["isolation_two_slots"])
        self.assertEqual(res["isolation"]["max_busy_slots"], 2)

    def test_serialized_server_is_no_isolation_evidence(self):
        # one request at a time: answers are right, but nothing was decoded together; client-side overlap still > 0
        res = self.harness("serial")
        self.assertEqual(self.failed(res), ["isolation_two_slots"])
        self.assertTrue(all(c["pass"] for c in res["isolation"]["cmp"]))
        self.assertEqual(res["isolation"]["max_busy_slots"], 1)
        self.assertGreater(res["isolation"]["overlap_s"], 0)

    def test_nondeterministic_server_is_caught(self):
        self.assertIn("determinism", self.failed(self.harness("nondeterministic")))

    def test_cli_writes_json_and_fails_on_fault(self):
        out = os.path.join(self.tmp.name, "r.json")
        os.environ["STUB_FAULT"] = "restore_noop"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = hc.main(["--server", self.binary, "--model", "m.gguf", "--out", out, "--gen", "6"])
        finally:
            os.environ.pop("STUB_FAULT", None)
        self.assertEqual(rc, 1)
        self.assertTrue(os.path.getsize(out) > 0)


if __name__ == "__main__":
    unittest.main()
