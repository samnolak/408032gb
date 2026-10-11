"""N20: scripts/r1/rs_rollback_rule.py, the rule that keeps unpatched llama.cpp llama-server away from the recurrent
rollback hazards of docs/evidence/2026-10-11-n20-recurrent-rollback-guard.md (stdlib only)."""
import glob
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rs_rollback_rule", ROOT / "scripts" / "r1" / "rs_rollback_rule.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


class Rule(unittest.TestCase):
    def test_no_rollback_snapshots_no_constraint(self):
        self.assertEqual(R.check(["-m", "x.gguf"]), [])
        self.assertEqual(R.check(["--spec-type", "ngram-mod,ngram-simple", "-np", "8"]), [])
        self.assertEqual(R.check(["--spec-type", "draft-simple", "--parallel", "4"]), [])

    def test_snapshot_types_need_one_slot(self):
        for t in sorted(R.RS_TYPES):
            with self.subTest(t=t):
                self.assertEqual(R.check(["--spec-type", t, "-np", "1"]), [])
                self.assertEqual(len(R.check(["--spec-type", t, "-np", "2"])), 1)
                self.assertEqual(len(R.check(["--spec-type", f"ngram-mod,{t}", "--parallel=3"])), 1)

    def test_the_default_slot_count_is_auto_four(self):
        self.assertEqual(R.n_slots([]), 4)
        self.assertEqual(R.n_slots(["-np", "-1"]), 4)
        self.assertIn("not 4 slot(s)", R.check(["--spec-type", "draft-mtp"])[0])

    def test_environment_when_the_flag_is_absent(self):
        env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp", "LLAMA_ARG_N_PARALLEL": "1"}
        self.assertEqual(R.check([], env), [])
        self.assertEqual(len(R.check(["-np", "2"], env)), 1, "the flag wins over the environment")
        self.assertEqual(R.check(["--spec-type", "ngram-mod"], env), [])

    def test_r1_llamacpp_configs_follow_the_rule(self):
        paths = sorted(glob.glob(str(ROOT / "scripts" / "r1" / "configs" / "*.json")))
        self.assertTrue(paths)
        self.assertEqual(R.main(paths), 0)


if __name__ == "__main__":
    unittest.main()
