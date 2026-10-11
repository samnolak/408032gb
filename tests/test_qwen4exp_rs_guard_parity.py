"""N20: the recurrent rollback guard (patches/llama.cpp/<pin>/0001-recurrent-rollback-guard.patch) on llama.cpp qwen4exp,
against ref/qwen4exp.py MultiSeqContext (guard=True with QWEN4EXP_RS_GUARD=1, else the pin's own behaviour).

The four runs of tools/tinygen/qwen4exp_rs_guard_plans.py go through the N19 machinery unchanged
(tests/test_qwen4exp_multiseq_parity.py: every graph row by row against (A) each sequence alone and (B) the cell model,
cross-sequence leakage, the recurrent state each sequence reads, seq_rm results and positions, divergences), with the
expectations of the plans module: a rollback onto a snapshot plane the last write did not reach is refused by the
patch and accepted (silently wrong) without it; a seq_cp hands the source's pending rollback to the copy.
GuardLog checks the patch's own refusal message in every run's log (the N19 runs included): with the patch exactly
plan["guard_log"] lines (default 0), without it none; a build that is not the patched one fails here even where the
refusals look alike (n_rs_seq bound, pending, shared cell).

Env as tests/test_qwen4exp_multiseq_parity.py (QWEN4EXP_MULTISEQ_DUMP, QWEN4EXP_TINY_GGUF, LLAMA_CPP_GGUF_PY,
PARITY_LLAMA_COMMIT, QWEN4EXP_RS_GUARD); driver tools/tinygen/run_parity_qwen4exp_rs_guard.sh. Without the dumps every
test skips.
"""
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "tinygen"))
import tests.test_qwen4exp_multiseq_parity as B  # noqa: E402

_spec = importlib.util.spec_from_file_location("qwen4exp_rs_guard_plans",
                                               ROOT / "tools" / "tinygen" / "qwen4exp_rs_guard_plans.py")
PLANS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PLANS)

MESSAGE = "refusing a rollback of"          # LLAMA_LOG_WARN of the patch's seq_rm refusal

for _run in PLANS.RUNS:
    globals()[f"Run_{_run}"] = B._class(_run, PLANS, __name__)
del _run


@unittest.skipUnless(B.DUMP and B.GGUF and B.GGUF_PY, B.SKIP)
class GuardLog(unittest.TestCase):
    def test_refusal_messages(self):
        """Every run's dump-tensors log: the guard's refusal message exactly plan["guard_log"] times with the patch
        (the rollbacks it refuses because their plane was not written), never without it."""
        runs = {**B.PLANS.RUNS, **PLANS.RUNS}
        seen = 0
        for run, plan in runs.items():
            log = pathlib.Path(B.DUMP, f"{run}.log")
            if not log.exists():
                self.fail(f"no log of run {run} under {B.DUMP}")
            n = sum(MESSAGE in line for line in log.read_text(errors="replace").splitlines())
            want = plan.get("guard_log", 0) if B.GUARD else 0
            with self.subTest(run=run):
                self.assertEqual(n, want, f"{run}: guard refusal messages")
            seen += 1
        self.assertEqual(seen, len(runs))


def tearDownModule():
    B.tearDownModule()


if __name__ == "__main__":
    unittest.main()
