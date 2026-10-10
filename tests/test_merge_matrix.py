"""tools/research/merge_matrix.py on synthetic matrix rows (SYNTHETIC: properties of the tool, not of any project)."""
import contextlib
import copy
import io
import json
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "research"))
import merge_matrix as mm  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40


def row(**over):
    """A fully valid, fully labelled row."""
    r = {
        "name": "Engine X",
        "url": "https://github.com/Owner/engine-x",
        "sha": SHA_A,
        "last_activity": "2026-10-10",
        "license": "MIT",
        "models_supported": "VERIFIED: GLM-5.3-Flash (src/model.cpp:1)",
        "hardware": "VERIFIED: CUDA sm_89",
        "offload_methods": "VERIFIED: VRAM cache + CPU experts",
        "scheduler": "VERIFIED: doorbell per layer",
        "kernels": "VERIFIED: AVX2 Q4_K",
        "multi_gpu": "VERIFIED: layer split",
        "long_context_262k": "REPORTED: 262K (README)",
        "overlap_gpu_cpu_ssd": "VERIFIED: GPU || CPU",
        "benchmarks": "REPORTED: 40 tok/s on 1x3090",
        "independent_confirmation": "NOT FOUND: searched issues",
        "limitations": "VERIFIED: single GPU",
        "key_code": ["src/a.cpp:f — does a"],
        "portability_to_us": "HYPOTHESIS: medium",
        "adaptation_cost": "HYPOTHESIS: weeks",
        "testable_without_target_hw": "HYPOTHESIS: partly",
        "potential_impact": "HYPOTHESIS: high",
        "confidence": "medium",
        "status": "active",
    }
    r.update(over)
    return r


def codes(issues, severity=None):
    return sorted({(i["code"], i["field"]) for i in issues if severity is None or i["severity"] == severity})


class Validation(unittest.TestCase):
    def test_clean_row_has_no_issues(self):
        self.assertEqual(mm.validate_row("R5.json", 0, row()), [])

    def test_missing_and_extra_keys(self):
        r = row()
        del r["scheduler"]
        r["notes"] = "x"
        got = codes(mm.validate_row("R5.json", 0, r), "error")
        self.assertIn(("E_MISSING_KEY", "scheduler"), got)
        self.assertIn(("E_EXTRA_KEY", "notes"), got)

    def test_unlabelled_claim_is_an_error_but_na_is_not(self):
        got = codes(mm.validate_row("R5.json", 0, row(kernels="AVX2 Q4_K", models_supported="n/a (library)")))
        self.assertIn(("E_UNLABELED", "kernels"), got)
        self.assertNotIn(("E_UNLABELED", "models_supported"), got)

    def test_absence_without_label_is_flagged_with_hint(self):
        issues = mm.validate_row("R5.json", 0, row(independent_confirmation="none found"))
        msg = [i["message"] for i in issues if i["field"] == "independent_confirmation"]
        self.assertTrue(msg and "NOT FOUND" in msg[0])

    def test_label_must_be_a_word(self):
        # "UNVERIFIED" or "REPORTEDLY" are not labels; "NOT-FOUND" is non-canonical
        got = codes(mm.validate_row("R5.json", 0, row(hardware="UNVERIFIEDish", scheduler="REPORTEDLY fast")))
        self.assertIn(("E_UNLABELED", "hardware"), got)
        self.assertIn(("E_UNLABELED", "scheduler"), got)
        got = codes(mm.validate_row("R5.json", 0, row(limitations="NOT-FOUND (code)")))
        self.assertIn(("W_NONCANONICAL_LABEL", "limitations"), got)
        self.assertIn(("E_UNLABELED", "limitations"), got)

    def test_url_and_sha_formats(self):
        bad = codes(mm.validate_row("R5.json", 0, row(url="github.com/x/y", sha="abc1234")), "error")
        self.assertIn(("E_URL", "url"), bad)
        self.assertIn(("E_SHA", "sha"), bad)
        for sha in ("UNKNOWN", "UNKNOWN (HF blocked)", "NOT FOUND", "NOT FOUND (no code)", SHA_B):
            self.assertNotIn(("E_SHA", "sha"), codes(mm.validate_row("R5.json", 0, row(sha=sha))), sha)
        for sha in ("NOT-FOUND (no code)", "n/a", "master " + SHA_A, SHA_A.upper()):
            self.assertIn(("E_SHA", "sha"), codes(mm.validate_row("R5.json", 0, row(sha=sha))), sha)
        self.assertNotIn(("E_URL", "url"),
                         codes(mm.validate_row("R5.json", 0, row(url="NOT FOUND", sha="NOT FOUND",
                                                                  status="not-found"))))

    def test_types_and_status(self):
        got = codes(mm.validate_row("R5.json", 0, row(key_code="src/a.cpp", benchmarks=3, status="UNKNOWN")))
        self.assertIn(("E_TYPE", "key_code"), got)
        self.assertIn(("E_TYPE", "benchmarks"), got)
        self.assertIn(("E_STATUS", "status"), got)
        got = codes(mm.validate_row("R5.json", 0, row(status="active (slow)")))
        self.assertIn(("W_STATUS_NOTE", "status"), got)
        self.assertEqual(codes(mm.validate_row("R5.json", 0, ["not", "a", "dict"])), [("E_NOT_OBJECT", None)])

    def test_assessment_fields_are_warnings(self):
        issues = mm.validate_row("R5.json", 0, row(potential_impact="high"))
        self.assertEqual(codes(issues), [("W_IMPLICIT_HYPOTHESIS", "potential_impact")])
        self.assertEqual(codes(issues, "error"), [])


class Keys(unittest.TestCase):
    def test_repo_key_normalisation(self):
        for url in ("https://github.com/Owner/Engine-X", "https://github.com/owner/engine-x.git",
                    "https://github.com/owner/engine-x/tree/dev", "https://www.github.com/owner/engine-x/"):
            self.assertEqual(mm.repo_key(url), "github.com/owner/engine-x", url)
        self.assertIsNone(mm.repo_key("https://arxiv.org/abs/2609.18110"))
        self.assertIsNone(mm.repo_key("NOT FOUND"))

    def test_project_key_by_name_and_alias(self):
        self.assertEqual(mm.project_key(row(name="HOBBIT (paper)", url="NOT FOUND"), {}), "name:hobbit")
        self.assertEqual(mm.project_key(row(name="HOBBIT", url="https://arxiv.org/pdf/2411.01433"), {}),
                         "name:hobbit")
        aliases = {"name:hobbit": "github.com/x/hobbit"}
        self.assertEqual(mm.project_key(row(name="HOBBIT", url="NOT FOUND"), aliases), "github.com/x/hobbit")


class MergeDir(unittest.TestCase):
    def write(self, files):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        mdir = os.path.join(d, "matrix")
        os.makedirs(mdir)
        for name, rows in files.items():
            with open(os.path.join(mdir, name), "w", encoding="utf-8") as fh:
                if isinstance(rows, str):
                    fh.write(rows)
                else:
                    json.dump(rows, fh)
        return d, mdir

    def project(self, merged, key):
        hits = [p for p in merged["projects"] if p["key"] == key]
        self.assertEqual(len(hits), 1, [p["key"] for p in merged["projects"]])
        return hits[0]

    def test_duplicates_merge_and_keep_every_source(self):
        _, mdir = self.write({
            "R5.json": [row(), row(name="Other", url="https://github.com/o/other")],
            "R10.json": [row(name="Engine X fork view", url="https://github.com/owner/engine-x.git")],
        })
        merged = mm.build(mdir, aliases={})
        p = self.project(merged, "github.com/owner/engine-x")
        self.assertEqual([(s["report"], s["index"]) for s in p["sources"]], [("R5", 0), ("R10", 0)])
        self.assertEqual(p["name"], "Engine X")
        self.assertEqual(p["names"], ["Engine X", "Engine X fork view"])
        self.assertEqual(merged["summary"]["projects"], 2)
        self.assertEqual(merged["summary"]["rows"], 3)
        # identical values collapse into one value with both sources
        self.assertEqual(p["fields"]["kernels"],
                         {"agreement": "single", "values": [{"value": row()["kernels"], "sources": ["R5#0", "R10#0"]}]})
        self.assertEqual(p["fields"]["url"]["agreement"], "multi")

    def test_conflicts_are_kept_not_resolved(self):
        _, mdir = self.write({
            "R5.json": [row(license="MIT", status="active", sha=SHA_A,
                            kernels="VERIFIED: AVX2 kernels (src/k.cpp)", scheduler="VERIFIED: doorbell")],
            "R8.json": [row(license="Apache-2.0", status="stale", sha=SHA_B,
                            kernels="NOT FOUND: no CPU kernels", scheduler="REPORTED: doorbell + DMA share")],
        })
        p = self.project(mm.build(mdir, aliases={}), "github.com/owner/engine-x")
        self.assertEqual(sorted(p["conflicts"]), ["kernels", "license", "sha", "status"])
        for f in ("license", "status", "sha", "kernels"):
            vals = [g["value"] for g in p["fields"][f]["values"]]
            self.assertEqual(len(vals), 2, f)
        self.assertEqual(p["fields"]["scheduler"]["agreement"], "multi")   # complementary, not contradictory

    def test_not_found_row_joined_by_alias_shows_conflict(self):
        _, mdir = self.write({
            "R5.json": [row(name="ProMoE", url="NOT FOUND", sha="NOT FOUND", status="not-found",
                            kernels="UNKNOWN", key_code=[])],
            "R8.json": [row(name="ProMoE (predictor)", url="https://github.com/p/promoe", sha=SHA_A, status="stale")],
        })
        merged = mm.build(mdir, aliases={"name:promoe": "github.com/p/promoe"})
        p = self.project(merged, "github.com/p/promoe")
        self.assertEqual(len(p["sources"]), 2)
        self.assertIn("url", p["conflicts"])
        self.assertIn("sha", p["conflicts"])
        self.assertIn("status", p["conflicts"])
        without = mm.build(mdir, aliases={})
        self.assertEqual(without["summary"]["projects"], 2)

    def test_sha_overlap_and_shared_absence_are_not_conflicts(self):
        _, mdir = self.write({
            "R6.json": [row(sha="NOT FOUND", url="NOT FOUND", name="Paper P", status="not-found")],
            "R8.json": [row(sha="NOT FOUND (no code)", url="https://arxiv.org/abs/1", name="Paper P (2024)",
                            status="not-found")],
        })
        p = self.project(mm.build(mdir, aliases={}), "name:paperp")
        self.assertEqual(p["conflicts"], [])

    def test_invalid_rows_and_bad_files_are_reported(self):
        bad = row()
        del bad["sha"]
        _, mdir = self.write({"R5.json": [row(), bad], "R6.json": "{not json", "R7.json": {"a": 1}})
        merged = mm.build(mdir, aliases={})
        self.assertEqual(merged["invalid_rows"], [{"file": "R5.json", "index": 1}])
        file_codes = sorted(i["code"] for i in merged["issues"] if i["index"] is None)
        self.assertEqual(file_codes, ["E_FILE_NOT_LIST", "E_JSON"])
        md = mm.render_md(merged, os.path.dirname(mdir))
        self.assertIn("`R5.json` #1", md)
        self.assertIn("E_MISSING_KEY: sha", md)

    def test_render_is_deterministic_and_order_independent(self):
        rows5 = [row(name=f"P{i}", url=f"https://github.com/o/p{i}", license=random.Random(i).choice(["MIT", "BSD"]))
                 for i in range(6)]
        rows9 = [row(name="P3 again", url="https://github.com/o/p3", kernels="REPORTED: other text")]
        _, d1 = self.write({"R5.json": rows5, "R9.json": rows9})
        shuffled = copy.deepcopy(rows5)
        random.Random(1).shuffle(shuffled)
        _, d2 = self.write({"R9.json": rows9, "R5.json": rows5})
        m1, m2 = mm.build(d1), mm.build(d2)
        self.assertEqual(mm.dump_json(m1), mm.dump_json(m2))
        self.assertEqual(mm.render_md(m1, os.path.dirname(d1)), mm.render_md(m2, os.path.dirname(d2)))
        self.assertEqual(mm.render_md(m1, os.path.dirname(d1)), mm.render_md(mm.build(d1), os.path.dirname(d1)))
        names = [p["name"] for p in m1["projects"]]
        self.assertEqual(names, sorted(names, key=str.casefold))
        # row order inside a report changes the row indices (sources), but not the set of projects
        _, d3 = self.write({"R5.json": shuffled, "R9.json": rows9})
        self.assertEqual([p["key"] for p in mm.build(d3)["projects"]], [p["key"] for p in m1["projects"]])

    def test_markdown_escaping(self):
        _, mdir = self.write({"R5.json": [row(limitations="VERIFIED: a | b <c>")]})
        md = mm.render_md(mm.build(mdir, aliases={}), os.path.dirname(mdir))
        self.assertIn("VERIFIED: a \\| b &lt;c&gt;", md)

    def test_cli_write_then_check(self):
        d, mdir = self.write({"R5.json": [row()], "R10.json": [row(name="Y", url="https://github.com/o/y")]})
        out_md = os.path.join(d, "MATRIX.md")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mm.main(["--matrix-dir", mdir]), 0)
            self.assertTrue(os.path.isfile(os.path.join(mdir, "merged.json")))
            self.assertTrue(os.path.isfile(out_md))
            self.assertEqual(mm.main(["--matrix-dir", mdir, "--check"]), 0)
            with open(out_md, "a", encoding="utf-8") as fh:
                fh.write("edit\n")
            self.assertEqual(mm.main(["--matrix-dir", mdir, "--check"]), 1)
            # merged.json in the matrix dir is not an input (glob R*.json)
            self.assertEqual(mm.build(mdir)["summary"]["rows"], 2)
        bad = row(kernels="no label")
        _, mdir2 = self.write({"R5.json": [bad]})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mm.main(["--matrix-dir", mdir2, "--strict"]), 1)
            self.assertEqual(mm.main(["--matrix-dir", mdir2]), 0)


class RealRows(unittest.TestCase):
    """Smoke test on the repository's own rows: every row is accounted for exactly once."""

    def test_every_real_row_lands_in_one_project(self):
        if not os.path.isdir(mm.MATRIX_DIR):
            self.skipTest("no matrix dir")
        merged = mm.build(mm.MATRIX_DIR)
        tags = [(s["file"], s["index"]) for p in merged["projects"] for s in p["sources"]]
        self.assertEqual(len(tags), len(set(tags)))
        expected = sum(i["rows"] for i in merged["inputs"])
        objects = expected - sum(1 for i in merged["issues"] if i["code"] == "E_NOT_OBJECT")
        self.assertEqual(len(tags), objects)


if __name__ == "__main__":
    unittest.main()
