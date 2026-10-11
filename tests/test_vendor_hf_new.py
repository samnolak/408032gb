"""scripts/vendor_hf_new.py: only requests without a PIN.json are vendored; the fetch is vendor_hf_files.one()."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import vendor_hf_new  # noqa: E402


class Pending(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.reqs = []
        for name, pinned in (("a", True), ("b", False), ("c", False)):
            os.makedirs(os.path.join(self.d.name, name))
            p = os.path.join(self.d.name, name, "REQUEST.json")
            json.dump({"repo": f"x/{name}", "files": []}, open(p, "w"))
            if pinned:
                open(os.path.join(self.d.name, name, "PIN.json"), "w").write("{}")
            self.reqs.append(p)

    def tearDown(self):
        self.d.cleanup()

    def test_only_unpinned_requests_are_fetched(self):
        with mock.patch.object(vendor_hf_new.vendor_hf_files, "one") as one:
            self.assertEqual(vendor_hf_new.main(self.reqs), 0)
        self.assertEqual([c.args[0] for c in one.call_args_list], self.reqs[1:])

    def test_nothing_pending_fetches_nothing(self):
        with mock.patch.object(vendor_hf_new.vendor_hf_files, "one") as one, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(vendor_hf_new.main(self.reqs[:1]), 0)
        one.assert_not_called()
        self.assertIn("nothing pending", out.getvalue())

    def test_a_fetch_error_fails_with_an_annotation(self):
        with mock.patch.object(vendor_hf_new.vendor_hf_files, "one", side_effect=RuntimeError("blob mismatch")), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(vendor_hf_new.main(self.reqs[1:2]), 1)
        self.assertIn("::error::RuntimeError: blob mismatch", out.getvalue())


if __name__ == "__main__":
    unittest.main()
