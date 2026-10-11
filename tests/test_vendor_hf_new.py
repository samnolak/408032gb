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


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


class Pending(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.reqs = []
        for name, pinned in (("a", True), ("b", False), ("c", False)):
            os.makedirs(os.path.join(self.d.name, name))
            p = os.path.join(self.d.name, name, "REQUEST.json")
            write(p, json.dumps({"repo": f"x/{name}", "files": []}))
            if pinned:
                write(os.path.join(self.d.name, name, "PIN.json"), "{}")
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


class DataRequests(unittest.TestCase):
    """data/hf-requests/<name>/REQUEST.json (agents cannot write under third_party/, scripts/hooks/guard-upstream.sh)
    is vendored into third_party/hf/<name>/, where the copied request, the files and PIN.json land."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.d.name)
        self.req = os.path.join(self.root, "data", "hf-requests", "m", "REQUEST.json")
        os.makedirs(os.path.dirname(self.req))
        write(self.req, json.dumps({"repo": "x/m", "files": ["config.json"]}))
        self.dest = os.path.join(self.root, "third_party", "hf", "m")

    def tearDown(self):
        self.d.cleanup()

    def test_destination_is_third_party_hf(self):
        self.assertEqual(vendor_hf_new.dest_of(self.req), self.dest)
        other = os.path.join(self.root, "third_party", "hf", "n", "REQUEST.json")
        self.assertEqual(vendor_hf_new.dest_of(other), os.path.dirname(other))

    def test_request_is_copied_and_fetched_from_the_destination(self):
        with mock.patch.object(vendor_hf_new.vendor_hf_files, "one") as one:
            self.assertEqual(vendor_hf_new.main([self.req]), 0)
        staged = os.path.join(self.dest, "REQUEST.json")
        self.assertEqual([c.args[0] for c in one.call_args_list], [staged])
        self.assertEqual(read(staged), read(self.req))

    def test_pinned_destination_is_not_fetched_again(self):
        os.makedirs(self.dest)
        write(os.path.join(self.dest, "PIN.json"), "{}")
        with mock.patch.object(vendor_hf_new.vendor_hf_files, "one") as one, redirect_stdout(io.StringIO()):
            self.assertEqual(vendor_hf_new.main([self.req]), 0)
        one.assert_not_called()

    def test_default_globs_find_both_places(self):
        t = os.path.join(self.root, "third_party", "hf", "t")
        os.makedirs(t)
        write(os.path.join(t, "REQUEST.json"), json.dumps({"repo": "x/t", "files": []}))
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            with mock.patch.object(vendor_hf_new.vendor_hf_files, "one") as one:
                self.assertEqual(vendor_hf_new.main([]), 0)
        finally:
            os.chdir(cwd)
        got = sorted(os.path.relpath(os.path.join(self.root, c.args[0]), self.root) for c in one.call_args_list)
        self.assertEqual(got, [os.path.join("third_party", "hf", "m", "REQUEST.json"),
                               os.path.join("third_party", "hf", "t", "REQUEST.json")])


if __name__ == "__main__":
    unittest.main()
