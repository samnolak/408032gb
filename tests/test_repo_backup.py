"""scripts/offline/repo_backup.py: a backup bundle counts only if it restores in an empty directory. Reproduces the
failure of 2026-10-10 (a bundle made in a shallow clone passes `git bundle verify` there but cannot be cloned) on a
small local repository, and checks that `create` refuses a shallow clone and that a full clone restores."""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "offline"))
import repo_backup as rb  # noqa: E402

ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def call(argv, cwd):
    out = io.StringIO()
    old = os.getcwd()
    os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = rb.main(argv)
    finally:
        os.chdir(old)
    return rc, out.getvalue()


class Backup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="backup-")
        self.up = os.path.join(self.tmp, "up")
        git("init", "-q", "-b", "main", self.up)
        for i in range(4):
            with open(os.path.join(self.up, "f.txt"), "w") as f:
                f.write("%d\n" % i)
            git("-C", self.up, "add", "f.txt")
            git("-C", self.up, "commit", "-q", "-m", "c%d" % i)
        git("-C", self.up, "checkout", "-q", "-b", "claude/work")
        with open(os.path.join(self.up, "g.txt"), "w") as f:
            f.write("work\n")
        git("-C", self.up, "add", "g.txt")
        git("-C", self.up, "commit", "-q", "-m", "work")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def clone(self, name, *extra):
        dst = os.path.join(self.tmp, name)
        git("clone", "-q", "--no-single-branch", *extra, "--branch", "claude/work", "file://" + self.up, dst)
        git("-C", dst, "branch", "-q", "main", "origin/main")
        return dst

    def test_full_clone_bundle_restores(self):
        repo = self.clone("full")
        out = os.path.join(self.tmp, "full.bundle")
        rc, text = call(["create", "--out", out], repo)
        self.assertEqual(rc, 0, text)
        self.assertIn("ok   git clone --mirror", text)
        self.assertIn("ok   git fsck --full", text)
        self.assertIn("ok   ref refs/heads/claude/work", text)
        self.assertIn("5 commits", text)                     # the whole history, not just the tip
        self.assertIn("ok   git clone --branch claude/work", text)
        self.assertIn("restore check: ok", text)

    def test_shallow_clone_is_refused(self):
        repo = self.clone("shallow", "--depth", "2")
        self.assertEqual(git("-C", repo, "rev-parse", "--is-shallow-repository"), "true")
        out = os.path.join(self.tmp, "shallow.bundle")
        rc, text = call(["create", "--out", out], repo)
        self.assertEqual(rc, 2, text)
        self.assertIn("git fetch --unshallow", text)
        self.assertFalse(os.path.exists(out))

    def test_shallow_bundle_passes_verify_but_fails_the_restore_check(self):
        # the 2026-10-10 failure: made with plain git in a shallow clone, `git bundle verify` there says ok
        repo = self.clone("shallow", "--depth", "2")
        out = os.path.join(self.tmp, "raw.bundle")
        git("-C", repo, "bundle", "create", out, "main", "claude/work")
        verify = subprocess.run(["git", "-C", repo, "bundle", "verify", out], capture_output=True, text=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("complete history", verify.stderr + verify.stdout)
        rc, text = call(["check", out], self.tmp)
        self.assertEqual(rc, 1, text)
        self.assertIn("FAIL git clone --mirror", text)
        self.assertIn("did not send all necessary objects", text)
        # after unshallowing the same clone, create works and restores
        git("-C", repo, "fetch", "-q", "--unshallow", "origin")
        rc, text = call(["create", "--out", os.path.join(self.tmp, "fixed.bundle")], repo)
        self.assertEqual(rc, 0, text)

    def test_check_compares_expected_commits(self):
        repo = self.clone("full")
        out = os.path.join(self.tmp, "x.bundle")
        self.assertEqual(call(["create", "--out", out], repo)[0], 0)
        rc, text = call(["check", out, "--expect", "claude/work=" + "0" * 40], self.tmp)
        self.assertEqual(rc, 1, text)
        self.assertIn("FAIL expected refs/heads/claude/work", text)
        tip = git("-C", repo, "rev-parse", "claude/work")
        rc, text = call(["check", out, "--expect", "claude/work=" + tip], self.tmp)
        self.assertEqual(rc, 0, text)
        rc, text = call(["create", "--out", out], repo)
        self.assertEqual(rc, 2, "an existing file must not be overwritten")


if __name__ == "__main__":
    unittest.main()
