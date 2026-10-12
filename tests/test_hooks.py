"""Git publication guards (docs/reassessment/ADR-012.1-git-publication-policy.md, AGENTS.md "Git and sync").

scripts/hooks/guard-bash.sh (Bash), scripts/hooks/guard-github.sh (GitHub MCP tools) and scripts/hooks/secret_scan.py.
The hook scripts are run exactly as Claude Code runs them: JSON on stdin, exit 2 = blocked. Repository conditions (origin
URL, push config, aliases, branch existence, secrets in outgoing commits) are tested in throw-away repositories.
Secret-like strings are assembled at run time so that this file itself never contains one.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOKS = ROOT / "scripts" / "hooks"
sys.path.insert(0, str(HOOKS))
import guard_bash  # noqa: E402
import guard_github  # noqa: E402
import secret_scan  # noqa: E402

P = "pu" + "sh"            # keeps literal shell commands with the push verb out of this file's own Bash invocations


def hook(script, payload, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(["bash", str(HOOKS / script)], input=json.dumps(payload), capture_output=True, text=True, env=e)
    return p.returncode, p.stderr


def sh(repo, *args, env=None):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    e.update(env or {})
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=e).stdout


class Repo:
    """A repository shaped like the project: origin = samnolak/408032gb, main on origin, a local claude/x branch."""

    def __init__(self, url="https://github.com/samnolak/408032gb"):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(os.path.realpath(self.tmp.name)) / "repo"
        self.path.mkdir()
        sh(self.path, "init", "-q", "-b", "main")
        sh(self.path, "remote", "add", "origin", url)
        self.commit("README.md", "hello\n")
        sh(self.path, "update-ref", "refs/remotes/origin/main", "HEAD")
        sh(self.path, "checkout", "-q", "-b", "claude/x")

    def commit(self, name, text):
        (self.path / name).write_text(text)
        sh(self.path, "add", name)
        sh(self.path, "commit", "-q", "-m", "add " + name)

    def verdict(self, cmd, cwd=None):
        old = os.environ.get("CLAUDE_PROJECT_DIR")
        os.environ["CLAUDE_PROJECT_DIR"] = str(self.path)
        try:
            return guard_bash.verdict(cmd, str(cwd or self.path))
        finally:
            if old is None:
                del os.environ["CLAUDE_PROJECT_DIR"]
            else:
                os.environ["CLAUDE_PROJECT_DIR"] = old

    def close(self):
        self.tmp.cleanup()


def fake(kind):
    return {
        "aws": "AK" + "IA" + "QWERTYUIOPASDFGH",
        "github": "gh" + "p_" + "a1B2" * 9,
        "anthropic": "sk-" + "ant-" + "api03-" + "x" * 30,
        "openai": "sk-" + "proj-" + "Ab1" * 12,
        "hf": "h" + "f_" + "Qw3" * 11,
        "key": "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
        "url": "https://" + "user" + ":" + "s3cretpass" + "@example.com/x",
    }[kind]


class BashSyntax(unittest.TestCase):
    """Command classification without repository checks."""

    def v(self, cmd):
        return guard_bash.verdict(cmd, str(ROOT), run_checks=False)

    def test_allowed(self):
        for c in [
            "git status",
            "git log --oneline -n 5",
            "git log --format=%s | grep -i " + P,
            "git commit -m 'git " + P + " policy for claude/* branches'",
            "grep -rn 'git " + P + "' AGENTS.md",
            "git " + P + " -u origin claude/relaxed-dijkstra-1d7i5o",
            "git " + P + " origin claude/research-x",
            "git " + P + " --set-upstream origin claude/a/b",
            "git " + P + " origin HEAD:claude/research-x",
            "git " + P + " origin HEAD:refs/heads/claude/research-x",
            "cd /home/user/408032gb && git " + P + " -u origin claude/relaxed-dijkstra-1d7i5o",
            "git " + P + " -u origin claude/relaxed-dijkstra-1d7i5o 2>&1 | tail -15",
            "git " + P + " origin claude/x > /tmp/push.log 2> /tmp/push.err",
            "git " + P + " origin claude/x &>/dev/null",
            "git config --get remote.origin.url",
            "git remote -v",
            "gh pr view 1",
            "gh api repos/samnolak/408032gb/actions/runs",
            "curl -s https://api.github.com/repos/samnolak/408032gb",
        ]:
            self.assertIsNone(self.v(c), c)

    def test_blocked(self):
        for c in [
            "git " + P,
            "git " + P + " origin",
            "git " + P + " origin main",
            "git " + P + " origin main 2>&1 | tail -5",
            "git " + P + " origin claude/x > out main",
            "git " + P + " -u origin main",
            "git " + P + " origin HEAD:main",
            "git " + P + " origin claude/x:main",
            "git " + P + " origin claude/x:refs/heads/main",
            "git " + P + " origin claude/x main",
            "git " + P + " origin :claude/x",
            "git " + P + " --force origin claude/x",
            "git " + P + " -f origin claude/x",
            "git " + P + " --force-with-lease origin claude/x",
            "git " + P + " origin +claude/x",
            "git " + P + " --mirror origin",
            "git " + P + " --all origin",
            "git " + P + " --delete origin claude/x",
            "git " + P + " -d origin claude/x",
            "git " + P + " --tags origin claude/x",
            "git " + P + " --no-verify origin claude/x",
            "git " + P + " --dry-run origin main",
            "git " + P + " upstream claude/x",
            "git " + P + " https://github.com/other/repo claude/x",
            "git -C /tmp/other " + P + " origin claude/x",
            "git -c core.sshCommand=x " + P + " origin claude/x",
            "git -c remote.origin.push=refs/heads/claude/x:refs/heads/main status",
            "true && git " + P + " origin main",
            "git " + P + " -u origin claude/x; git " + P + " origin main",
            "bash -c 'git " + P + " origin main'",
            "sh -c \"git " + P + " -u origin claude/x\"",
            "env git " + P + " origin claude/x",
            "GIT_CONFIG_COUNT=1 git " + P + " origin claude/x",
            "timeout 60 git " + P + " origin claude/x",
            "echo git " + P,
            "git remote set-url origin https://github.com/other/repo",
            "git remote add fork https://github.com/other/repo",
            "git config remote.origin.push refs/heads/claude/x:refs/heads/main",
            "git config --global url.https://evil/.pushInsteadOf https://github.com/",
            "git config alias.p " + P,
            "git reset --hard HEAD~1",
            "git rebase -i HEAD~3",
            "git rebase --interactive HEAD~3",
            "git filter-branch --all",
            "git filter-repo --path x",
            "gh pr merge 1",
            "gh repo delete samnolak/408032gb",
            "gh api -X DELETE repos/samnolak/408032gb/git/refs/heads/main",
            "gh api repos/samnolak/408032gb/pulls/1/merge -X PUT",
            "gh api repos/samnolak/408032gb/git/refs -f ref=refs/heads/main",
            "curl -X DELETE https://api.github.com/repos/samnolak/408032gb/git/refs/heads/main",
            "curl -d '{}' https://api.github.com/repos/samnolak/408032gb/merges",
        ]:
            self.assertIsNotNone(self.v(c), c)


class BashRepoConditions(unittest.TestCase):
    """The checks that look at the repository the push runs in."""

    def setUp(self):
        self.r = Repo()
        self.r.commit("notes.md", "plain work\n")
        self.push = "git " + P + " -u origin claude/x"

    def tearDown(self):
        self.r.close()

    def test_clean_branch_is_allowed(self):
        self.assertIsNone(self.r.verdict(self.push))
        self.assertIsNone(self.r.verdict("git " + P + " origin HEAD:claude/x"))

    def test_secret_in_outgoing_commit_blocks(self):
        for kind in ("aws", "github", "anthropic", "openai", "hf", "key", "url"):
            with self.subTest(kind=kind):
                r = Repo()
                try:
                    r.commit("cfg.txt", "token = %s\n" % fake(kind))
                    why = r.verdict(self.push)
                    self.assertIsNotNone(why)
                    self.assertIn("secret scan", why)
                    self.assertNotIn(fake(kind), why)          # the value is never echoed
                finally:
                    r.close()

    def test_secret_already_on_origin_is_not_rescanned(self):
        r = Repo()
        try:
            r.commit("old.txt", fake("aws") + "\n")
            sh(r.path, "update-ref", "refs/remotes/origin/claude/x", "HEAD")
            r.commit("new.txt", "clean\n")
            self.assertIsNone(r.verdict(self.push))
        finally:
            r.close()

    def test_credential_file_name_blocks(self):
        self.r.commit(".env", "A=1\n")
        self.assertIn("secret scan", self.r.verdict(self.push))

    def test_wrong_origin_blocks(self):
        r = Repo(url="https://github.com/someone/else")
        try:
            self.assertIn("origin must be", r.verdict(self.push))
        finally:
            r.close()

    def test_push_refspec_config_blocks(self):
        sh(self.r.path, "config", "remote.origin.push", "refs/heads/claude/x:refs/heads/main")
        self.assertIn("remote.origin.push", self.r.verdict(self.push))

    def test_insteadof_blocks(self):
        sh(self.r.path, "config", "url.https://evil.example/.pushInsteadOf", "https://github.com/")
        self.assertIn("origin must be", self.r.verdict(self.push))

    def test_alias_blocks(self):
        sh(self.r.path, "config", "alias.pp", P + " origin main")
        self.assertIn("alias", self.r.verdict("git pp"))

    def test_tag_named_like_branch_blocks(self):
        sh(self.r.path, "tag", "claude/t")
        self.assertIn("not a local branch", self.r.verdict("git " + P + " origin claude/t"))

    def test_other_directory_blocks(self):
        other = Repo()
        try:
            self.assertIn("project repository", self.r.verdict("cd %s && %s" % (other.path, self.push)))
            self.assertIn("project repository", self.r.verdict(self.push, cwd=other.path))
        finally:
            other.close()


class BashHookScript(unittest.TestCase):
    """The shell entry point with the JSON Claude Code sends."""

    def test_exit_codes_and_message(self):
        r = Repo()
        try:
            env = {"CLAUDE_PROJECT_DIR": str(r.path)}
            ok = {"tool_input": {"command": "git " + P + " -u origin claude/x"}, "cwd": str(r.path)}
            bad = {"tool_input": {"command": "git " + P + " origin main"}, "cwd": str(r.path)}
            self.assertEqual(hook("guard-bash.sh", ok, env)[0], 0)
            rc, err = hook("guard-bash.sh", bad, env)
            self.assertEqual(rc, 2)
            self.assertIn("Blocked", err)
            self.assertEqual(hook("guard-bash.sh", {"tool_input": {"command": "ls"}}, env)[0], 0)
        finally:
            r.close()

    def test_guard_that_cannot_run_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            broken = pathlib.Path(d) / "guard-bash.sh"
            broken.write_text((HOOKS / "guard-bash.sh").read_text())      # guard_bash.py is missing next to it
            p = subprocess.run(["bash", str(broken)], input="{}", capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)


class GitHubTools(unittest.TestCase):
    def v(self, tool, **args):
        return guard_github.verdict("mcp__github__" + tool, args)

    def test_allowed(self):
        own = {"owner": "samnolak", "repo": "408032gb"}
        self.assertIsNone(self.v("create_pull_request", head="claude/x", base="main", draft=True, **own))
        self.assertIsNone(self.v("push_files", branch="claude/x", **own))
        self.assertIsNone(self.v("create_branch", branch="claude/y", **own))
        self.assertIsNone(self.v("update_pull_request", pullNumber=1, draft=True, **own))
        self.assertIsNone(self.v("get_file_contents", owner="other", repo="x"))
        self.assertIsNone(self.v("list_pull_requests", **own))

    def test_blocked(self):
        own = {"owner": "samnolak", "repo": "408032gb"}
        for tool, args in [
            ("merge_pull_request", own),
            ("enable_pr_auto_merge", own),
            ("delete_file", dict(own, branch="claude/x")),
            ("create_repository", {"name": "x"}),
            ("fork_repository", own),
            ("push_files", dict(own, branch="main")),
            ("create_or_update_file", dict(own, branch="feature")),
            ("create_branch", dict(own, branch="main2")),
            ("push_files", {"owner": "other", "repo": "x", "branch": "claude/x"}),
            ("create_pull_request", dict(own, head="feature", base="main")),
            ("create_pull_request", dict(own, head="claude/x", base="release")),
            ("update_pull_request", dict(own, base="release")),
        ]:
            self.assertIsNotNone(self.v(tool, **args), (tool, args))

    def test_hook_script(self):
        bad = {"tool_name": "mcp__github__merge_pull_request", "tool_input": {"owner": "samnolak", "repo": "408032gb"}}
        ok = {"tool_name": "mcp__github__get_me", "tool_input": {}}
        self.assertEqual(hook("guard-github.sh", bad)[0], 2)
        self.assertEqual(hook("guard-github.sh", ok)[0], 0)


class SecretScan(unittest.TestCase):
    def test_patterns(self):
        for kind in ("aws", "github", "anthropic", "openai", "hf", "key", "url"):
            self.assertTrue(secret_scan.scan_line("x = " + fake(kind)), kind)

    def test_no_false_positives_on_repo_shaped_text(self):
        for line in [
            "fe607938eef4d4da2c0838ae465bbd1dd2860c53fd81f4b4be893187748f114d  tools/gateway/gateway.py",
            "task-scheduler and mask-" + "a" * 40,
            "http://local_proxy@127.0.0.1:8080/git/samnolak/408032gb",
            "AKIA is the prefix of an AWS access key id",
            "export HF_HOME=/srv/llm/hf",
            "-----BEGIN CERTIFICATE-----",
        ]:
            self.assertEqual(secret_scan.scan_line(line), [], line)

    def test_bad_file_names(self):
        self.assertEqual(secret_scan.bad_file_names([".env", "a/id_rsa", "k.pem", ".env.example", "env.md", "x/key.py"]),
                         [".env", "a/id_rsa", "k.pem"])

    def test_cli_on_outgoing_commits(self):
        r = Repo()
        try:
            self.assertEqual(secret_scan.main(["--repo", str(r.path)]), 0)
            r.commit("c.txt", fake("github") + "\n")
            self.assertEqual(secret_scan.main(["--repo", str(r.path)]), 1)
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
