"""The Bash guard hook: which git commands it lets through (docs/reassessment/ADR-012-push-policy.md)."""
import json
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts" / "hooks" / "guard-bash.sh"


def run(command):
    payload = json.dumps({"tool_input": {"command": command}})
    p = subprocess.run(["bash", str(HOOK)], input=payload, capture_output=True, text=True)
    return p.returncode


class GuardBash(unittest.TestCase):
    def test_allowed(self):
        for c in [
            "git status",
            "git log --oneline -n 5",
            "git push -u origin claude/relaxed-dijkstra-1d7i5o",
            "git push origin claude/research-x",
            "git push origin HEAD:claude/research-x",
            "cd /home/user/408032gb && git push -u origin claude/relaxed-dijkstra-1d7i5o",
            "echo git-push-is-not-a-command",
        ]:
            self.assertEqual(run(c), 0, c)

    def test_blocked(self):
        for c in [
            "git push",
            "git push origin",
            "git push origin main",
            "git push -u origin main",
            "git push origin HEAD:main",
            "git push origin claude/x main",
            "git push --force origin claude/x",
            "git push -f origin claude/x",
            "git push --force-with-lease origin claude/x",
            "git push origin +claude/x",
            "git push --mirror origin",
            "git push --delete origin claude/x",
            "git push --tags origin claude/x",
            "git push upstream claude/x",
            "git push https://github.com/other/repo claude/x",
            "git -C /tmp/other push origin claude/x",
            "git -c core.sshCommand=x push origin claude/x",
            "true && git push origin main",
            "git push -u origin claude/x; git push origin main",
            "git reset --hard HEAD~1",
            "git rebase -i HEAD~3",
            "git filter-branch --all",
        ]:
            self.assertEqual(run(c), 2, c)


if __name__ == "__main__":
    unittest.main()
