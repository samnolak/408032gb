"""Every subagent and skill file must parse, or Claude Code skips it silently.

Regression test for the bug fixed in the '[evidence-auditor] fix' commit: an unquoted
description containing ': ' broke YAML parsing of .claude/agents/engine-integrator.md.
"""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FM = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _frontmatter(path):
    m = FM.match(path.read_text(encoding="utf-8"))
    if not m:
        return None
    try:
        import yaml  # optional on the VM
    except ImportError:
        yaml = None
    if yaml is not None:
        return yaml.safe_load(m.group(1))
    # Fallback: flat "key: value" parse; descriptions must be JSON-quoted strings.
    out = {}
    for line in m.group(1).splitlines():
        if line.startswith((" ", "-")) or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        out[k] = json.loads(v) if v.startswith('"') else v
    return out


class TeamConfig(unittest.TestCase):
    def test_agents_and_skills_parse(self):
        files = sorted((ROOT / ".claude").rglob("*.md"))
        self.assertGreaterEqual(len(files), 13)
        names = set()
        for f in files:
            fm = _frontmatter(f)
            self.assertIsInstance(fm, dict, f"{f}: no parseable frontmatter")
            self.assertIn("name", fm, f)
            self.assertIn("description", fm, f)
            self.assertNotIn(":", fm["name"], f)
            self.assertFalse(fm["name"].startswith("-"), f)
            self.assertNotIn(fm["name"], names, f"duplicate name {fm['name']}")
            names.add(fm["name"])

    def test_descriptions_are_quoted(self):
        for f in sorted((ROOT / ".claude").rglob("*.md")):
            for line in FM.match(f.read_text(encoding="utf-8")).group(1).splitlines():
                if line.startswith("description:"):
                    self.assertTrue(line.startswith('description: "'), f"{f}: quote the description")

    def test_agent_skills_exist(self):
        skills = {p.parent.name for p in (ROOT / ".claude/skills").glob("*/SKILL.md")}
        for f in sorted((ROOT / ".claude/agents").glob("*.md")):
            text = FM.match(f.read_text(encoding="utf-8")).group(1)
            for s in re.findall(r"^\s+- ([a-z0-9-]+)$", text, re.M):
                self.assertIn(s, skills, f"{f}: skill {s} missing")

    def test_routing_declared(self):
        for f in sorted((ROOT / ".claude/agents").glob("*.md")):
            self.assertRegex(f.read_text(encoding="utf-8"), r"Routing: (Sol|Terra|Luna)\.", f)

    def test_settings_json_valid(self):
        s = json.loads((ROOT / ".claude/settings.json").read_text())
        self.assertIn("PreToolUse", s["hooks"])


if __name__ == "__main__":
    unittest.main()
