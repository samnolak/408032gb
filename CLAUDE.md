@AGENTS.md

# Claude Code specifics

- Team roles are project subagents in `.claude/agents/`; shared know-how is in `.claude/skills/`.
- Delegate by role: "Use the parity-qa subagent to ...". The lead-architect coordinates.
- Hooks in `.claude/settings.json` block `git push` and edits to upstream pins; sync is by bundle.
- Before starting work: read docs/PLAN.md (current gate) and the newest file in docs/evidence/.
