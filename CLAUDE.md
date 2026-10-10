@AGENTS.md

# Claude Code specifics

- Team roles are project subagents in `.claude/agents/`; shared know-how is in `.claude/skills/`.
- Delegate by role: "Use the parity-qa subagent to ...". The lead-architect coordinates.
- Hooks in `.claude/settings.json` allow only a plain push of a `claude/*` branch to `origin` (after a secret scan),
  block force/delete/main/other remotes, merges and other GitHub writes outside `claude/*`, history rewrites and edits
  to upstream pins (AGENTS.md "Git and sync"). The inference VM still syncs by bundle.
- Work that must survive the container: commit to the session's `claude/*` branch and push it; open a Draft PR into
  `main` for review. Never merge.
- Before starting work: read docs/PLAN.md (current gate) and the newest file in docs/evidence/.
