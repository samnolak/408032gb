---
name: gate-review
description: Checklist the lead-architect uses to accept or reject a gate in docs/PLAN.md. Use when deciding whether a gate is complete.
---

# Gate review

A gate moves to ACCEPTED only when every item below is true. Otherwise list what is missing.

1. Every acceptance criterion of the gate in docs/PLAN.md has an evidence file in
   docs/evidence/ with literal command output.
2. evidence-auditor returned no FAIL on the reports for this gate.
3. All tests pass at HEAD: `python3 -m unittest discover -s tests` plus the C++/CUDA parity
   tests named in the gate; output pasted in the acceptance note.
4. Upstream pins unchanged (AGENTS.md), or a new ADR explains the bump.
5. Open risks for the next gate are written down.

Record the decision in docs/PLAN.md: status, date, evidence links, the commit hash copied
from `git log` output (never typed).
