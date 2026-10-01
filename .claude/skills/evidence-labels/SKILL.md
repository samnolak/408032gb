---
name: evidence-labels
description: "How to label claims CONFIRMED / PROVISIONAL / UNKNOWN and how to report work as done with literal command output. Use for every report, doc edit, commit message and status update in this repo."
---

# Evidence labels

Every number or factual claim gets exactly one label.

| label | meaning | must include |
|---|---|---|
| CONFIRMED | checked against a primary source | `path:line@commit`, a URL, or a command whose literal output is saved |
| PROVISIONAL | derived, estimated, or from a secondary source | how it was derived |
| UNKNOWN | not known | what would confirm it |

Examples:
- `n_routed_experts = 288` CONFIRMED (HF zai-org/GLM-5.3-Flash config.json)
- `GLM-5.3-Flash IQ3_XXS experts = 111.1 GiB` PROVISIONAL (tools/fitplan.py, uniform bpw, no per-tensor overrides)
- `RAM channels` UNKNOWN (confirm with `dmidecode -t memory`)

## Reporting "done"

Paste, literally and unedited:

```
$ git log --oneline -n 5
$ git status --short
$ <every test command you ran>
```

If you did not run something, write `NOT RUN: <reason>`. Never type a commit hash or a test
result from memory. A shorter honest report beats a complete invented one.

## Line references

Before citing `file:line`, print the line (`sed -n '<n>p' <file>`) and check it says what
you claim. Line numbers drift; cite the pinned commit.
