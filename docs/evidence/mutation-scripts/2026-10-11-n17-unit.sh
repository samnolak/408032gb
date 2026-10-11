#!/usr/bin/env bash
# N17: the unit discovery, the CI checker and hook tests, and a YAML parse of the changed workflows, on the worktree HEAD.
set -uo pipefail
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=/home/user/wt-n17
cd $WT
echo "HEAD $(git rev-parse HEAD)"
echo '$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests'
LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > $S/n17/unit-discover.log 2>&1
echo "rc=$?"; tail -n 4 $S/n17/unit-discover.log
echo '$ python3 -m unittest tests.test_check_unittest tests.test_hooks'
python3 -m unittest tests.test_check_unittest tests.test_hooks > $S/n17/unit-check.log 2>&1
echo "rc=$?"; tail -n 4 $S/n17/unit-check.log
echo '$ python3 -c "import yaml; ..." (YAML parse of the changed workflows)'
python3 - <<'PY'
import yaml
for wf in (".github/workflows/ci.yml", ".github/workflows/parity-r1-pin.yml"):
    d = yaml.safe_load(open(wf))
    steps = [(j, s.get("name")) for j, job in d["jobs"].items() for s in job["steps"] if "N17" in str(s.get("name", ""))]
    print(wf, "parsed; jobs", list(d["jobs"]), "; N17 steps", steps)
PY
echo "status: $(git status --short | wc -l) changed or untracked files"
