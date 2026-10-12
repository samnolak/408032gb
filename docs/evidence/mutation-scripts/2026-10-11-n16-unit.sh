#!/usr/bin/env bash
# N16: the repository's unit tests and the CI checker tests on the worktree HEAD (outputs pasted into the evidence)
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
cd /home/user/wt-n16
echo "HEAD $(git rev-parse HEAD)"
echo '$ LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests'
LLAMA_CPP_GGUF_PY=$S/llama-ec7630a/gguf-py python3 -m unittest discover -s tests > $S/n16/unit-discover.log 2>&1; echo "rc=$?"
tail -4 $S/n16/unit-discover.log
echo '$ python3 -m unittest tests.test_check_unittest tests.test_hooks'
python3 -m unittest tests.test_check_unittest tests.test_hooks > $S/n16/unit-check.log 2>&1; echo "rc=$?"
tail -4 $S/n16/unit-check.log
echo '$ python3 -c "import yaml; ..." (YAML parse of the changed workflows)'
python3 -c "
import yaml
for f in ('.github/workflows/ci.yml', '.github/workflows/parity-r1-pin.yml'):
    d = yaml.safe_load(open(f))
    steps = {j: [s.get('name', s.get('uses')) for s in d['jobs'][j]['steps']] for j in d['jobs']}
    n16 = [(j, n) for j, ss in steps.items() for n in ss if n and n.startswith('N16')]
    print(f, 'parsed; jobs', list(d['jobs']), '; N16 steps', n16)
"
echo "status: $(git status --short | wc -l) changed or untracked files"
