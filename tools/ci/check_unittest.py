#!/usr/bin/env python3
"""Fail a CI parity step whose tests did not really run (gap 11 (a) of docs/reassessment/CORRECTNESS.md).

Every dump-parity test module skips cleanly when its environment variables or its dumps are missing
(`unittest.skipUnless(DUMP and GGUF and GGUF_PY, ...)`, `raise unittest.SkipTest(f"no run ... under ...")`), and
`python3 -m unittest` then exits 0: a driver whose env plumbing broke would leave its GitHub step green with nothing
checked. This checker runs the step's command (or reads a saved log of it), finds the one summary of the verbose
unittest run it contains and requires, for the step and the commit the dumps come from, exactly what
tools/ci/parity_expectations.json says:
  * the run is OK: no failures, errors, expected failures or unexpected successes;
  * `Ran N tests` with the expected N (a class skipped in setUpClass is not counted in N, a class skipped by its
    decorator is counted, so a missing dump changes N or the skips, usually both);
  * the skips are exactly the expected ones: the summary's `skipped=K` equals the number of expected skips, every
    `... skipped '<reason>'` line of the log is attributed (K lines), and each matches a distinct expected entry by
    test id and by reason (re.fullmatch). A legitimate pin-dependent skip (the glm5-next MTP class at ec7630a, whose
    graph has no glm5-next MTP) is listed for that commit only.
The commit is the full sha of --checkout (git rev-parse HEAD) or --commit. Before the command runs, the checkout it
runs on must be at that commit: a driver's first argument when it is a directory (tools/tinygen/run_parity*.sh
<checkout> <build>), else, for an inline unittest command of a llama.cpp step, the checkout holding LLAMA_CPP_GGUF_PY
(the pinned gguf-py the tests read); a mismatch is an error and nothing runs. The drivers export PARITY_LLAMA_COMMIT /
PARITY_IK_COMMIT only to their own unittest process, so the checker does not see that; if the step's pin variable is
set in the checker's own environment, it must agree too. A commit without expectations fails: a new pin needs one run
whose counts are reviewed and committed.

Usage (CI wraps the step's command; its output is streamed unchanged):
  python3 tools/ci/check_unittest.py --step n9-glm5next-decode --checkout ../llama.cpp -- \\
      tools/tinygen/run_parity_glm5next_decode.sh ../llama.cpp ../build-tools
  python3 tools/ci/check_unittest.py --step n9-glm5next-decode --commit <40-hex sha> --log saved.log
Exit status: 0 as expected; 1 a mismatch; 2 a usage or expectations-file error; the command's own status if it
failed (the parity verdict is then the command's, the checker only reports).
Python 3.11+ (the `name (module.Class.name)` test descriptions of TextTestResult); stdlib only.
"""
import argparse
import ast
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPECT = ROOT / "tools" / "ci" / "parity_expectations.json"
PIN_ENV = {"llama.cpp": "PARITY_LLAMA_COMMIT", "ik_llama.cpp": "PARITY_IK_COMMIT"}
SHA = re.compile(r"[0-9a-f]{40}")

RAN = re.compile(r"^Ran (\d+) tests? in [0-9.]+s$")
STATUS = re.compile(r"^(OK|FAILED|NO TESTS RAN)(?: \((.*)\))?$")
# `<description> ... skipped '<reason>'` (the reason is repr()'d by TextTestResult.addSkip); the description may be on
# the line before when the test has a docstring, and is absent when the test printed between start and skip
SKIP = re.compile(r"^(?:(?P<pre>.*) \.\.\. )?skipped (?P<reason>'.*'|\".*\")\s*$")
# `name (qualname)` plus the parameters of a subtest: test_x (m.C.test_x), setUpClass (m.C), test_x (m.C.test_x) (i=1)
DESC = re.compile(r"(?P<name>\w+) \((?P<qual>[\w.]+)\)(?: (?P<sub>\(.*\)))?")


class CheckError(Exception):
    """The run does not match its expectations (exit 1); .problems lists every difference."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


class ConfigError(Exception):
    """Bad arguments or expectations file (exit 2)."""


def test_id(text):
    """Normalized id of a unittest description, or None: `m.C.test_x`, `m.C.setUpClass`, `m.C.test_x (i=1)`."""
    m = None
    for m in DESC.finditer(text):
        pass
    if m is None:
        return None
    name, qual = m["name"], m["qual"]
    tid = qual if qual.endswith("." + name) or qual == name else f"{qual}.{name}"
    return f"{tid} {m['sub']}" if m["sub"] else tid


def parse(text):
    """The unittest summary of a verbose log: dict(ran, status, counts, skips=[(id or None, reason)]).

    Raises CheckError unless the log holds exactly one `Ran N tests` line followed by its status line."""
    lines = text.splitlines()
    summaries = []
    for i, line in enumerate(lines):
        m = RAN.match(line.strip())
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        s = STATUS.match(lines[j].strip()) if j < len(lines) else None
        if s:
            summaries.append((int(m[1]), s[1], s[2]))
    if len(summaries) != 1:
        raise CheckError(f"expected exactly one unittest summary ('Ran N tests' + status line), found {len(summaries)}")
    ran, status, infos = summaries[0]
    counts = {}
    for part in (infos.split(", ") if infos else []):
        k, _, v = part.partition("=")
        if not v.isdigit():
            raise CheckError(f"cannot read the unittest status '{status} ({infos})'")
        counts[k] = int(v)
    skips = []
    for i, line in enumerate(lines):
        m = SKIP.match(line.strip())
        if not m:
            continue
        try:
            reason = ast.literal_eval(m["reason"])
        except (ValueError, SyntaxError):
            reason = m["reason"][1:-1]
        tid = test_id(m["pre"]) if m["pre"] else None
        if tid is None and m["pre"] is not None and i > 0:      # docstring: the description is on the line before
            tid = test_id(lines[i - 1])
        skips.append((tid, reason))
    return dict(ran=ran, status=status, counts=counts, skips=skips)


def load_expectations(path=EXPECT):
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    steps = data.get("steps")
    if not isinstance(steps, dict) or not steps:
        raise ConfigError(f"{path}: no 'steps'")
    for name, st in steps.items():
        if st.get("pin") not in PIN_ENV:
            raise ConfigError(f"{path}: step {name}: 'pin' must be one of {sorted(PIN_ENV)}")
        if not isinstance(st.get("modules"), list) or not st["modules"]:
            raise ConfigError(f"{path}: step {name}: 'modules' must list the unittest modules the step runs")
        exp = st.get("expect")
        if not isinstance(exp, dict) or not exp:
            raise ConfigError(f"{path}: step {name}: no 'expect'")
        for sha, e in exp.items():
            if not SHA.fullmatch(sha):
                raise ConfigError(f"{path}: step {name}: '{sha}' is not a full 40-hex commit")
            if not isinstance(e.get("ran"), int) or e["ran"] < 1:
                raise ConfigError(f"{path}: step {name} @ {sha[:7]}: 'ran' must be a positive integer")
            sk = e.get("skipped")
            if not isinstance(sk, list):
                raise ConfigError(f"{path}: step {name} @ {sha[:7]}: 'skipped' must be a list")
            for s in sk:
                if not (isinstance(s, dict) and isinstance(s.get("test"), str) and isinstance(s.get("reason"), str)):
                    raise ConfigError(f"{path}: step {name} @ {sha[:7]}: a skip needs 'test' and 'reason'")
                try:
                    re.compile(s["reason"])
                except re.error as err:
                    raise ConfigError(f"{path}: step {name} @ {sha[:7]}: bad reason regex: {err}") from err
            if not isinstance(e.get("source"), str) or not e["source"]:
                raise ConfigError(f"{path}: step {name} @ {sha[:7]}: 'source' (where the numbers come from) is required")
    return data


def check(result, exp, where):
    """Compare a parse() result with one expectations entry; raise CheckError listing every difference."""
    problems = []
    if result["status"] != "OK":
        problems.append(f"status {result['status']} {result['counts']}, expected OK")
    other = {k: v for k, v in result["counts"].items() if k != "skipped"}
    if other:
        problems.append(f"unexpected outcomes {other}")
    if result["ran"] != exp["ran"]:
        problems.append(f"Ran {result['ran']} tests, expected {exp['ran']}")
    want = exp["skipped"]
    got_k = result["counts"].get("skipped", 0)
    if got_k != len(want):
        problems.append(f"skipped={got_k}, expected skipped={len(want)}")
    if len(result["skips"]) != got_k:
        problems.append(f"{len(result['skips'])} skip lines in the log for skipped={got_k} (unattributed skips)")
    free = list(range(len(want)))
    for tid, reason in result["skips"]:
        hit = next((k for k in free if tid == want[k]["test"]
                    and re.fullmatch(want[k]["reason"], reason, re.S)), None)
        if hit is None:
            problems.append(f"skip not allowed here: {tid or '<no test id>'}: {reason!r}")
        else:
            free.remove(hit)
    for k in free:
        problems.append(f"expected skip did not happen: {want[k]['test']}: /{want[k]['reason']}/")
    if problems:
        raise CheckError([f"{where}: {p}" for p in problems])
    return f"{where}: Ran {result['ran']}, skipped {got_k} as expected" + \
        (" (" + ", ".join(t for t, _ in result["skips"]) + ")" if result["skips"] else "")


def git_head(checkout):
    try:
        out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise ConfigError(f"cannot read the commit of {checkout}: {e}") from e
    return out


def resolve_commit(step, commit=None, checkout=None, env=None):
    """The full commit the step's dumps come from: --commit, else the checkout's HEAD, else the pin variable.

    The pin variable is read from the checker's own environment only (the drivers export theirs to their unittest
    process); in CI it is unset, and the binding of the command to the commit is check_command_commit()."""
    env = os.environ if env is None else env
    var = PIN_ENV[step["pin"]]
    if commit and checkout:
        raise ConfigError("give --commit or --checkout, not both")
    sha = commit or (git_head(checkout) if checkout else env.get(var))
    if not sha:
        raise ConfigError(f"no commit: give --checkout <{step['pin']} checkout>, --commit or set {var}")
    if not SHA.fullmatch(sha):
        raise ConfigError(f"'{sha}' is not a full 40-hex commit")
    if env.get(var) and env[var] != sha:
        raise ConfigError(f"{var}={env[var]} disagrees with the commit {sha}")
    return sha


def command_target(cmd):
    """The directory a wrapped driver runs on: its first argument when that is an existing directory, after an optional
    `env VAR=value ...` prefix (tools/tinygen/run_parity*.sh <checkout> <build>); None for other commands."""
    args = list(cmd)
    if args and os.path.basename(args[0]) == "env":
        args = args[1:]
        while args and re.match(r"\w+=", args[0]):
            args = args[1:]
    if len(args) >= 2 and os.path.isdir(args[1]):
        return args[1]
    return None


def check_command_commit(step, sha, cmd, env=None):
    """The checkout the wrapped command runs on must be at the step's commit (--checkout / --commit): the driver's
    checkout argument, or, for an inline unittest command of a llama.cpp step, the checkout holding LLAMA_CPP_GGUF_PY.
    Returns a description of what was compared, or None if the command names no checkout (the binding then rests on
    the workflow text, tests/test_check_unittest.py). Raises ConfigError on a mismatch."""
    env = os.environ if env is None else env
    target, what = command_target(cmd), "the driver's checkout"
    if target is None and step["pin"] == "llama.cpp" and env.get("LLAMA_CPP_GGUF_PY"):
        target, what = env["LLAMA_CPP_GGUF_PY"], "LLAMA_CPP_GGUF_PY"
    if target is None:
        return None
    head = git_head(target)
    if head != sha:
        raise ConfigError(f"{what} {target} is at {head}, but the step's commit (--checkout / --commit) is {sha}: "
                          f"the expectations would be those of another commit")
    return f"{what} {target}"


def expectation(data, name, sha):
    st = data["steps"].get(name)
    if st is None:
        raise ConfigError(f"unknown step '{name}' (known: {', '.join(sorted(data['steps']))})")
    e = st["expect"].get(sha)
    if e is None:
        raise CheckError(f"{name}: no expectations at {st['pin']} {sha} (known: "
                         f"{', '.join(k[:7] for k in st['expect'])}); run the step once at this commit, review its "
                         f"counts and add them to tools/ci/parity_expectations.json")
    return e


def run_streamed(cmd, log_out=None):
    """Run cmd with stdout+stderr merged, echo it unchanged, return (rc, text). Unbuffered Python children keep
    print() output in order with unittest's status lines."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except OSError as e:
        raise ConfigError(f"cannot run {cmd[0]}: {e}") from e
    chunks = []
    out = sys.stdout.buffer
    for raw in proc.stdout:
        out.write(raw)
        out.flush()
        chunks.append(raw)
    rc = proc.wait()
    data = b"".join(chunks)
    if log_out:
        pathlib.Path(log_out).write_bytes(data)
    return rc, data.decode("utf-8", "replace")


def annotate(kind, *msgs):
    for msg in msgs:
        print(f"::{kind}::{msg[:900]}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", required=True, help="step id in the expectations file")
    ap.add_argument("--checkout", help="checkout whose HEAD is the commit the dumps come from")
    ap.add_argument("--commit", help="the full commit instead of --checkout")
    ap.add_argument("--expect", default=str(EXPECT), help="expectations file (default: %(default)s)")
    ap.add_argument("--log", help="check this saved log instead of running a command")
    ap.add_argument("--log-out", help="also save the command's output here")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    try:
        if bool(cmd) == bool(a.log):
            raise ConfigError("give either --log <file> or -- <command>")
        data = load_expectations(a.expect)
        if a.step not in data["steps"]:
            raise ConfigError(f"unknown step '{a.step}' (known: {', '.join(sorted(data['steps']))})")
        sha = resolve_commit(data["steps"][a.step], a.commit, a.checkout)
        where = f"{a.step} @ {data['steps'][a.step]['pin']} {sha[:7]}"
        bound = check_command_commit(data["steps"][a.step], sha, cmd) if cmd else None
        if cmd:
            rc, text = run_streamed(cmd, a.log_out)
            if rc != 0:
                annotate("error", f"{where}: the command exited with status {rc}")
                return rc
        else:
            text = pathlib.Path(a.log).read_text(errors="replace")
        exp = expectation(data, a.step, sha)
        try:
            result = parse(text)
        except CheckError as e:
            raise CheckError([f"{where}: {p}" for p in e.problems]) from None
        annotate("notice", check(result, exp, where) + (f"; {bound} is at {sha[:7]}" if bound else ""))
        return 0
    except CheckError as e:
        annotate("error", *e.problems)
        return 1
    except ConfigError as e:
        annotate("error", f"check_unittest: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
