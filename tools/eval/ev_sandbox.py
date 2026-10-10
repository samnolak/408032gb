"""Run a code task's test command in a throw-away directory with no network and limited resources.

Layers (each one is recorded per run in the result's "isolation" field):
  network : a new network namespace via util-linux `unshare` (only a down loopback inside; tried as
            `unshare --net --pid --fork --kill-child`, then with `--user --map-root-user` for non-root,
            then without the PID namespace); plus Landlock TCP bind/connect denial when the kernel has
            Landlock ABI >= 4; plus a soft Python socket guard (sitecustomize) for Python commands.
  files   : Landlock (sandbox_exec.py) - writes only beneath the attempt directory and /dev/null.
            Reads are NOT restricted (the interpreter and libraries must stay readable).
  limits  : RLIMIT_CPU / AS / FSIZE / NOFILE / CORE set in sandbox_exec.py; wall-clock timeout enforced
            here by killing the whole process group (and, with --pid --kill-child, the PID namespace).
  env     : built from scratch: PATH of the runner, HOME and TMPDIR inside the attempt directory,
            C.UTF-8 locale, PYTHONDONTWRITEBYTECODE, PYTHONHASHSEED=0. No other variable is inherited
            (API keys of the runner never reach the test process).
Policy: by default a code check runs only if BOTH network isolation (namespace or Landlock TCP) and
Landlock file isolation are available; otherwise it is a harness error (never a model failure).
`--weak-sandbox` in run_eval.py runs with whatever is available and records it.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX_EXEC = os.path.join(HERE, "sandbox_exec.py")
GUARD_DIR = os.path.join(HERE, "sandbox_guard")
MARK = "@@EVAL-SANDBOX@@"
NETNS_CANDIDATES = (
    ("unshare", "--net", "--pid", "--fork", "--kill-child"),
    ("unshare", "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child"),
    ("unshare", "--net"),
    ("unshare", "--user", "--map-root-user", "--net"),
)
DEFAULT_LIMITS = {"timeout_s": 60, "as_mb": 2048, "fsize_mb": 64, "nofile": 256, "nproc": 0}
TAIL = 8000

_CAPS = None


def capabilities(refresh=False):
    """Probe once: which unshare form works, which Landlock ABI the kernel offers."""
    global _CAPS
    if _CAPS is not None and not refresh:
        return _CAPS
    caps = {"unshare": shutil.which("unshare"), "netns_prefix": None, "netns_probe": [], "landlock_abi": 0}
    if caps["unshare"]:
        for cand in NETNS_CANDIDATES:
            try:
                p = subprocess.run(list(cand) + ["true"], capture_output=True, text=True, timeout=10)
                caps["netns_probe"].append({"cmd": list(cand), "rc": p.returncode, "stderr": p.stderr[-300:]})
                if p.returncode == 0:
                    caps["netns_prefix"] = list(cand)
                    break
            except (OSError, subprocess.SubprocessError) as e:
                caps["netns_probe"].append({"cmd": list(cand), "rc": None, "error": str(e)})
    try:
        p = subprocess.run([sys.executable, "-I", SANDBOX_EXEC, "--probe"], capture_output=True, text=True, timeout=10)
        caps["landlock_abi"] = int(json.loads(p.stdout)["landlock_abi"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        caps["landlock_probe_error"] = str(e)
    caps["netns"] = caps["netns_prefix"] is not None
    caps["landlock_fs"] = caps["landlock_abi"] >= 1
    caps["landlock_net"] = caps["landlock_abi"] >= 4
    _CAPS = caps
    return caps


def missing_isolation(caps):
    """What the strict policy needs but this host lacks ([] = strict sandbox available)."""
    out = []
    if not (caps.get("netns") or caps.get("landlock_net")):
        out.append("network isolation (unshare --net failed and Landlock ABI < 4)")
    if not caps.get("landlock_fs"):
        out.append("file-write isolation (Landlock not available)")
    return out


def _tail(path, n=TAIL):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - n, 0))
            data = f.read()
        return data.decode("utf-8", errors="replace"), size
    except OSError:
        return "", 0


def _head(path, n=4000):
    try:
        with open(path, "rb") as f:
            return f.read(n).decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_command(attempt_dir, argv, limits=None, caps=None, use_netns=True, use_landlock=True, py_guard=True,
                env_extra=None):
    """Run argv with cwd attempt_dir/repo. Returns a dict; never raises for the child's behaviour."""
    lim = dict(DEFAULT_LIMITS)
    lim.update({k: v for k, v in (limits or {}).items() if v is not None})
    caps = caps or capabilities()
    argv = [sys.executable if a == "{python}" else a for a in argv]
    tmp = os.path.join(attempt_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    wrapper = [sys.executable, "-I", SANDBOX_EXEC, "--root", attempt_dir,
               "--cpu", str(int(lim["timeout_s"]) + 5), "--as-mb", str(int(lim["as_mb"] or 0)),
               "--fsize-mb", str(int(lim["fsize_mb"] or 0)), "--nofile", str(int(lim["nofile"] or 0))]
    if lim.get("nproc"):
        wrapper += ["--nproc", str(int(lim["nproc"]))]
    if not use_landlock:
        wrapper.append("--no-landlock")
    prefix = list(caps["netns_prefix"]) if (use_netns and caps.get("netns_prefix")) else []
    full = prefix + wrapper + ["--"] + argv
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp,
           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONHASHSEED": "0", "EVAL_SANDBOX": "1"}
    if py_guard:
        env["PYTHONPATH"] = GUARD_DIR
        env["EVAL_SANDBOX_PY_GUARD"] = "1"
    for k, v in (env_extra or {}).items():
        env[str(k)] = str(v)
    logs = tempfile.mkdtemp(prefix="eval-logs-")
    out_p, err_p = os.path.join(logs, "stdout"), os.path.join(logs, "stderr")
    res = {"cmd": argv, "rc": None, "timed_out": False, "elapsed_ms": None, "error": None,
           "limits": lim, "isolation": None}
    t0 = time.perf_counter()
    try:
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            try:
                p = subprocess.Popen(full, cwd=attempt_dir, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                                     start_new_session=True)
            except OSError as e:
                res["error"] = "cannot start sandbox: %s" % e
                return res
            try:
                res["rc"] = p.wait(timeout=float(lim["timeout_s"]))
            except subprocess.TimeoutExpired:
                res["timed_out"] = True
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            try:
                os.killpg(p.pid, signal.SIGKILL)         # leftovers of the group (background children)
            except (ProcessLookupError, PermissionError):
                pass
        res["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        err_head = _head(err_p)
        stdout, out_size = _tail(out_p)
        stderr, err_size = _tail(err_p)
        iso = None
        for line in err_head.splitlines():
            if line.startswith(MARK):
                try:
                    iso = json.loads(line[len(MARK):].strip())
                except ValueError:
                    pass
                break
        stderr = "\n".join(l for l in stderr.splitlines() if not l.startswith(MARK))
        res.update({"stdout_tail": stdout, "stderr_tail": stderr, "stdout_bytes": out_size, "stderr_bytes": err_size})
        if iso is None:
            res["error"] = "sandbox wrapper did not report (exec failed before the test command?)"
            iso = {}
        ll = iso.get("landlock") or {}
        res["isolation"] = {"netns": bool(prefix), "netns_cmd": prefix or None,
                            "landlock_abi": ll.get("abi"), "landlock_fs": bool(ll.get("fs")),
                            "landlock_net": bool(ll.get("net")), "landlock_error": ll.get("error"),
                            "rlimits": iso.get("rlimits"), "python_guard": py_guard}
        return res
    finally:
        shutil.rmtree(logs, ignore_errors=True)
