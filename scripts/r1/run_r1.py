#!/usr/bin/env python3
"""R1 orchestration (docs/reassessment/ROADMAP.md, track N4): measure one R1 cell (model x engine x mode) with one
command, the same way every time, into one auditable run directory that ends with a verification pack.

  python3 scripts/r1/run_r1.py --config scripts/r1/configs/<name>.json --out RUNS_ROOT
                               [--plan [--json]] [--only step1,step2] [--set KEY=VALUE ...]

Steps, in this order; each can be disabled in the config ("steps": {"<step>": {"enabled": false}}) or left out with
--only. A failing step is recorded and every later step that does not depend on it still runs.
  weights_verify       scripts/offline/weights.py verify MANIFEST --dest DIR (+ a byte copy of the manifest)
  quant_audit          tools/gguf_quant_audit.py FIRST.gguf --json
  roofline_prediction  tools/roofline.py ... --json, one call per context, then prereg/prereg.json with the sha256 of
                       every prediction: written BEFORE the engine starts (preregistration rule of
                       docs/reassessment/EXPERIMENTS.md); the same mode flags later go to vs_roofline
  nvidia_smi_before    nvidia-smi -q (when nvidia-smi exists)
  engine_build_check   engine.build_info (build-info.json of scripts/engines/build_engine.py) against
                       tools/engines/pins.json[engine.pin]: sha, checked-out commit, deps, status, sm_<arch> cubins,
                       sha256 of every built file, engine.binary inside that build, git HEAD of engine.source_dir;
                       a mismatch fails it and engine_start is not run (skipped without engine.pin / build_info)
  engine_version      the template's version command and --help; flags of the engine argv absent from --help are listed
  engine_start         the engine argv in its own process group; stdout+stderr -> artifacts/engine/engine.log;
                       the exact argv, cwd and environment overrides -> artifacts/engine/argv.json
  health_wait          GET the health URL until the expected status; fails at the timeout or as soon as the engine exits
  matrix               tools/bench/run_matrix.py, one run directory per configured entry
  probes               tools/bench/probes.py, one run directory per configured entry
  slot_persist         tools/bench/prompts.py --save-text (prompt fitted with the server tokenizer), then
                       tools/kv/slot_persist.py; only when the engine supports /slots
  logprob_compare      tools/quality/logprob_compare.py --ref REFERENCE --cand ENGINE; only with a reference endpoint
  vs_roofline          tools/bench/vs_roofline.py over the matrix runs, with the same mode flags as the prediction
  engine_stop          SIGTERM to the engine's process group, SIGKILL after the timeout; the exit code is recorded.
                       Always runs when the engine was started (try/finally; also on Ctrl-C, SIGTERM, SIGHUP or an error)
  nvidia_smi_after     nvidia-smi -q
  pack                 tools/verify/pack.py --runs RUN/artifacts --out RUN/pack, then pack.py --check RUN/pack

Run directory RUNS_ROOT/<UTC>-<config name>/:
  steps.jsonl   journal: one "start" and one "end" record per command or action (step, argv, UTC start/end, exit
                code, status ok|failed|skipped|interrupted, reason, output paths and their sha256)
  run.json      the resolved configuration, the plan, the status of every step and the pack verdict
  artifacts/    everything the pack is built from (tool outputs, engine log/argv/version, nvidia-smi, prereg, journal)
  pack/         the verification pack (tools/verify/pack.py)
  slots/        default ${SLOT_DIR} for llama-server --slot-save-path (outside artifacts: state files are large)

--plan prints every command that would run and runs nothing (no directory is created, no process is started).
Exit codes: 0 every selected step ok or skipped by the config; 1 a step failed (or was skipped because a step it
depends on failed); 128+N interrupted by signal N (130 = Ctrl-C); 2 configuration or usage error. The pack verdict
(PASS / FAIL / NOT RUN) is printed and stored in run.json; it is not the exit code.

Python 3 stdlib only; the tools are called as subprocesses (none of them is modified). Docs: scripts/r1/README.md.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PINS = os.path.join(REPO, "tools", "engines", "pins.json")
sys.path.insert(0, os.path.join(REPO, "tools", "bench"))
import runlog  # noqa: E402  (tools/bench/runlog.py: UTC stamps, git and host info)

SCHEMA = "408032gb-r1-run/1"
STEPS = ["weights_verify", "quant_audit", "roofline_prediction", "nvidia_smi_before", "engine_build_check",
         "engine_version", "engine_start", "health_wait", "matrix", "probes", "slot_persist", "logprob_compare", "vs_roofline",
         "engine_stop", "nvidia_smi_after", "pack"]
TAIL = ("engine_stop", "nvidia_smi_after", "pack")      # run in the finally block
NEEDS_ENGINE = ("matrix", "probes", "slot_persist", "logprob_compare")
DEPENDS = dict({"engine_start": ("engine_build_check",), "health_wait": ("engine_start",)},
               **{s: ("engine_start", "health_wait") for s in NEEDS_ENGINE})
TOOLS = {"weights": "scripts/offline/weights.py", "quant": "tools/gguf_quant_audit.py", "roofline": "tools/roofline.py",
         "matrix": "tools/bench/run_matrix.py", "probes": "tools/bench/probes.py", "prompts": "tools/bench/prompts.py",
         "slot": "tools/kv/slot_persist.py", "logprob": "tools/quality/logprob_compare.py",
         "vs": "tools/bench/vs_roofline.py", "pack": "tools/verify/pack.py"}
OWNED_FLAGS = {"matrix": ("--url", "--out-dir", "--engine-desc", "--model-desc"),
               "probes": ("--url", "--out-dir", "--engine-desc", "--model-desc"),
               "slot_persist": ("--url", "--prompt-file", "--out"),
               "logprob_compare": ("--ref", "--cand", "--corpus", "--out")}
TOP_KEYS = {"name", "mock", "vars", "engine", "model", "roofline", "mode", "reference", "steps", "orchestrator"}
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
FIXED_VARS = ("REPO", "RUNS_ROOT", "RUN_DIR", "ARTIFACTS", "ENGINE_DIR", "NAME", "UTC", "CONFIG_DIR", "MODEL")
DEFAULT_VARS = {"PYTHON": sys.executable or "python3", "HOST": "127.0.0.1", "PORT": "auto", "CTX": "262144",
                "SLOT_DIR": "${RUN_DIR}/slots", "NVIDIA_SMI": "nvidia-smi"}
ENV_OF_INTEREST = re.compile(r"^(CUDA_|GGML_|LLAMA_|STRATA_|NCCL_|OMP_|NVIDIA_|HIP_)")
FLAG_RE = re.compile(r"^(--?[A-Za-z][A-Za-z0-9_-]*)(=.*)?$")


class ConfigError(Exception):
    pass


class Interrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


def signame(signum):
    try:
        return signal.Signals(signum).name
    except ValueError:
        return "signal %d" % signum


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_port(host):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def load_pins():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)["engines"]


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def shell(argv):
    return shlex.join([str(a) for a in argv])


# ---------------------------------------------------------------------------- configuration
class Vars:
    """${NAME} resolution: built-ins, defaults, config "vars", --set overrides; lazily, with cycle detection."""

    def __init__(self, raw):
        self.raw = dict(raw)
        self.done = {}

    def get(self, name, stack=()):
        if name in self.done:
            return self.done[name]
        if name in stack:
            raise ConfigError("variable cycle: %s" % " -> ".join(stack + (name,)))
        if name not in self.raw:
            raise ConfigError("unknown variable ${%s} (define it in \"vars\" or pass --set %s=...)" % (name, name))
        val = self.sub(str(self.raw[name]), stack + (name,))
        if name == "PORT" and val == "auto":
            val = str(free_port(self.get("HOST", stack + (name,))))
        self.done[name] = val
        return val

    def sub(self, s, stack=()):
        return VAR_RE.sub(lambda m: self.get(m.group(1), stack), s)


def substitute(obj, var, path, typed=False):
    """Every string of the config gets ${VAR} replaced; keys starting with "_" are free text and stay as written.
    typed (engine.config_files): a string that is exactly one ${VAR} whose value is an integer becomes a JSON number."""
    if isinstance(obj, dict):
        return {k: (v if k.startswith("_") else substitute(v, var, path + "." + k, typed or k == "config_files"))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, var, "%s[%d]" % (path, i), typed) for i, v in enumerate(obj)]
    if isinstance(obj, str):
        out = var.sub(obj)
        if "${" in out:
            raise ConfigError("%s: unresolved or malformed variable reference in %r" % (path, obj))
        if typed and VAR_RE.fullmatch(obj) and re.fullmatch(r"-?\d+", out):
            return int(out)
        return out
    return obj


def num(v, what, lo=None, hi=None, integer=False):
    try:
        if isinstance(v, bool):
            raise ValueError
        x = int(v) if integer else float(v)
        if integer and float(v) != x:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError("%s must be a%s number, got %r" % (what, "n integer" if integer else "", v))
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ConfigError("%s = %r is outside [%s, %s]" % (what, v, lo, hi))
    return x


def flag(v, what):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "yes", "1", "on", "false", "no", "0", "off"):
        return v.lower() in ("true", "yes", "1", "on")
    raise ConfigError("%s must be true or false, got %r" % (what, v))


def str_list(v, what, allow_empty=True):
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
        raise ConfigError("%s must be a list of strings" % what)
    if not v and not allow_empty:
        raise ConfigError("%s must not be empty" % what)
    return [str(x) for x in v]


def obj(v, what):
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ConfigError("%s must be a JSON object" % what)
    return v


class Item:
    """One command or action of the plan."""

    def __init__(self, step, sub=None, kind="cmd", argv=None, stdout=None, stderr=None, cwd=None, env=None,
                 timeout=None, ok_codes=(0,), skip=None, action=None, json_out=False, pre=(), after=None, **extra):
        self.step, self.sub, self.kind, self.argv = step, sub, kind, argv
        self.stdout, self.stderr, self.cwd, self.env = stdout, stderr, cwd, env or {}
        self.timeout, self.ok_codes, self.skip, self.action = timeout, tuple(ok_codes), skip, action
        self.json_out, self.pre, self.after, self.extra = json_out, list(pre), after, extra

    @property
    def key(self):
        return "%s[%s]" % (self.step, self.sub) if self.sub else self.step

    def describe(self, run_dir):
        rel = (lambda p: os.path.relpath(p, run_dir) if p else p)
        if self.skip:
            return "SKIP (%s)" % self.skip
        pre = "".join("(first: %s) " % " ".join(str(x) for x in p) for p in self.pre_text(rel))
        if self.kind == "cmd":
            s = shell(self.argv) + " > " + rel(self.stdout) + (" 2> " + rel(self.stderr) if self.stderr else " 2>&1")
            if self.cwd and self.cwd != REPO:
                s = "cd %s && %s" % (shlex.quote(self.cwd), s)
            return pre + s
        if self.kind == "engine":
            env = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in sorted(self.env.items()))
            return pre + "start in the background, own process group, cwd %s: %s%s > %s 2>&1" % (
                self.cwd, (env + " ") if env else "", shell(self.argv), rel(self.stdout))
        return pre + (self.action or self.kind)

    def pre_text(self, rel):
        out = []
        for p in self.pre:
            if p[0] == "copy":
                out.append(("copy", p[1], "->", rel(p[2])))
            elif p[0] == "mkdir":
                out.append(("mkdir", p[1]))
            elif p[0] == "write":
                out.append(("write", rel(p[1])))
        return out

    def as_dict(self, run_dir):
        rel = (lambda p: os.path.relpath(p, run_dir) if p else None)
        d = {"step": self.step, "sub": self.sub, "kind": self.kind, "argv": self.argv, "stdout": rel(self.stdout),
             "stderr": rel(self.stderr), "cwd": self.cwd, "env": self.env, "timeout_s": self.timeout,
             "ok_codes": list(self.ok_codes), "skip": self.skip, "action": self.action,
             "pre": [list(x) for x in self.pre_text(rel)], "text": self.describe(run_dir)}
        for k in ("url", "expect_status", "interval_s"):
            if k in self.extra:
                d[k] = self.extra[k]
        return d


class Cell:
    """A loaded, fully substituted configuration and its plan."""

    def __init__(self, config_path, runs_root, sets, stamp=None):
        self.config_path = os.path.abspath(config_path)
        try:
            with open(self.config_path, "rb") as f:
                self.config_bytes = f.read()
            raw = json.loads(self.config_bytes.decode("utf-8"))
        except (OSError, ValueError) as e:
            raise ConfigError("cannot read %s: %s" % (config_path, e))
        if not isinstance(raw, dict):
            raise ConfigError("the config must be a JSON object")
        unknown = sorted(k for k in raw if k not in TOP_KEYS and not k.startswith("_"))
        if unknown:
            raise ConfigError("unknown top-level keys: %s (known: %s)" % (", ".join(unknown), ", ".join(sorted(TOP_KEYS))))
        self.name = str(raw.get("name") or os.path.splitext(os.path.basename(config_path))[0])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ConfigError("name %r: use letters, digits, '.', '_' or '-'" % self.name)
        self.label = raw.get("_label")
        self.mock = flag(raw.get("mock", False), "mock")
        self.runs_root = os.path.abspath(runs_root)
        self.stamp = stamp or runlog.utc_stamp()
        self.run_dir = self.pick_run_dir()
        self.artifacts = os.path.join(self.run_dir, "artifacts")
        self.engine_dir = os.path.join(self.artifacts, "engine")
        values = dict(DEFAULT_VARS)
        for k, v in obj(raw.get("vars"), "vars").items():
            if k in FIXED_VARS:
                raise ConfigError("vars.%s is built in and cannot be redefined" % k)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, (str, int, float)) or isinstance(v, bool):
                raise ConfigError("vars.%s: names are [A-Za-z_][A-Za-z0-9_]*, values strings or numbers" % k)
            values[k] = v
        for s in sets:
            k, sep, v = s.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise ConfigError("--set %r: expected KEY=VALUE" % s)
            if k in FIXED_VARS:
                raise ConfigError("--set %s: built-in variable, cannot be overridden" % k)
            values[k] = v
        values.update(REPO=REPO, RUNS_ROOT=self.runs_root, RUN_DIR=self.run_dir, ARTIFACTS=self.artifacts,
                      ENGINE_DIR=self.engine_dir, NAME=self.name, UTC=self.stamp,
                      CONFIG_DIR=os.path.dirname(self.config_path))
        files = obj(raw.get("model"), "model").get("files")
        if isinstance(files, list) and files and isinstance(files[0], str):
            values["MODEL"] = files[0]
        self.var = Vars(values)
        self.vars = {k: self.var.get(k) for k in sorted(values)}
        body = {k: v for k, v in raw.items() if k != "vars"}
        self.cfg = substitute(body, self.var, "config")
        self.validate()
        self.plan = self.build_plan()

    def pick_run_dir(self):
        base = os.path.join(self.runs_root, "%s-%s" % (self.stamp, self.name))
        path, n = base, 1
        while os.path.exists(path):
            n += 1
            path = "%s-%d" % (base, n)
        return path

    # ------------------------------------------------------------------ validation
    def validate(self):
        c = self.cfg
        e = obj(c.get("engine"), "engine")
        self.engine = {
            "name": str(e.get("name") or ""), "commit": str(e.get("commit") or ""),
            "argv": str_list(e.get("argv"), "engine.argv"),
            "version_argv": str_list(e.get("version_argv"), "engine.version_argv"),
            "help_argv": str_list(e.get("help_argv"), "engine.help_argv"),
            "help_ignore": str_list(e.get("help_ignore"), "engine.help_ignore"),
            "env": {str(k): str(v) for k, v in obj(e.get("env"), "engine.env").items()},
            "cwd": str(e.get("cwd") or self.run_dir),
            "config_files": obj(e.get("config_files"), "engine.config_files"),
            "base_url": str(e.get("base_url") or "http://%s:%s" % (self.vars["HOST"], self.vars["PORT"])).rstrip("/"),
            "supports_slots": flag(e.get("supports_slots", False), "engine.supports_slots"),
            "mkdirs": str_list(e.get("mkdirs"), "engine.mkdirs"),
            "after_ready": [str_list(x, "engine.after_ready[%d]" % i, allow_empty=False)
                            for i, x in enumerate(e.get("after_ready") or [])],
            "pin": str(e.get("pin") or ""), "build_info": e.get("build_info"), "binary": e.get("binary"),
            "source_dir": e.get("source_dir"), "arch": num(e.get("arch", 89), "engine.arch", 30, 200, True),
        }
        self.validate_pin()
        if not self.engine["name"]:
            raise ConfigError("engine.name is required (free text: engine and build, e.g. 'llama.cpp llama-server')")
        for fname in self.engine["config_files"]:
            if os.path.isabs(fname) or ".." in fname.replace("\\", "/").split("/") or not fname:
                raise ConfigError("engine.config_files key %r must be a plain relative file name" % fname)
        h = obj(e.get("health"), "engine.health")
        self.health = {"url": str(h.get("url") or self.engine["base_url"] + "/health"),
                       "timeout_s": num(h.get("timeout_s", 1800), "engine.health.timeout_s", 0),
                       "interval_s": num(h.get("interval_s", 2), "engine.health.interval_s", 0.01),
                       "expect_status": num(h.get("expect_status", 200), "engine.health.expect_status", 100, 599, True)}
        st = obj(e.get("stop"), "engine.stop")
        self.stop_timeout = num(st.get("timeout_s", 60), "engine.stop.timeout_s", 0)
        m = obj(c.get("model"), "model")
        self.model = {"desc": str(m.get("desc") or ""), "files": str_list(m.get("files"), "model.files"),
                      "weights_manifest": m.get("weights_manifest"), "weights_dest": m.get("weights_dest")}
        if not self.model["desc"]:
            raise ConfigError("model.desc is required (model, quant, revision)")
        for k in ("weights_manifest", "weights_dest"):
            if self.model[k] is not None and not isinstance(self.model[k], str):
                raise ConfigError("model.%s must be a string or null" % k)
        r = obj(c.get("roofline"), "roofline")
        self.roofline = None
        if r.get("spec"):
            self.roofline = {"spec": repo_path(str(r["spec"])),
                             "hw": repo_path(str(r.get("hw") or "hardware/target-4x4080-32g.unverified.json")),
                             "expert_bpw": num(r.get("expert_bpw", 4.5), "roofline.expert_bpw", 0.5, 32),
                             "dense_bpw": num(r.get("dense_bpw", 8.5), "roofline.dense_bpw", 0.5, 32),
                             "kv_bytes": num(r.get("kv_bytes", 2.0), "roofline.kv_bytes", 0.1, 8)}
        md = obj(c.get("mode"), "mode")
        self.mode = {"overlap": str(md.get("overlap", "no")), "strategy": str(md.get("strategy", "pp")),
                     "miss": str(md.get("miss", "cpu")), "hit": num(md.get("hit", 1.0), "mode.hit", 0, 1),
                     "spec_k": num(md.get("spec_k", 0), "mode.spec_k", 0, 64, True),
                     "spec_accept": num(md.get("spec_accept", 0.0), "mode.spec_accept", 0, 1),
                     "draft_gb": (None if md.get("draft_gb") in (None, "") else
                                  num(md.get("draft_gb"), "mode.draft_gb", 0))}
        for k, allowed in (("overlap", ("yes", "no")), ("strategy", ("pp", "ep", "tp")), ("miss", ("cpu", "pcie"))):
            if self.mode[k] not in allowed:
                raise ConfigError("mode.%s must be one of %s, got %r" % (k, "/".join(allowed), self.mode[k]))
        ref = c.get("reference")
        ref = obj(ref, "reference")
        self.reference = {"url": ref.get("url"), "model": ref.get("model"), "cand_model": ref.get("cand_model")}
        self.steps = obj(c.get("steps"), "steps")
        unknown = sorted(k for k in self.steps if k not in STEPS and not k.startswith("_"))
        if unknown:
            raise ConfigError("unknown steps: %s (known: %s)" % (", ".join(unknown), ", ".join(STEPS)))
        for s in STEPS:
            obj(self.steps.get(s), "steps.%s" % s)
        if "enabled" in (self.steps.get("engine_stop") or {}):
            raise ConfigError("steps.engine_stop cannot be configured: an engine started by this run is always stopped")
        o = obj(c.get("orchestrator"), "orchestrator")
        self.child_grace_s = num(o.get("child_grace_s", 30), "orchestrator.child_grace_s", 0)
        self.version_timeout_s = num(o.get("version_timeout_s", 120), "orchestrator.version_timeout_s", 1)
        self.engine_desc = "%s @ %s%s; server argv: %s" % (
            self.engine["name"], self.engine["commit"] or "commit not given",
            " (pin %r of tools/engines/pins.json)" % self.engine["pin"] if self.engine["pin"] else "",
            shell(self.engine["argv"]) if self.engine["argv"] else "(started outside this run)")
        first = self.vars.get("MODEL")
        self.model_desc = self.model["desc"] + ("; first file: %s" % os.path.basename(first) if first else "")

    def validate_pin(self):
        """engine.pin names an engine of tools/engines/pins.json (the builds of scripts/engines/build_engine.py); the
        commit then comes from the pin. engine.build_info / binary / source_dir say where that build is on this host."""
        e = self.engine
        self.pin_spec = None
        for k in ("build_info", "binary", "source_dir"):
            if e[k] is not None and not isinstance(e[k], str):
                raise ConfigError("engine.%s must be a string or null" % k)
            if e[k] and not e["pin"]:
                raise ConfigError("engine.%s needs engine.pin (the pinned build it belongs to)" % k)
            if e[k]:
                e[k] = os.path.normpath(e[k])
        if not e["pin"]:
            return
        pins = load_pins()
        if e["pin"] not in pins:
            raise ConfigError("engine.pin %r is not an engine of tools/engines/pins.json (known: %s)"
                              % (e["pin"], ", ".join(sorted(pins))))
        self.pin_spec = pins[e["pin"]]
        sha = self.pin_spec["sha"]
        if not e["commit"]:
            e["commit"] = sha
        elif not (re.fullmatch(r"[0-9a-f]{7,40}", e["commit"]) and sha.startswith(e["commit"])):
            raise ConfigError("engine.commit %r contradicts engine.pin %r = %s (drop engine.commit: the pin gives it)"
                              % (e["commit"], e["pin"], sha))

    def step_cfg(self, s):
        return self.steps.get(s) or {}

    def enabled(self, s):
        return flag(self.step_cfg(s).get("enabled", True), "steps.%s.enabled" % s)

    def runs(self, s):
        runs = self.step_cfg(s).get("runs") or []
        if not isinstance(runs, list):
            raise ConfigError("steps.%s.runs must be a list" % s)
        out, seen = [], set()
        for i, r in enumerate(runs):
            r = obj(r, "steps.%s.runs[%d]" % (s, i))
            name = str(r.get("name") or "")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in seen:
                raise ConfigError("steps.%s.runs[%d].name %r: unique, letters/digits/._-" % (s, i, name))
            seen.add(name)
            args = str_list(r.get("args"), "steps.%s.runs[%d].args" % (s, i))
            self.check_owned(s, args)
            timeout = r.get("timeout_s")
            out.append((name, args, None if timeout in (None, "") else num(timeout, "timeout_s", 1)))
        return out

    def check_owned(self, s, args):
        for a in args:
            f = a.split("=", 1)[0]
            if f in OWNED_FLAGS.get(s, ()):
                raise ConfigError("steps.%s: %s is set by the orchestrator, remove it from args" % (s, f))

    # ------------------------------------------------------------------ plan
    def a(self, *parts):
        return os.path.join(self.artifacts, *parts)

    def tool(self, key, *args):
        return [self.vars["PYTHON"], os.path.join(REPO, TOOLS[key])] + [str(x) for x in args]

    def mode_flags(self):
        m = self.mode
        out = ["--strategy", m["strategy"], "--miss", m["miss"], "--overlap", m["overlap"], "--spec-k", str(m["spec_k"]),
               "--spec-accept", repr(float(m["spec_accept"]))]
        if m["draft_gb"] is not None:
            out += ["--draft-gb", repr(float(m["draft_gb"]))]
        return out

    def quant_flags(self):
        r = self.roofline
        return ["--hw", r["hw"], "--expert-bpw", repr(float(r["expert_bpw"])), "--dense-bpw", repr(float(r["dense_bpw"])),
                "--kv-bytes", repr(float(r["kv_bytes"]))]

    def roofline_contexts(self):
        ctxs = self.step_cfg("roofline_prediction").get("contexts")
        if ctxs:
            return [num(x, "steps.roofline_prediction.contexts", 1, None, True) for x in str_list(ctxs, "contexts")]
        found = set()
        for _name, args, _t in self.runs("matrix"):
            for i, x in enumerate(args):
                val = args[i + 1] if x == "--inputs" and i + 1 < len(args) else (x.split("=", 1)[1] if x.startswith("--inputs=") else None)
                if val:
                    found.update(int(v) for v in val.split(",") if v.strip())
        return sorted(found) or [num(self.vars["CTX"], "CTX", 1, None, True)]

    def build_plan(self):
        P = []
        log = lambda name: (self.a("logs", name + ".stdout.txt"), self.a("logs", name + ".stderr.txt"))  # noqa: E731
        model = self.vars.get("MODEL")

        # weights_verify
        man = self.model["weights_manifest"]
        if not self.enabled("weights_verify"):
            P.append(Item("weights_verify", skip="disabled in the config"))
        elif not man:
            P.append(Item("weights_verify", skip="no model.weights_manifest in the config: weights integrity (V2) stays "
                                                 "NOT RUN"))
        else:
            man = repo_path(man)
            dest = self.model["weights_dest"] or (os.path.dirname(model) if model else None)
            if not dest:
                P.append(Item("weights_verify", skip="no model.weights_dest and no model.files"))
            else:
                P.append(Item("weights_verify", argv=self.tool("weights", "verify", man, "--dest", dest),
                              stdout=self.a("weights", "weights_verify.txt"),
                              stderr=self.a("logs", "weights_verify.stderr.txt"),
                              pre=[("copy", man, self.a("weights", os.path.basename(man)))]))
        # quant_audit
        if not self.enabled("quant_audit"):
            P.append(Item("quant_audit", skip="disabled in the config"))
        elif not model:
            P.append(Item("quant_audit", skip="no model.files in the config: V1 stays NOT RUN"))
        else:
            extra = str_list(self.step_cfg("quant_audit").get("args"), "steps.quant_audit.args")
            P.append(Item("quant_audit", argv=self.tool("quant", repo_path(model), "--json", *extra),
                          stdout=self.a("quant", "quant_audit.json"), stderr=self.a("logs", "quant_audit.stderr.txt"),
                          json_out=True))
        # roofline_prediction (preregistration)
        if not self.enabled("roofline_prediction"):
            P.append(Item("roofline_prediction", skip="disabled in the config: no preregistered prediction"))
        elif not self.roofline:
            P.append(Item("roofline_prediction", skip="no roofline.spec in the config: no preregistered prediction"))
        else:
            preds = []
            for ctx in self.roofline_contexts():
                out = self.a("prereg", "roofline-ctx%d.json" % ctx)
                preds.append((ctx, out))
                P.append(Item("roofline_prediction", sub="ctx=%d" % ctx, stdout=out,
                              stderr=self.a("logs", "roofline-ctx%d.stderr.txt" % ctx), json_out=True,
                              argv=self.tool("roofline", self.roofline["spec"], "--ctx", ctx, *self.quant_flags(),
                                             "--sweep-hit", repr(float(self.mode["hit"])),
                                             "--strategies", self.mode["strategy"], "--miss", self.mode["miss"],
                                             "--overlap", self.mode["overlap"], "--spec-k", self.mode["spec_k"],
                                             "--spec-accept", repr(float(self.mode["spec_accept"])),
                                             *(["--draft-gb", repr(float(self.mode["draft_gb"]))]
                                               if self.mode["draft_gb"] is not None else []), "--json")))
            P.append(Item("roofline_prediction", sub="record", kind="prereg", predictions=preds,
                          stdout=self.a("prereg", "prereg.json"),
                          action="write artifacts/prereg/prereg.json (sha256 of every prediction, engine argv, mode, "
                                 "planned measurement commands) before the engine starts"))
        # nvidia_smi_before
        P.append(self.smi_item("nvidia_smi_before", "before"))
        # engine_build_check
        e = self.engine
        if not self.enabled("engine_build_check"):
            P.append(Item("engine_build_check", skip="disabled in the config"))
        elif not e["pin"]:
            P.append(Item("engine_build_check", skip="no engine.pin: the engine is not tied to tools/engines/pins.json "
                                                     "(commit as written in engine.commit)"))
        elif not e["build_info"]:
            P.append(Item("engine_build_check", skip="no engine.build_info: the build of pin %r is not checked"
                                                     % e["pin"]))
        else:
            binary = e["binary"] or (e["argv"][0] if e["argv"] else None)
            P.append(Item("engine_build_check", kind="buildcheck", stdout=self.a("engine", "build_check.json"),
                          build_info=e["build_info"], binary=binary, source_dir=e["source_dir"],
                          action="compare %s with tools/engines/pins.json[%s] = %s: engine, repo, sha, checked-out "
                                 "commit, deps, status ok, arch and sm_%d cubins, sha256 of every built file%s%s; copy "
                                 "it to artifacts/engine/build-info.json; any mismatch fails this step and the engine "
                                 "is not started" % (e["build_info"], e["pin"], self.pin_spec["sha"], e["arch"],
                                                     "; %s is a binary of that build" % binary if binary else "",
                                                     "; git HEAD of %s is the pin" % e["source_dir"]
                                                     if e["source_dir"] else "")))
        # engine_version
        if not self.enabled("engine_version"):
            P.append(Item("engine_version", skip="disabled in the config"))
        elif not (e["version_argv"] or e["help_argv"]):
            P.append(Item("engine_version", skip="the template has no version_argv / help_argv"))
        else:
            if e["version_argv"]:
                P.append(Item("engine_version", sub="version", argv=e["version_argv"], cwd=e["cwd"], env=e["env"],
                              stdout=self.a("engine", "version.txt"), timeout=self.version_timeout_s))
            if e["help_argv"]:
                P.append(Item("engine_version", sub="help", argv=e["help_argv"], cwd=e["cwd"], env=e["env"],
                              stdout=self.a("engine", "help.txt"), timeout=self.version_timeout_s, flags_check=True))
        # engine_start
        if not self.enabled("engine_start"):
            P.append(Item("engine_start", skip="disabled in the config: the engine is managed outside this run"))
        elif not e["argv"]:
            P.append(Item("engine_start", skip="no engine.argv: the engine is managed outside this run"))
        else:
            pre = [("mkdir", d) for d in e["mkdirs"]]
            pre += [("write", os.path.join(self.engine_dir, f)) for f in sorted(e["config_files"])]
            pre.append(("write", self.a("engine", "argv.json")))
            P.append(Item("engine_start", kind="engine", argv=e["argv"], cwd=e["cwd"], env=e["env"],
                          stdout=self.a("engine", "engine.log"), pre=pre))
        # health_wait
        h = self.health
        if not self.enabled("health_wait"):
            P.append(Item("health_wait", skip="disabled in the config"))
        else:
            P.append(Item("health_wait", kind="health", url=h["url"], expect_status=h["expect_status"],
                          interval_s=h["interval_s"], timeout=h["timeout_s"],
                          action="GET %s until HTTP %d (timeout %g s, poll every %g s; fails as soon as the engine exits)"
                                 % (h["url"], h["expect_status"], h["timeout_s"], h["interval_s"])))
            for i, argv in enumerate(e["after_ready"], 1):
                P.append(Item("health_wait", sub="after-ready-%d" % i, argv=argv, after="health_wait", timeout=120,
                              stdout=self.a("hw", "after-ready-%d.txt" % i), soft=True, needs_binary=True))
        # matrix / probes
        for s, key, sub in (("matrix", "matrix", "matrix"), ("probes", "probes", "probes")):
            runs = self.runs(s)
            if not self.enabled(s):
                P.append(Item(s, skip="disabled in the config"))
            elif not runs:
                P.append(Item(s, skip="no steps.%s.runs in the config" % s))
            for name, args, timeout in (runs if self.enabled(s) else []):
                o, err = log("%s-%s" % (s, name))
                P.append(Item(s, sub=name, argv=self.tool(key, "--url", e["base_url"], "--out-dir", self.a(sub, name),
                                                          "--engine-desc", self.engine_desc,
                                                          "--model-desc", self.model_desc, *args),
                              stdout=o, stderr=err, timeout=timeout, run_dir=self.a(sub, name)))
        # slot_persist
        sp = self.step_cfg("slot_persist")
        if not self.enabled("slot_persist"):
            P.append(Item("slot_persist", skip="disabled in the config"))
        elif not e["supports_slots"]:
            P.append(Item("slot_persist", skip="engine.supports_slots is false (no /slots save/restore): V7 stays "
                                               "NOT RUN"))
        else:
            args = str_list(sp.get("args"), "steps.slot_persist.args")
            self.check_owned("slot_persist", args)
            pf = sp.get("prompt_file")
            pre = []
            if pf:
                pf = repo_path(str(pf))
                prompt = self.a("kv", "prompt", os.path.basename(pf))
                pre = [("copy", pf, prompt)]
            else:
                n = num(sp.get("prompt_tokens", 131072), "steps.slot_persist.prompt_tokens", 16, None, True)
                prompt = self.a("kv", "prompt", "prose-%d.txt" % n)
                o, err = log("slot_persist-prompt")
                P.append(Item("slot_persist", sub="prompt", stdout=o, stderr=err,
                              argv=self.tool("prompts", "--url", e["base_url"], "--kind", "prose", "--targets", n,
                                             "--seed", str(sp.get("seed", 0)), "--save-text", self.a("kv", "prompt"),
                                             "--out", self.a("kv", "prompt-fit.json"))))
            o, err = log("slot_persist")
            P.append(Item("slot_persist", sub="persist", stdout=o, stderr=err, pre=pre,
                          after=None if pf else "slot_persist[prompt]",
                          argv=self.tool("slot", "--url", e["base_url"], "--prompt-file", prompt,
                                         "--out", self.a("kv", "slot_persist.json"), *args)))
        # logprob_compare
        lp = self.step_cfg("logprob_compare")
        corpus = [x for x in str_list(lp.get("corpus"), "steps.logprob_compare.corpus") if x.strip()]
        if not self.enabled("logprob_compare"):
            P.append(Item("logprob_compare", skip="disabled in the config"))
        elif not self.reference["url"]:
            P.append(Item("logprob_compare", skip="no reference endpoint (\"reference\": {\"url\": ...}): V6 stays NOT RUN"))
        elif not corpus:
            P.append(Item("logprob_compare", skip="no steps.logprob_compare.corpus: V6 stays NOT RUN"))
        else:
            args = str_list(lp.get("args"), "steps.logprob_compare.args")
            self.check_owned("logprob_compare", args)
            models = []
            if self.reference["model"]:
                models += ["--ref-model", str(self.reference["model"])]
            if self.reference["cand_model"]:
                models += ["--cand-model", str(self.reference["cand_model"])]
            o, err = log("logprob_compare")
            P.append(Item("logprob_compare", stdout=o, stderr=err, pre=[("mkdir", self.a("quality"))],
                          argv=self.tool("logprob", "--ref", str(self.reference["url"]), "--cand", e["base_url"],
                                         "--corpus", *[repo_path(x) for x in corpus], *models,
                                         "--out", self.a("quality", "logprob_compare.json"), *args)))
        # vs_roofline
        mdirs = [it.extra["run_dir"] for it in P if it.step == "matrix" and not it.skip]
        if not self.enabled("vs_roofline"):
            P.append(Item("vs_roofline", skip="disabled in the config"))
        elif not self.roofline:
            P.append(Item("vs_roofline", skip="no roofline.spec in the config"))
        elif not mdirs:
            P.append(Item("vs_roofline", skip="no matrix runs configured"))
        else:
            out = self.a("roofline", "vs_roofline.json")
            argv = self.tool("vs", *mdirs, "--spec", self.roofline["spec"], *self.quant_flags(),
                             "--hit", repr(float(self.mode["hit"])), *self.mode_flags(), "--json")
            P.append(Item("vs_roofline", argv=argv, stdout=out, stderr=self.a("logs", "vs_roofline.stderr.txt"),
                          json_out=True, matrix_dirs=mdirs,
                          pre=[("write", self.a("roofline", "vs_roofline.cmd.txt"))]))
        # tail: engine_stop, nvidia_smi_after, pack
        P.append(Item("engine_stop", kind="stop", timeout=self.stop_timeout,
                      action="SIGTERM to the engine's process group, SIGKILL after %g s; record the exit code "
                             "(always, when this run started the engine)" % self.stop_timeout))
        P.append(self.smi_item("nvidia_smi_after", "after"))
        pk = self.step_cfg("pack")
        if not self.enabled("pack"):
            P.append(Item("pack", skip="disabled in the config"))
        else:
            pack_dir = os.path.join(self.run_dir, "pack")
            crit = pk.get("criteria")
            argv = self.tool("pack", "--runs", self.artifacts, "--out", pack_dir,
                             *(["--criteria", repo_path(str(crit))] if crit else []))
            P.append(Item("pack", sub="build", argv=argv, ok_codes=(0, 1, 3),
                          stdout=os.path.join(self.run_dir, "logs", "pack.stdout.txt"),
                          stderr=os.path.join(self.run_dir, "logs", "pack.stderr.txt"), pack_dir=pack_dir,
                          pre=[("copy", os.path.join(self.run_dir, "steps.jsonl"),
                                self.a("orchestrator", "steps.jsonl"))]))
            P.append(Item("pack", sub="check", argv=self.tool("pack", "--check", pack_dir), after="pack[build]",
                          stdout=os.path.join(self.run_dir, "logs", "pack-check.stdout.txt"),
                          stderr=os.path.join(self.run_dir, "logs", "pack-check.stderr.txt")))
        for it in P:
            if it.stdout:
                it.stdout = os.path.normpath(it.stdout)
            if it.kind == "cmd" and it.cwd is None:
                it.cwd = REPO
        P.sort(key=lambda it: STEPS.index(it.step))      # STEPS is the order (stable: sub-items keep theirs)
        return P

    def smi_item(self, step, when):
        if not self.enabled(step):
            return Item(step, skip="disabled in the config")
        return Item(step, argv=[self.vars["NVIDIA_SMI"], "-q"], stdout=self.a("hw", "nvidia-smi-q.%s.txt" % when),
                    stderr=self.a("logs", "%s.stderr.txt" % step), timeout=120, needs_binary=True)

    # ------------------------------------------------------------------ outputs
    def plan_dict(self, selected):
        return {"schema": SCHEMA, "name": self.name, "label": self.label, "mock": self.mock,
                "config": self.config_path, "run_dir": self.run_dir, "vars": self.vars, "selected": sorted(selected),
                "engine_desc": self.engine_desc, "model_desc": self.model_desc, "mode": self.mode,
                "roofline": self.roofline,
                "items": [dict(it.as_dict(self.run_dir), index=i + 1, selected=it.step in selected)
                          for i, it in enumerate(self.plan)]}

    def plan_text(self, selected):
        L = ["# R1 plan: %s (config %s)" % (self.name, self.config_path)]
        if self.label:
            L.append("# %s" % self.label)
        if self.mock:
            L.append("# MOCK CONFIG: tools/bench/mock_server.py is the 'engine'; nothing it produces is evidence")
        L.append("# run directory (created when the run starts): %s" % self.run_dir)
        L.append("# variables: " + ", ".join("%s=%s" % (k, v) for k, v in sorted(self.vars.items())))
        L.append("# engine desc: %s" % self.engine_desc)
        L.append("# model desc: %s" % self.model_desc)
        for i, it in enumerate(self.plan, 1):
            tag = "[%02d] %s" % (i, it.key)
            if it.step not in selected:
                L.append("%s: SKIP (not selected by --only)" % tag)
            elif it.skip:
                L.append("%s: SKIP (%s)" % (tag, it.skip))
            else:
                L.append("%s: %s" % (tag, it.describe(self.run_dir)))
        return "\n".join(L)


def parse_only(only):
    if not only:
        return set(STEPS)
    sel = {s.strip() for s in only.split(",") if s.strip()}
    bad = sorted(sel - set(STEPS))
    if bad:
        raise ConfigError("--only: unknown steps %s (known: %s)" % (", ".join(bad), ", ".join(STEPS)))
    if "engine_start" in sel:
        sel.add("engine_stop")              # an engine started by this run is always stopped by it
        sel.add("engine_build_check")       # and is never started without its build check (a no-op without a pin)
    return sel


# ---------------------------------------------------------------------------- execution
class Run:
    def __init__(self, cell, selected, out=None, argv=None):
        self.cell, self.selected = cell, selected
        self.argv = argv
        self.out = out or sys.stdout
        self.run_dir = cell.run_dir
        self.journal_path = os.path.join(self.run_dir, "steps.jsonl")
        self.item_state, self.step_state = {}, {}
        self.soft_failures = []
        self.records = []
        self.engine = None
        self.engine_log = None
        self.child = None
        self.critical = 0
        self.pending_signal = None
        self.finishing = False
        self.current = None
        self.interrupted = None
        self.seq = 0
        self.pack_verdict = None
        self.saved_handlers = {}
        self.utc_start = runlog.utc_now()

    # ------------------------------------------------------------------ bookkeeping
    def say(self, msg):
        print("[r1] %s" % msg, file=self.out, flush=True)

    def rel(self, p):
        return os.path.relpath(p, self.run_dir) if p else None

    def journal(self, rec):
        with self.shielded():
            self.seq += 1
            rec = dict(rec, seq=self.seq)
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    def set_state(self, it, status):
        self.item_state[it.key] = status
        if it.extra.get("soft") and status in ("failed", "blocked"):
            self.soft_failures.append(it.key)         # recorded and counted in the exit code, never blocks a step
            return
        order = {"interrupted": 4, "failed": 3, "blocked": 2, "ok": 1, "skipped": 0}
        cur = self.step_state.get(it.step)
        if cur is None or order[status] > order[cur]:
            self.step_state[it.step] = status

    def skip(self, it, reason, blocked_by=None):
        rec = {"event": "end", "step": it.step, "sub": it.sub, "kind": it.kind, "status": "skipped",
               "reason": reason, "utc_end": runlog.utc_now(), "argv": it.argv if it.kind != "health" else None}
        if blocked_by:
            rec["blocked_by"] = blocked_by
        with self.shielded():
            self.records.append(self.journal(rec))
            self.set_state(it, "blocked" if blocked_by else "skipped")
        self.say("%-34s skipped: %s" % (it.key, reason))
        self.raise_pending()

    def begin(self, it):
        self.journal({"event": "start", "step": it.step, "sub": it.sub, "kind": it.kind, "argv": it.argv,
                      "action": it.action, "cwd": it.cwd, "stdout": self.rel(it.stdout),
                      "stderr": self.rel(it.stderr), "utc_start": runlog.utc_now()})
        self.raise_pending()
        return runlog.utc_now(), time.monotonic()

    def end(self, it, t, status, exit_code=None, reason=None, outputs=(), **extra):
        utc_start, t0 = t
        outs = [p for p in outputs if p]
        sha = {}
        for p in outs:
            if os.path.isfile(p) and os.path.getsize(p) <= (256 << 20):
                sha[self.rel(p)] = sha256_file(p)
        rec = {"event": "end", "step": it.step, "sub": it.sub, "kind": it.kind, "argv": it.argv,
               "action": it.action, "cwd": it.cwd, "env": it.env or None, "utc_start": utc_start,
               "utc_end": runlog.utc_now(), "duration_s": round(time.monotonic() - t0, 3), "exit_code": exit_code,
               "status": status, "reason": reason, "outputs": [self.rel(p) for p in outs], "sha256": sha}
        rec.update(extra)
        with self.shielded():
            self.records.append(self.journal(rec))
            self.set_state(it, status)
        self.say("%-34s %s%s%s" % (it.key, status, "" if exit_code is None else " (exit %s)" % exit_code,
                                   (": " + reason) if reason else ""))
        if status != "interrupted":
            self.raise_pending()
        return rec

    # ------------------------------------------------------------------ signals
    def on_signal(self, signum, frame):
        if self.critical:
            self.pending_signal = self.pending_signal or signum
            return
        raise Interrupted(signum)

    @contextlib.contextmanager
    def shielded(self):
        """A signal that arrives inside is deferred (journal writes, engine start/stop, ending a tool)."""
        self.critical += 1
        try:
            yield
        finally:
            self.critical -= 1

    def raise_pending(self):
        if self.pending_signal and not self.critical and not self.finishing:
            sig, self.pending_signal = self.pending_signal, None
            raise Interrupted(sig)

    def install_signals(self):
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.saved_handlers[sig] = signal.signal(sig, self.on_signal)

    def restore_signals(self):
        for sig, h in self.saved_handlers.items():
            signal.signal(sig, h)
        self.saved_handlers = {}

    # ------------------------------------------------------------------ main
    def main(self):
        os.makedirs(os.path.join(self.cell.artifacts, "orchestrator"), exist_ok=True)
        with open(os.path.join(self.cell.artifacts, "orchestrator", "config.json"), "wb") as f:
            f.write(self.cell.config_bytes)
        with open(os.path.join(self.cell.artifacts, "orchestrator", "plan.txt"), "w", encoding="utf-8") as f:
            f.write(self.cell.plan_text(self.selected) + "\n")
        self.write_json(os.path.join(self.cell.artifacts, "orchestrator", "resolved.json"),
                        self.cell.plan_dict(self.selected))
        self.write_run_json("running")
        self.say("run directory: %s" % self.run_dir)
        if self.cell.mock:
            self.say("MOCK CONFIG: the 'engine' is tools/bench/mock_server.py; nothing in this run is evidence")
        self.install_signals()
        try:
            self.execute()
        finally:
            self.emergency_kill()
            self.restore_signals()
            code = self.exit_code()
            self.write_run_json("interrupted" if self.interrupted else
                                ("completed_with_failures" if code else "completed"), code)
        self.summary()
        return code

    def execute(self):
        head = [it for it in self.cell.plan if it.step not in TAIL]
        tail = [it for it in self.cell.plan if it.step in TAIL]
        done = 0
        try:
            for it in head:
                done += 1
                self.do(it)
        except KeyboardInterrupt as e:
            self.interrupted = getattr(e, "signum", signal.SIGINT)
            cur = self.current
            if cur is not None and cur.key not in self.item_state:
                with self.shielded():
                    self.records.append(self.journal({
                        "event": "end", "step": cur.step, "sub": cur.sub, "kind": cur.kind, "argv": cur.argv,
                        "status": "interrupted", "exit_code": None, "utc_end": runlog.utc_now(),
                        "reason": "interrupted by %s" % signame(self.interrupted)}))
                    self.set_state(cur, "interrupted")
            self.say("interrupted by %s: stopping the engine, skipping the remaining steps" % signame(self.interrupted))
            for it in head[done:]:
                self.skip(it, "not run: interrupted by %s" % signame(self.interrupted))
        finally:
            self.finish(tail)

    def finish(self, tail):
        self.finishing = True
        for it in tail:
            try:
                if self.pending_signal and not self.interrupted:
                    self.interrupted = self.pending_signal
                if it.step == "engine_stop":
                    self.do(it)
                elif self.interrupted:
                    self.skip(it, "not run: interrupted by %s" % signame(self.interrupted))
                else:
                    self.do(it)
            except KeyboardInterrupt as e:
                self.interrupted = self.interrupted or getattr(e, "signum", signal.SIGINT)
        if self.pending_signal and not self.interrupted:
            self.interrupted = self.pending_signal

    def do(self, it):
        self.current = it
        if it.step not in self.selected:
            return self.skip(it, "not selected (--only)")
        if it.skip:
            return self.skip(it, it.skip)
        for dep in DEPENDS.get(it.step, ()):
            st = self.step_state.get(dep)
            if st in ("failed", "interrupted", "blocked"):
                return self.skip(it, "depends on %s, which %s" % (dep, "was skipped (dependency failed)"
                                                                  if st == "blocked" else st), blocked_by=dep)
        if it.after and self.item_state.get(it.after) != "ok":
            return self.skip(it, "depends on %s, which is %s" % (it.after, self.item_state.get(it.after)),
                             blocked_by=it.after)
        handler = {"cmd": self.run_cmd, "engine": self.start_engine, "health": self.wait_health,
                   "stop": self.stop_engine, "prereg": self.write_prereg, "buildcheck": self.check_build}[it.kind]
        n = len(self.records)
        try:
            return handler(it)
        except Exception as e:  # noqa: BLE001 - an orchestrator bug fails this step, not the whole run
            if it.kind == "stop":
                self.emergency_kill()
            if len(self.records) == n:
                rec = {"event": "end", "step": it.step, "sub": it.sub, "kind": it.kind, "argv": it.argv,
                       "status": "failed", "reason": "orchestrator error: %s: %s" % (type(e).__name__, e),
                       "utc_end": runlog.utc_now(), "exit_code": None}
                self.records.append(self.journal(rec))
                self.set_state(it, "failed")
                self.say("%-34s failed: orchestrator error: %s: %s" % (it.key, type(e).__name__, e))

    def exit_code(self):
        if self.interrupted:
            return 128 + int(self.interrupted)
        bad = [s for s, st in self.step_state.items() if st in ("failed", "blocked", "interrupted")]
        return 1 if bad or self.soft_failures else 0

    # ------------------------------------------------------------------ actions
    def run_pre(self, it):
        for p in it.pre:
            if p[0] == "mkdir":
                os.makedirs(p[1], exist_ok=True)
            elif p[0] == "copy":
                os.makedirs(os.path.dirname(p[2]), exist_ok=True)
                shutil.copyfile(p[1], p[2])
            elif p[0] == "write":
                self.write_generated(it, p[1])

    def write_generated(self, it, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base = os.path.basename(path)
        if it.step == "vs_roofline":
            with open(path, "w", encoding="utf-8") as f:
                f.write(shell(it.argv) + " > " + it.stdout + "\n")
        elif it.step == "engine_start" and base == "argv.json":
            self.write_json(path, {
                "argv": it.argv, "cwd": it.cwd, "env_overrides": it.env, "utc": runlog.utc_now(),
                "inherited_env_of_interest": {k: v for k, v in sorted(os.environ.items()) if ENV_OF_INTEREST.match(k)},
                "engine": self.cell.engine["name"], "commit": self.cell.engine["commit"],
                "pin": self.cell.engine["pin"] or None, "engine_desc": self.cell.engine_desc})
        elif it.step == "engine_start":
            content = self.cell.engine["config_files"][os.path.relpath(path, self.cell.engine_dir)]
            with open(path, "w", encoding="utf-8") as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=2)
                    f.write("\n")

    def run_cmd(self, it):
        argv = it.argv
        if it.extra.get("needs_binary") and not shutil.which(argv[0]):
            return self.skip(it, "%s not found on this host" % argv[0])
        if it.step == "vs_roofline":
            have = [d for d in it.extra["matrix_dirs"] if os.path.isfile(os.path.join(d, "summary.json"))
                    and os.path.isfile(os.path.join(d, "meta.json"))]
            if not have:
                return self.skip(it, "no matrix run directory with summary.json and meta.json")
            argv = [a for a in argv if a not in it.extra["matrix_dirs"] or a in have]
            it.argv = argv
        t = self.begin(it)
        try:
            self.run_pre(it)
        except OSError as e:
            return self.end(it, t, "failed", reason="preparation failed: %s" % e)
        for p in (it.stdout, it.stderr):
            if p:
                os.makedirs(os.path.dirname(p), exist_ok=True)
        target = it.stdout + ".part" if it.json_out else it.stdout
        timed_out, rc = False, None
        with open(target, "wb") as fo, (open(it.stderr, "wb") if it.stderr else contextlib.nullcontext()) as fe:
            try:
                p = subprocess.Popen(argv, stdout=fo, stderr=fe if it.stderr else subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=it.cwd,
                                     env=dict(os.environ, **it.env) if it.env else None, start_new_session=True)
            except OSError as e:
                fo.close()
                self.settle_json(it, target)
                return self.end(it, t, "failed", reason="cannot start: %s" % e, outputs=[it.stdout, it.stderr])
            self.child = p
            try:
                rc = p.wait(timeout=it.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = self.end_child(p)
            except KeyboardInterrupt as e:
                rc = self.end_child(p)
                fo.close()
                outs = [self.settle_json(it, target), it.stderr] + self.extra_outputs(it)
                self.end(it, t, "interrupted", rc, reason="interrupted by %s; the tool got SIGINT"
                         % signame(getattr(e, "signum", signal.SIGINT)), outputs=outs)
                raise
            finally:
                self.child = None
        out = self.settle_json(it, target)
        status = "ok" if rc in it.ok_codes and not timed_out else "failed"
        reason = None
        if timed_out:
            reason = "timeout after %g s: SIGINT, then SIGTERM/SIGKILL" % it.timeout
        elif status == "failed":
            reason = "exit code %s (accepted: %s)" % (rc, ", ".join(str(c) for c in it.ok_codes))
        if it.json_out and out and not out.endswith(".json"):
            reason = (reason + "; " if reason else "") + "stdout is not JSON, kept as %s" % self.rel(out)
            status = "failed"
        extra = {}
        if it.extra.get("flags_check"):
            extra = self.flags_check(it)
        if it.step == "pack" and it.sub == "build":
            extra = self.read_verdict(it)
        if it.step == "quant_audit" and out and out.endswith(".json"):
            try:
                with open(out, encoding="utf-8") as f:
                    extra = {"verdict": json.load(f).get("verdict")}
            except (OSError, ValueError, AttributeError):
                pass
        return self.end(it, t, status, rc, reason, outputs=[out, it.stderr] + self.extra_outputs(it), **extra)

    def extra_outputs(self, it):
        outs = []
        for p in it.pre:
            if p[0] == "copy":
                outs.append(p[2])
            elif p[0] == "write":
                outs.append(p[1])
        if it.step == "slot_persist" and it.sub == "persist":
            outs.append(self.cell.a("kv", "slot_persist.json"))
        if it.step == "slot_persist" and it.sub == "prompt":
            outs.append(self.cell.a("kv", "prompt-fit.json"))
        if it.step == "logprob_compare":
            outs.append(self.cell.a("quality", "logprob_compare.json"))
        for k in ("run_dir", "pack_dir"):
            if it.extra.get(k):
                outs.append(it.extra[k])
        return [p for p in outs if os.path.exists(p)]

    def settle_json(self, it, target):
        """json_out: keep stdout as the .json file only if it parses; otherwise as .stdout.txt (the pack would call a
        broken .json unparseable and refuse a PASS for the whole pack; a missing artifact is an honest NOT RUN)."""
        if not it.json_out:
            return it.stdout
        try:
            with open(target, encoding="utf-8") as f:
                json.load(f)
            os.replace(target, it.stdout)
            return it.stdout
        except (OSError, ValueError):
            alt = it.stdout[:-len(".json")] + ".stdout.txt"
            if os.path.exists(target):
                os.replace(target, alt)
                return alt
            return None

    def end_child(self, p):
        """SIGINT (the bench tools then write their run directory as interrupted), SIGTERM, SIGKILL."""
        with self.shielded():
            for sig, wait in ((signal.SIGINT, self.cell.child_grace_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 10.0)):
                if p.poll() is not None:
                    break
                try:
                    os.killpg(p.pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        return p.poll()

    def flags_check(self, it):
        try:
            with open(it.stdout, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return {}
        flags = []
        for a in self.cell.engine["argv"][1:]:
            m = FLAG_RE.match(a)
            if m and m.group(1) not in flags:
                flags.append(m.group(1))
        ignore = set(self.cell.engine["help_ignore"])
        missing = [f for f in flags if f not in ignore and not re.search(
            r"(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])" % re.escape(f), text)]
        path = self.cell.a("engine", "flags_check.json")
        self.write_json(path, {"help_argv": it.argv, "flags_checked": flags, "ignored": sorted(ignore),
                               "missing_from_help": missing,
                               "_note": "a flag of the engine argv that --help does not mention is a typo or a version "
                                        "difference: check it before trusting the run"})
        if missing:
            self.say("WARNING: engine flags not found in --help: %s" % " ".join(missing))
        return {"flags_missing_from_help": missing}

    def read_verdict(self, it):
        try:
            with open(os.path.join(it.extra["pack_dir"], "verdict.json"), encoding="utf-8") as f:
                v = json.load(f)
        except (OSError, ValueError):
            return {}
        crit = v.get("criteria_results") or []
        self.pack_verdict = {
            "overall": v.get("overall"), "overall_reason": v.get("overall_reason"),
            "runs_dir": v.get("runs_dir"),
            "by_status": {s: sorted(c["id"] for c in crit if c.get("status") == s) for s in ("PASS", "FAIL", "NOT RUN")},
            "mock_criteria": sorted(c["id"] for c in crit if c.get("reason_code") == "mock"),
            "synthetic_markers": len(v.get("synthetic_markers") or []),
            "verdict_md": self.rel(os.path.join(it.extra["pack_dir"], "VERDICT.md"))}
        return {"verdict": self.pack_verdict["overall"], "verdict_reason": self.pack_verdict["overall_reason"],
                "mock_criteria": self.pack_verdict["mock_criteria"]}

    def write_prereg(self, it):
        t = self.begin(it)
        preds, missing = [], []
        for ctx, path in it.extra["predictions"]:
            if not (os.path.isfile(path) and path.endswith(".json")):
                missing.append(ctx)
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            rows = []
            for rep in data if isinstance(data, list) else []:
                for d in rep.get("decode") or []:
                    rows.append({k: d.get(k) for k in ("strategy", "miss", "hit_vram", "overlap", "tok_s",
                                                        "tok_s_envelope", "envelope_hit_vram", "tokens_per_step")})
            preds.append({"ctx": ctx, "file": self.rel(path), "sha256": sha256_file(path), "decode": rows,
                          "ttft_s_bound": [(rep.get("ttft") or {}).get("ttft_s_bound") for rep in data]
                          if isinstance(data, list) else None})
        planned = {s: [x.argv for x in self.cell.plan if x.step == s and not x.skip and x.step in self.selected]
                   for s in ("matrix", "probes", "slot_persist", "logprob_compare", "vs_roofline")}
        rec = {"_label": "PREREGISTRATION written by scripts/r1/run_r1.py before the engine started. The predictions are "
                         "PROVISIONAL bounds of tools/roofline.py (memory-bandwidth model: no kernel launch overheads, "
                         "no GPU compute limit on decode, stated efficiencies; see its docstring), not expectations.",
               "utc": runlog.utc_now(), "config": {"name": self.cell.name, "path": self.cell.config_path,
                                                   "sha256": hashlib.sha256(self.cell.config_bytes).hexdigest()},
               "repo_git": runlog.git_info(), "engine": {"name": self.cell.engine["name"],
                                                         "commit": self.cell.engine["commit"],
                                                         "argv": self.cell.engine["argv"]},
               "model": {"desc": self.cell.model["desc"], "files": self.cell.model["files"]},
               "mode": self.cell.mode, "roofline_inputs": self.cell.roofline, "predictions": preds,
               "missing_predictions": missing, "planned_commands": planned}
        try:
            self.write_json(it.stdout, rec)
        except OSError as e:
            return self.end(it, t, "failed", reason="cannot write %s: %s" % (self.rel(it.stdout), e))
        return self.end(it, t, "failed" if missing or not preds else "ok",
                        reason=("no prediction for ctx %s" % missing) if missing else None,
                        outputs=[it.stdout] + [p for _c, p in it.extra["predictions"] if os.path.isfile(p)])

    def check_build(self, it):
        """engine_build_check: the build the engine runs from is the pinned one, for this GPU architecture, unchanged."""
        t = self.begin(it)
        e, pin = self.cell.engine, self.cell.pin_spec
        path = it.extra["build_info"]
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
            if not isinstance(info, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as err:
            return self.end(it, t, "failed", reason="cannot read %s: %s" % (path, err))
        copy = os.path.join(self.cell.engine_dir, "build-info.json")
        os.makedirs(self.cell.engine_dir, exist_ok=True)
        shutil.copyfile(path, copy)
        build = os.path.dirname(os.path.abspath(path))
        checks = []

        def chk(name, ok, got, want):
            checks.append({"check": name, "ok": bool(ok), "got": got, "want": want})

        for k in ("engine", "repo", "sha"):
            want = e["pin"] if k == "engine" else pin[k]
            chk(k, info.get(k) == want, info.get(k), want)
        chk("checked_out", info.get("checked_out") == pin["sha"], info.get("checked_out"), pin["sha"])
        want_deps = {k: d["sha"] for k, d in (pin.get("deps") or {}).items()}
        deps = info.get("deps")
        got_deps = ({k: (d or {}).get("sha") if isinstance(d, dict) else d for k, d in deps.items()}
                    if isinstance(deps, dict) else deps)
        chk("deps", got_deps == want_deps, got_deps, want_deps)
        st = info.get("status")
        chk("status", isinstance(st, str) and st.startswith("ok"), st, "ok")
        chk("arch", info.get("arch") == e["arch"], info.get("arch"), e["arch"])
        sm = "sm_%d" % e["arch"]
        cub = info.get("cubin_archs")
        with_sm = sorted(f for f, v in cub.items() if isinstance(v, list) and sm in v) if isinstance(cub, dict) else []
        chk("cubins", with_sm, with_sm if isinstance(cub, dict) else cub, "a built file with %s code" % sm)
        bins = info.get("bins")
        chk("binaries", isinstance(bins, dict) and sorted(bins) == sorted(pin["bins"]) and all(bins.values()),
            bins, pin["bins"])
        files = info.get("files_sha256")
        if not isinstance(files, dict) or not files:
            chk("files_sha256", False, files, "sha256 of every built file (scripts/engines/build_engine.py writes it)")
        else:
            changed = []
            for rel, h in sorted(files.items()):
                p = os.path.join(build, rel)
                got = sha256_file(p) if os.path.isfile(p) else None
                if got != h:
                    changed.append({"file": rel, "got": got, "want": h})
            chk("files_sha256", not changed, changed or "%d files unchanged" % len(files),
                "every built file unchanged since the build")
        binary = it.extra.get("binary")
        if binary:
            rel = os.path.relpath(os.path.realpath(binary), os.path.realpath(build))
            chk("binary", rel in pin["bins"] and isinstance(files, dict) and rel in files, rel,
                "one of %s of the build in %s" % (pin["bins"], build))
        src = it.extra.get("source_dir")
        if src:
            try:
                p = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
                head = p.stdout.strip() if p.returncode == 0 else "git failed: " + p.stderr.strip()[-200:]
            except (OSError, subprocess.SubprocessError) as err:
                head = "git failed: %s" % err
            chk("source_head", head == pin["sha"], head, pin["sha"])
        bad = [c["check"] for c in checks if not c["ok"]]
        self.write_json(it.stdout, {"_label": "engine_build_check of scripts/r1/run_r1.py: the build-info.json of "
                                              "scripts/engines/build_engine.py against tools/engines/pins.json, before "
                                              "the engine starts", "utc": runlog.utc_now(), "build_info": path,
                                    "pin": e["pin"], "pin_sha": pin["sha"], "arch": e["arch"], "checks": checks,
                                    "ok": not bad})
        return self.end(it, t, "failed" if bad else "ok", outputs=[it.stdout, copy], checks_failed=bad,
                        reason=("the build does not match the pin: " + ", ".join(bad)) if bad else None)

    def start_engine(self, it):
        t = self.begin(it)
        try:
            self.run_pre(it)
        except (OSError, KeyError, TypeError, ValueError) as e:
            return self.end(it, t, "failed", reason="preparation failed: %s" % e)
        os.makedirs(os.path.dirname(it.stdout), exist_ok=True)
        log = open(it.stdout, "ab")
        with self.shielded():                      # no signal between the fork and self.engine = p
            try:
                p = subprocess.Popen(it.argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=it.cwd, env=dict(os.environ, **it.env), start_new_session=True)
            except OSError as e:
                p = e
            if not isinstance(p, OSError):
                self.engine, self.engine_log = p, log
        if isinstance(p, OSError):
            log.close()
            return self.end(it, t, "failed", reason="cannot start the engine: %s" % p,
                            outputs=[it.stdout] + self.extra_outputs(it))
        return self.end(it, t, "ok", reason="started (pid %d); health is checked by health_wait" % p.pid,
                        outputs=self.extra_outputs(it), pid=p.pid, pgid=p.pid)

    def wait_health(self, it):
        t = self.begin(it)
        url, expect = it.extra["url"], int(it.extra["expect_status"])
        timeout, interval = float(it.timeout), float(it.extra["interval_s"])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        t0, polls, last = time.monotonic(), 0, None
        while True:
            if self.engine is not None and self.engine.poll() is not None:
                return self.end(it, t, "failed", reason="the engine exited with code %s before %s answered %d (see %s)"
                                % (self.engine.returncode, url, expect, self.rel(self.cell.a("engine", "engine.log"))),
                                polls=polls, last=last, engine_exit_code=self.engine.returncode)
            remaining = timeout - (time.monotonic() - t0)
            polls += 1
            code = None
            try:
                with opener.open(url, timeout=max(0.2, min(5.0, remaining))) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
                e.close()
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = "%s: %s" % (type(e).__name__, getattr(e, "reason", e))
            if code is not None:
                last = "HTTP %d" % code
            if code == expect:
                return self.end(it, t, "ok", reason="%s answered HTTP %d after %.1f s" % (url, code, time.monotonic() - t0),
                                polls=polls, last=last)
            remaining = timeout - (time.monotonic() - t0)
            if remaining <= 0:
                return self.end(it, t, "failed", reason="timeout after %g s; last answer: %s" % (timeout, last),
                                polls=polls, last=last)
            time.sleep(min(interval, remaining))

    def stop_engine(self, it):
        p = self.engine
        if p is None:
            return self.skip(it, "the engine was not started by this run")
        t = self.begin(it)
        self.critical += 1                         # SIGINT/SIGTERM/SIGHUP are deferred until the engine is gone
        signals, leftovers, reason, status = [], False, None, "ok"
        try:
            if p.poll() is not None:
                status = "failed"
                reason = "the engine had already exited with code %s before the stop (crash or early exit; see %s)" % (
                    p.returncode, self.rel(it.extra.get("log") or self.cell.a("engine", "engine.log")))
            else:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    signals.append("SIGTERM")
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    p.wait(timeout=float(it.timeout))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(p.pid, signal.SIGKILL)
                    signals.append("SIGKILL")
                    p.wait(timeout=30)
                    reason = "did not exit within %g s of SIGTERM: killed" % float(it.timeout)
            try:
                os.killpg(p.pid, 0)
                leftovers = True
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        except BaseException:
            self.emergency_kill()
            raise
        finally:
            self.critical -= 1                     # (in the finish phase a deferred signal is not re-raised)
            if self.engine_log:
                self.engine_log.close()
                self.engine_log = None
        if leftovers:
            reason = (reason + "; " if reason else "") + "processes left in the engine's group were killed"
        rec = self.end(it, t, status, p.returncode, reason, outputs=[self.cell.a("engine", "engine.log")],
                       pid=p.pid, signals_sent=signals, group_leftovers_killed=leftovers)
        self.write_json(self.cell.a("engine", "exit.json"),
                        {k: rec.get(k) for k in ("pid", "exit_code", "signals_sent", "group_leftovers_killed",
                                                 "utc_start", "utc_end", "status", "reason")})
        return rec

    def emergency_kill(self):
        p = self.engine
        if p is not None and p.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=30)
        c = self.child
        if c is not None and c.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(c.pid, signal.SIGKILL)

    # ------------------------------------------------------------------ outputs
    @staticmethod
    def write_json(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def write_run_json(self, status, code=None):
        steps = []
        for r in self.records:
            steps.append({k: r.get(k) for k in ("step", "sub", "status", "exit_code", "reason", "blocked_by")})
        self.write_json(os.path.join(self.run_dir, "run.json"), {
            "schema": SCHEMA, "name": self.cell.name, "label": self.cell.label, "mock": self.cell.mock,
            "status": status, "exit_code": code, "interrupted_by": signame(self.interrupted) if self.interrupted else None,
            "utc_start": self.utc_start, "utc_end": runlog.utc_now() if code is not None else None,
            "orchestrator": {"argv": self.argv, "python": sys.version.split()[0], "git": runlog.git_info(),
                             "host": runlog.host_info()},
            "config": {"path": self.cell.config_path, "sha256": hashlib.sha256(self.cell.config_bytes).hexdigest()},
            "selected": sorted(self.selected), "step_states": self.step_state, "steps": steps,
            "pack": self.pack_verdict, "journal": "steps.jsonl", "plan": "artifacts/orchestrator/resolved.json"})

    def summary(self):
        self.say("steps: " + ", ".join("%s=%s" % (s, self.step_state.get(s, "-")) for s in STEPS))
        if self.pack_verdict:
            pv = self.pack_verdict
            self.say("verification pack: %s (%s) -> %s" % (pv["overall"], (pv["overall_reason"] or "")[:160],
                                                           pv["verdict_md"]))
            if pv["mock_criteria"]:
                self.say("criteria NOT RUN because the server was the mock: %s" % ", ".join(pv["mock_criteria"]))
        else:
            self.say("verification pack: not built")
        self.say("journal: %s" % self.journal_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="scripts/r1/configs/<name>.json")
    ap.add_argument("--out", required=True, metavar="RUNS_ROOT",
                    help="the run directory RUNS_ROOT/<UTC>-<config name>/ is created inside")
    ap.add_argument("--plan", action="store_true", help="print every command that would run; run nothing")
    ap.add_argument("--json", action="store_true", help="with --plan: print the plan as JSON")
    ap.add_argument("--only", default=None, help="comma list of steps to run (engine_start implies engine_stop): "
                    + ",".join(STEPS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a variable of the config (repeatable), e.g. --set MODEL_DIR=/srv/llm/models/x")
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    try:
        selected = parse_only(a.only)
        cell = Cell(a.config, a.out, a.set)
    except ConfigError as e:
        print("run_r1: config error: %s" % e, file=sys.stderr)
        return 2
    if a.plan:
        print(json.dumps(cell.plan_dict(selected), indent=1) if a.json else cell.plan_text(selected))
        return 0
    try:
        os.makedirs(cell.runs_root, exist_ok=True)
        os.makedirs(cell.run_dir)                    # reserve; never reuse an existing run directory
    except OSError as e:
        print("run_r1: cannot create the run directory: %s" % e, file=sys.stderr)
        return 2
    return Run(cell, selected, argv=["scripts/r1/run_r1.py"] + argv).main()


if __name__ == "__main__":
    sys.exit(main())
