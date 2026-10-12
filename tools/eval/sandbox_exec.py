#!/usr/bin/env python3
"""Exec wrapper for code-task test commands (stdlib only, Linux).

    python3 -I sandbox_exec.py --root DIR [limits] -- CMD [ARGS...]
    python3 -I sandbox_exec.py --probe            # print {"landlock_abi": N} and exit

Before exec'ing CMD (cwd = --cwd, default --root/repo if it exists, else --root) it:
  1. sets resource limits: RLIMIT_CORE 0, RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE (and RLIMIT_NPROC
     only when --nproc is given; it is per-user and unreliable as a sandbox limit);
  2. sets PR_SET_NO_NEW_PRIVS;
  3. applies a Landlock ruleset (Linux >= 5.13, unprivileged): every file-system WRITE right (write, truncate,
     create, remove, rename/link, mknod, symlink) is allowed only beneath --root and on /dev/null; reads and
     execution are not restricted. With Landlock ABI >= 4 (Linux >= 6.7) TCP bind and connect are denied too.
  4. writes one line "@@EVAL-SANDBOX@@ {json}" to stderr describing what was applied, then exec's CMD.
Network isolation by namespace (unshare) is done by the caller (ev_sandbox.py), outside this wrapper.
Landlock UAPI constants: include/uapi/linux/landlock.h (Linux 6.x); syscalls 444/445/446 are the same on
x86_64 and aarch64 (generic syscall table).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import sys

MARK = "@@EVAL-SANDBOX@@"
SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
# access_fs bits
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
REFER, TRUNCATE = 1 << 13, 1 << 14            # ABI 2, ABI 3
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1  # ABI 4


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def landlock_abi():
    try:
        libc = _libc()
        v = libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
        return int(v) if v > 0 else 0
    except (OSError, AttributeError):
        return 0


def apply_landlock(root, block_tcp=True):
    """Returns dict(fs=bool, net=bool, abi=int, error=str|None)."""
    out = {"abi": landlock_abi(), "fs": False, "net": False, "error": None}
    abi = out["abi"]
    if abi < 1:
        out["error"] = "landlock not available"
        return out
    libc = _libc()
    write = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_CHAR | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO \
        | MAKE_BLOCK | MAKE_SYM
    if abi >= 2:
        write |= REFER
    if abi >= 3:
        write |= TRUNCATE
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if (abi >= 4 and block_tcp) else 0
    attr = RulesetAttr(write, net)
    size = ctypes.sizeof(RulesetAttr) if net else 8           # ABI < 4 kernels only know the fs field
    rfd = libc.syscall(SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if rfd < 0:
        out["error"] = "landlock_create_ruleset: errno %d" % ctypes.get_errno()
        return out
    try:
        rules = [(root, write)]
        file_rights = WRITE_FILE | (TRUNCATE if abi >= 3 else 0)
        if os.path.exists("/dev/null"):
            rules.append(("/dev/null", file_rights))
        for path, rights in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                pb = PathBeneathAttr(rights, fd)
                r = libc.syscall(SYS_ADD_RULE, ctypes.c_int(rfd), ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                                 ctypes.byref(pb), ctypes.c_uint32(0))
                if r != 0:
                    out["error"] = "landlock_add_rule(%s): errno %d" % (path, ctypes.get_errno())
                    return out
            finally:
                os.close(fd)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            out["error"] = "prctl(NO_NEW_PRIVS): errno %d" % ctypes.get_errno()
            return out
        if libc.syscall(SYS_RESTRICT, ctypes.c_int(rfd), ctypes.c_uint32(0)) != 0:
            out["error"] = "landlock_restrict_self: errno %d" % ctypes.get_errno()
            return out
        out["fs"] = True
        out["net"] = bool(net)
    finally:
        os.close(rfd)
    return out


def set_limits(a):
    applied = {}

    def lim(name, res, val):
        if val is None or val <= 0:
            return
        try:
            soft, hard = resource.getrlimit(res)
            new_hard = val if hard == resource.RLIM_INFINITY or val < hard else hard
            resource.setrlimit(res, (min(val, new_hard), new_hard))
            applied[name] = min(val, new_hard)
        except (ValueError, OSError) as e:
            applied[name] = "failed: %s" % e
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        applied["core"] = 0
    except (ValueError, OSError) as e:
        applied["core"] = "failed: %s" % e
    lim("cpu_s", resource.RLIMIT_CPU, a.cpu)
    lim("as_bytes", resource.RLIMIT_AS, a.as_mb * 1024 * 1024 if a.as_mb else None)
    lim("fsize_bytes", resource.RLIMIT_FSIZE, a.fsize_mb * 1024 * 1024 if a.fsize_mb else None)
    lim("nofile", resource.RLIMIT_NOFILE, a.nofile)
    if a.nproc:
        lim("nproc", resource.RLIMIT_NPROC, a.nproc)
    return applied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--cwd")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--as-mb", type=int, default=0)
    ap.add_argument("--fsize-mb", type=int, default=0)
    ap.add_argument("--nofile", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--no-landlock", action="store_true", help="skip Landlock (reported in the sandbox line)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.probe:
        print(json.dumps({"landlock_abi": landlock_abi()}))
        return 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not a.root or not cmd:
        ap.error("--root and a command after -- are required")
    root = os.path.realpath(a.root)
    cwd = a.cwd or (os.path.join(root, "repo") if os.path.isdir(os.path.join(root, "repo")) else root)
    report = {"rlimits": set_limits(a)}
    report["landlock"] = {"fs": False, "net": False, "abi": landlock_abi(), "error": "disabled (--no-landlock)"} \
        if a.no_landlock else apply_landlock(root)
    report["pid"] = os.getpid()
    sys.stderr.write("%s %s\n" % (MARK, json.dumps(report, sort_keys=True)))
    sys.stderr.flush()
    try:
        os.chdir(cwd)
        os.execvpe(cmd[0], cmd, os.environ)
    except OSError as e:
        sys.stderr.write("sandbox_exec: cannot exec %r: %s\n" % (cmd[0], e))
        return 127


if __name__ == "__main__":
    sys.exit(main())
