"""Soft network guard for Python test commands run by tools/eval (defence in depth only).

Loaded through PYTHONPATH by ev_sandbox.run_command when EVAL_SANDBOX_PY_GUARD=1. It refuses IPv4/IPv6
connect/sendto from Python code. It is NOT a security boundary (a subprocess or C code bypasses it);
the real isolation is the network namespace and Landlock, see ev_sandbox.py.
"""
import os as _os

if _os.environ.get("EVAL_SANDBOX_PY_GUARD") == "1":
    import socket as _socket

    _INET = (_socket.AF_INET, _socket.AF_INET6)
    _orig_connect = _socket.socket.connect
    _orig_connect_ex = _socket.socket.connect_ex
    _orig_sendto = _socket.socket.sendto

    def _deny(what):
        raise PermissionError("network disabled by the tools/eval sandbox (python guard): %s" % what)

    def connect(self, address):
        if self.family in _INET:
            _deny("connect %r" % (address,))
        return _orig_connect(self, address)

    def connect_ex(self, address):
        if self.family in _INET:
            _deny("connect_ex %r" % (address,))
        return _orig_connect_ex(self, address)

    def sendto(self, *args):
        if self.family in _INET:
            _deny("sendto")
        return _orig_sendto(self, *args)

    _socket.socket.connect = connect
    _socket.socket.connect_ex = connect_ex
    _socket.socket.sendto = sendto
