#!/usr/bin/env python3
"""Stdlib parser for the GGUF header (KV metadata + tensor infos) of a byte buffer that may end early.

Used by scripts/gguf_headers_fetch.py (CI, HTTP Range reads: the buffer grows until the header parses) and by
tools/gguf_headers/analyze.py (offline, on the committed header bytes). Never reads tensor data.

Layout and checks follow llama.cpp ggml/src/gguf.cpp gguf_init_from_reader @ec7630a640789c393694fb194f1bbbf0369fc62d
(the same at 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6):
  magic "GGUF", u32 version (2 or 3), i64 n_tensors, i64 n_kv;
  n_kv x (string key, u32 value type, value); an array is u32 element type, u64 count, elements (no nested arrays);
  n_tensors x (string name < 64 bytes, u32 n_dims <= 4, n_dims x i64 ne, i32 ggml type, u64 offset);
  general.alignment (u32, power of two, default 32); the data section starts at the next multiple of the alignment
  after the tensor infos if n_tensors > 0, else right after them (gguf.cpp:781-788); tensor offsets are relative to the
  data section and must be the running sum of the padded tensor sizes (gguf.cpp:790-816).
A string is u64 length + bytes. Everything little endian.

parse(buf) raises NeedMore(n) when buf holds fewer than the n bytes the next step needs (n counts from the start of the
file), ValueError for a malformed header. The returned Header covers buf[:data_offset] exactly.
"""
import hashlib
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gguf_quant_audit import GGML_TYPES  # noqa: E402  (type id -> (name, block size, type size), checked vs gguf-py)

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
MAX_DIMS = 4             # GGML_MAX_DIMS
MAX_NAME = 64            # GGML_MAX_NAME: a tensor name must be shorter
MAX_STRING = 1 << 30     # sanity bound, not a format limit
MAX_COUNT = 1 << 40      # sanity bound for n_kv / n_tensors / array counts

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, \
    T_FLOAT64 = range(13)
TYPE_NAMES = {0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32", 6: "float32", 7: "bool",
              8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64"}
SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class NeedMore(Exception):
    """The buffer ends before the header does; `need` is the smallest total buffer length that lets parsing go on."""

    def __init__(self, need):
        super().__init__(f"need at least {need} bytes")
        self.need = need


class _Cursor:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def take(self, n):
        end = self.p + n
        if end > len(self.b):
            raise NeedMore(end)
        v = self.b[self.p:end]
        self.p = end
        return v

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        n = self.unpack("Q")
        if n > MAX_STRING:
            raise ValueError(f"implausible string length {n} at offset {self.p - 8}")
        return bytes(self.take(n))


def align_up(x, a):
    return (x + a - 1) // a * a


def tensor_nbytes(dims, type_id):
    """ggml_nbytes of a contiguous tensor; None for a type id the table does not know."""
    if type_id not in GGML_TYPES:
        return None
    _, block, size = GGML_TYPES[type_id]
    n = 1
    for d in dims:
        n *= d
    return n // block * size


class Header:
    __slots__ = ("version", "n_tensors", "n_kv", "kv", "tensors", "alignment", "infos_end", "data_offset")

    def get(self, key, default=None):
        for e in self.kv:
            if e["key"] == key:
                return e["value"]
        return default


def _value(c, vtype, in_array=False):
    if vtype in SCALAR_FMT:
        return c.unpack(SCALAR_FMT[vtype])
    if vtype == T_STRING:
        return c.string().decode("utf-8", errors="surrogateescape")
    if vtype == T_ARRAY and not in_array:
        etype = c.unpack("I")
        count = c.unpack("Q")
        if count > MAX_COUNT:
            raise ValueError(f"implausible array length {count}")
        if etype in SCALAR_FMT:
            fmt = SCALAR_FMT[etype]
            raw = c.take(count * struct.calcsize("<" + fmt))
            return etype, list(struct.unpack(f"<{count}{fmt}", raw))
        if etype == T_STRING:
            return etype, [_value(c, T_STRING, True) for _ in range(count)]
        raise ValueError(f"array element type {etype} is not allowed (gguf.cpp rejects nested arrays)")
    raise ValueError(f"unknown GGUF value type {vtype}")


def parse(buf):
    """Parse the header at the start of buf (bytes / bytearray / memoryview). See the module docstring."""
    c = _Cursor(memoryview(buf))
    if bytes(c.take(4)) != MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    h = Header()
    h.version = c.unpack("I")
    if h.version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {h.version}")
    h.n_tensors = c.unpack("q")
    h.n_kv = c.unpack("q")
    if not (0 <= h.n_tensors <= MAX_COUNT and 0 <= h.n_kv <= MAX_COUNT):
        raise ValueError(f"implausible counts n_tensors={h.n_tensors} n_kv={h.n_kv}")
    h.kv = []
    seen = set()
    for _ in range(h.n_kv):
        key = c.string().decode("utf-8")
        if not key or key in seen:
            raise ValueError(f"empty or duplicate key {key!r}")
        seen.add(key)
        start = c.p
        vtype = c.unpack("I")
        e = {"key": key, "type": vtype}
        if vtype == T_ARRAY:
            e["elem_type"], e["value"] = _value(c, T_ARRAY)
        else:
            e["value"] = _value(c, vtype)
        e["span"] = (start, c.p)
        h.kv.append(e)
    h.alignment = DEFAULT_ALIGNMENT
    for e in h.kv:
        if e["key"] == "general.alignment":
            if e["type"] != T_UINT32:
                raise ValueError("general.alignment must be uint32")
            h.alignment = e["value"]
            if h.alignment == 0 or h.alignment & (h.alignment - 1):
                raise ValueError(f"alignment {h.alignment} is not a power of 2")
    h.tensors = []
    names = set()
    for _ in range(h.n_tensors):
        name = c.string().decode("utf-8")
        if len(name.encode()) >= MAX_NAME or name in names:
            raise ValueError(f"tensor name too long or duplicate: {name!r}")
        names.add(name)
        n_dims = c.unpack("I")
        if n_dims > MAX_DIMS:
            raise ValueError(f"tensor {name}: {n_dims} dimensions")
        dims = [c.unpack("q") for _ in range(n_dims)]
        if any(d < 0 for d in dims):
            raise ValueError(f"tensor {name}: negative dimension {dims}")
        ttype = c.unpack("i")
        if ttype in GGML_TYPES and dims and dims[0] % GGML_TYPES[ttype][1]:
            raise ValueError(f"tensor {name}: row of {dims[0]} is not a multiple of the {GGML_TYPES[ttype][0]} block")
        offset = c.unpack("Q")
        h.tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    h.infos_end = c.p
    h.data_offset = align_up(c.p, h.alignment) if h.n_tensors > 0 else c.p
    if h.data_offset > len(buf):
        raise NeedMore(h.data_offset)
    return h


def data_layout(h):
    """Check the tensor offsets like gguf.cpp does; returns (status, data_size, size_without_last_padding).

    status: "contiguous", "mismatch at <tensor>" or "unknown type <id> at <tensor>" (sizes after it are unknown)."""
    run = 0
    last_pad = 0
    for t in h.tensors:
        nb = tensor_nbytes(t["dims"], t["type"])
        if nb is None:
            return f"unknown type {t['type']} at {t['name']}", None, None
        if t["offset"] != run:
            return f"mismatch at {t['name']}: offset {t['offset']}, expected {run}", None, None
        padded = align_up(nb, h.alignment)
        last_pad = padded - nb
        run += padded
    return "contiguous", run, run - last_pad


def _json_scalar(v):
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)          # "inf", "-inf", "nan": JSON has no literal for them
    if isinstance(v, str):
        return v.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    return v


def summarize(h, header_bytes, max_inline=4096):
    """JSON-able summary, a pure function of the header bytes. Arrays longer than max_inline elements and strings longer
    than 1 MiB are replaced by their count and the sha256 of the serialized value (type u32 [, element type u32,
    count u64], payload: header_bytes[span])."""
    kv = []
    for e in h.kv:
        d = {"key": e["key"], "type": TYPE_NAMES[e["type"]]}
        v = e["value"]
        big = False
        if e["type"] == T_ARRAY:
            d["elem_type"] = TYPE_NAMES[e["elem_type"]]
            d["count"] = len(v)
            big = len(v) > max_inline
        elif e["type"] == T_STRING:
            big = len(v) > (1 << 20)
            if big:
                d["length"] = len(v.encode("utf-8", errors="surrogateescape"))
        if big:
            s, t = e["span"]
            d["sha256"] = hashlib.sha256(bytes(header_bytes[s:t])).hexdigest()
        elif e["type"] == T_ARRAY:
            d["value"] = [_json_scalar(x) for x in v]
        else:
            d["value"] = _json_scalar(v)
        kv.append(d)
    status, data_size, data_size_unpadded = data_layout(h)
    tensors = []
    for t in h.tensors:
        tensors.append({"name": t["name"], "type": GGML_TYPES.get(t["type"], (None,))[0], "type_id": t["type"],
                        "shape": t["dims"], "offset": t["offset"], "nbytes": tensor_nbytes(t["dims"], t["type"])})
    return {"gguf_version": h.version, "n_kv": h.n_kv, "n_tensors": h.n_tensors, "alignment": h.alignment,
            "tensor_infos_end": h.infos_end, "data_offset": h.data_offset,
            "header_sha256": hashlib.sha256(bytes(header_bytes[:h.data_offset])).hexdigest(),
            "tensor_offsets": status, "data_size": data_size, "data_size_without_last_padding": data_size_unpadded,
            "kv": kv, "tensors": tensors}


def dumps(obj):
    """The serialization the fetcher commits (and the analyzer re-derives byte for byte)."""
    return json.dumps(obj, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def read_file_header(path, step=1 << 20, cap=1 << 30):
    """Header of a local file, reading only as much as the parser asks for."""
    with open(path, "rb") as f:
        buf = b""
        want = step
        while True:
            buf += f.read(want - len(buf))
            try:
                h = parse(buf)
                return h, buf[:h.data_offset]
            except NeedMore as e:
                if e.need > cap:
                    raise ValueError(f"{path}: header larger than {cap} bytes")
                if len(buf) < want:
                    raise ValueError(f"{path}: file ends inside the header") from None
                want = max(e.need, 2 * len(buf))


def rebuild(h, header_bytes, keep=lambda key: True, rename=lambda key: key, tensors=False):
    """A new GGUF (bytes) with the KV pairs of h for which keep(key) holds, keys renamed by rename(key), values copied
    byte for byte; no tensors unless tensors=True (then the infos are copied too and the caller appends the data)."""
    kv = [e for e in h.kv if keep(e["key"])]
    out = bytearray(MAGIC + struct.pack("<Iqq", h.version, len(h.tensors) if tensors else 0, len(kv)))
    for e in kv:
        k = rename(e["key"]).encode()
        s, t = e["span"]
        out += struct.pack("<Q", len(k)) + k + bytes(header_bytes[s:t])
    if tensors:
        for t in h.tensors:
            n = t["name"].encode()
            out += struct.pack("<Q", len(n)) + n + struct.pack("<I", len(t["dims"]))
            out += b"".join(struct.pack("<q", d) for d in t["dims"]) + struct.pack("<iQ", t["type"], t["offset"])
        out += b"\0" * (align_up(len(out), h.alignment) - len(out))
    return bytes(out)
