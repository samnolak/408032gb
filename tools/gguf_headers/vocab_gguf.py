#!/usr/bin/env python3
"""Write a vocab-only GGUF from a committed GGUF header, so that llama.cpp's own tokenizer can run on the real vocabulary
without the weights (llama-tokenize loads with vocab_only = true: tools/tokenize/tokenize.cpp:146 @79e2e74).

The KV values are copied byte for byte from the header (gguf_header.rebuild); no tensors. Dropped: split.* (the loader
would look for the other shards). --arch renames the architecture string and the keys prefixed with it, for a file
whose architecture the checkout does not know (the Unsloth glm5next files at both llama.cpp pins: tools/gguf_headers/
analyze.py section (e)); vocab-only loading stops before the arch hyperparameters (llama_model_base::load_hparams
returns for vocab_only, src/llama-model.cpp:1339-1341 @79e2e74), so only the tokenizer.* keys are read. The tokenizer
keys are never changed.

Usage: python3 tools/gguf_headers/vocab_gguf.py <file>.header.gz OUT.gguf [--arch glm5-next]
"""
import argparse
import gzip
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_header as G  # noqa: E402


def build(header_bytes, arch=None):
    h = G.parse(header_bytes)
    old = h.get("general.architecture")
    keep = lambda k: not k.startswith("split.")  # noqa: E731
    rename = (lambda k: arch + k[len(old):] if arch and old and k.startswith(old + ".") else k)
    out = G.rebuild(h, header_bytes, keep=keep, rename=rename)
    if arch and old != arch:
        # replace the value of general.architecture (a string) in place
        a, b = old.encode(), arch.encode()
        before = struct.pack("<Q", len(b"general.architecture")) + b"general.architecture" + \
            struct.pack("<I", G.T_STRING) + struct.pack("<Q", len(a)) + a
        after = before[:-(8 + len(a))] + struct.pack("<Q", len(b)) + b
        if out.count(before) != 1:
            raise ValueError("general.architecture not found exactly once")
        out = out.replace(before, after)
    h2 = G.parse(out)
    assert h2.n_tensors == 0 and h2.data_offset == len(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("header", help="<file>.header.gz (or raw header bytes)")
    ap.add_argument("out")
    ap.add_argument("--arch", help="architecture string to write instead of the file's")
    a = ap.parse_args(argv)
    with open(a.header, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = build(raw, a.arch)
    with open(a.out, "wb") as f:
        f.write(data)
    h = G.parse(data)
    print(f"wrote {a.out}: {len(data)} B, {h.n_kv} KV, architecture {h.get('general.architecture')!r}, "
          f"tokenizer {h.get('tokenizer.ggml.model')!r} / pre {h.get('tokenizer.ggml.pre')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
