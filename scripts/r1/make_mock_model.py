#!/usr/bin/env python3
"""Write the SYNTHETIC "model" that scripts/r1/configs/mock.json points at (stdlib only).

Two files in OUT_DIR:
  SYNTHETIC-mock-model-Q4_K.gguf           a header-only GGUF v3 (metadata + tensor infos, no tensor data) with
                                           Q4_K routed experts and Q8_0 / Q6_K / F32 other tensors, so that
                                           tools/gguf_quant_audit.py has something real to read
  SYNTHETIC-mock-model.manifest.json       a weights manifest of that file in the format of
                                           scripts/offline/weights.py (built with its build_manifest()), so that
                                           `weights.py verify` has something real to check

THIS IS TEST DATA, NOT A MODEL AND NOT EVIDENCE. tools/bench/mock_server.py (the "engine" of mock.json) never reads it;
it only lets the weights_verify and quant_audit steps of scripts/r1/run_r1.py run end to end without real weights.

Usage: python3 scripts/r1/make_mock_model.py OUT_DIR
"""
import argparse
import hashlib
import json
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "scripts", "offline"))
import gguf_quant_audit  # noqa: E402  (tools/gguf_quant_audit.py: the ggml type table)
import weights  # noqa: E402  (scripts/offline/weights.py: the manifest format)

GGUF_NAME = "SYNTHETIC-mock-model-Q4_K.gguf"
MANIFEST_NAME = "SYNTHETIC-mock-model.manifest.json"
REPO_ID = "SYNTHETIC/mock-model"
REVISION = hashlib.sha1(b"SYNTHETIC mock model, not evidence").hexdigest()     # a 40-hex id, as weights.py demands
TYPE_ID = {name: tid for tid, (name, _b, _s) in gguf_quant_audit.GGML_TYPES.items()}
TENSORS = [                                    # (name, dims, ggml type): classes as tools/gguf_quant_audit.classify()
    ("token_embd.weight", [256, 64], "Q8_0"),
    ("blk.0.attn_norm.weight", [256], "F32"),
    ("blk.0.attn_q.weight", [256, 256], "Q8_0"),
    ("blk.0.ffn_gate_inp.weight", [256, 8], "F32"),
    ("blk.0.ffn_gate_exps.weight", [256, 64, 8], "Q4_K"),
    ("blk.0.ffn_up_exps.weight", [256, 64, 8], "Q4_K"),
    ("blk.0.ffn_down_exps.weight", [256, 64, 8], "Q4_K"),
    ("output.weight", [256, 64], "Q6_K"),
]


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_header(tensors, arch="synthetic-mock"):
    kv = [("general.architecture", arch), ("general.name", "SYNTHETIC mock model, NOT EVIDENCE")]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, value in kv:
        out += _s(key) + struct.pack("<I", 8) + _s(value)          # 8 = GGUF string
    off = 0
    for name, dims, tname in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", TYPE_ID[tname], off)
        off += 4096
    return out


def write(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gguf = os.path.join(out_dir, GGUF_NAME)
    data = gguf_header(TENSORS)
    with open(gguf, "wb") as f:
        f.write(data)
    entry = {"type": "file", "path": GGUF_NAME, "size": len(data), "oid": hashlib.sha1(data).hexdigest(),
             "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}}
    man = weights.build_manifest(REPO_ID, REVISION, [entry], "*.gguf")
    man["_label"] = ("SYNTHETIC test data written by scripts/r1/make_mock_model.py for the mock R1 run; "
                     "NOT a Hugging Face repository and NOT EVIDENCE")
    mpath = os.path.join(out_dir, MANIFEST_NAME)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    return gguf, mpath


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    gguf, mpath = write(a.out_dir)
    print("SYNTHETIC mock model (not evidence): %s, manifest %s" % (gguf, mpath))
    return 0


if __name__ == "__main__":
    sys.exit(main())
