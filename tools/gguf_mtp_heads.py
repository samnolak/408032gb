#!/usr/bin/env python3
"""Keep only the first K NextN (MTP) heads of a single-file mimo2 GGUF (N21 workaround for an unpatched llama.cpp,
docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md section 5).

With one head, common/speculative.cpp does not chain heads (chain_heads = n_mtp_layers > 1, speculative.cpp:1482
@79e2e74): process() is one decode and the draft grows the one head's KV, so the per-head re-placement that leaves head
0 with K/V of other positions never happens. llama.cpp's --override-kv <arch>.nextn_predict_layers=int:1 does not do
this: the loader then refuses the file ("done_getting_tensors: wrong number of tensors", CONFIRMED on the tiny model at
79e2e74, evidence file). This tool writes a new file: the trunk and MTP layers n_layer .. n_layer + K - 1 unchanged,
blk.<i>.* of the dropped heads removed, <arch>.block_count and <arch>.nextn_predict_layers lowered, and every per-layer
array of length block_count cut to the new block count. Everything else (tensor data, types, other keys) is copied.

Single-file GGUF only: a split file (split.count > 1) is refused; merge it first with llama-gguf-split --merge.
Needs llama.cpp's gguf-py on PYTHONPATH (the pinned checkout's gguf-py).

usage: python3 tools/gguf_mtp_heads.py --keep 1 in.gguf out.gguf
"""
import argparse
import re
import sys


def keep_heads(src, dst, keep):
    """The copy follows gguf-py's own gguf_new_metadata.py (copy_with_new_metadata, gguf-py/gguf/scripts @79e2e74):
    fields re-added with their types, tensor infos then data streamed from the reader, the alignment kept."""
    import gguf
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    if "split.count" in r.fields and int(r.fields["split.count"].contents()) > 1:
        raise SystemExit(f"{src}: a split GGUF (split.count {r.fields['split.count'].contents()}); merge it first")
    n_block = int(r.fields[f"{arch}.block_count"].contents())
    n_nextn = int(r.fields[f"{arch}.nextn_predict_layers"].contents())
    if not 1 <= keep <= n_nextn:
        raise SystemExit(f"--keep {keep}: the file has {n_nextn} NextN heads")
    n_layer = n_block - n_nextn
    new_block = n_layer + keep
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    if "general.alignment" in r.fields:
        w.data_alignment = int(r.fields["general.alignment"].contents())
    cut = []
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        sub = f.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        v = f.contents()
        if f.name == f"{arch}.block_count":
            v = new_block
        elif f.name == f"{arch}.nextn_predict_layers":
            v = keep
        elif t == gguf.GGUFValueType.ARRAY and f.name.startswith(f"{arch}.") and len(v) == n_block:
            v = v[:new_block]
            cut.append(f.name)
        w.add_key_value(f.name, v, t, sub_type=sub)
    kept = [t for t in r.tensors if not (m := re.match(r"blk\.(\d+)\.", t.name)) or int(m[1]) < new_block]
    for t in kept:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in kept:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    return dict(arch=arch, n_layer=n_layer, heads=(n_nextn, keep), block_count=(n_block, new_block),
                tensors=(len(r.tensors), len(kept)), arrays_cut=cut)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep", type=int, default=1)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args(argv)
    print(keep_heads(a.src, a.dst, a.keep))


if __name__ == "__main__":
    sys.exit(main())
