#!/usr/bin/env bash
# N17: the dump tool after N17 issues the same calls as before for one MTP head: N16's mtp run (n16.gguf, target 240 +
# 24 x 1, DUMP_MTP=1 DUMP_MTP_DRAFT=4, tools/tinygen/run_parity_mimo2_decode.sh) with the tool built from 0b0eace and
# with the N17 tool, every dumped file compared, the MTP manifests compared without the fields N17 added.
# usage: 2026-10-11-n17-n16compat.sh <old dump-tensors> <new dump-tensors> <llama.cpp (gguf-py)> <repo> <scratch>
set -euo pipefail
OLD=$1; NEW=$2; LLAMA=$3; REPO=$4; T=$5
rm -rf "$T"; mkdir -p "$T"
PYTHONPATH="$LLAMA/gguf-py" python3 "$REPO/tools/tinygen/mimo2_tiny.py" "$T/n16.gguf" --fused-qkv --sinks swa \
  --value-scale 0.75 --mtp >/dev/null
TOK=$(python3 -c "import random; random.seed(16); t=[random.randrange(3,96) for _ in range(300)]; print(','.join(map(str,t[:264])))")
PFX="embd,attn_norm,wqkv,Qcur,Kcur,Vcur,kq,attn_out,ffn_inp,ffn_norm,ffn_up,ffn_gate,ffn_swiglu,ffn_out,ffn_moe_logits,ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_out,l_out,result_,h_nextn,mtp_"
CALLS="240,$(python3 -c "print(','.join(['1']*24))")"
for v in old new; do
  BIN=$OLD; [ $v = new ] && BIN=$NEW
  env DUMP_DIR="$T/$v" DUMP_TOKENS="$TOK" DUMP_PER_GRAPH=1 DUMP_PREFIXES="$PFX" DUMP_CALLS="$CALLS" DUMP_MTP=1 \
    DUMP_MTP_DRAFT=4 "$BIN" -m "$T/n16.gguf" -c 512 -b 512 -ub 64 -t 1 -fa off -ctk f32 -ctv f32 > "$T/$v.log" 2>&1
done
python3 - "$T" <<'PY'
import filecmp, json, os, sys
T = sys.argv[1]
files = sorted(os.path.relpath(os.path.join(d, f), f"{T}/old") for d, _, fs in os.walk(f"{T}/old") for f in fs
               if not f.endswith("manifest.json"))
new = sorted(os.path.relpath(os.path.join(d, f), f"{T}/new") for d, _, fs in os.walk(f"{T}/new") for f in fs
             if not f.endswith("manifest.json"))
diff = [f for f in files if not filecmp.cmp(f"{T}/old/{f}", f"{T}/new/{f}", shallow=False)]
print(f"files: old {len(files)}, new {len(new)}, same names {files == new}, differing {len(diff)} {diff[:5]}")
for m in ("manifest.json", "mtp/manifest.json"):
    a, b = json.load(open(f"{T}/old/{m}")), json.load(open(f"{T}/new/{m}"))
    added = sorted({k for c in b["calls"] for k in c} - {k for c in a["calls"] for k in c})
    for c in b["calls"]:
        for k in added:
            c.pop(k, None)
    print(f"{m}: equal without the N17 fields {added}: {a == b}")
PY
