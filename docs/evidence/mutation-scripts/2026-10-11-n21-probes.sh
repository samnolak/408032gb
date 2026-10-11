#!/usr/bin/env bash
# N21 probes (outputs in docs/evidence/2026-10-11-n21-mimo-mtp-heads-fix.md 8.1, 8.5, 8.7): git apply --check of the patch
# at both pins, the loader's answer to --override-kv mimo2.nextn_predict_layers=int:1, the one-head GGUF tool against
# N16's model, the --swa-full KV arithmetic from the MiMo-V2.6-Flash config, and the placement model with one draft step.
set -u
S=/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad
WT=${1:-/home/user/wt-n21}; OUT=$S/n21/probes
rm -rf $OUT; mkdir -p $OUT
P=$WT/patches/llama.cpp/0001-mtp-heads-slot-replay.patch
echo "== git apply --check of $(basename $P) (sha256 $(sha256sum $P | cut -c1-16)...)"
for c in 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 ec7630a640789c393694fb194f1bbbf0369fc62d; do
  R=$S/llama.cpp; [ ${c:0:7} = ec7630a ] && R=$S/llama-ec7630a
  d=$OUT/apply-${c:0:7}; mkdir -p $d; git -C $R archive $c src common | tar -x -C $d
  echo "-- at $c"; (cd $d && git apply --check -v $P 2>&1 | grep -E '^(Checking|error)'); rm -rf $d
done
L=$S/llama.cpp
PYTHONPATH=$L/gguf-py python3 $WT/tools/tinygen/mimo2_tiny.py $OUT/n17.gguf --fused-qkv --sinks swa --value-scale 0.75 --mtp-layers 3 >/dev/null
PYTHONPATH=$L/gguf-py python3 $WT/tools/tinygen/mimo2_tiny.py $OUT/n16.gguf --fused-qkv --sinks swa --value-scale 0.75 --mtp >/dev/null
echo "== --override-kv mimo2.nextn_predict_layers=int:1 on the 3-head n17.gguf (dump tool, llama.cpp $(git -C $L rev-parse --short HEAD))"
TOK=$(python3 -c "import random; random.seed(16); print(','.join(str(random.randrange(3,96)) for _ in range(40)))")
env DUMP_DIR=$OUT/ovr DUMP_TOKENS="$TOK" DUMP_PREFIXES=mtp_tok_embd DUMP_PER_GRAPH=1 DUMP_CALLS="30,1,1,d3,4,r1,1,1,1" DUMP_MTP=1 \
  $S/n21/build-r1/dump-tensors -m $OUT/n17.gguf -c 512 -b 512 -ub 64 -t 1 -fa off \
  --override-kv mimo2.nextn_predict_layers=int:1 2>&1 | grep -E ' E |error|failed' | sed -E 's/^[0-9.]+ //'
echo "exit status ${PIPESTATUS[0]}"
echo "== tools/gguf_mtp_heads.py --keep 1 n17.gguf (3 heads) vs N16's n16.gguf (mimo2_tiny.py --mtp)"
PYTHONPATH=$L/gguf-py python3 $WT/tools/gguf_mtp_heads.py --keep 1 $OUT/n17.gguf $OUT/n17-keep1.gguf
cmp $OUT/n16.gguf $OUT/n17-keep1.gguf && echo "cmp: byte-identical ($(stat -c %s $OUT/n16.gguf) bytes)"
echo "== --swa-full: extra f16 KV per SWA layer, MiMo-V2.6-Flash config"
python3 - "$WT/third_party/hf/mimo-v2.6-flash-chat/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
n_swa_layers = sum(c["hybrid_layer_pattern"]); kv = c["swa_num_key_value_heads"]; dk, dv = c["swa_head_dim"], c["swa_v_head_dim"]
mtp = c["num_nextn_predict_layers"]; ring = 768
print(f"SWA layers {n_swa_layers} of {c['num_hidden_layers']}, MTP layers {mtp}, KV heads {kv}, K {dk}, V {dv}, ring {ring} cells")
for ctx in (32768, 131072, 262144):
    per = (ctx - ring) * kv * (dk + dv) * 2
    print(f"ctx {ctx}: per layer +{per} B = {per / 2**30:.3f} GiB; target {n_swa_layers} layers +{n_swa_layers * per / 2**30:.2f} GiB;"
          f" MTP {mtp} layers +{mtp * per / 2**30:.2f} GiB; total +{(n_swa_layers + mtp) * per / 2**30:.2f} GiB")
PY
echo "== placement model, upstream, one draft step (--spec-draft-n-max 1): process still runs 3 heads"
(cd $WT && python3 - <<'PY'
from ref.mtp_cells import serve, summary
for a in (0.3, 0.6, 0.85):
    n = bad = 0
    for seed in (1, 2, 3):
        s = summary(serve(700, 4000, a, 768, 128, n_draft=1, seed=seed)); n += s["n_drafts"]; bad += s["bad"][0]
    print(f"prompt 700, accept {a}: drafts {n}, head-0 drafts on foreign K/V {bad} ({bad / n:.1%})")
PY
)
rm -rf $OUT
