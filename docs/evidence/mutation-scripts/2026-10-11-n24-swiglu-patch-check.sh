#!/usr/bin/env bash
# N24: does patches/llama.cpp/n24-swiglu-clamp-after-silu.patch switch llama.cpp's clamped SwiGLU to the post-SiLU
# semantics, and nothing else? CPU, tiny glm5-next model (L = 0.5, so the clamp fires), the G1 parity flow of
# tools/tinygen/run_parity.sh run twice on the same model and tokens: once with a stock dump-tensors build of ec7630a,
# once with the patched build. tests.test_glm_tiny_parity compares every dumped op with the numpy oracle (pre semantics)
# on llama.cpp's own inputs: with the patch only test_swiglu_clamp may fail. Then the dumped
# ffn_moe_swiglu_limited is compared with both formulas on the dumped gate/up of each MoE layer.
# usage: bash docs/evidence/mutation-scripts/2026-10-11-n24-swiglu-patch-check.sh <repo> <stock build dir> <patched build dir> <llama.cpp ec7630a> <work>
set -u
REPO=$1; STOCK=$2; POST=$3; LLAMA=$4; WORK=$5
mkdir -p "$WORK"
export LLAMA_CPP_GGUF_PY="$LLAMA/gguf-py"
PYTHONPATH="$LLAMA_CPP_GGUF_PY" python3 "$REPO/tools/tinygen/glm5next_tiny.py" "$WORK/tiny.gguf" --layers 12 >/dev/null
sha256sum "$WORK/tiny.gguf"
TOK=$(python3 -c "import random; random.seed(2); print(','.join(str(random.randrange(3,96)) for _ in range(64)))")
for v in stock post; do
  B=$STOCK; [ $v = post ] && B=$POST
  echo "== $v: $(grep -m1 '^LLAMA_CPP_DIR' "$B/CMakeCache.txt" | sed 's|.*=||')"
  DUMP_DIR="$WORK/dump-$v" DUMP_TOKENS="$TOK" \
  DUMP_PREFIXES="kda_,hc_,attn_norm,ffn_norm,ffn_moe_,ffn_shexp,ffn_out,l_out,indexer,kq_mask,q_absorbed,q_resid,kv_cmpr,kqv_out,attn_out" \
    "$B/dump-tensors" -m "$WORK/tiny.gguf" -c 512 -b 512 -ub 512 -t 1 -fa off -ctk f32 -ctv f32 >"$WORK/run-$v.log" 2>&1
  echo "dump-tensors exit $?"
  (cd "$REPO" && GLM_TINY_DUMP="$WORK/dump-$v" GLM_TINY_GGUF="$WORK/tiny.gguf" python3 -m unittest tests.test_glm_tiny_parity 2>&1 \
    | grep -E "^(FAIL|ERROR):|^Ran |^OK|^FAILED")
  python3 - "$WORK/dump-$v" <<'EOF'
import json, pathlib, sys
import numpy as np
d = pathlib.Path(sys.argv[1])
def load(n):
    j = json.loads((d / (n + ".json")).read_text())
    return np.fromfile(d / (n + ".bin"), dtype=np.float32).astype(np.float64)
silu = lambda x: x / (1.0 + np.exp(-x))
L = 0.5
worst = {"pre": 0.0, "post": 0.0}; above = 0; n = 0
for p in sorted(d.glob("ffn_moe_swiglu_limited-*__0.json")):
    il = p.name.split("-")[1].split("__")[0]
    g, u, out = load(f"ffn_moe_gate-{il}__0"), load(f"ffn_moe_up-{il}__0"), load(f"ffn_moe_swiglu_limited-{il}__0")
    uc = np.clip(u, -L, L)
    worst["pre"] = max(worst["pre"], float(np.max(np.abs(silu(np.minimum(g, L)) * uc - out))))
    worst["post"] = max(worst["post"], float(np.max(np.abs(np.minimum(silu(g), L) * uc - out))))
    above += int(np.sum(g > L)); n += g.size
print("ffn_moe_swiglu_limited vs pre formula: max abs err %.3e; vs post formula: %.3e; gate > L in %d of %d elements"
      % (worst["pre"], worst["post"], above, n))
EOF
done
