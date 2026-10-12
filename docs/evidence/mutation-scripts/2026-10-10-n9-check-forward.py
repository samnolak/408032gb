"""quick check: ref/glm5next.forward vs a G1 single-ubatch dump; usage: check_forward.py <g1 dir> <gguf-py>"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/user/408032gb/.claude/worktrees/agent-ad00a79701ebcb394")
d = pathlib.Path(sys.argv[1])
sys.path.append(sys.argv[2])
from gguf import GGUFReader  # noqa: E402
from ref import glm5next as G  # noqa: E402

r = GGUFReader(str(d / "tiny.gguf"))
W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
hp = G.hparams({f.name: f for f in r.fields.values()})
print(hp)
toks = [int(x) for x in (d / "tokens.txt").read_text().split(",")]


def dump(name):
    j = json.loads((d / "dump" / (name + ".json")).read_text())
    dt = np.float32 if j["type"] == "f32" else np.int32
    return np.fromfile(d / "dump" / (name + ".bin"), dtype=dt).astype(np.float64), j["ne"]


A = G.forward(toks, W, hp)
T = len(toks)


def cmp(what, got, want):
    want = np.asarray(want)
    got = got.reshape(want.shape)
    e = np.abs(got - want).max() / (np.abs(want).max() + 1e-30)
    flag = "" if e < 1e-4 else "   <-- FAIL"
    print(f"{what:40s} rel {e:.2e}{flag}")


for il in range(hp["n_layer"]):
    L = A["layers"][il]
    cmp(f"hc_attn_pre-{il}", dump(f"hc_attn_pre-{il}__0")[0], L["attn_pre"])
    cmp(f"attn_norm-{il}", dump(f"attn_norm-{il}__0")[0], L["attn_norm"])
    if hp["kda"][il]:
        cmp(f"kda_scan_out-{il}", dump(f"kda_scan_out-{il}__0")[0], L["kda"]["scan"])
        cmp(f"kda_out-{il}", dump(f"kda_out-{il}__0")[0], L["kda"]["out"])
    else:
        cmp(f"kqv_out-{il}", dump(f"kqv_out-{il}__0")[0], L["dsa"]["kqv"])
        cmp(f"attn_out-{il}", dump(f"attn_out-{il}__0")[0], L["dsa"]["out"])
    cmp(f"hc_attn_post-{il}", dump(f"hc_attn_post-{il}__0")[0], L["hc_attn_post"])
    cmp(f"ffn_out-{il}", dump(f"ffn_out-{il}__0")[0], L["ffn"]["out"])
    cmp(f"l_out-{il}", dump(f"l_out-{il}__0")[0], L["l_out"])
cmp("hc_head", dump("hc_head__0")[0], A["hc_head"])
cmp("result_norm", dump("result_norm__0")[0], A["result_norm"])
cmp("result_output", dump("result_output__0")[0], A["logits"])
lo = np.fromfile(d / "dump" / "logits.bin", dtype=np.float32).astype(np.float64)
cmp("logits.bin", lo, A["logits"])
