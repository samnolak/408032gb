"""where does kda_normed's 9.7e-5 come from? per-head rms of the scan at the worst element; usage: run ub18 g0 L10"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/user/408032gb/.claude/worktrees/agent-ad00a79701ebcb394")
S = pathlib.Path("/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/n9/work-ec")
sys.path.append("/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad/llama-ec7630a/gguf-py")
from gguf import GGUFReader  # noqa: E402
from ref import glm5next as G  # noqa: E402

r = GGUFReader(str(S / "glm5next.gguf"))
W = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
hp = G.hparams({f.name: f for f in r.fields.values()})
run, g, il = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = S / "n9" / run / f"g{g:03d}"


def dump(name):
    j = json.loads((d / (name + "__0.json")).read_text())
    return np.fromfile(d / (name + "__0.bin"), dtype=np.float32).astype(np.float64).reshape(j["ne"][::-1])


toks = [int(x) for x in (S / "n9" / "tokens.txt").read_text().split(",")]
A = G.forward(toks, W, hp)
K = A["layers"][il]["kda"]
man = json.loads((S / "n9" / run / "manifest.json").read_text())
ub = man["n_ubatch"]
p0 = g * ub
n = dump(f"kda_normed-{il}").reshape(-1, 4, 16).shape[0]
got_n = dump(f"kda_normed-{il}").reshape(n, 4, 16)
want_n = K["normed"][p0:p0 + n]
got_s = dump(f"kda_scan_out-{il}").reshape(n, 4, 16)
want_s = K["scan"][p0:p0 + n]
e = np.abs(got_n - want_n)
t, h, c = np.unravel_index(e.argmax(), e.shape)
print("worst normed elem", (t + p0, h, c), "abs", e.max(), "max|normed|", np.abs(want_n).max())
rms = np.sqrt((want_s ** 2).mean(-1))
es = np.abs(got_s - want_s)
print("scan normwise rel (whole tensor, this graph/layer)", es.max() / np.abs(want_s).max(), "max abs", es.max())
