"""list the MTP DSA tie replays at R1: position, graph, reference cells, oracle cells, their scores"""
import os
import sys

S = "/tmp/claude-0/-home-user-408032gb/83634621-ad4f-5747-933c-0d86b3c4b300/scratchpad"
os.environ.update(GLM5NEXT_DECODE_DUMP=f"{S}/n9/work-r1/n9", GLM5NEXT_MTP_GGUF=f"{S}/n9/work-r1/n9/../glm5next_mtp.gguf",
                  LLAMA_CPP_GGUF_PY=f"{S}/llama.cpp/gguf-py",
                  PARITY_LLAMA_COMMIT="79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6")
sys.path.insert(0, "/home/user/408032gb/.claude/worktrees/agent-ad00a79701ebcb394")
import numpy as np  # noqa: E402
from tests import test_glm5next_mtp_parity as M  # noqa: E402

M.Mtp.setUpClass()
c = M.Mtp
kp = c.hp["kpool"]
c._oracles({})              # the oracle's own choice, before replay
for gr in c.mg:
    ob = c.B[gr["g"]]["dsa"]
    for r in range(gr["n"]):
        t, s = gr["p0"] + r, c.seen[(gr["g"], r)]
        if s != ob["cells"][r]:
            sc = ob["scores"][r]
            vis = np.where(np.isfinite(sc))[0]
            print(f"pos {t} graph {gr['g']} ({gr['kind']}): reference {s} oracle {ob['cells'][r]}; "
                  f"visible pool scores {dict(zip(vis.tolist(), np.round(sc[vis], 9).tolist()))}")
