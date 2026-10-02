#!/usr/bin/env python3
"""Does the reference decode (one token per step) reproduce its own prefill?

    python3 tools/ds41/check_decode_consistency.py            # reference as vendored
    python3 tools/ds41/check_decode_consistency.py --republish # test patch, see below

Prints the max relative logit error of prefill(37) + 6 decode steps against rows 36..42 of one
43-token prefill, on the tiny model of run_reference.py in "exact" mode (no quantisation noise).

--republish applies a TEST-ONLY patch in memory (the vendored file is not touched): an index-key
owner publishes its key cache on every call, not only when its compression group just completed
(inference/model.py:548 assigns shared_attn.index_k inside `if self.owns_k and latent is not None`,
line 554 reads it unconditionally). Run each variant in its own process.
"""
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import run_reference as RR  # noqa: E402

R = RR.R


def main(republish):
    RR.kernel_cpu.MODE = "exact"
    torch.manual_seed(41)
    args, tok = RR.make_args()
    net = R.Transformer(args, tok)
    RR.init_params(net, 41)
    net.eval()
    if republish:
        real = R.Indexer.forward

        def forward(self, x, qr, latent, start_pos, offset):
            if self.owns_k:
                R.shared_attn.index_k = self.k_cache
            return real(self, x, qr, latent, start_pos, offset)
        R.Indexer.forward = forward
    n, k = RR.N_PREFILL, RR.N_DECODE
    tokens = torch.randint(3, args.vocab_size, (1, n + k), generator=torch.Generator().manual_seed(42))
    rec = RR.Recorder()
    rec.install(net)
    net(tokens, 0)
    last = args.n_layers - 1
    with torch.inference_mode():
        hid = net.layers[last].hc_pre(torch.from_numpy(rec.rec[f"L{last}.out"]), torch.from_numpy(rec.rec[f"L{last}.ffn_pre"]))
        want = net.head(net.norm(hid), full_logits=True)[0].numpy()
    rec.on = False
    errs = []
    for t in range(n - 1, n + k):
        _, lg, _ = net(tokens[:, :n], 0) if t == n - 1 else net(tokens[:, t:t + 1], t)
        errs.append(float(np.abs(lg[0].numpy() - want[t]).max() / np.abs(want[t]).max()))
    label = "test patch (owner republishes index_k)" if republish else f"reference as vendored"
    print(f"{label}: positions {n - 1}..{n + k - 1} max rel logit error " + " ".join(f"{e:.1e}" for e in errs))
    return max(errs)


if __name__ == "__main__":
    main("--republish" in sys.argv)
