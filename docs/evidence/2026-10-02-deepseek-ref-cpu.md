# DeepSeek-V4.1-Flash reference on CPU, 2026-10-02 (sandbox, no GPU)

Reference: third_party/deepseek-v41-flash @ HF revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 (PIN.json;
17 files, git blob ids verified 17/17 by CI check-run 110916696516). Tools: tools/ds41/kernel_cpu.py (overlay for the
TileLang `kernel.py`), tools/ds41/run_reference.py (tiny seeded model, unchanged `model.py`), torch 2.14.1 on CPU.

## torch on CPU in the sandbox (tools/sandbox/cuda_stub_gen.py)
```
$ LD_LIBRARY_PATH=~/cudastub python3 -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available()); ..."
torch 2.14.1+cu130 cuda available: False
torch.float8_e4m3fn torch.float8_e8m0fnu torch.float4_e2m1fn_x2
fp8 roundtrip tensor([ 3.1250e-01, -1.7500e+00,  4.4800e+02,  4.4800e+02])
```

## Kernel overlay self-checks
```
fp8 cast: [1.0, 1.25, 0.3125, 448.0, 448.0, -1.0, 0.0, 0.0]            # inputs 1.0625 1.1875 0.3 447.9 449 -1.0625 1e-9 0
fp4 round: [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 0.0, -1.0, -4.0, 0.0]
fp4 pack/unpack roundtrip ok; packed dtype torch.float4_e2m1fn_x2 (4, 32)
pow2_ceil: [1.0, 2.0, 0.5, 1.0, 4.0, 4.0, 2.384185791015625e-07]
act_quant: torch.float8_e4m3fn torch.float8_e8m0fnu (3, 5, 2) max rel err 0.058699995279312134
fp8_gemm vs dequant matmul: 0.0 (3, 5, 48)
fp4_gemm vs dequant matmul: 0.0
sparse_attn all-invalid row max abs: 0.0 | row0 == p*kv0: True
sinkhorn sums over dim-2 (exact): 0.9999989867210388 0.9999990463256836 | over dim-1: 0.9999976754188538 1.0000015497207642
exact mode identity ok
```
CONFIRMED: torch's CPU cast to FP8 E4M3 rounds half to even and saturates at 448. UNKNOWN: tie rounding of the GPU
FP4/FP8 casts in TileLang (cannot be run here); the overlay uses round-half-to-even.

## The unchanged reference model runs on CPU
```
$ python3 tools/ds41/run_reference.py --mode reference --out /tmp/ds41_reference.npz
{"mode": "reference", "seed": 41, "n_prefill": 37, "n_decode": 6, "torch": "2.14.1+cu130", "decode_vs_prefill_max_rel": 0.4385627210140228, "pin": "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"}
621 arrays, 4422843 bytes -> /tmp/ds41_reference.npz
$ python3 tools/ds41/run_reference.py --mode exact --out /tmp/ds41_exact.npz
{"mode": "exact", "seed": 41, "n_prefill": 37, "n_decode": 6, "torch": "2.14.1+cu130", "decode_vs_prefill_max_rel": 0.3292400538921356, "pin": "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"}
621 arrays, 4557207 bytes -> /tmp/ds41_exact.npz
```
Tiny config: 8 layers, compress_ratios (0,2,2,2,1,1,1,1), KV sources 1,3,4, index sources 1,3,4,6, candidate source 4,
Engram on layers 1 and 3, FP8 dense + FP4 expert weights, window 8, 43 tokens. The numpy oracle against these records:
NOT WRITTEN yet (ref/ds41.py).

## Finding A: the reference decode does not reproduce its own prefill (CONFIRMED on the tiny model)
```
$ python3 tools/ds41/check_decode_consistency.py
reference as vendored: positions 36..42 max rel logit error 4.7e-07 3.1e-07 3.9e-01 6.7e-02 5.5e-01 2.0e-01 2.8e-01
$ python3 tools/ds41/check_decode_consistency.py --republish
test patch (owner republishes index_k): positions 36..42 max rel logit error 4.7e-07 3.1e-07 5.0e-07 4.3e-07 5.2e-07 4.6e-07 3.9e-07
$ grep -n 'shared_attn.index_k\|if self.owns_k and latent is not None' third_party/deepseek-v41-flash/inference/model.py
537:        if self.owns_k and latent is not None:
548:            shared_attn.index_k = self.k_cache
554:        index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
```
Cause (CONFIRMED by the test patch): an index-key owner assigns `shared_attn.index_k` only when its compression group
just completed. On a decode step where a ratio-2 group is incomplete, the ratio-2 owners read the slot as the last
forward left it, i.e. pointing at the key cache of the last owner in the stack (the ratio-1 source). Position 37 is fine
(group complete), 38 is the first wrong step, and later steps stay wrong because caches were written from wrong hidden
states. With the released config the same pattern exists: owners 2, 8, 14 have ratio 2, owner 20 has ratio 1
(inference/config.json) - PROVISIONAL for the real model (logic read from code, not run on real weights).
Whether DeepSeek intends this: UNKNOWN. Consequence for us: the oracle is defined by PREFILL semantics (each layer scores
against its own source's keys); token parity against the reference `generate.py` decode loop is not a valid acceptance test.

## Finding B: exact-zero ties in the indexer top-k are implementation-defined (CONFIRMED on the tiny model)
With 4 index heads (first version of the tiny config), 37-token and 43-token prefills disagreed on row 34 of layer 1:
```
L1.attn.o                    rel 3.848e-01 rows [34]
layer-1 indexer, query 34: reachable keys 17, scores exactly 0.0: 4; lowest scores: [-2.7273, -1.5907, -0.8101, -0.5037, -0.3455, -0.1604, -0.1014, -0.0951]
```
Scores are `sum_h relu(q_h . k) * w_h`; when every head is rectified the score is exactly 0, several keys tie at the
top-k cut and `topk` picks differently depending on the tensor width. With 16 index heads (current tiny config):
```
prefix consistency 37 vs 43 prefill: 0 records differ []
```
Same class of effect as GLM finding 5 in 2026-10-02-g1-parity.md. Parity tests must accept any valid top-k on ties.

## Facts read from the vendored code and config (CONFIRMED at the pin)
| fact | source |
|---|---|
| Engram tables: 384,006,168 and 384,016,682 rows x 256, 8 heads x 3 n-gram sizes, bucket base 16,000,000, compressed vocab 99,092 | inference/config.json `engram_*`; fixes the fact sheet (it had 2 x 384,006,168) |
| row counts equal the sums of the bucket primes drawn by `EngramLayout.from_args` | computed in the sandbox: 384006168, 384016682 |
| compress_ratios: layers 0-1 = 0, 2-19 = 2, 20-39 = 1, 3 MTP layers = 0 | inference/config.json |
| KV sources 2, 8, 14, 20; index sources 2, 8, 14, 20, 24, 28, 32, 36; candidate source 20, 2048 blocks x 8, index top-k 512 | inference/config.json |
| mHC: coefficient formulas as in ref/mhc.py; the `pre` mix a sublayer computes is used by the NEXT sublayer; no hc_head; first mix is one-hot | inference/model.py `Block.forward`, `Transformer.forward`, kernel.py:406-474 |
| no hash routing (V4's tid2eid): the gate bias steers selection on every layer; weights = sqrt(softplus) / (sum + 1e-20) x 1.5 | inference/model.py `Gate.forward` |
| SwiGLU clamp before the activation: gate <= 10, up in [-10, 10] | inference/model.py `Expert.forward` |
| activations are quantised to FP8 (blocks of 32, power-of-two scales) before every quantised Linear | inference/model.py `linear()` |
| index queries/keys FP4 with E8M0 scales (32); compressed KV FP4 with E4M3 scales (16); window KV FP8 | inference/model.py `Indexer.forward`, `Attention._compress_kv`, `_window_kv` |
| FP4 experts: two values per byte, low nibble first, one E8M0 scale per 32 along K | inference/convert.py `cast_e2m1fn_to_e4m3fn` |
