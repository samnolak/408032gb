"""CPU stand-in for the DeepSeek-V4.1-Flash reference `inference/kernel.py` (ADR-008, item 4).

The reference kernels are TileLang GPU JIT code and cannot run on a CPU. This module implements
the same six public functions in plain torch so that the unchanged reference `model.py` runs on a
CPU. tools/ds41/run_reference.py installs it as the module named `kernel` before importing model.

Transcribed from third_party/deepseek-v41-flash/inference/kernel.py (pin: PIN.json):
    act_quant            kernel.py:40-124     block FP8, scale amax/448 or 2^ceil(log2(amax/448))
    fp4_act_quant        kernel.py:127-204    block FP4, scale 2^ceil(log2(amax/6)) or FP8(amax/6)
    fp8_gemm             kernel.py:207-307    C = A_fp8 @ B_fp8^T with per-block scales
    sparse_attn          kernel.py:310-403    gather top-k KV, softmax with a per-head sink
    hc_split_sinkhorn    kernel.py:406-474    pre / post / comb coefficients of Hyper-Connections
    fp4_gemm             kernel.py:477-591    C = A_fp8 @ B_fp4^T, B packed two values per byte

MODE:
    "reference"  quantize exactly where the reference kernels quantize (FP8 activations, FP4 index
                 queries/keys, FP4 compressed KV). This is the model function DeepSeek ships.
    "exact"      no activation/cache quantization: act_quant and fp4_act_quant are identities.
                 Weights stay as stored. Separates algorithm errors from quantization noise.

Known differences from the GPU kernels (they do not change the algorithm; documented in
docs/evidence): accumulation is fp32 matmul instead of per-K-block accumulation; sparse_attn works
in fp32 where the GPU kernel casts probabilities to bf16; rounding of exact ties in FP4 is
round-half-to-even here, the GPU cast is UNKNOWN (cannot be run without a GPU).
"""
import torch

MODE = "reference"          # or "exact"; set by the runner before the model is built

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive normal float32, by exponent bits (kernel.py fast_log2_ceil,
    fast_pow2)."""
    x = x.to(torch.float32).contiguous()
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x), e)


def fp8_round(v: torch.Tensor) -> torch.Tensor:
    """Round float32 to the FP8 E4M3 grid (saturating at +-448), returned as float32."""
    return v.to(torch.float32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def fp4_codes(v: torch.Tensor) -> torch.Tensor:
    """Round float32 to FP4 E2M1 codes 0..15 (bit 3 = sign), round-half-to-even on the code."""
    a = v.to(torch.float32).abs().clamp(max=FP4_MAX)
    lo = torch.bucketize(a, _FP4_MIDS, right=False)      # ties go down
    hi = torch.bucketize(a, _FP4_MIDS, right=True)       # ties go up
    code = torch.where((lo != hi) & (hi % 2 == 0), hi, lo)
    return code + 8 * (torch.signbit(v) & (code > 0)).to(code.dtype)


def fp4_round(v: torch.Tensor) -> torch.Tensor:
    return FP4_TABLE[fp4_codes(v)]


def fp4_pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] codes -> [..., K//2] float4_e2m1fn_x2; the first value of a pair is the low nibble
    (inference/convert.py cast_e2m1fn_to_e4m3fn: low = x & 0x0F, high = x >> 4)."""
    c = codes.to(torch.uint8).unflatten(-1, (-1, 2))
    return (c[..., 0] | (c[..., 1] << 4)).contiguous().view(torch.float4_e2m1fn_x2)


def fp4_unpack(b: torch.Tensor) -> torch.Tensor:
    """[..., K//2] float4_e2m1fn_x2 -> [..., K] float32."""
    x = b.contiguous().view(torch.uint8)
    return torch.stack([FP4_TABLE[(x & 0x0F).long()], FP4_TABLE[((x >> 4) & 0x0F).long()]], dim=-1).flatten(-2)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    return x.to(torch.float32).unflatten(-1, (-1, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 quantization along the last dim. Not inplace: returns (y, s) with y the FP8
    mantissas and s one scale per block; inplace: writes the dequantized values back into x."""
    n = x.size(-1)
    assert n % block_size == 0
    if MODE == "exact":
        if inplace:
            return x
        return x.to(torch.float32), torch.ones(*x.shape[:-1], n // block_size, dtype=torch.float32)
    v = _blocks(x, block_size)
    amax = v.abs().amax(dim=-1).clamp_min(1e-4)
    inv = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    s = pow2_ceil(amax * inv) if scale_fmt is not None else amax * inv
    q = fp8_round(v / s.unsqueeze(-1))
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2).to(torch.float8_e4m3fn), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 with E8M0 (power of two) or E4M3 scales. Only the inplace form is used by
    model.py (index queries/keys with E8M0, compressed KV with E4M3 over groups of 16)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert x.size(-1) % block_size == 0
    if not inplace:
        raise NotImplementedError("model.py only calls fp4_act_quant(..., inplace=True)")
    if MODE == "exact":
        return x
    v = _blocks(x, block_size)
    amax = v.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        s = fp8_round(amax / FP4_MAX)
    else:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        s = pow2_ceil(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    q = fp4_round((v / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
    return x


def _dequant_act(a, a_s, block):
    return (_blocks(a, block) * a_s.to(torch.float32).unsqueeze(-1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[..., N] = A[..., K] @ B[N, K]^T; A has one scale per K block, B one per (N block, K block)."""
    n, k = b.shape
    a_f = _dequant_act(a, a_s, block_size)
    sb = b_s.to(torch.float32).repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    return (a_f @ (b.to(torch.float32) * sb).T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C = A_fp8 @ B_fp4^T; B is [N, K//2] packed FP4 with one E8M0 scale per 32 values along K."""
    a_f = _dequant_act(a, a_s, act_block_size)
    w = fp4_unpack(b) * b_s.to(torch.float32).repeat_interleave(32, 1)
    return (a_f @ w.T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b, m, h, d]; kv [b, n, d], key and value at once and shared by all heads; topk_idxs
    [b, m, k] with -1 for an empty slot. One learned sink logit per head joins the softmax
    denominator only. A row with no valid slot yields zeros (kernel.py:352-354)."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv.to(torch.float32)[torch.arange(b)[:, None, None], idx.clamp_min(0)]          # [b, m, k, d]
    s = torch.einsum("bmhd,bmkd->bmhk", q.to(torch.float32), g) * softmax_scale
    s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.to(torch.float32).view(1, 1, h, 1).expand(b, m, h, 1)
    p = torch.cat([s, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", p, g).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """mixes [b, s, (2+hc)*hc] -> pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc]."""
    hc = hc_mult
    m = mixes.to(torch.float32)
    sc, base = hc_scale.to(torch.float32), hc_base.to(torch.float32)
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
