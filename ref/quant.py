"""Oracle: quantized GGUF weights and the activation quantization of ggml's CPU matrix products (N10, gap 2 of
docs/reassessment/CORRECTNESS.md).

Transcribed from llama.cpp @ec7630a640789c393694fb194f1bbbf0369fc62d ("C" = ggml/src/ggml-cpu/ggml-cpu.c, "X" =
ggml/src/ggml-cpu/arch/x86/quants.c, "XR" = ggml/src/ggml-cpu/arch/x86/repack.cpp, "RP" =
ggml/src/ggml-cpu/repack.cpp, "GQ" = ggml/src/ggml-quants.c), for the AVX2 / F16C / FMA build that
tools/g0/CMakeLists.txt makes with GGML_NATIVE=OFF (-mavx -mavx2 -mbmi2 -mf16c -mfma, no AVX-512): the instruction set
of the server's Zen 3 CPU. At the R1 pin 79e2e74 X, XR and gguf-py/gguf/quants.py are identical and every cited line
of C and RP is the same (diffed; C differs only by `!params->use_ref &&` in two conditions, the default path).

Weights. A quantized (or F16) weight is held as gguf.quants.dequantize of the pinned gguf-py (gguf-py/gguf/quants.py,
float32 arithmetic in the order of ggml's dequantize_row_*). ggml_get_rows of such a tensor (the token embedding, the
PLE table) and ggml_cast to F32 return exactly these values; the tests check that bit for bit (tolerance 0).

Matrix products. ggml_compute_forward_mul_mat converts src1 (the f32 activations) row by row to the vec_dot_type of
src0's type (C:1278-1279 vec_dot_type and its from_float, C:1360-1371 the conversion; mul_mat_id the same per
token/slot row, C:1631-1640 with the conversion at C:1637; C:1620-1629 is an #if 0 branch) and then sums the
products of weight blocks and activation blocks with integer dots scaled in f32. Exactly: sum over blocks of
d_w d_a sum(q_w q_a) (minus the k-quant minimum terms), which is the dot product of the dequantized weight row with
the dequantized activation row; only the f32 scaling and summation remain as rounding. So the oracle computes
act_quant(x) @ W_deq.T in float64, with act_quant per vec_dot_type (type traits C:215-410):
  F32                                   -> F32 (C:219): no conversion
  F16 (C:222-226)                       -> F16: ggml_cpu_fp32_to_fp16, round to nearest even (C:3480-3493)
  Q8_0, Q4_0, Q5_0, IQ4_NL, MXFP4       -> Q8_0 (C:275, 243, 263, 383, 290): quantize_row_q8_0, the AVX2 version
        X:302-398: per block of 32: amax = max|x|, d = amax / 127 stored as f16, id = 127 / amax (f32, 0 for
        amax 0), q = round_to_nearest_even(x * id) (_mm256_round_ps(.., _MM_ROUND_NEAREST), X:343-346); the dot sees
        f16(d) * q. (The scalar reference GQ:276 quantize_row_q8_0_ref uses id = 1/d and roundf, half away from
        zero: not the code this build runs.)
  Q2_K..Q6_K, IQ2/IQ3/IQ1, IQ4_XS, TQ*  -> Q8_K (C:302-330, 340-377, 389, 404-410): quantize_row_q8_K, which on
        x86 is quantize_row_q8_K_ref (X:505-507, GQ:2768-2805): per block of 256: max = the first element of
        largest |x|, iscale = -127 / max, q = nearest_int(iscale * x) (round to nearest even, GQ:621-626),
        MIN(127, q), d = 1 / iscale (f32). The AVX2 repacked Q4_K path (RP:5006-5010: Q4_K weights with a multiple
        of 8 rows are repacked to q4_K_8x8_q8_K) quantizes the activations four rows at a time with
        ggml_quantize_mat_q8_K_4x8 (RP:4697-4700, XR:290-512; iscale = +-127 / amax, the sign chosen by whether a
        +amax element exists, XR:382-388; rounding _MM_ROUND_NEAREST) and the remaining rows with from_float
        (RP:4703-4705); its mul_mat_id converts every row with from_float (RP:4839-4841). Both give the same values:
        q and d change sign together, d * q does not.
        Below: s = 127 / amax in f32, value = (1 / s) * rint(x * s).
  Q4_1, Q5_1                            -> Q8_1 (C:253, 269): the dot uses s = d * sum(q) stored as f16 for the
        minimum term, so it is not the dot of dequantized rows; not emulated (NotImplementedError). No tensor of the
        models used here has such a type.
The f32 input of the conversion is the graph's f32 activation: the oracle's float64 value is rounded to f32 first.

QWeight wraps one weight: `x @ w.T` (activations as rows, any leading shape) and `w @ x` (x one row) apply act_quant;
`np.asarray(w)` and `w[rows]` of a 2-D weight give the dequantized values (get_rows, cast); `w[e]` of a 3-D expert
tensor gives the expert's QWeight. act=False keeps the activations unquantized (dequantized weights only: the
comparison that shows what the activation quantization contributes).
"""
import numpy as np

QK8_0 = 32
QK_K = 256

VEC_DOT = {"F32": None, "F16": "F16", "BF16": "BF16",
           "Q8_0": "Q8_0", "Q4_0": "Q8_0", "Q5_0": "Q8_0", "IQ4_NL": "Q8_0", "MXFP4": "Q8_0", "Q1_0": "Q8_0",
           "Q2_0": "Q8_0", "NVFP4": "Q8_0",
           "Q4_1": "Q8_1", "Q5_1": "Q8_1",
           "Q2_K": "Q8_K", "Q3_K": "Q8_K", "Q4_K": "Q8_K", "Q5_K": "Q8_K", "Q6_K": "Q8_K", "IQ2_XXS": "Q8_K",
           "IQ2_XS": "Q8_K", "IQ2_S": "Q8_K", "IQ3_XXS": "Q8_K", "IQ3_S": "Q8_K", "IQ1_S": "Q8_K", "IQ1_M": "Q8_K",
           "IQ4_XS": "Q8_K", "TQ1_0": "Q8_K", "TQ2_0": "Q8_K"}


def _blocks(x, n):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % n:
        raise ValueError(f"row length {x.shape[-1]} is no multiple of the block size {n}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // n, n)


def q8_0(x):
    """quantize_row_q8_0 (AVX2, X:302-360) then the value the dot product sees: (q, d16) with value d16 * q."""
    b = _blocks(x, QK8_0)
    amax = np.abs(b).max(axis=-1, keepdims=True)
    d16 = (amax / np.float32(127)).astype(np.float16)
    with np.errstate(divide="ignore"):
        inv = np.where(amax != 0, np.float32(127) / np.where(amax != 0, amax, np.float32(1)), np.float32(0))
    q = np.rint((b * inv.astype(np.float32)).astype(np.float32))
    return q, d16


def q8_K(x):
    """quantize_row_q8_K_ref (GQ:2768-2805) / ggml_quantize_mat_q8_K_4x8 (XR:290-420): (q, d) with value d * q,
    in the sign convention q(+amax) = +127 (the reference's d and q may both carry the opposite sign)."""
    b = _blocks(x, QK_K)
    amax = np.abs(b).max(axis=-1, keepdims=True)
    nz = amax != 0
    s = np.where(nz, np.float32(127) / np.where(nz, amax, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.minimum(np.rint((b * s).astype(np.float32)), 127)
    d = np.where(nz, np.float32(1) / np.where(nz, s, np.float32(1)), np.float32(0)).astype(np.float32)
    return q, d


def act_quant(x, vec_dot_type):
    """The activation row(s) x as the dot product of a weight with this vec_dot_type sees them (float64)."""
    if vec_dot_type is None:
        return np.asarray(x, dtype=np.float64)
    shape = np.shape(x)
    if vec_dot_type == "F16":
        return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float16).astype(np.float64)
    if vec_dot_type == "Q8_0":
        q, d = q8_0(x)
    elif vec_dot_type == "Q8_K":
        q, d = q8_K(x)
    else:
        raise NotImplementedError(f"activation conversion to {vec_dot_type} is not emulated")
    return (q.astype(np.float64) * d.astype(np.float64)).reshape(shape)


def flip_bound(x, w, ulps=8):
    """Bound per output of x @ w.T (w a QWeight with Q8_0 / Q8_K activations) when x is llama.cpp's f32 activation
    only up to `ulps` f32 spacings (an input the test recomputes instead of reading it from a dump): an element whose
    scaled value x * 127 / amax lies within that distance of a rounding boundary (a half-integer) may round to the
    neighbouring integer on llama.cpp's side, which moves its value by one step of its block. Returns
    sum_j [x_j near a boundary] step_j |w_ij|. A change of amax itself by that much changes s by a relative `ulps`
    2^-23 only (covered by the f32 tolerance of the caller), but for Q8_0 it can move the f16 scale d = f16(amax / 127)
    across a binary16 rounding boundary: every value of that block then moves by |q_j| spacing16(d), added for such
    blocks."""
    vdt = w.vdt
    if vdt not in ("Q8_0", "Q8_K"):
        raise NotImplementedError(f"flip_bound for {vdt}")
    shape = np.shape(x)
    b = _blocks(x, QK8_0 if vdt == "Q8_0" else QK_K).astype(np.float64)
    amax = np.abs(b).max(axis=-1, keepdims=True)
    nz = amax != 0
    s = np.where(nz, 127.0 / np.where(nz, amax, 1.0), 0.0)
    v = np.abs(b) * s
    dist = np.abs(v - (np.floor(v) + 0.5))
    near = dist <= ulps * 2.0 ** -23 * np.maximum(v, 1.0)
    if vdt == "Q8_0":
        d = amax / 127
        d16 = d.astype(np.float32).astype(np.float16).astype(np.float64)
        sp = 2.0 ** (np.floor(np.log2(np.maximum(d, 2.0 ** -14))) - 10)          # binary16 spacing at d
        mid = np.floor(d / sp) * sp + sp / 2                                  # the rounding boundary next to d
        d_near = np.abs(d - mid) <= ulps * 2.0 ** -23 * d
        extra = d_near * np.rint(v) * sp
        contrib = near * d16 + extra
    else:
        contrib = near * np.where(nz, 1.0 / np.where(nz, s, 1.0), 0.0)
    return contrib.reshape(shape) @ np.abs(w.w).T


class QWeight:
    """A non-F32 weight: dequantized values plus the activation conversion of its vec_dot_type (module docstring)."""
    __array_ufunc__ = None          # ndarray @ QWeight -> QWeight.__rmatmul__ (via .T) instead of a silent ndarray

    def __init__(self, deq, qtype, act=True):
        self.w = np.asarray(deq, dtype=np.float64)
        self.qtype = qtype
        self.act = act
        if qtype not in VEC_DOT:
            raise NotImplementedError(qtype)
        self.vdt = VEC_DOT[qtype] if act else None

    shape = property(lambda self: self.w.shape)
    ndim = property(lambda self: self.w.ndim)

    def __array__(self, dtype=None, copy=None):
        return self.w if dtype is None else self.w.astype(dtype)

    def __getitem__(self, key):
        if self.w.ndim == 3 and isinstance(key, (int, np.integer)):
            return QWeight(self.w[key], self.qtype, self.act)            # one expert of a 3-D tensor
        return self.w[key]                                                # get_rows: dequantized values

    @property
    def T(self):
        return _Transposed(self)

    def __matmul__(self, x):
        """w @ x for one activation row x (the experts of ref/qwen4exp.py moe_layer)."""
        if np.ndim(x) != 1:
            raise ValueError("w @ x takes one activation row; use x @ w.T for several")
        return self.w @ act_quant(x, self.vdt)

    def __repr__(self):
        return f"QWeight({self.qtype}, shape={self.w.shape}, act={self.act})"


class _Transposed:
    __array_ufunc__ = None

    def __init__(self, qw):
        self.qw = qw

    def __rmatmul__(self, x):
        return act_quant(x, self.qw.vdt) @ self.qw.w.T


def tensor(t, dequantize, act=True):
    """A gguf ReaderTensor as the oracle uses it: F32 -> float64 ndarray, anything else -> QWeight.
    dequantize: gguf.quants.dequantize of the pinned gguf-py."""
    name = t.tensor_type.name
    if name == "F32":
        return np.array(t.data, dtype=np.float64)
    return QWeight(dequantize(t.data, t.tensor_type), name, act)


def load(reader, dequantize, act=True):
    """All tensors of a GGUFReader: {name: ndarray or QWeight}."""
    return {t.name: tensor(t, dequantize, act) for t in reader.tensors}
