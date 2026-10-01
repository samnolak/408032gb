"""Bits per weight of storage formats, including block scales.

ggml types: CONFIRMED from llama.cpp gguf-py/gguf/constants.py GGML_QUANT_SIZES
@ec7630a640789c393694fb194f1bbbf0369fc62d as (block_size, type_size_bytes); the extraction
command and its literal output are in docs/evidence/2026-10-01-recon.md.

Non-ggml formats are PROVISIONAL and documented per entry.
"""

# name: (block_size, type_size_bytes)  -- CONFIRMED, gguf-py @ec7630a
GGML_BLOCKS = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q6_K": (256, 210),
    "Q5_K": (256, 176),
    "Q4_K": (256, 144),
    "IQ4_XS": (256, 136),
    "IQ4_NL": (32, 18),
    "MXFP4": (32, 17),
    "Q3_K": (256, 110),
    "IQ3_S": (256, 110),
    "IQ3_XXS": (256, 98),
    "Q2_K": (256, 84),
    "IQ2_S": (256, 82),
    "IQ2_XS": (256, 74),
    "IQ2_XXS": (256, 66),
    "IQ1_M": (256, 56),
    "IQ1_S": (256, 50),
}

# PROVISIONAL formats as shipped in safetensors checkpoints.
OTHER_BPW = {
    # FP8 E4M3 weights with one fp32 scale per 128x128 block (GLM-5.3-Flash quantization_config).
    "FP8": 8.0 + 32.0 / (128 * 128),
    # FP8 E4M3 with one E8M0 scale per 32 values (DeepSeek-V4.1 Engram tables).
    "FP8_E8M0_32": 8.0 + 8.0 / 32,
}


def bpw(fmt: str) -> float:
    """Bits per weight for a storage format name, scales included."""
    if fmt in GGML_BLOCKS:
        block, size = GGML_BLOCKS[fmt]
        return size * 8.0 / block
    if fmt in OTHER_BPW:
        return OTHER_BPW[fmt]
    raise KeyError(f"unknown format {fmt!r}; known: {sorted(GGML_BLOCKS) + sorted(OTHER_BPW)}")


def all_formats():
    return sorted(GGML_BLOCKS) + sorted(OTHER_BPW)
