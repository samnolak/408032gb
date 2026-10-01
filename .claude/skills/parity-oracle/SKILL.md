---
name: parity-oracle
description: How to build numpy oracles in ref/ and compare kernels against llama.cpp tensor dumps. Use when adding or verifying any numerical operation.
---

# Parity oracles

## 1. Oracle from the reference source

Write the op in numpy (float64 for the oracle, cast at the boundary) from the pinned
reference: `llama.cpp/src/models/<arch>.cpp` for the graph, `ggml/src/ggml-cpu/ops.cpp` for
op semantics. Cite `path:line@ec7630a` in the docstring. Example: ref/kda.py.

## 2. Property tests

stdlib unittest + numpy only. Test reductions to known special cases (e.g. KDA with a gate
constant across channels equals scalar-gate GDN), state continuity (split a sequence in
two, carry the state, same output), and algebraic identities.

## 3. Dumps from llama.cpp

Tensor names are the strings in `cb(tensor, "<name>", il)`; the graph scheduler appends
`-<layer>`, e.g. `kda_scan_out-4`, `ffn_moe_topk-10`. Capture with an eval callback
(`params.cb_eval`, see llama.cpp examples/eval-callback and tools/g0/expert_hist.cpp):
return true for wanted names when `ask` is true, then copy with `ggml_backend_tensor_get`.

## 4. Comparison

Report per layer: max abs error, max relative error, argmax of the error. Tolerances:
fp32 1e-4 rel, bf16 1e-2 rel. Quantized weights: compare against the dequantized weights,
never against original fp32/fp8 weights.
