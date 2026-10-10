# parity-qa memory: recurring mismatch causes and dump facts (llama.cpp @ec7630a)

Source of each item: docs/evidence/2026-10-10-qwen4exp-decode-parity.md (N6) unless noted.

## Layouts / transposes
- Delta-rule state (ggml_gated_delta_net, `state_predelta-*`, `new_state-*`): ggml keeps S[i][j] at [j*S_v + i]
  per head (ggml/src/ggml-cpu/ops.cpp:11228). A dump reshaped to (H_v, S, S) is the ref/kda.py state[h].T.
- `conv_state_at-<il>` (qwen4exp build_conv_state_at): ne [K-1, C] = time fastest; numpy (C, K-1).T gives rows.
  On a PLE layer occurrence 0 is the PLE history, occurrence 1 the GDN conv history.
- `kq_mask_qsa-*` rows are KV cells; for one sequence cell = position (also after seq_rm rollback).

## Names
- cb() twice on one tensor keeps the later name (kqv_out -> attn_pregate, 2nd hc_combine -> l_last) (N2).
- One layer index used several times: occurrence k in `<name>__<k>` (qwen4exp MTP: hc_norm-8 __0 attn, __1 MoE,
  __2 head; the head's hc_mixed is renamed result_norm).
- Per-ubatch dumps: dump_tensors with DUMP_PER_GRAPH=1 (one g<NNN>/ per graph); default mode appends k globally.

## Traps
- QSA tie replay must validate BOTH the reference's and the oracle's own selection (qsa_tie_variant with the
  absolute position), else an oracle selection bug is replayed away (N6 S9; N2's tests still have this hole).
- Errors vs an end-to-end oracle include propagation (N6 worst 5.4e-6 at the gated RMS norm of the GDN output);
  per-component checks from dumped inputs are ~10x tighter (N2 7.4e-7).
- MTP context: call a setter that requests a scheduler reserve after llama_set_embeddings_nextn, else the first MTP
  graph segfaults in ggml_gallocr_init_tensor (out_ids buffer_id -1) at this pin.
- unittest: gguf-py ships a regular package `tests`; import gguf with gguf-py appended to sys.path, then remove it
  before importing the repo's namespace package `tests`.
- tests/test_bench.py timing test (decode tok/s floor) can fail under load; rerun alone before blaming a change.
