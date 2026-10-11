# Source of `config.json` (copy, not a request)

This directory has no `REQUEST.json`, so the `gguf-headers` workflow ignores it. `config.json` is a byte-exact copy
for `tools/gguf_headers/analyze.py`: the MiMo header requests cross-check against it (`"analysis": {"hf_config": ...}`).
N16 is not merged into this branch, so the vendored file is not under `third_party/hf/` here.

- Upstream: Hugging Face `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @`b12f5573350bb1704513fcf119fca48e498b0423`, file
  `config.json`, 8068 B, sha256 `61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`, git blob
  `9e0d9673d479e810bb2cd0ff8e581e9f16f702b4`. These values come from that branch's
  `third_party/hf/mimo-v2.6-flash-chat/PIN.json` (`"revision"` on line 3, and the `config.json` entry).
- Vendored by bot commit `1ce4d32c20e9476a33c62046ea392538b80d904f` on `claude/n16-mimo-decode-parity` (workflow
  `vendor-hf-branch`, run 38137721895), at `third_party/hf/mimo-v2.6-flash-chat/config.json`.
- Copied with `git show origin/claude/n16-mimo-decode-parity:third_party/hf/mimo-v2.6-flash-chat/config.json`. The copy
  has the same sha256 (`61bea4a0…e621`) and the same git blob id (`9e0d9673…02b4`, `git hash-object`) as the vendored
  file (N18, 2026-10-11).
