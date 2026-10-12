# data/gguf-headers

Requests for the workflow `gguf-headers` (`.github/workflows/gguf-headers.yml`, `scripts/gguf_headers_fetch.py`):
read ONLY the GGUF header (KV metadata + tensor infos, up to the aligned data offset) of Hugging Face GGUF files with
HTTP Range requests, on a GitHub runner (the agent sandbox cannot reach huggingface.co). docs/reassessment/CORRECTNESS.md
gap 9.

- `<name>/REQUEST.json` is written by an agent; pushing it to a `claude/*` branch starts the workflow.
- The results go to `third_party/gguf-headers/<name>/` (CI-fetched upstream bytes; agents' Edit/Write tools are blocked
  under `third_party/`): `<file>.header.gz` (the header bytes, gzip), `<file>.summary.json`, `PIN.json`, or
  `ERROR.json` when the request failed (it then stays pending). A discover request writes `LISTING.json` and reads no file;
  besides hub searches it may name repos (`"repos": [{"repo", "revision"}]`), whose `.gguf` paths are listed with size
  and LFS sha256 at the pinned revision (N18).
- A request is done once its result exists; to repeat one, change the request and remove the result in a reviewed commit.
- Byte caps: at most 64 MiB per file (`max_header_bytes`), at most 1 GiB worst case per run; `--plan` checks them and
  fetches nothing. A header larger than the cap is an error, never a full download.
- Offline analysis of the committed headers: `tools/gguf_headers/analyze.py` (glm5-next, qwen4exp, mimo2; with
  `"analysis": {"mode": "mtp_files"}` every file is a separate MTP model, checked against what each llama.cpp pin's
  loader asks of one).

| request | what |
|---|---|
| `glm-5.3-flash-ud-q4_k_xl` | the 6 GLM-5.3-Flash UD-Q4_K_XL shards (`glm5next`) of the 3090 host; repo id UNKNOWN, `unsloth/GLM-5.3-Flash-GGUF` tried |
| `qwen3.8-flash-next-ud-q4_k_xl` | the 4 Qwen3.8-Flash-Next UD-Q4_K_XL shards of R1 (`unsloth/Qwen3.8-Flash-Next-GGUF` @`38bb39e`) |
| `discover-glm-mimo` | listing of GLM-5.3-Flash and MiMo-V2.6-Flash GGUF repos with their architecture |
| `qwen3.8-flash-next-mtp` | N18: the 6 separate MTP files `MTP/mtp-Qwen3.8-Flash-Next-{,shared-}{BF16,Q4_K_M,Q8_0}.gguf` of `38bb39e` |
| `glm-5.3-flash-ggml-org-q4_k` | N18: `ggml-org/GLM-5.3-Flash-GGUF` @`c259a9f` Q4_K, 2 shards (`glm5-next`, llama.cpp mainline) |
| `glm-5.3-flash-ggml-org-mtp` | N18: its separate MTP files `mtp-GLM-5.3-Flash-{Q4_0,Q8_0}.gguf` |
| `discover-mimo` | N18: file lists of 4 named MiMo-V2.6-Flash GGUF repos (ggml-org MOPD / RL, 2 MOPD MXFP4) and 2 unsloth names; reads no file |
| `mimo-v2.6-flash-mopd-ggml-org-mxfp4` | N18: `ggml-org/MiMo-V2.6-Flash-MOPD-GGUF` @`e6d4bce` MXFP4, 2 shards (`mimo2`, llama.cpp mainline) |
| `mimo-v2.6-flash-mopd-ggml-org-mtp` | N18: its separate MTP file `mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf` |
| `mimo-v2.6-flash-mopd-quantaplanta-mxfp4` | N18: `QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF` @`ac33b37`, one MXFP4 file |
| `mimo-v2.6-flash-mopd-kernelpool-mxfp4` | N18: `kernelpool/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF` @`44d9191`, one MXFP4 file |
| `mimo-v2.6-flash-config` | not a request: copy of the vendored MOPD `config.json` for the analyzer cross-check (`SOURCE.md`) |
