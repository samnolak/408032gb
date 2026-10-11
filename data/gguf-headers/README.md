# data/gguf-headers

Requests for the workflow `gguf-headers` (`.github/workflows/gguf-headers.yml`, `scripts/gguf_headers_fetch.py`):
read ONLY the GGUF header (KV metadata + tensor infos, up to the aligned data offset) of Hugging Face GGUF files with
HTTP Range requests, on a GitHub runner (the agent sandbox cannot reach huggingface.co). docs/reassessment/CORRECTNESS.md
gap 9.

- `<name>/REQUEST.json` is written by an agent; pushing it to a `claude/*` branch starts the workflow.
- The results go to `third_party/gguf-headers/<name>/` (CI-fetched upstream bytes; agents' Edit/Write tools are blocked
  under `third_party/`): `<file>.header.gz` (the header bytes, gzip), `<file>.summary.json`, `PIN.json`, or
  `ERROR.json` when the request failed (it then stays pending). A discover request writes `LISTING.json` and reads no file.
- A request is done once its result exists; to repeat one, change the request and remove the result in a reviewed commit.
- Byte caps: at most 64 MiB per file (`max_header_bytes`), at most 1 GiB worst case per run; `--plan` checks them and
  fetches nothing. A header larger than the cap is an error, never a full download.
- Offline analysis of the committed headers: `tools/gguf_headers/analyze.py`.

| request | what |
|---|---|
| `glm-5.3-flash-ud-q4_k_xl` | the 6 GLM-5.3-Flash UD-Q4_K_XL shards (`glm5next`) of the 3090 host; repo id UNKNOWN, `unsloth/GLM-5.3-Flash-GGUF` tried |
| `qwen3.8-flash-next-ud-q4_k_xl` | the 4 Qwen3.8-Flash-Next UD-Q4_K_XL shards of R1 (`unsloth/Qwen3.8-Flash-Next-GGUF` @`38bb39e`) |
| `discover-glm-mimo` | listing of GLM-5.3-Flash and MiMo-V2.6-Flash GGUF repos with their architecture |
