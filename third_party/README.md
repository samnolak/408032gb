# third_party

Upstream sources kept in the repo because the agent sandbox cannot download them
(huggingface.co is not reachable from it). Rules (AGENTS.md, Upstream pins):

- files are fetched by CI, byte-exact, at one pinned revision; nothing here is typed or edited by an agent;
- changes to upstream code live in `patches/` or in overlay modules, never in place;
- every directory carries `PIN.json` (revision, sizes, sha256, git blob ids) and `SHA256SUMS`.

| directory | upstream | license | how it gets here |
|---|---|---|---|
| `deepseek-v41-flash/` | huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash: `inference/` (reference implementation), `config.json`, `LICENSE` | MIT | job `vendor-deepseek-ref` runs `scripts/vendor_deepseek_ref.py`; write the wanted revision (40 hex, or `main`) into `deepseek-v41-flash/REQUEST` and push |
| `vllm-dflash2/` | github.com/vllm-project/vllm: DFlash2 draft model, speculators and their tests (reference for ADR-011) | Apache-2.0 | `scripts/vendor_vllm_dflash2.sh [commit]` (GitHub is reachable from the sandbox; git blob ids verified) |
| `hf/<name>/` | small text files of Hugging Face repos (configs, cards, licences; never weights) listed in `hf/<name>/REQUEST.json` | per repo, recorded in `PIN.json` | job `vendor-hf` runs `scripts/vendor_hf_files.py`; edit `REQUEST.json` and push |
