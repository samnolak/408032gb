# third_party

Upstream sources kept in the repo because the agent sandbox cannot download them
(huggingface.co is not reachable from it). Rules (AGENTS.md, Upstream pins):

- files are fetched by CI, byte-exact, at one pinned revision; nothing here is typed or edited by an agent;
- changes to upstream code live in `patches/` or in overlay modules, never in place;
- every directory carries `PIN.json` (revision, sizes, sha256, git blob ids) and `SHA256SUMS`.

| directory | upstream | license | how it gets here |
|---|---|---|---|
| `deepseek-v41-flash/` | huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash: `inference/` (reference implementation), `config.json`, `LICENSE` | MIT | job `vendor-deepseek-ref` runs `scripts/vendor_deepseek_ref.py`; write the wanted revision (40 hex, or `main`) into `deepseek-v41-flash/REQUEST` and push |
