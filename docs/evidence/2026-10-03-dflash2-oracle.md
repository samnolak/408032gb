# DFlash2: drafter choice and oracle against vLLM, 2026-10-03 (sandbox + CI, no GPU)

## Drafter with an open licence (CONFIRMED)
Job `vendor` of workflow vendor-hf on commit 2c02925, check-run 111268345038. Annotations, literal:
```
[notice] canada-quant/GLM-5.3-Flash-DFlash2-G @ bd03d3a38c55490fdcd9cafcfe0825b9957933ec: 3 files vendored, git blob ids verified; gated=False; ['license:apache-2.0']
[notice] canada-quant/GLM-5.3-Flash-DFlash2-G files upstream:
        1519     .gitattributes
         881     PROVENANCE.txt
        9619     README.md
        1971     config.json
        6092     launch_dflash2_tp2.sh
        9882 lfs mask_embedding.pt
  6222153560 lfs model.safetensors
[notice] vendor commit 572324bf1c1ee94a7d91ffb85b6bda41a66b02e4
```
The repo has no LICENSE file; the licence is the Hugging Face metadata tag and the model card ("Apache-2.0, no NC/ND terms").

## Reference pinned and vendored (CONFIRMED)
```
$ scripts/vendor_vllm_dflash2.sh
vendored 6 files of vllm-project/vllm @ bc21cba9673cfc2256a2726b4bc32f062237cd35 into third_party/vllm-dflash2 (git blob ids verified)
```

## Oracle against executed reference code
`tests/test_dflash2_ref.py` takes `_grouped_conv` and `_score_edges` out of the unchanged vendored file and runs them with torch.
```
$ LD_LIBRARY_PATH=~/cudastub python3 -m unittest tests.test_dflash2_ref -v          # sandbox, torch 2.14.1+cu130 on CPU
test_score_edges (tests.test_dflash2_ref.AgainstVllm.test_score_edges) ... ok
...
Ran 7 tests in 1.934s

OK
[notice] dflash2 oracle vs vLLM: Ran 7 tests in 0.032s OK  torch 2.14.1+cpu        # CI job ref-torch-cpu, check-run 111268345246
```
Mutation check (the tests can fail):
```
mutation 1 (no block-boundary mask): FAILED (failures=3)
mutation 2 (wrong predecessor shift): FAILED (failures=1)
restored: OK
```
| primitive | status |
|---|---|
| grouped convolution (`grouped_conv`, `conv_prepare`, `conv_finish`) | CONFIRMED against vLLM `_grouped_conv` (CPU branch), 1e-12, incl. the real geometry 256 groups x 16 |
| candidate edge scores (`score_edges`) | CONFIRMED against vLLM `_score_edges`, 1e-12, top-16 |
| greedy candidate walk (`selector_walk`) | PROVISIONAL: transcribed from a Triton GPU kernel; checked only against its written definition |
| context K/V (`context_kv`) | PROVISIONAL: transcribed by reading; vLLM's path uses its fused ops, not executed here |
| drafter attention layer, full block forward, mask embedding, RoPE of context keys | NOT WRITTEN |
| acceptance and speed with any target | NOT RUN (no GPU, no weights on the server) |
