# Roofline: calibration against the one real-weights measurement, and mutation checks (2026-10-09, cloud sandbox)

Written after the adversarial review (docs/reassessment/REVIEW-adversarial.md items 4 and 11): the calibration and the mutation
checks now have literal, reproducible output in the repository.

## Calibration (PROVISIONAL inputs; the command and its output are literal)
Measurement: ik_llama.cpp, GLM-5.3-Flash UD-Q4_K_XL, one RTX 3090, `-ncmoe 39` (experts of 39 of 42 MoE layers on the CPU), prompt
2027 tokens, tg **15.0 tok/s** (PROVISIONAL: measured by the operator of host `epyc-ai`, `PROBLEMS.md`). File bpw from the same
log: experts 173.4 GiB -> 4.89 bpw; dense 8550 MiB -> 8.04 bpw if the dense part excludes the MTP layer (8.47 if it includes it:
PROVISIONAL, review item 11). `--overlap no` because with `-ncmoe` a layer's experts are entirely on the CPU or entirely on the GPU.
```
$ for ch in 4 8; do for eff in 0.5 0.6 0.7 0.8 0.9 1.0; do python3 tools/roofline.py models/specs/glm-5.3-flash.json \
    --hw hardware/epyc-ai-1x3090.partial.json --ctx 2091 --expert-bpw 4.89 --dense-bpw 8.04 --kv-bytes 1.0625 \
    --sweep-hit 0.0714 --strategies pp --ram-channels $ch --ram-eff $eff --overlap no; done; done   (tok/s column only)
ch=4 ram_eff=0.5: 9.3 tok/s
ch=4 ram_eff=0.6: 10.9 tok/s
ch=4 ram_eff=0.7: 12.4 tok/s
ch=4 ram_eff=0.8: 13.8 tok/s
ch=4 ram_eff=0.9: 15.2 tok/s
ch=4 ram_eff=1.0: 16.5 tok/s
ch=8 ram_eff=0.5: 16.5 tok/s
ch=8 ram_eff=0.6: 19.0 tok/s
ch=8 ram_eff=0.7: 21.2 tok/s
ch=8 ram_eff=0.8: 23.3 tok/s
ch=8 ram_eff=0.9: 25.2 tok/s
ch=8 ram_eff=1.0: 26.9 tok/s
```
Reading: 15.0 tok/s is consistent with 4 channels at RAM efficiency ~0.9 **or** 8 channels at ~0.45. Channels and efficiency
cannot be separated from this one number (the earlier claim "the host has more than 4-channel bandwidth" followed only from the
assumed efficiency 0.7 and is withdrawn). This is a calibration of ik_llama's CPU-expert path on that host, not of "a mature engine"
in general; it must not be transferred to all-in-VRAM configurations or used as an acceptance threshold. The server's own
calibration comes from R0 (membw) + R1 (bench) in the same mode.

## Mutation checks of tests/test_roofline.py (literal)
Each mutation is applied to tools/roofline.py, the test module is run, the file is restored from a copy.
```
M1 kv_split always n_gpu (no replication under TP):
FAIL: test_mla_latent_is_replicated_under_tp (tests.test_roofline.ReviewFixes.test_mla_latent_is_replicated_under_tp) FAILED (failures=1) 
M2 draft cost not charged in PP:
FAIL: test_draft_cost_is_charged (tests.test_roofline.ReviewFixes.test_draft_cost_is_charged) FAILED (failures=1) 
M3 TTFT back on CUDA-core FP32 (52.2):
FAIL: test_ttft_uses_tensor_core_rate (tests.test_roofline.ReviewFixes.test_ttft_uses_tensor_core_rate) FAILED (failures=1) 
M4 serial mode identical to overlap:
FAIL: test_serial_mode_is_slower_and_monotone (tests.test_roofline.ReviewFixes.test_serial_mode_is_slower_and_monotone) FAILED (failures=1) 
M5 (old) full-attention KV read independent of ctx:
FAIL: test_kv_read_scaling (tests.test_roofline.Formulas.test_kv_read_scaling) FAILED (failures=1) 
M6 (old) recurrent state not read:
FAIL: test_kv_read_scaling (tests.test_roofline.Formulas.test_kv_read_scaling) FAILED (failures=1) 
M7 (old) CPU path priced at PCIe bandwidth:
FAIL: test_all_experts_from_ram_priced_at_ram_bandwidth (tests.test_roofline.Physics.test_all_experts_from_ram_priced_at_ram_bandwidth) FAILED (failures=1) 
restored:
OK
```
7 of 7 mutations are caught (M1-M4 exercise the review fixes: TP replication of undividable KV, the draft cost, tensor-core TTFT,
the serial CPU/GPU mode; M5-M7 are the three original mutations whose output had not been recorded before).
Note (review item 11): `test_cpu_compute_beats_pcie_copy_for_batch_one` checks the code, not physics - with 143 GB/s RAM and
25 GB/s PCIe it is true by construction. The physics is experiment M4.
