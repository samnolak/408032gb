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

## Аудит доказательств 2026-10-10, F1/F2: огибающая roofline и режим движка в vs_roofline

Изменения: `tools/roofline.py` печатает и пишет в JSON огибающую (`tok_s_envelope`, `envelope_hit_vram` — максимум по hit' ≤ hit;
с перекрытием CPU/GPU граница не монотонна); `ep_imbalance` кэшируется (без этого полный прогон CLI занимал 95 с вместо 0.5 с);
`tools/bench/vs_roofline.py` получил `--overlap`, `--spec-k`, `--spec-accept`, `--draft-gb`, сравнивает с огибающей, пишет
`bound_at_hit`, `bound_hit` и входы границы (`inputs`); из докстринга убрана «калибровка ~0.70» (ROOFLINE §1 запрещает её как порог).
Мутации — в временных worktree от `abfdfd0` с незакоммиченными файлами (sha256 на момент прогона: `tools/roofline.py`
`67ca3e9fcf6a9c1eb86b1f1c5fa6a97536940822a363371aa7a07fc26b088b62` — после правки `miss_split` для ps=None; прогон E1–E3 шёл на
предыдущей версии `d68a5fd0…` того же файла, отличие — только ветка `ps is None`; `tests/test_roofline.py`
`e352b924e6c9a4c4f8e59b310db2fce6de1d016cb5b3d49fb33ee263d7d8c876`; `tools/bench/vs_roofline.py`
`3d9104d526f918f0b1b9da211a3a97773b313e992d5288e5f9b2bd371e88e702`; `tests/test_vs_roofline.py`
`a5c56ebbe308af4cb42570be2b5fe2fc46e0b07276ec05e9be5291b33610ae23`).

Огибающая в `tools/roofline.py` (`tests.test_roofline`); метка E3 в выводе неточна — мутация меняет `max` на `min` при сведении
огибающей и значения при доле попаданий:
```
baseline: Ran 26 tests in 0.490s / OK
E1 envelope ignores lower hits -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    FAIL: test_overlap_bound_peaks_below_full_hit (tests.test_roofline.Envelope.test_overlap_bound_peaks_below_full_hit)
    Ran 26 tests in 0.499s
    FAILED (failures=2)
E2 envelope searches above the hit -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    Ran 26 tests in 0.520s
    FAILED (failures=1)
E3 envelope applied to the serial mode is replaced by min -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    FAIL: test_overlap_bound_peaks_below_full_hit (tests.test_roofline.Envelope.test_overlap_bound_peaks_below_full_hit)
    Ran 26 tests in 0.499s
    FAILED (failures=2)
restored identical: True
```

`tools/bench/vs_roofline.py` (`tests.test_vs_roofline`). Первый круг: V5 (флаги спекуляции не доходят до значения при доле
попаданий) **выжила** — тест сверял только итоговую границу, которую огибающая всё равно считала со спекуляцией:
```
V5 speculation flags ignored -> SURVIVED
    Ran 3 tests in 0.011s
    OK
```
Тест дополнен сверкой `bound_at_hit` с прямым `roofline.decode(..., spec_k=2, spec_accept=0.7)`, добавлена V8; повторный прогон:
```
baseline: Ran 3 tests in 0.009s / OK
V5 speculation flags ignored -> CAUGHT
    FAIL: test_engine_mode_flags_envelope_and_inputs (tests.test_vs_roofline.VsRoofline.test_engine_mode_flags_envelope_and_inputs)
    Ran 3 tests in 0.010s
    FAILED (failures=1)
V6 bound at the hit instead of the envelope -> CAUGHT
    FAIL: test_engine_mode_flags_envelope_and_inputs (tests.test_vs_roofline.VsRoofline.test_engine_mode_flags_envelope_and_inputs)
    Ran 3 tests in 0.010s
    FAILED (failures=1)
V7 overlap flag ignored -> CAUGHT
    FAIL: test_engine_mode_flags_envelope_and_inputs (tests.test_vs_roofline.VsRoofline.test_engine_mode_flags_envelope_and_inputs)
    Ran 3 tests in 0.013s
    FAILED (failures=1)
V8 speculation dropped from the envelope -> CAUGHT
    FAIL: test_engine_mode_flags_envelope_and_inputs (tests.test_vs_roofline.VsRoofline.test_engine_mode_flags_envelope_and_inputs)
    Ran 3 tests in 0.010s
    FAILED (failures=1)
restored identical: True
```

Повтор E1–E3 на итоговом `tools/roofline.py` (`67ca3e9f…`):
```
baseline: Ran 26 tests in 0.518s / OK
E1 envelope ignores lower hits -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    FAIL: test_overlap_bound_peaks_below_full_hit (tests.test_roofline.Envelope.test_overlap_bound_peaks_below_full_hit)
    Ran 26 tests in 0.480s
    FAILED (failures=2)
E2 envelope searches above the hit -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    Ran 26 tests in 0.482s
    FAILED (failures=1)
E3 envelope applied to the serial mode is replaced by min -> CAUGHT
    FAIL: test_envelope_is_an_upper_envelope_and_monotone (tests.test_roofline.Envelope.test_envelope_is_an_upper_envelope_and_monotone)
    FAIL: test_overlap_bound_peaks_below_full_hit (tests.test_roofline.Envelope.test_overlap_bound_peaks_below_full_hit)
    Ran 26 tests in 0.482s
    FAILED (failures=2)
restored identical: True
```
