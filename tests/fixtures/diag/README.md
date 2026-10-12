# SYNTHETIC TEST FIXTURE: NOT EVIDENCE

Everything under `tests/fixtures/diag/` is **invented test data** for `tests/test_diag_kit.py`.
None of it was collected on any real machine. Do not cite any value from these files in
docs, ADRs, evidence or reports.

- `synthetic-full/`: a diag-out directory in the exact format written by
  `scripts/diag/collect.sh` (probe files with COMMAND/STDOUT/STDERR/RESULT sections plus
  `manifest.json`) and by the bench scripts (`bench/`). It is collected "as root" and has
  deliberate red flags so the checks in `scripts/diag/summarize.py` are exercised: 4 of 8
  DIMMs, DDR4-2933, an x8 GPU, a Gen3 GPU, a 16 GB GPU, a 256 MiB BAR1, P2P "NS", an Xid
  line, AER counters, ACS redirect, an NVMe at x2, a Gen3 NVMe, NVMe > 70 C, VRAM
  mismatches, a memlock limit of 8192 KiB and a shared-upstream H2D ratio.
- `synthetic-nonroot/`: a minimal run "as a normal user" inside a VM. Root-only probes are
  `needs-root` and missing tools are `not-available`. The summarizer must report UNKNOWN for
  them and must not fill anything in.

Both `manifest.json` files carry `"synthetic": true`. For such input, `summarize.py` prints a
"SYNTHETIC TEST FIXTURE - NOT EVIDENCE" banner. Model names, serials, BIOS/driver/toolkit
versions (`999.99.99`, `99.9`, `SYNTH-0.1`, ...) are fake on purpose.

The output formats imitate dmidecode, nvidia-smi, lscpu, numactl, nvme-cli, ipmitool and fio
as the author understands them. They are PROVISIONAL: they were not captured from those tools
in this sandbox, because none of them are installed here. When real outputs from the
server arrive, re-check the parsers against them.
