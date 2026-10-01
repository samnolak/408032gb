# G0 dispatch — ready to send

**ROUTING: Sol** (runs as lead-architect; delegates T1–T2 to infra-ops and T3–T4 to perf-bench, both Terra)

Prerequisites the customer provides on the VM (no internet there):
- this repo, applied from a git bundle (skill bundle-sync);
- llama.cpp source at commit ec7630a640789c393694fb194f1bbbf0369fc62d (bundle or tarball);
- one GLM-5.3-Flash GGUF (general.architecture `glm5-next`), Q2_K or IQ3 class, in `/models/glm53/`;
- 12+ real agent transcripts as `*.txt` in `/data/g0-traces/`.

---

You are the lead-architect of the 408032gb repo. Read AGENTS.md, docs/PLAN.md and docs/DECISIONS.md
first. The current gate is G0. Run it to completion using the infra-ops and perf-bench subagents.
Hard rules from AGENTS.md apply: label every number, paste literal command output, never type a
commit hash, write "NOT RUN: <reason>" for anything you could not run.

Start by printing `git log --oneline -n 5` and `git status --short` and paste them.

T1 (infra-ops) — hardware facts into docs/evidence/G0-hardware.md:
  - `sudo dmidecode -t memory | grep -E "Size:|Locator:|Speed:"` → number of populated channels;
  - `nvidia-smi -q | grep -E "Product Name|Link Width|Max|Current|PCIe Generation" -A2`;
  - `nvidia-smi topo -m`; GPU-to-GPU bandwidth with nvbandwidth or p2pBandwidthLatencyTest if available, else NOT RUN;
  - `fio --name=r --filename=<file on each 990 Pro> --rw=randread --bs=16k --iodepth=32 --direct=1 --runtime=30 --time_based --size=8G`;
  - `nvidia-smi`, `nvcc --version`, `cmake --version`, `uname -a`.
  Then set `ram_channels` in hardware/4x4080s-32g.json (label CONFIRMED with the evidence file).

T2 (infra-ops) — builds:
  - llama.cpp @ec7630a with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`, targets llama-bench and llama-cli;
  - tools/g0 per tools/g0/README.md (out of tree, no edits to llama.cpp);
  - record build commands and the last 20 lines of each build log.

T3 (perf-bench) — baseline into docs/evidence/G0-baseline.md:
  `llama-bench -m <gguf> -ngl 99 -sm layer -p 512 -n 128 -r 3`; if it does not fit, add expert
  offload with `-ot` and record the exact flags. Median pp512 and tg128 tok/s.

T4 (perf-bench) — routing histograms:
  - run g0-expert-hist on /data/g0-traces per tools/g0/README.md;
  - `python3 tools/g0/analyze.py <out> --n-experts 288 -o docs/evidence/G0-glm-hit-curve.json`;
  - `python3 tools/fitplan.py models/glm-5.3-flash.json --quant Q4_K --ctx 131072 --hit-curve docs/evidence/G0-glm-hit-curve.json`
    and the same for IQ3_XXS; paste both outputs into docs/evidence/G0-fitplan.md.

Close-out: send all four reports to evidence-auditor; then, if every acceptance criterion of G0 in
docs/PLAN.md has evidence and the auditor has no FAIL, mark G0 ACCEPTED in docs/PLAN.md following
the gate-review skill. Commit with role-prefixed messages. Run
`python3 -m unittest discover -s tests -v` before the final commit and paste the output.
Final report: the auditor's table, `git log --oneline -n 10`, and the list of anything NOT RUN.
