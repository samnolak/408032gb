# DeepSeek-V4.1-Flash recon, 2026-10-02 (sandbox CPU, no GPU)

Literal command output. Pins: llama.cpp ec7630a640789c393694fb194f1bbbf0369fc62d.

## Baseline before the track (main = 0cbb1a1)
```
$ git rev-parse HEAD
0cbb1a187c58aa06a237454400cbb8aba6bb7442
$ python3 -m unittest discover -s tests | tail -3
Ran 53 tests in 1.000s

OK (skipped=17)
```
The 17 skipped tests need llama.cpp dumps (they run in CI and in tools/tinygen/run_parity*.sh).

## The sandbox cannot reach Hugging Face
```
$ curl -sS -o /dev/null -w "huggingface.co http=%{http_code}\n" -m 15 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
huggingface.co http=403
$ curl -sS -D - -o /dev/null -m 15 https://huggingface.co/ | grep -i -E 'x-deny-reason|HTTP/'
HTTP/2 403
x-deny-reason: host_not_allowed
```
CONFIRMED. Consequence: ADR-008 item 3 (the reference code is vendored by CI).

## llama.cpp has a V4 graph, not V4.1
```
$ ls src/models | grep -iE 'deepseek'          # llama.cpp @ec7630a
deepseek.cpp
deepseek2.cpp
deepseek2ocr.cpp
deepseek32.cpp
deepseek4.cpp
$ wc -l src/models/deepseek4.cpp
1501 src/models/deepseek4.cpp
$ grep -n 'DSV4_CSA_RATIO\|DSV4_HCA_RATIO' src/models/deepseek4.cpp | head -2
266:static constexpr int64_t DSV4_CSA_RATIO  = 4;
267:static constexpr int64_t DSV4_HCA_RATIO  = 128;
```
CONFIRMED: V4 compresses with ratios 4 (CSA + lightning indexer) and 128 (HCA); the V4.1 fact sheet
(docs/models/deepseek-v4.1-flash.md) lists ratios 1 and 2 with a two-level indexer, which this graph does not build.

## fitplan, native MXFP4 experts
```
$ python3 tools/fitplan.py models/deepseek-v4.1-flash.json --ctx 32768
# fitplan: DeepSeek-V4.1-Flash  experts=MXFP4 (4.2500 bpw)  dense=Q8_0  ctx=32768
All figures PROVISIONAL (see tools/fitplan.py docstring).
experts: 15744 x 35.39M params = 275.7 GiB (17.93 MiB each)
dense 9.3 GiB | KV 0.03 GiB | recurrent state 0.00 GiB
  GPU0: 32.0 GiB = fixed 4.08 + expert cache 27.92
  GPU1: 32.0 GiB = fixed 4.08 + expert cache 27.92
  GPU2: 32.0 GiB = fixed 4.08 + expert cache 27.92
  GPU3: 32.0 GiB = fixed 4.08 + expert cache 27.92
VRAM-resident experts: 40.5% (6378)  | spill to RAM: 164.0 GiB of 116 usable
Strata mode (all experts in RAM): DOES NOT FIT | disjoint VRAM/RAM: DOES NOT FIT | NVMe spill 48.0 GiB
n-gram/Engram tables on NVMe: 188.8 GiB
decode traffic per token: GPU 9349 MiB, CPU 2560 MiB  (hit 40.5%, uniform routing (worst case for a frequency cache))
  ceiling, 4-channel RAM: pipeline 38 tok/s | ideal TP4 38 tok/s  (upper bounds, no MTP)
  ceiling, 8-channel RAM: pipeline 75 tok/s | ideal TP4 76 tok/s  (upper bounds, no MTP)
```
PROVISIONAL (model, not a measurement). 275.7 GiB = routed 268.9 GiB + DSpark experts.
