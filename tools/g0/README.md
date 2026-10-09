# Gate 0 tools: expert routing histograms

1. Build (on the VM, CUDA, sm_89), out of tree; llama.cpp itself is not edited:
   ```
   cmake -S tools/g0 -B build-g0 -DLLAMA_CPP_DIR=/path/to/llama.cpp \
         -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
   cmake --build build-g0 -j --target g0-expert-hist
   ```
   llama.cpp must be checked out at ec7630a640789c393694fb194f1bbbf0369fc62d.
2. Traces: put 12+ representative prompts as `*.txt` into a directory: real agent sessions
   (coding-agent transcripts, tool calls included), each 4K-64K tokens. Do not use synthetic text:
   routing depends on the content.
3. Capture:
   ```
   G0_TRACES_DIR=traces/ G0_OUT_DIR=g0-out/ build-g0/g0-expert-hist \
       -m GLM-5.3-Flash-<quant>.gguf -ngl 99 -sm layer -c 65536 -b 2048 -ub 512
   ```
4. Analyse (288 = GLM-5.3-Flash routed experts per layer):
   ```
   python3 tools/g0/analyze.py g0-out/ --n-experts 288 -o docs/evidence/G0-glm-hit-curve.json
   python3 tools/fitplan.py models/glm-5.3-flash.json --quant Q4_K --ctx 131072 \
       --hit-curve docs/evidence/G0-glm-hit-curve.json
   ```

Limitations: prompt tokens only (generated tokens are not captured); the quantized model may
route slightly differently from the FP8 original; the held-out split is by file.

## Per-token traces and cache policies (added 2026-10-09, reassessment experiment M1/M2)

`g0-expert-trace` (same build, target `g0-expert-trace`) records, for every prompt token AND every greedily generated token,
the top-k experts of every MoE layer (`*.trace`, format in `cachesim.py`). Inputs: `*.txt` (tokenized by the model) or `*.ids`
(token ids, exact and reproducible). `G0_GEN_TOKENS` sets the generated length (default 256).
```
G0_TRACES_DIR=traces/ G0_OUT_DIR=g0-trace/ G0_GEN_TOKENS=512 build-g0/g0-expert-trace -m model.gguf -ngl 99 -c 65536 -b 2048 -ub 512
python3 tools/g0/cachesim.py g0-trace/ --n-experts 288 --capacity 0.25,0.5,0.67,0.8 --holdout-every 4 -o docs/evidence/G0-cachesim.json
```
Also reports the mean number of DISTINCT experts per layer touched by windows of k = 1, 2, 4, 8 consecutive tokens versus
independent routing: the expert-traffic multiplier of verifying a k-token speculation window (experiment S1/S3).
Policies: static (train profile), LRU, LFU with aging, TinyLFU-style admission, hybrid static+LRU (Strata's shape), and the
Belady-with-bypass offline bound. Record `-b/-ub` with every trace: routing depends slightly on batch shape
(docs/evidence/2026-10-09-g0-trace-cpu.md).
