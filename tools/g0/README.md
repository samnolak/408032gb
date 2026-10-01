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
