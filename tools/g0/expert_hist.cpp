// Gate 0: per-layer routed-expert histograms from llama.cpp (any MoE architecture).
//
// Captures the `ffn_moe_topk-<layer>` tensors (llama.cpp src/llama-graph.cpp:2134 @ec7630a:
// `cb(selected_experts, "ffn_moe_topk", il)`; the "-<layer>" suffix is added in
// src/llama-context.cpp:2638) through params.cb_eval, the same hook as examples/eval-callback.
//
// Usage (paths via environment, everything else is normal llama.cpp arguments):
//   G0_TRACES_DIR=traces/ G0_OUT_DIR=out/ ./expert_hist -m model.gguf -ngl 99 -c 32768 -b 2048
// For every *.txt file in G0_TRACES_DIR it clears the context, feeds the file's tokens in
// n_batch chunks, and writes G0_OUT_DIR/<file>.csv with rows "layer,expert,count".
// Prompt tokens only: routing of generated tokens is not captured (stated limitation).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const char * kPrefix = "ffn_moe_topk-";

struct HistState {
    std::map<int, std::map<int32_t, uint64_t>> counts;  // layer -> expert -> count
    std::vector<int32_t> row;
    uint64_t captured = 0;
};

static bool g0_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<HistState *>(user_data);
    const size_t plen = std::strlen(kPrefix);
    const bool want = std::strncmp(t->name, kPrefix, plen) == 0 && t->type == GGML_TYPE_I32;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int layer = std::atoi(t->name + plen);
    const int64_t k = t->ne[0];        // experts per token
    const int64_t n_tok = t->ne[1];    // tokens in this ubatch
    st->row.resize((size_t) k);
    // read row by row: the tensor may be a strided view of the argsort result
    for (int64_t r = 0; r < n_tok; ++r) {
        ggml_backend_tensor_get(t, st->row.data(), (size_t) r * t->nb[1], (size_t) k * sizeof(int32_t));
        auto & layer_counts = st->counts[layer];
        for (int64_t j = 0; j < k; ++j) {
            layer_counts[st->row[(size_t) j]]++;
        }
    }
    st->captured++;
    return true;
}

static std::string read_file(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * traces_dir = std::getenv("G0_TRACES_DIR");
    const char * out_dir = std::getenv("G0_OUT_DIR");
    if (!traces_dir || !out_dir) {
        std::fprintf(stderr, "set G0_TRACES_DIR and G0_OUT_DIR\n");
        return 2;
    }
    fs::create_directories(out_dir);

    HistState st;
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = g0_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<fs::path> files;
    for (const auto & e : fs::directory_iterator(traces_dir)) {
        if (e.is_regular_file() && e.path().extension() == ".txt") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto & path : files) {
        st.counts.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> tokens = common_tokenize(ctx, read_file(path), add_bos, true);
        const int32_t n_ctx = (int32_t) llama_n_ctx(ctx);
        if ((int32_t) tokens.size() > n_ctx) {
            tokens.resize((size_t) n_ctx);  // truncated: recorded in the CSV header
        }
        const int32_t n_batch = params.n_batch;
        for (int32_t i = 0; i < (int32_t) tokens.size(); i += n_batch) {
            const int32_t n = std::min(n_batch, (int32_t) tokens.size() - i);
            common_batch batch = common_batch_get_one(ctx, tokens.data() + i, n);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed on %s at token %d\n", path.c_str(), i);
                return 1;
            }
        }
        fs::path out = fs::path(out_dir) / (path.stem().string() + ".csv");
        std::ofstream o(out);
        o << "# trace=" << path.filename().string() << " tokens=" << tokens.size() << "\n";
        o << "layer,expert,count\n";
        for (const auto & [layer, m] : st.counts) {
            for (const auto & [expert, c] : m) {
                o << layer << "," << expert << "," << c << "\n";
            }
        }
        LOG_INF("%s: %zu tokens, %zu MoE layers -> %s\n", path.filename().c_str(), tokens.size(),
                st.counts.size(), out.c_str());
    }
    llama_backend_free();
    return 0;
}
