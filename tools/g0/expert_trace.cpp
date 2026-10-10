// Gate 0 / reassessment: per-token routed-expert traces from llama.cpp (any MoE architecture), prompt AND
// generated tokens, for trace-driven cache simulation (tools/g0/cachesim.py).
//
// Same capture hook as expert_hist.cpp: the `ffn_moe_topk-<layer>` tensors (llama.cpp src/llama-graph.cpp:2134
// @ec7630a, `cb(selected_experts, "ffn_moe_topk", il)`) through params.cb_eval.
//
// Usage (paths via environment, everything else is normal llama.cpp arguments):
//   G0_TRACES_DIR=traces/ G0_OUT_DIR=out/ G0_GEN_TOKENS=256 ./g0-expert-trace -m model.gguf -ngl 99 -c 65536 -b 2048
// For every *.txt (text, tokenized by the model) or *.ids (token ids separated by commas or whitespace; exact and
// reproducible, and usable with test models that have no real tokenizer) in G0_TRACES_DIR: clear the context, feed
// the prompt in n_batch chunks (every prompt token a batch output, see prompt_batch), then generate
// G0_GEN_TOKENS tokens greedily (argmax, stops at end-of-generation), and write G0_OUT_DIR/<file>.trace:
//   b"G0TR", u32 version=1, u32 n_layers, u32 k, u32 n_tokens, u32 n_prompt, u32 layer_ids[n_layers],
//   then n_tokens x n_layers x k u16 expert ids (token-major).
// Rows of one layer arrive in token order (ubatches are processed in order), so each layer is a simple append.
// A prompt that does not fit (prompt + G0_GEN_TOKENS > n_ctx) is an error; G0_TRUNCATE=1 keeps the old behaviour (cut the
// tail of the prompt) and says so on stderr. *.txt is tokenized as plain text: NO chat template is applied, so agent
// traces must be rendered with the model's template beforehand (or given as *.ids from the serving engine's /tokenize).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
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

struct TraceState {
    std::map<int, std::vector<uint16_t>> rows;  // layer -> flattened [token][k]
    std::vector<int32_t> row;
    int64_t k = -1;
    bool bad = false;
};

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<TraceState *>(user_data);
    const size_t plen = std::strlen(kPrefix);
    const bool want = std::strncmp(t->name, kPrefix, plen) == 0 && t->type == GGML_TYPE_I32;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int layer = std::atoi(t->name + plen);
    const int64_t k = t->ne[0];
    const int64_t n_tok = t->ne[1];
    if (st->k < 0) {
        st->k = k;
    } else if (st->k != k) {
        st->bad = true;  // mixed top-k across layers: not supported by the format
    }
    st->row.resize((size_t) k);
    auto & dst = st->rows[layer];
    for (int64_t r = 0; r < n_tok; ++r) {
        ggml_backend_tensor_get(t, st->row.data(), (size_t) r * t->nb[1], (size_t) k * sizeof(int32_t));
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = st->row[(size_t) j];
            if (e < 0 || e > 65535) {
                st->bad = true;
            }
            dst.push_back((uint16_t) e);
        }
    }
    return true;
}

static std::string read_file(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void put_u32(std::ofstream & o, uint32_t v) {
    unsigned char b[4] = {(unsigned char) (v & 0xff), (unsigned char) ((v >> 8) & 0xff),
                          (unsigned char) ((v >> 16) & 0xff), (unsigned char) ((v >> 24) & 0xff)};
    o.write(reinterpret_cast<const char *>(b), 4);
}


// Every prompt token is an output. With only the last token as output (common_batch_get_one), the model graphs drop
// the other rows before the last layer's FFN (`if (il == n_layer - 1 && inp_out_ids) cur = ggml_get_rows(...)`,
// e.g. src/models/qwen3moe.cpp:123-125 and deepseek2.cpp:636-638 @ec7630a), so the last MoE layer would see one row
// per prompt batch instead of n. Asking for all outputs keeps all rows; the routing itself is unchanged.
static common_batch prompt_batch(llama_context * ctx, const llama_token * tokens, int32_t n) {
    common_batch batch = common_batch_get_one(ctx, tokens, n);
    for (int32_t j = 0; j < n; ++j) {
        batch.set_output(j, true);
    }
    return batch;
}

static int run(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * traces_dir = std::getenv("G0_TRACES_DIR");
    const char * out_dir = std::getenv("G0_OUT_DIR");
    const char * gen_env = std::getenv("G0_GEN_TOKENS");
    if (!traces_dir || !out_dir) {
        std::fprintf(stderr, "set G0_TRACES_DIR and G0_OUT_DIR (optional G0_GEN_TOKENS, default 256)\n");
        return 2;
    }
    const int32_t n_gen = gen_env ? std::atoi(gen_env) : 256;
    const char * trunc_env = std::getenv("G0_TRUNCATE");
    const bool allow_truncate = trunc_env && std::strcmp(trunc_env, "1") == 0;
    fs::create_directories(out_dir);

    TraceState st;
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = trace_cb;
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
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<fs::path> files;
    for (const auto & e : fs::directory_iterator(traces_dir)) {
        if (e.is_regular_file() && (e.path().extension() == ".txt" || e.path().extension() == ".ids")) {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto & path : files) {
        st.rows.clear();
        st.k = -1;
        st.bad = false;
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> tokens;
        if (path.extension() == ".ids") {
            std::string txt = read_file(path);
            for (char & c : txt) {
                if (c == ',') {
                    c = ' ';
                }
            }
            std::istringstream is(txt);
            long long id;
            while (is >> id) {
                if (id < 0 || id >= n_vocab) {
                    LOG_ERR("%s: token id %lld out of range [0, %d)\n", path.c_str(), id, n_vocab);
                    return 1;
                }
                tokens.push_back((llama_token) id);
            }
        } else {
            tokens = common_tokenize(ctx, read_file(path), add_bos, true);
        }
        if (tokens.empty()) {
            LOG_ERR("%s: no tokens\n", path.c_str());
            return 1;
        }
        const int32_t n_ctx = (int32_t) llama_n_ctx(ctx);
        if ((int32_t) tokens.size() + n_gen > n_ctx) {
            if (!allow_truncate) {
                LOG_ERR("%s: prompt %zu + G0_GEN_TOKENS %d exceeds n_ctx %d; raise -c or set G0_TRUNCATE=1\n",
                        path.c_str(), tokens.size(), n_gen, n_ctx);
                return 1;
            }
            LOG_WRN("%s: prompt truncated from %zu to %d tokens (G0_TRUNCATE=1)\n", path.c_str(), tokens.size(),
                    std::max(1, n_ctx - n_gen));
            tokens.resize((size_t) std::max(1, n_ctx - n_gen));
        }
        const int32_t n_prompt = (int32_t) tokens.size();
        const int32_t n_batch = params.n_batch;
        for (int32_t i = 0; i < n_prompt; i += n_batch) {
            const int32_t n = std::min(n_batch, n_prompt - i);
            common_batch batch = prompt_batch(ctx, tokens.data() + i, n);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed on %s at token %d\n", path.c_str(), i);
                return 1;
            }
        }
        int32_t generated = 0;
        for (; generated < n_gen; ++generated) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            llama_token best = 0;
            for (llama_token v = 1; v < n_vocab; ++v) {
                if (logits[v] > logits[best]) {
                    best = v;
                }
            }
            if (llama_vocab_is_eog(vocab, best)) {
                break;
            }
            common_batch batch = common_batch_get_one(ctx, &best, 1);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed on %s at generated token %d\n", path.c_str(), generated);
                return 1;
            }
        }
        // the last generated token was fed, so traced tokens = prompt + generated
        const uint32_t n_tokens = (uint32_t) (n_prompt + generated);
        std::vector<int> layers;
        for (const auto & [layer, v] : st.rows) {
            layers.push_back(layer);
            if (st.k > 0 && v.size() != (size_t) n_tokens * (size_t) st.k) {
                st.bad = true;
            }
        }
        if (st.bad || st.k <= 0 || layers.empty()) {
            LOG_ERR("%s: inconsistent capture (k=%lld, layers=%zu); no trace written\n", path.c_str(),
                    (long long) st.k, layers.size());
            return 1;
        }
        fs::path out = fs::path(out_dir) / (path.stem().string() + ".trace");
        std::ofstream o(out, std::ios::binary);
        o.write("G0TR", 4);
        put_u32(o, 1);
        put_u32(o, (uint32_t) layers.size());
        put_u32(o, (uint32_t) st.k);
        put_u32(o, n_tokens);
        put_u32(o, (uint32_t) n_prompt);
        for (int l : layers) {
            put_u32(o, (uint32_t) l);
        }
        for (uint32_t t = 0; t < n_tokens; ++t) {
            for (int l : layers) {
                const uint16_t * src = st.rows[l].data() + (size_t) t * (size_t) st.k;
                for (int64_t j = 0; j < st.k; ++j) {
                    unsigned char b[2] = {(unsigned char) (src[j] & 0xff), (unsigned char) (src[j] >> 8)};
                    o.write(reinterpret_cast<const char *>(b), 2);
                }
            }
        }
        LOG_INF("%s: %d prompt + %d generated tokens, %zu MoE layers, k=%lld -> %s\n", path.filename().c_str(),
                n_prompt, generated, layers.size(), (long long) st.k, out.c_str());
    }
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    const int rc = run(argc, argv);
    // the logger is asynchronous: without a flush, an error logged just before `return 1` can be lost
    common_log_flush(common_log_main());
    return rc;
}
