// Expert-prediction probe: for a window of tokens (the tail of each prompt plus every greedily generated token) records,
// per MoE layer, the router input x (the `ffn_norm-<il>` tensor, stored as fp16), the router logits (`ffn_moe_logits-<il>`,
// fp32) and the selected experts (`ffn_moe_topk-<il>`). tools/g0/predict_eval.py then asks: how well does layer l's router
// input, pushed through layer l+d's router weights, predict the experts layer l+d will select (cross-layer "pre-gating")?
// That is the question behind predictive expert prefetch (docs/reassessment/research/R8-*.md).
//
// Same capture hook as expert_trace.cpp (params.cb_eval); llama.cpp is not edited. Tensor names at ec7630a:
// src/llama-graph.cpp:2051 `ffn_moe_logits`, :2134 `ffn_moe_topk`; `ffn_norm` is the router input in qwen3moe.cpp:134,
// deepseek2.cpp:644, glm5-next.cpp:614 (the self-check in predict_eval.py verifies W_l x_l == logits_l per model).
// Architectures without a `ffn_norm` callback before the router (e.g. qwen4exp) are reported as an inconsistent
// capture (n_embd=-1). Every prompt token is a batch output (prompt_batch below), as in expert_trace.cpp.
//
// Usage (environment as expert_trace.cpp, plus):
//   G0_PROBE_TAIL=256   prompt tokens recorded before generation starts (default 256; all prompt tokens if larger)
//   G0_TRACES_DIR=ids/ G0_OUT_DIR=out/ G0_GEN_TOKENS=128 ./g0-expert-probe -m model.gguf -c 8192 -b 512 -ub 512
// Output G0_OUT_DIR/<file>.probe, little endian:
//   b"G0PB", u32 version=1, u32 n_layers, u32 k, u32 n_embd, u32 n_expert, u32 n_rec, u32 first, u32 n_prompt,
//   u32 n_total, u32 layer_ids[n_layers], then for each recorded token (token index first .. first+n_rec-1) and each layer:
//   u16 topk[k], f16 x[n_embd], f32 logits[n_expert].
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml.h"
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

struct LayerRec {
    std::vector<uint16_t> topk;     // [tok][k]
    std::vector<ggml_fp16_t> x;     // [tok][n_embd]
    std::vector<float> logits;      // [tok][n_expert]
};

struct ProbeState {
    std::map<int, LayerRec> layers;
    int64_t pos0 = 0;               // absolute index of the first token of the ubatch being computed
    int64_t first = 0;              // first recorded token index
    int64_t k = -1, n_embd = -1, n_expert = -1;
    bool bad = false;
};

enum Kind { NONE, TOPK, NORM, LOGITS };

static Kind kind_of(const char * name, int * layer) {
    static const struct { const char * p; Kind k; } table[] = {
        {"ffn_moe_topk-", TOPK}, {"ffn_norm-", NORM}, {"ffn_moe_logits-", LOGITS}};
    for (const auto & e : table) {
        const size_t n = std::strlen(e.p);
        if (std::strncmp(name, e.p, n) == 0) {
            char * end = nullptr;
            const long v = std::strtol(name + n, &end, 10);
            if (end && *end == '\0') {
                *layer = (int) v;
                return e.k;
            }
        }
    }
    return NONE;
}

static bool probe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<ProbeState *>(user_data);
    int layer = -1;
    const Kind kind = kind_of(t->name, &layer);
    const bool want = kind != NONE && ((kind == TOPK && t->type == GGML_TYPE_I32) ||
                                       (kind != TOPK && t->type == GGML_TYPE_F32));
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int64_t width = t->ne[0];
    const int64_t n_tok = t->ne[1];
    auto & rec = st->layers[layer];
    std::vector<char> row((size_t) width * 4);
    for (int64_t r = 0; r < n_tok; ++r) {
        if (st->pos0 + r < st->first) {
            continue;
        }
        ggml_backend_tensor_get(t, row.data(), (size_t) r * t->nb[1], (size_t) width * 4);
        if (kind == TOPK) {
            if (st->k < 0) st->k = width; else if (st->k != width) st->bad = true;
            const int32_t * v = reinterpret_cast<const int32_t *>(row.data());
            for (int64_t j = 0; j < width; ++j) {
                if (v[j] < 0 || v[j] > 65535) st->bad = true;
                rec.topk.push_back((uint16_t) v[j]);
            }
        } else if (kind == NORM) {
            if (st->n_embd < 0) st->n_embd = width; else if (st->n_embd != width) st->bad = true;
            const float * v = reinterpret_cast<const float *>(row.data());
            for (int64_t j = 0; j < width; ++j) rec.x.push_back(ggml_fp32_to_fp16(v[j]));
        } else {
            if (st->n_expert < 0) st->n_expert = width; else if (st->n_expert != width) st->bad = true;
            const float * v = reinterpret_cast<const float *>(row.data());
            rec.logits.insert(rec.logits.end(), v, v + width);
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
    if (!traces_dir || !out_dir) {
        std::fprintf(stderr, "set G0_TRACES_DIR and G0_OUT_DIR (optional G0_GEN_TOKENS=128, G0_PROBE_TAIL=256)\n");
        return 2;
    }
    const int32_t n_gen = std::getenv("G0_GEN_TOKENS") ? std::atoi(std::getenv("G0_GEN_TOKENS")) : 128;
    const int32_t tail = std::getenv("G0_PROBE_TAIL") ? std::atoi(std::getenv("G0_PROBE_TAIL")) : 256;
    fs::create_directories(out_dir);

    ProbeState st;
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = probe_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    if (params.n_batch != params.n_ubatch) {
        // one llama_process call must be one graph, so that the callback's row r is token pos0 + r
        LOG_ERR("g0-expert-probe needs -b equal to -ub (got -b %d -ub %d)\n", params.n_batch, params.n_ubatch);
        return 1;
    }

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<fs::path> files;
    for (const auto & e : fs::directory_iterator(traces_dir)) {
        if (e.is_regular_file() && e.path().extension() == ".ids") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto & path : files) {
        st.layers.clear();
        st.k = st.n_embd = st.n_expert = -1;
        st.bad = false;
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string txt = read_file(path);
        std::replace(txt.begin(), txt.end(), ',', ' ');
        std::istringstream is(txt);
        std::vector<llama_token> tokens;
        long long id;
        while (is >> id) {
            if (id < 0 || id >= n_vocab) {
                LOG_ERR("%s: token id %lld out of range [0, %d)\n", path.c_str(), id, n_vocab);
                return 1;
            }
            tokens.push_back((llama_token) id);
        }
        const int32_t n_prompt = (int32_t) tokens.size();
        if (n_prompt == 0 || n_prompt + n_gen > (int32_t) llama_n_ctx(ctx)) {
            LOG_ERR("%s: %d prompt tokens + %d generated do not fit n_ctx %u\n", path.c_str(), n_prompt, n_gen,
                    llama_n_ctx(ctx));
            return 1;
        }
        st.first = std::max<int64_t>(0, (int64_t) n_prompt - tail);
        const int32_t n_batch = params.n_batch;
        for (int32_t i = 0; i < n_prompt; i += n_batch) {
            const int32_t n = std::min(n_batch, n_prompt - i);
            st.pos0 = i;   // -b == -ub: one graph per call, rows are tokens i .. i+n-1
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
                if (logits[v] > logits[best]) best = v;
            }
            if (llama_vocab_is_eog(vocab, best)) break;
            st.pos0 = n_prompt + generated;
            common_batch batch = common_batch_get_one(ctx, &best, 1);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed on %s at generated token %d\n", path.c_str(), generated);
                return 1;
            }
        }
        const int64_t n_total = n_prompt + generated;
        const int64_t n_rec = n_total - st.first;
        std::vector<int> moe;
        for (const auto & [layer, rec] : st.layers) {
            if (!rec.topk.empty()) moe.push_back(layer);
        }
        for (int l : moe) {
            const auto & rec = st.layers[l];
            if (rec.topk.size() != (size_t) (n_rec * st.k) || rec.x.size() != (size_t) (n_rec * st.n_embd) ||
                rec.logits.size() != (size_t) (n_rec * st.n_expert)) {
                st.bad = true;
            }
        }
        if (st.bad || moe.empty() || st.k <= 0 || st.n_embd <= 0 || st.n_expert <= 0) {
            LOG_ERR("%s: inconsistent capture (k=%lld n_embd=%lld n_expert=%lld moe_layers=%zu n_rec=%lld)\n", path.c_str(),
                    (long long) st.k, (long long) st.n_embd, (long long) st.n_expert, moe.size(), (long long) n_rec);
            return 1;
        }
        fs::path out = fs::path(out_dir) / (path.stem().string() + ".probe");
        std::ofstream o(out, std::ios::binary);
        o.write("G0PB", 4);
        for (uint32_t v : {1u, (uint32_t) moe.size(), (uint32_t) st.k, (uint32_t) st.n_embd, (uint32_t) st.n_expert,
                           (uint32_t) n_rec, (uint32_t) st.first, (uint32_t) n_prompt, (uint32_t) n_total}) {
            put_u32(o, v);
        }
        for (int l : moe) put_u32(o, (uint32_t) l);
        for (int64_t t = 0; t < n_rec; ++t) {
            for (int l : moe) {
                const auto & rec = st.layers[l];
                o.write(reinterpret_cast<const char *>(rec.topk.data() + t * st.k), st.k * 2);
                o.write(reinterpret_cast<const char *>(rec.x.data() + t * st.n_embd), st.n_embd * 2);
                o.write(reinterpret_cast<const char *>(rec.logits.data() + t * st.n_expert), st.n_expert * 4);
            }
        }
        LOG_INF("%s: %d prompt + %d generated tokens, recorded %lld (from %lld), %zu MoE layers, k=%lld, n_embd=%lld, "
                "n_expert=%lld -> %s\n", path.filename().c_str(), n_prompt, generated, (long long) n_rec,
                (long long) st.first, moe.size(), (long long) st.k, (long long) st.n_embd, (long long) st.n_expert,
                out.c_str());
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
