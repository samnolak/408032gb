// Dump named intermediate tensors and per-position logits from llama.cpp (CPU or GPU build).
//
// Environment:
//   DUMP_DIR       output directory
//   DUMP_PREFIXES  comma-separated tensor name prefixes, e.g. "kda_,hc_comb,ffn_moe_topk"
//                  (llama.cpp names are "<cb name>-<layer>", src/llama-context.cpp:2638 @ec7630a)
//   DUMP_TOKENS    comma-separated token ids (fed directly, no tokenizer involved)
// Usage: DUMP_DIR=d DUMP_PREFIXES=... DUMP_TOKENS=1,5,9 ./dump-tensors -m model.gguf -c 512 -b 512 -ub 512
// Output: <DUMP_DIR>/<tensor name>__<k>.bin (raw float32/int32, ggml order) + .json sidecar with
// type and ne, k = occurrence index within the graph; logits.bin (n_tokens x n_vocab float32).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml-backend.h"

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

struct DumpState {
    std::string dir;
    std::vector<std::string> prefixes;
    std::map<std::string, int> seen;
};

static std::vector<std::string> split(const char * s) {
    std::vector<std::string> out;
    std::stringstream ss(s ? s : "");
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static bool wanted(const DumpState & st, const char * name) {
    for (const auto & p : st.prefixes) {
        if (std::strncmp(name, p.c_str(), p.size()) == 0) return true;
    }
    return false;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = static_cast<DumpState *>(ud);
    const bool ok_type = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_F16;
    if (ask) return ok_type && wanted(*st, t->name);
    if (!ok_type || !wanted(*st, t->name)) return true;
    const size_t es = ggml_type_size(t->type);
    std::vector<char> buf((size_t) ggml_nelements(t) * es);
    char * dst = buf.data();
    // copy element rows honouring strides (views and permutes are common in these graphs)
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        const size_t base = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->nb[0] == es) {
            ggml_backend_tensor_get(t, dst, base, t->ne[0] * es);
            dst += t->ne[0] * es;
        } else {
            for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                ggml_backend_tensor_get(t, dst, base + i0 * t->nb[0], es);
                dst += es;
            }
        }
    }
    if (t->type == GGML_TYPE_F16) {                       // widen to f32 so every dump reads the same way
        std::vector<char> f32(buf.size() * 2);
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(buf.data()),
                              reinterpret_cast<float *>(f32.data()), ggml_nelements(t));
        buf.swap(f32);
    }
    const int k = st->seen[t->name]++;
    const std::string stem = st->dir + "/" + t->name + "__" + std::to_string(k);
    std::ofstream(stem + ".bin", std::ios::binary).write(buf.data(), (std::streamsize) buf.size());
    std::ofstream(stem + ".json") << "{\"type\":\"" << (t->type == GGML_TYPE_I32 ? "i32" : "f32")
        << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]}\n";
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    DumpState st;
    const char * dir = std::getenv("DUMP_DIR");
    const auto toks = split(std::getenv("DUMP_TOKENS"));
    st.prefixes = split(std::getenv("DUMP_PREFIXES"));
    if (!dir || toks.empty()) {
        std::fprintf(stderr, "set DUMP_DIR, DUMP_TOKENS (and DUMP_PREFIXES)\n");
        return 2;
    }
    st.dir = dir;
    fs::create_directories(st.dir);

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    common_batch batch(ctx);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.add((llama_token) std::atoi(toks[i].c_str()), (llama_pos) i, 0, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { LOG_ERR("decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::ofstream lo(st.dir + "/logits.bin", std::ios::binary);
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        lo.write(reinterpret_cast<const char *>(l), (std::streamsize) n_vocab * sizeof(float));
    }
    std::ofstream(st.dir + "/logits.json") << "{\"n_tokens\":" << toks.size() << ",\"n_vocab\":" << n_vocab << "}\n";
    llama_backend_free();
    return 0;
}
