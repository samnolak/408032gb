// Dump named intermediate tensors and per-token logits from ik_llama.cpp (CPU build), N12 (gap 5 of
// docs/reassessment/CORRECTNESS.md). The ik_llama.cpp counterpart of tools/dump/dump_tensors.cpp, written against ik's
// own API at the pin of tools/engines/pins.json (5194a9e1451b885e24dd5a065f650864e8807c59): gpt_params /
// gpt_params_parse / llama_init_from_gpt_params (common/common.h:643-744), llama_batch / llama_decode /
// llama_get_logits_ith (include/llama.h:326-344, 1104-1170), the scheduler eval callback (llama.h:474-475).
//
// Environment:
//   DUMP_DIR       output directory
//   DUMP_PREFIXES  comma-separated tensor name prefixes ("" or unset = no tensors, logits only); ik names a tensor
//                  "<cb name>-<layer>" like llama.cpp
//   DUMP_TOKENS    comma-separated token ids (fed directly, no tokenizer involved)
//   DUMP_CALLS     optional call plan: comma-separated counts, "<n>" = one llama_decode call with the next n tokens at
//                  the next n positions (default: one call with every token). Every token must be consumed.
// Output: <DUMP_DIR>/g<NNN>/<tensor name>__<k>.bin (raw float32 or int32, ggml order; f16 widened to float32) + .json
// sidecar {type, ggml_type, ne}; k = occurrence index within the graph; <DUMP_DIR>/logits.bin (one float32 row of
// n_vocab per fed token, in call order); <DUMP_DIR>/manifest.json (calls, the graphs each call computed, positions).
// Graph boundaries are found as in dump_tensors.cpp: the scheduler asks the callback about the nodes of a graph in
// node order, so a graph starts where the name of the first node ever asked comes round again.
// Usage: DUMP_DIR=d DUMP_PREFIXES=hc_,dsa_ DUMP_TOKENS=5,9,11 ./ik-dump-tensors -m tiny.gguf -c 512 -b 512 -ub 512 \
//        -t 1 -fa off -ctk f32 -ctv f16 -ictk f32 [--dsa]
// (ik refuses -ctv f32 without flash attention, src/llama.cpp:9257-9259 @5194a9e; with -fa off and the default
// -mla 3 the MLA latent cache takes the -ctv type, :1496-1497.) Run it from a scratch directory: ik's logger
// writes llama.log into the current directory.
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

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

static std::vector<std::string> split(const char * s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

struct dump_state {
    fs::path dir;
    std::vector<std::string> prefixes;
    std::string first_name;           // name of the first node ever asked: marks a graph start
    int graph = -1;                    // index of the current graph
    std::map<std::string, int> seen;   // occurrences within the current graph
    std::vector<uint8_t> buf;
};

static bool wanted(const dump_state & st, const char * name) {
    for (const auto & p : st.prefixes) {
        if (strncmp(name, p.c_str(), p.size()) == 0) return true;
    }
    return false;
}

// ik's ggml_backend_sched_eval_callback returns int (ggml/include/ggml-backend.h:180 @5194a9e); 1 = synchronize, call back
static int dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto & st = *(dump_state *) user;
    if (ask) {
        if (st.first_name.empty()) st.first_name = t->name;
        if (st.first_name == t->name) {   // a new graph
            st.graph++;
            st.seen.clear();
        }
        return wanted(st, t->name) ? 1 : 0;
    }
    if (!wanted(st, t->name)) return 1;
    const int k = st.seen[t->name]++;
    const size_t n = ggml_nelements(t);
    const size_t nbytes = ggml_nbytes(t);
    st.buf.resize(nbytes);
    ggml_backend_tensor_get(t, st.buf.data(), 0, nbytes);
    std::vector<float> f;
    const char * type = "f32";
    const void * out = nullptr;
    size_t out_bytes = 0;
    if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "ik-dump: %s is not contiguous, skipped\n", t->name);
        return 1;
    }
    if (t->type == GGML_TYPE_F32) {
        out = st.buf.data(); out_bytes = n * sizeof(float);
    } else if (t->type == GGML_TYPE_F16) {
        f.resize(n);
        const ggml_fp16_t * h = (const ggml_fp16_t *) st.buf.data();
        for (size_t i = 0; i < n; ++i) f[i] = ggml_fp16_to_fp32(h[i]);
        out = f.data(); out_bytes = n * sizeof(float);
    } else if (t->type == GGML_TYPE_I32) {
        type = "i32"; out = st.buf.data(); out_bytes = n * sizeof(int32_t);
    } else {
        fprintf(stderr, "ik-dump: %s has type %s, skipped\n", t->name, ggml_type_name(t->type));
        return 1;
    }
    char g[16];
    snprintf(g, sizeof g, "g%03d", st.graph);
    fs::path d = st.dir / g;
    fs::create_directories(d);
    const std::string base = std::string(t->name) + "__" + std::to_string(k);
    std::ofstream(d / (base + ".bin"), std::ios::binary).write((const char *) out, out_bytes);
    std::ofstream js(d / (base + ".json"));
    js << "{\"type\": \"" << type << "\", \"ggml_type\": \"" << ggml_type_name(t->type) << "\", \"ne\": ["
       << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]}\n";
    return 1;
}

int main(int argc, char ** argv) {
    const char * dir = getenv("DUMP_DIR");
    const char * toks = getenv("DUMP_TOKENS");
    if (!dir || !toks) {
        fprintf(stderr, "set DUMP_DIR and DUMP_TOKENS\n");
        return 2;
    }
    dump_state st;
    st.dir = dir;
    st.prefixes = split(getenv("DUMP_PREFIXES"));
    fs::create_directories(st.dir);

    std::vector<llama_token> tokens;
    for (const auto & s : split(toks)) tokens.push_back((llama_token) std::stoi(s));
    std::vector<int> plan;
    for (const auto & s : split(getenv("DUMP_CALLS"))) plan.push_back(std::stoi(s));
    if (plan.empty()) plan.push_back((int) tokens.size());
    size_t total = 0;
    for (int n : plan) total += n;
    if (total != tokens.size()) {
        fprintf(stderr, "DUMP_CALLS consumes %zu tokens, DUMP_TOKENS has %zu\n", total, tokens.size());
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) return 1;
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model * model = init.model;
    llama_context * ctx = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "ik-dump: failed to load the model\n");
        return 1;
    }
    const int n_vocab = llama_n_vocab(model);
    std::ofstream logits_out(st.dir / "logits.bin", std::ios::binary);
    std::ostringstream man;
    man << "{\"n_vocab\": " << n_vocab << ", \"n_ubatch\": " << params.n_ubatch << ", \"calls\": [";

    size_t tok0 = 0;
    for (size_t c = 0; c < plan.size(); ++c) {
        const int n = plan[c];
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[tok0 + i];
            batch.pos[i] = (llama_pos) (tok0 + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        const int g0 = st.graph + 1;
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "ik-dump: llama_decode returned %d at call %zu\n", rc, c);
            return 1;
        }
        for (int i = 0; i < n; ++i) {
            const float * row = llama_get_logits_ith(ctx, i);
            logits_out.write((const char *) row, sizeof(float) * n_vocab);
        }
        llama_batch_free(batch);
        man << (c ? ", " : "") << "{\"n\": " << n << ", \"tok0\": " << tok0 << ", \"pos0\": " << tok0
            << ", \"graphs\": [" << g0 << ", " << st.graph + 1 << "]}";
        tok0 += n;
    }
    man << "], \"n_graphs\": " << st.graph + 1 << "}\n";
    std::ofstream(st.dir / "manifest.json") << man.str();
    fprintf(stderr, "ik-dump: %zu calls, %d graphs, %zu tokens\n", plan.size(), st.graph + 1, tokens.size());

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
