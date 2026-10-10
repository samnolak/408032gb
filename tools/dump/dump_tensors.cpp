// Dump named intermediate tensors and per-position logits from llama.cpp (CPU or GPU build).
//
// Environment:
//   DUMP_DIR       output directory
//   DUMP_PREFIXES  comma-separated tensor name prefixes, e.g. "kda_,hc_comb,ffn_moe_topk"
//                  (llama.cpp names are "<cb name>-<layer>", src/llama-context.cpp:2638 @ec7630a)
//   DUMP_TOKENS    comma-separated token ids (fed directly, no tokenizer involved)
// Optional (all unset = one llama_process call with every token, the behaviour the N2 parity tests rely on):
//   DUMP_CALLS     the call plan, comma-separated: "<n>" = one llama_process call with the next n tokens of
//                  DUMP_TOKENS at the next n positions; "r<n>" = roll sequence 0 back by n positions
//                  (llama_memory_seq_rm(mem, 0, pos - n, -1)), so the next call re-decodes from there.
//                  Every token of DUMP_TOKENS must be consumed. Example (prefill 48, then decode one by one):
//                  "48,1,1,1"; with -ub 16 the 48-token call runs as 3 ubatches, i.e. 3 graphs.
//   DUMP_PER_GRAPH 1 = write the dumps of every computed graph (one per ubatch) into <DUMP_DIR>/g<NNN>/, with the
//                  occurrence index k counted within that graph, and write <DUMP_DIR>/manifest.json (calls, the
//                  graphs each call computed, positions). Without it k counts over the whole process (so the
//                  dumps of later ubatches are appended as k = m, m+1, ..., never overwritten).
//   DUMP_N_RS_SEQ  context parameter n_rs_seq (recurrent-state rollback snapshots per sequence), default 0.
//                  When set, the context is created here from the model instead of by common_init_from_params.
// Usage: DUMP_DIR=d DUMP_PREFIXES=... DUMP_TOKENS=1,5,9 ./dump-tensors -m model.gguf -c 512 -b 512 -ub 512
// Output: <DUMP_DIR>/<tensor name>__<k>.bin (raw float32/int32, ggml order) + .json sidecar with
// type and ne, k = occurrence index within the graph; logits.bin (n_tokens x n_vocab float32, one row per decoded
// token in call order).
//
// Graph boundaries: ggml_backend_sched_compute_splits asks the eval callback about every node of a graph once, in
// node order (ggml/src/ggml-backend.cpp:1820-1850 @ec7630a), so a graph starts where the name of the first node
// ever asked comes round again. The manifest records how many graphs each call computed, and the run fails if that
// differs from ceil(n_tokens / n_ubatch) (one graph per ubatch for a single sequence).
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

struct GraphInfo {
    int call    = -1;
    int n_asked = 0;
    int n_dumped = 0;
};

struct DumpState {
    std::string dir;
    std::vector<std::string> prefixes;
    std::map<std::string, int> seen;
    // per-graph mode
    bool per_graph = false;
    std::string first_node;      // name of the first node ever asked: marks the start of every graph
    int graph = -1;              // index of the graph being computed
    int call  = -1;              // index of the llama_process call in progress
    std::vector<GraphInfo> graphs;
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

static char graph_dir_buf[32];

static std::string graph_dir(const DumpState & st) {
    std::snprintf(graph_dir_buf, sizeof(graph_dir_buf), "g%03d", st.graph);
    return st.dir + "/" + graph_dir_buf;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = static_cast<DumpState *>(ud);
    if (ask && st->per_graph) {
        if (st->graph < 0 || (st->first_node == t->name && st->graphs[st->graph].n_asked > 0)) {
            if (st->graph < 0) st->first_node = t->name;
            st->graph++;
            st->graphs.push_back(GraphInfo{st->call, 0, 0});
            st->seen.clear();
            fs::create_directories(graph_dir(*st));
        }
        st->graphs[st->graph].n_asked++;
    }
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
    const std::string where = st->per_graph ? graph_dir(*st) : st->dir;
    const std::string stem = where + "/" + t->name + "__" + std::to_string(k);
    std::ofstream(stem + ".bin", std::ios::binary).write(buf.data(), (std::streamsize) buf.size());
    std::ofstream(stem + ".json") << "{\"type\":\"" << (t->type == GGML_TYPE_I32 ? "i32" : "f32")
        << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]}\n";
    if (st->per_graph) st->graphs[st->graph].n_dumped++;
    return true;
}

struct CallRec {
    bool rollback = false;
    int  n        = 0;    // tokens decoded, or positions rolled back
    int  pos0     = 0;    // first position decoded, or the position the sequence was cut at
    int  tok0     = 0;    // index into DUMP_TOKENS of the first token decoded
    int  g0 = -1, g1 = -1; // graphs [g0, g1) computed by this call
};

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
    const char * per_graph = std::getenv("DUMP_PER_GRAPH");
    st.per_graph = per_graph && std::strcmp(per_graph, "0") != 0 && *per_graph;
    const auto plan = split(std::getenv("DUMP_CALLS"));
    const char * n_rs_env = std::getenv("DUMP_N_RS_SEQ");
    const int n_rs_seq = n_rs_env ? std::atoi(n_rs_env) : 0;

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    common_init_result_ptr init;
    llama_context_ptr own_ctx;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    if (n_rs_seq > 0) {
        // common_context_params_to_llama derives n_rs_seq from the speculative settings only, so set it here
        init = common_init_from_params(params, /*model_only=*/true);
        model = init->model();
        if (model) {
            auto cparams = common_context_params_to_llama(params);
            cparams.n_rs_seq = (uint32_t) n_rs_seq;
            own_ctx.reset(llama_init_from_model(model, cparams));
            ctx = own_ctx.get();
            if (ctx && llama_n_rs_seq(ctx) != (uint32_t) n_rs_seq) {
                LOG_ERR("n_rs_seq: asked %d, the context has %u\n", n_rs_seq, llama_n_rs_seq(ctx));
                return 1;
            }
        }
    } else {
        init = common_init_from_params(params);
        model = init->model();
        ctx = init->context();
    }
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_ubatch = (int) llama_n_ubatch(ctx);
    std::ofstream lo(st.dir + "/logits.bin", std::ios::binary);
    std::vector<CallRec> calls;
    common_batch batch(ctx);

    auto decode = [&](int tok0, int pos0, int n) -> bool {
        batch.clear();
        for (int i = 0; i < n; ++i) {
            batch.add((llama_token) std::atoi(toks[tok0 + i].c_str()), (llama_pos) (pos0 + i), 0, true);
        }
        CallRec rec;
        rec.n = n; rec.pos0 = pos0; rec.tok0 = tok0;
        st.call = (int) calls.size();
        rec.g0 = st.graph + 1;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { LOG_ERR("decode failed\n"); return false; }
        rec.g1 = st.graph + 1;
        for (int i = 0; i < n; ++i) {
            const float * l = llama_get_logits_ith(ctx, (int32_t) i);
            lo.write(reinterpret_cast<const char *>(l), (std::streamsize) n_vocab * sizeof(float));
        }
        if (st.per_graph && rec.g1 - rec.g0 != (n + n_ubatch - 1) / n_ubatch) {
            LOG_ERR("call %d: %d tokens computed %d graphs, expected %d (one per ubatch of %d)\n",
                    st.call, n, rec.g1 - rec.g0, (n + n_ubatch - 1) / n_ubatch, n_ubatch);
            return false;
        }
        calls.push_back(rec);
        return true;
    };

    int n_decoded = 0;
    if (plan.empty()) {
        if (!decode(0, 0, (int) toks.size())) return 1;
        n_decoded = (int) toks.size();
    } else {
        int ti = 0, pos = 0;
        for (const auto & step : plan) {
            if (step[0] == 'r') {
                const int n = std::atoi(step.c_str() + 1);
                if (n <= 0 || n > pos) { LOG_ERR("bad rollback step '%s' at pos %d\n", step.c_str(), pos); return 1; }
                pos -= n;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) {
                    LOG_ERR("llama_memory_seq_rm(0, %d, -1) refused a rollback of %d\n", pos, n);
                    return 1;
                }
                CallRec rec;
                rec.rollback = true; rec.n = n; rec.pos0 = pos; rec.tok0 = ti;
                rec.g0 = rec.g1 = st.graph + 1;
                calls.push_back(rec);
                continue;
            }
            const int n = std::atoi(step.c_str());
            if (n <= 0 || ti + n > (int) toks.size()) { LOG_ERR("bad call step '%s'\n", step.c_str()); return 1; }
            if (!decode(ti, pos, n)) return 1;
            ti += n; pos += n; n_decoded += n;
        }
        if (ti != (int) toks.size()) { LOG_ERR("DUMP_CALLS consumed %d of %zu tokens\n", ti, toks.size()); return 1; }
    }
    lo.close();
    std::ofstream(st.dir + "/logits.json") << "{\"n_tokens\":" << n_decoded << ",\"n_vocab\":" << n_vocab << "}\n";

    if (st.per_graph || !plan.empty()) {
        std::ofstream mf(st.dir + "/manifest.json");
        mf << "{\"n_ubatch\":" << n_ubatch << ",\"n_rs_seq\":" << llama_n_rs_seq(ctx)
           << ",\"per_graph\":" << (st.per_graph ? "true" : "false")
           << ",\"first_node\":\"" << st.first_node << "\",\"calls\":[";
        for (size_t c = 0; c < calls.size(); ++c) {
            const auto & r = calls[c];
            mf << (c ? "," : "") << "{\"kind\":\"" << (r.rollback ? "rollback" : "decode") << "\",\"n\":" << r.n
               << ",\"pos0\":" << r.pos0 << ",\"tok0\":" << r.tok0 << ",\"graphs\":[" << r.g0 << "," << r.g1 << "]}";
        }
        mf << "],\"graphs\":[";
        for (size_t g = 0; g < st.graphs.size(); ++g) {
            const auto & gi = st.graphs[g];
            mf << (g ? "," : "") << "{\"call\":" << gi.call << ",\"n_asked\":" << gi.n_asked
               << ",\"n_dumped\":" << gi.n_dumped << "}";
        }
        mf << "]}\n";
    }
    own_ctx.reset();
    llama_backend_free();
    return 0;
}
