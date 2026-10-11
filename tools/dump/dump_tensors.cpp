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
//   DUMP_MTP       1 = also run the model's MTP block (nextn_predict_layers > 0) the way common/speculative.cpp
//                  drives it (lines 1511-1529 and 1700-1704 @ec7630a): the model is loaded with load_mtp, the target
//                  context exports h_nextn (llama_set_embeddings_nextn(ctx, true, false)), and after every target
//                  call a second context of type LLAMA_CONTEXT_TYPE_MTP processes the same tokens at the same
//                  positions, token k paired with the target h_nextn row k-1 (the first token of a call with the
//                  last row of the previous call, zeros at the start).
//                  Its dumps go to <DUMP_DIR>/mtp/ (same layout), with logits.bin, h_in.bin (the h row fed with
//                  every MTP token), h_nextn.bin (the MTP output rows) and manifest.json; <DUMP_DIR>/h_nextn.bin
//                  holds the target rows.
//                  Several MTP heads (N17; nextn_predict_layers = H > 1, common/speculative.cpp chain_heads,
//                  lines 1482, 1598-1621, 1700-1708, 1767-1777 @79e2e74, the same code at 1421, 1534-1557, 1623-1631,
//                  1686-1696 @ec7630a): every target call is processed by every head h = 0..H-1 in turn, each after
//                  llama_memory_seq_rm(mtp, 0, first position of the call, -1) and llama_set_nextn_layer_offset(h),
//                  all with the target's h rows; the offset is set back to 0 afterwards.
//                  Plan steps in this mode (DUMP_CALLS): "<n>" as above; "r<n>" rolls back both contexts by n
//                  positions the way llama-server rejects a draft (common_memory::seq_rm on the target and the MTP
//                  memory, tools/server/server-context.cpp:4505 @79e2e74, 3998 @ec7630a) and pairs the next token with
//                  the target row of the new last position (common_speculative accept(), speculative.cpp:1824-1826
//                  @79e2e74); n must be smaller than the last call (the server keeps at least its first token);
//                  "d<n>" drafts at the current position with the next token of DUMP_TOKENS as the target's sampled
//                  token (not consumed: the next call feeds it), n steps (H > 1: at most H, speculative.cpp:1485),
//                  then llama_memory_seq_rm(mtp, 0, position, -1) as the server does after drafting
//                  (server-context.cpp:3423 @79e2e74, 3063 @ec7630a).
//                  Draft steps (common/speculative.cpp draft()): one head: step 1 the token with the pending target
//                  row at the position, step j > 1 the argmax of step j-1 at the next position with step j-1's MTP
//                  h_nextn row. H > 1: step j runs head j-1 on the whole draft prefix (the token, then drafts 1..j-1
//                  at the following positions) after llama_memory_seq_rm(mtp, 0, position, -1), the rows paired with
//                  the pending target row, then the h_nextn row each earlier step output at its last position; only
//                  the last row is an output (speculative.cpp:1700-1708, 1767-1777 @79e2e74). Greedy (p_min 0, the
//                  default, common/common.h:333 @79e2e74). Process calls mark every row as an output (the server
//                  marks none, speculative.cpp:1586) so their logits and h rows can be compared; the block's tensors
//                  before the output crop (mimo2.cpp:378 @79e2e74) do not depend on it.
//                  The MTP manifest records per call its kind (process | draft | rollback | seq_rm, the last the
//                  server's removal after a "d" draft), head, tokens, out0 (first row in logits.bin / h_nextn.bin) and
//                  outputs (the output batch indices).
//   DUMP_MTP_DRAFT n = after the plan, a draft of n steps as "d<n>", from the target's greedy token after the last
//                  position (argmax of the last target logits).
//   DUMP_MTP_NO_RESERVE 1 = skip the scheduler re-reserve after llama_set_embeddings_nextn (see main()); at
//                  @ec7630a the first MTP graph then crashes in ggml_gallocr_init_tensor (kept to reproduce it).
// When DUMP_N_RS_SEQ or DUMP_MTP is set, the model and the context(s) are created here instead of by
// common_init_from_params.
// Usage: DUMP_DIR=d DUMP_PREFIXES=... DUMP_TOKENS=1,5,9 ./dump-tensors -m model.gguf -c 512 -b 512 -ub 512
// Output: <DUMP_DIR>/<tensor name>__<k>.bin (raw float32/int32, ggml order; f16 tensors widened to float32) + .json
// sidecar with type (of the .bin), ggml_type (of the tensor) and ne, k = occurrence index within the graph; logits.bin
// (n_tokens x n_vocab float32, one row per decoded token in call order).
//
// Graph boundaries: ggml_backend_sched_compute_splits asks the eval callback about every node of a graph once, in
// node order (ggml/src/ggml-backend.cpp:1820-1850 @ec7630a), so a graph starts where the name of the first node
// ever asked comes round again. The manifest records how many graphs each call computed, and the run fails if that
// differs from ceil(n_tokens / n_ubatch) (one graph per ubatch for a single sequence).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-ext.h"   // staging API (llama.cpp src/): llama_set_embeddings_nextn, llama_get_embeddings_nextn_ith
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
    bool new_call = false;       // MTP context: the next node asked starts a graph (heads name their nodes by layer)
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

static std::string graph_dir(const DumpState & st) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "g%03d", st.graph);
    return st.dir + "/" + buf;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = static_cast<DumpState *>(ud);
    if (ask && st->per_graph) {
        if (st->graph < 0 || st->new_call || (st->first_node == t->name && st->graphs[st->graph].n_asked > 0)) {
            if (st->graph < 0 || st->new_call) st->first_node = t->name;
            st->new_call = false;
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
    // "type" is the type of the .bin (f16 widened to f32); "ggml_type" the tensor's own type (e.g. an f16 KV cache view)
    std::ofstream(stem + ".json") << "{\"type\":\"" << (t->type == GGML_TYPE_I32 ? "i32" : "f32")
        << "\",\"ggml_type\":\"" << ggml_type_name(t->type)
        << "\",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]}\n";
    if (st->per_graph) st->graphs[st->graph].n_dumped++;
    return true;
}

struct CallRec {
    std::string kind = "decode"; // decode | rollback (target); process | draft | rollback (MTP context)
    int  n        = 0;    // tokens decoded, or positions rolled back
    int  pos0     = 0;    // first position decoded, or the position the sequence was cut at
    int  tok0     = 0;    // index into DUMP_TOKENS of the first token decoded (MTP: first row of h_in.bin)
    int  g0 = -1, g1 = -1; // graphs [g0, g1) computed by this call
    std::vector<int> tokens; // MTP calls: the token ids fed
    int  head     = -1;   // MTP calls: nextn_layer_offset
    int  out0     = -1;   // MTP calls: first row of logits.bin / h_nextn.bin
    std::vector<int> outputs; // MTP calls: batch indices of the output rows
};

static void write_manifest(const DumpState & st, const std::vector<CallRec> & calls, int n_ubatch, uint32_t n_rs_seq,
                           int n_embd_out) {
    std::ofstream mf(st.dir + "/manifest.json");
    mf << "{\"n_ubatch\":" << n_ubatch << ",\"n_rs_seq\":" << n_rs_seq
       << ",\"per_graph\":" << (st.per_graph ? "true" : "false");
    if (n_embd_out > 0) mf << ",\"n_embd_out\":" << n_embd_out;
    mf << ",\"first_node\":\"" << st.first_node << "\",\"calls\":[";
    for (size_t c = 0; c < calls.size(); ++c) {
        const auto & r = calls[c];
        mf << (c ? "," : "") << "{\"kind\":\"" << r.kind << "\",\"n\":" << r.n
           << ",\"pos0\":" << r.pos0 << ",\"tok0\":" << r.tok0 << ",\"graphs\":[" << r.g0 << "," << r.g1 << "]";
        if (!r.tokens.empty()) {
            mf << ",\"tokens\":[";
            for (size_t i = 0; i < r.tokens.size(); ++i) mf << (i ? "," : "") << r.tokens[i];
            mf << "]";
        }
        if (r.head >= 0) mf << ",\"head\":" << r.head;
        if (r.out0 >= 0) {
            mf << ",\"out0\":" << r.out0 << ",\"outputs\":[";
            for (size_t i = 0; i < r.outputs.size(); ++i) mf << (i ? "," : "") << r.outputs[i];
            mf << "]";
        }
        mf << "}";
    }
    mf << "],\"graphs\":[";
    for (size_t g = 0; g < st.graphs.size(); ++g) {
        const auto & gi = st.graphs[g];
        mf << (g ? "," : "") << "{\"call\":" << gi.call << ",\"n_asked\":" << gi.n_asked
           << ",\"n_dumped\":" << gi.n_dumped << "}";
    }
    mf << "]}\n";
}

static int argmax(const float * x, int n) {
    return (int) (std::max_element(x, x + n) - x);
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
    const char * per_graph = std::getenv("DUMP_PER_GRAPH");
    st.per_graph = per_graph && std::strcmp(per_graph, "0") != 0 && *per_graph;
    const auto plan = split(std::getenv("DUMP_CALLS"));
    const char * n_rs_env = std::getenv("DUMP_N_RS_SEQ");
    const int n_rs_seq = n_rs_env ? std::atoi(n_rs_env) : 0;
    const char * mtp_env = std::getenv("DUMP_MTP");
    const bool mtp = mtp_env && std::strcmp(mtp_env, "0") != 0 && *mtp_env;
    const char * draft_env = std::getenv("DUMP_MTP_DRAFT");
    const int n_draft = draft_env ? std::atoi(draft_env) : 0;

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    common_init_result_ptr init;
    llama_model_ptr own_model;
    llama_context_ptr own_ctx, mtp_ctx;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    DumpState st_mtp;
    if (n_rs_seq > 0 || mtp) {
        // common_context_params_to_llama derives n_rs_seq (and common_model_params_to_llama load_mtp) from the
        // speculative settings only, so set them here
        auto mparams = common_model_params_to_llama(params);
        mparams.load_mtp = mtp;
        own_model.reset(llama_model_load_from_file(params.model.path.c_str(), mparams));
        model = own_model.get();
        if (model) {
            auto cparams = common_context_params_to_llama(params);
            cparams.n_rs_seq = (uint32_t) n_rs_seq;
            own_ctx.reset(llama_init_from_model(model, cparams));
            ctx = own_ctx.get();
            if (ctx && llama_n_rs_seq(ctx) != (uint32_t) n_rs_seq) {
                LOG_ERR("n_rs_seq: asked %d, the context has %u\n", n_rs_seq, llama_n_rs_seq(ctx));
                return 1;
            }
            if (ctx && mtp) {
                st_mtp.dir = st.dir + "/mtp";
                st_mtp.prefixes = st.prefixes;
                st_mtp.per_graph = st.per_graph;
                fs::create_directories(st_mtp.dir);
                auto cparams_mtp = common_context_params_to_llama(params);
                cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                cparams_mtp.n_rs_seq = 0;
                cparams_mtp.cb_eval_user_data = &st_mtp;
                mtp_ctx.reset(llama_init_from_model(model, cparams_mtp));
                if (!mtp_ctx) { LOG_ERR("failed to create the MTP context\n"); return 1; }
                llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
                llama_set_embeddings_nextn(mtp_ctx.get(), true, /*masked*/ true);
                // set_embeddings_nextn does not ask for a scheduler reserve (src/llama-context.cpp:1230-1235 @ec7630a),
                // so the first MTP graph crashes (SIGSEGV at ggml-alloc.c:987, out_ids with buffer_id -1; CONFIRMED,
                // docs/evidence/2026-10-10-qwen4exp-decode-parity.md finding 12). PROVISIONAL mechanism: it reuses the
                // allocation of the graph reserved at context creation (same node and leaf counts, the out_ids leaf
                // elsewhere), because ggml_gallocr_needs_realloc compares counts and nodes only
                // (ggml/src/ggml-alloc.c:1009-1050). Re-setting a default that does ask for the reserve (src/llama-context.cpp:
                // 1237-1245) avoids it; embeddings_layer_inp[0] stays false. DUMP_MTP_NO_RESERVE=1 skips this, to
                // reproduce the crash.
                const char * no_reserve = std::getenv("DUMP_MTP_NO_RESERVE");
                if (!(no_reserve && std::strcmp(no_reserve, "0") != 0 && *no_reserve)) {
                    llama_set_embeddings_layer_inp(ctx, 0, false);
                    llama_set_embeddings_layer_inp(mtp_ctx.get(), 0, false);
                }
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
    const int n_embd_out = mtp ? llama_model_n_embd_out(model) : 0;
    // common/speculative.cpp: n_mtp_layers = max(1, n_layer_nextn), chain_heads = n_mtp_layers > 1 && !is_mem_shared
    // (1439, 1482 @79e2e74; 1378, 1421 @ec7630a); the MTP context here never shares the target's memory
    const int n_heads = mtp ? std::max(1, (int) llama_model_n_layer_nextn(model)) : 0;
    const bool chain = n_heads > 1;
    std::ofstream lo(st.dir + "/logits.bin", std::ios::binary);
    std::ofstream h_tgt_out, mtp_lo, mtp_h_in, mtp_h_out;
    if (mtp) {
        h_tgt_out.open(st.dir + "/h_nextn.bin", std::ios::binary);
        mtp_lo.open(st_mtp.dir + "/logits.bin", std::ios::binary);
        mtp_h_in.open(st_mtp.dir + "/h_in.bin", std::ios::binary);
        mtp_h_out.open(st_mtp.dir + "/h_nextn.bin", std::ios::binary);
    }
    std::vector<CallRec> calls, mtp_calls;
    common_batch batch(ctx);
    std::vector<float> pending(n_embd_out, 0.0f);   // MTP: target row paired with the next call's first token
    std::vector<float> last_logits(n_vocab, 0.0f);  // target logits of the last decoded token
    std::map<int, std::vector<float>> h_at;         // MTP: the target h_nextn row of every position decoded
    int n_mtp_rows = 0, n_mtp_out = 0, last_call_n = 0;

    // run ids at positions pos0.. through the MTP context under head `head`, each with its h row; outputs: the batch
    // indices that are outputs (empty = all)
    auto mtp_call = [&](const std::string & kind, const std::vector<int> & ids, int pos0,
                        const std::vector<const float *> & h_rows, int head, std::vector<int> outputs) -> bool {
        if (outputs.empty()) {
            for (int k = 0; k < (int) ids.size(); ++k) outputs.push_back(k);
        }
        common_batch mb(mtp_ctx.get());
        for (size_t k = 0; k < ids.size(); ++k) {
            const bool out = std::find(outputs.begin(), outputs.end(), (int) k) != outputs.end();
            const int32_t idx = mb.add((llama_token) ids[k], (llama_pos) (pos0 + (int) k), 0, out);
            mb.set_embd(idx, llama_embd{h_rows[k], 1, (size_t) n_embd_out});
            mtp_h_in.write(reinterpret_cast<const char *>(h_rows[k]), (std::streamsize) n_embd_out * sizeof(float));
        }
        CallRec rec;
        rec.kind = kind; rec.n = (int) ids.size(); rec.pos0 = pos0; rec.tok0 = n_mtp_rows; rec.tokens = ids;
        rec.head = head; rec.out0 = n_mtp_out; rec.outputs = outputs;
        st_mtp.call = (int) mtp_calls.size();
        st_mtp.new_call = true;
        rec.g0 = st_mtp.graph + 1;
        if (llama_process(mtp_ctx.get(), LLAMA_PROCESS_TYPE_DECODE, mb.get())) {
            LOG_ERR("MTP decode failed\n");
            return false;
        }
        rec.g1 = st_mtp.graph + 1;
        for (const int i : outputs) {
            mtp_lo.write(reinterpret_cast<const char *>(llama_get_logits_ith(mtp_ctx.get(), i)),
                         (std::streamsize) n_vocab * sizeof(float));
            mtp_h_out.write(reinterpret_cast<const char *>(llama_get_embeddings_nextn_ith(mtp_ctx.get(), i)),
                            (std::streamsize) n_embd_out * sizeof(float));
        }
        n_mtp_rows += rec.n;
        n_mtp_out += (int) outputs.size();
        mtp_calls.push_back(rec);
        return true;
    };

    // common/speculative.cpp process(): one head, or every head after seq_rm from the call's first position
    auto mtp_process = [&](const std::vector<int> & ids, int pos0, const std::vector<const float *> & h_rows) -> bool {
        if (!chain) return mtp_call("process", ids, pos0, h_rows, 0, {});
        for (int head = 0; head < n_heads; ++head) {
            llama_memory_seq_rm(llama_get_memory(mtp_ctx.get()), 0, pos0, -1);
            llama_set_nextn_layer_offset(mtp_ctx.get(), head);
            if (!mtp_call("process", ids, pos0, h_rows, head, {})) return false;
        }
        llama_set_nextn_layer_offset(mtp_ctx.get(), 0);
        return true;
    };

    // common/speculative.cpp draft() for one sequence, greedy: id_last at pos0 with the pending row
    auto mtp_draft = [&](int id_last, int pos0, int n_req) -> bool {
        const int n_max = chain ? std::min(n_req, n_heads) : n_req;
        std::vector<std::vector<float>> chain_h = {pending};
        std::vector<int> result, ids = {id_last};
        std::vector<std::vector<float>> rows = {pending};
        std::vector<int> outs = {0};
        int pos_call = pos0;
        for (int i = 0; i < n_max; ++i) {
            if (chain) {
                llama_memory_seq_rm(llama_get_memory(mtp_ctx.get()), 0, pos0, -1);
                llama_set_nextn_layer_offset(mtp_ctx.get(), i);
            }
            std::vector<const float *> h_rows;
            for (const auto & r : rows) h_rows.push_back(r.data());
            if (!mtp_call("draft", ids, pos_call, h_rows, chain ? i : 0, outs)) return false;
            const int last = outs.back();
            const int id = argmax(llama_get_logits_ith(mtp_ctx.get(), last), n_vocab);
            const float * hn = llama_get_embeddings_nextn_ith(mtp_ctx.get(), last);
            result.push_back(id);
            if ((int) result.size() >= n_max) break;
            if (chain) {           // the whole prefix again under the next head, rows from the chain
                chain_h.emplace_back(hn, hn + n_embd_out);
                ids = {id_last};
                ids.insert(ids.end(), result.begin(), result.end());
                rows = chain_h;
                outs = {(int) ids.size() - 1};
            } else {               // the new token alone at the next position with this step's h row
                ids = {id};
                rows = {std::vector<float>(hn, hn + n_embd_out)};
                outs = {0};
                pos_call = pos0 + i + 1;
            }
        }
        if (chain) llama_set_nextn_layer_offset(mtp_ctx.get(), 0);
        return true;
    };

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
            if (i == n - 1) std::copy(l, l + n_vocab, last_logits.begin());
        }
        if (st.per_graph && rec.g1 - rec.g0 != (n + n_ubatch - 1) / n_ubatch) {
            LOG_ERR("call %d: %d tokens computed %d graphs, expected %d (one per ubatch of %d)\n",
                    st.call, n, rec.g1 - rec.g0, (n + n_ubatch - 1) / n_ubatch, n_ubatch);
            return false;
        }
        calls.push_back(rec);
        last_call_n = n;
        if (mtp) {
            // common/speculative.cpp:1511-1529: token k pairs with the target row k-1, the first with the pending row
            std::vector<std::vector<float>> rows(n);
            std::vector<const float *> h_rows(n);
            std::vector<int> ids(n);
            for (int i = 0; i < n; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx, i);
                rows[i].assign(h, h + n_embd_out);
                h_at[pos0 + i] = rows[i];
                h_tgt_out.write(reinterpret_cast<const char *>(h), (std::streamsize) n_embd_out * sizeof(float));
                ids[i] = std::atoi(toks[tok0 + i].c_str());
            }
            for (int i = 0; i < n; ++i) h_rows[i] = i == 0 ? pending.data() : rows[i - 1].data();
            if (!mtp_process(ids, pos0, h_rows)) return false;
            pending = rows[n - 1];
        }
        return true;
    };

    int n_decoded = 0, pos = 0;
    if (plan.empty()) {
        if (!decode(0, 0, (int) toks.size())) return 1;
        n_decoded = (int) toks.size();
        pos = n_decoded;
    } else {
        int ti = 0;
        for (const auto & step : plan) {
            if (step[0] == 'r') {
                const int n = std::atoi(step.c_str() + 1);
                if (n <= 0 || n > pos) { LOG_ERR("bad rollback step '%s' at pos %d\n", step.c_str(), pos); return 1; }
                if (mtp && n >= last_call_n) {
                    LOG_ERR("DUMP_MTP: rollback '%s' must keep a token of the last call (%d tokens)\n", step.c_str(),
                            last_call_n);
                    return 1;
                }
                pos -= n;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) {
                    LOG_ERR("llama_memory_seq_rm(0, %d, -1) refused a rollback of %d\n", pos, n);
                    return 1;
                }
                CallRec rec;
                rec.kind = "rollback"; rec.n = n; rec.pos0 = pos; rec.tok0 = ti;
                rec.g0 = rec.g1 = st.graph + 1;
                calls.push_back(rec);
                if (mtp) {
                    // llama-server: common_speculative_accept (pending = the target row of the last kept position),
                    // then common_memory::seq_rm on both contexts
                    if (!llama_memory_seq_rm(llama_get_memory(mtp_ctx.get()), 0, pos, -1)) {
                        LOG_ERR("MTP: llama_memory_seq_rm(0, %d, -1) refused\n", pos);
                        return 1;
                    }
                    pending = h_at.at(pos - 1);
                    CallRec mrec;
                    mrec.kind = "rollback"; mrec.n = n; mrec.pos0 = pos; mrec.tok0 = n_mtp_rows;
                    mrec.g0 = mrec.g1 = st_mtp.graph + 1;
                    mtp_calls.push_back(mrec);
                }
                last_call_n = 0;
                continue;
            }
            if (step[0] == 'd') {
                const int n = std::atoi(step.c_str() + 1);
                if (!mtp || n <= 0 || ti >= (int) toks.size()) {
                    LOG_ERR("bad draft step '%s' (needs DUMP_MTP and a next token)\n", step.c_str());
                    return 1;
                }
                if (!mtp_draft(std::atoi(toks[ti].c_str()), pos, n)) return 1;
                if (!llama_memory_seq_rm(llama_get_memory(mtp_ctx.get()), 0, pos, -1)) {
                    LOG_ERR("MTP: llama_memory_seq_rm(0, %d, -1) after the draft refused\n", pos);
                    return 1;
                }
                CallRec mrec;
                mrec.kind = "seq_rm"; mrec.n = 0; mrec.pos0 = pos; mrec.tok0 = n_mtp_rows;
                mrec.g0 = mrec.g1 = st_mtp.graph + 1;
                mtp_calls.push_back(mrec);
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

    if (mtp && n_draft > 0) {
        // common/speculative.cpp draft(): the target's greedy token after the last position with the last target row
        if (!mtp_draft(argmax(last_logits.data(), n_vocab), pos, n_draft)) return 1;
    }
    if (mtp) {
        mtp_lo.close(); mtp_h_in.close(); mtp_h_out.close(); h_tgt_out.close();
        write_manifest(st_mtp, mtp_calls, (int) llama_n_ubatch(mtp_ctx.get()), 0, n_embd_out);
    }
    if (st.per_graph || !plan.empty() || mtp) {
        write_manifest(st, calls, n_ubatch, llama_n_rs_seq(ctx), n_embd_out);
    }
    mtp_ctx.reset();
    own_ctx.reset();
    own_model.reset();
    llama_backend_free();
    return 0;
}
