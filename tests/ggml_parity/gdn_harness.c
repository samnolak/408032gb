// Runs ggml_gated_delta_net (CPU backend) on inputs written by tests/test_kda_vs_ggml.py.
// Input file (little-endian): int32 T, H_k, H_v, S, kda; then float32 arrays in ggml order:
//   q [T][H_k][S], k [T][H_k][S], v [T][H_v][S], g [T][H_v][S or 1], beta [T][H_v],
//   state [H_v][S_v(j)][S_k(i)]   (memory index j*S + i holds S[i][j], as in ggml ops.cpp)
// Output file: float32 attn [T][H_v][S] then final state [H_v][S][S] in the same layout.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rd(FILE * f, void * p, size_t n) { if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); } }

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]); return 2; }
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror("in"); return 2; }
    int32_t hdr[5]; rd(f, hdr, sizeof hdr);
    const int64_t T = hdr[0], Hk = hdr[1], Hv = hdr[2], S = hdr[3]; const int kda = hdr[4];
    struct ggml_init_params ip = { .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, 1);
    struct ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kda ? S : 1, Hv, T, 1);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
    struct ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
    rd(f, q->data, ggml_nbytes(q)); rd(f, k->data, ggml_nbytes(k)); rd(f, v->data, ggml_nbytes(v));
    rd(f, g->data, ggml_nbytes(g)); rd(f, b->data, ggml_nbytes(b)); rd(f, s->data, ggml_nbytes(s));
    fclose(f);
    struct ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 3; }
    FILE * o = fopen(argv[2], "wb");
    const size_t n = (size_t)(S * Hv * T + S * S * Hv);
    if (fwrite(out->data, sizeof(float), n, o) != n) { fprintf(stderr, "short write\n"); return 2; }
    fclose(o);
    ggml_free(ctx);
    return 0;
}
