// Isolated rope -> view -> set_rows test for issue #67.
// Builds the exact op chain the KV cache uses, runs it on the GPU backend and
// compares every node bit-for-bit against the CPU backend.
// Run once normally (rope_set_rows fusion available) and once with
// GGML_CUDA_DISABLE_ROPE_SET_ROWS=1; if the two GPU results differ while only
// one matches CPU, the fused kernel is the culprit.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

static const char * g_mode = "neox";

static bool cb(int node, struct ggml_tensor * t1, struct ggml_tensor * t2, void * ud) {
    (void) ud;
    const size_t n = ggml_nbytes(t1);
    if (n != ggml_nbytes(t2)) { printf("node %d: size mismatch\n", node); return false; }
    std::vector<uint8_t> a(n), b(n);
    ggml_backend_tensor_get(t1, a.data(), 0, n);
    ggml_backend_tensor_get(t2, b.data(), 0, n);
    int ndiff = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) { if (a[i] != b[i]) { if (ndiff == 0) first = i; ++ndiff; } }
    printf("node %2d %-14s nbytes=%zu  %s", node, ggml_op_desc(t1), n, ndiff ? "DIFFER" : "identical");
    if (ndiff) printf("  (%d bytes, first at %zu: gpu=%02x cpu=%02x)", ndiff, first, a[first], b[first]);
    printf("\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc > 1) g_mode = argv[1];

    ggml_backend_load_all();

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu) { printf("no GPU backend\n"); return 1; }
    if (!cpu) { printf("no CPU backend\n"); return 1; }
    printf("gpu=%s  cpu=%s  mode=%s  fusion=%s\n",
           ggml_backend_name(gpu), ggml_backend_name(cpu), g_mode,
           getenv("GGML_CUDA_DISABLE_ROPE_SET_ROWS") ? "OFF" : "ON(default)");

    const int nh = 4;       // n_head_kv
    const int hd = 256;     // n_embd_head
    const int nt = 8;       // tokens
    const int kv = 64;      // cache rows
    const int neg = nh*hd;

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*64 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, nt);
    struct ggml_tensor * pos = nullptr;
    struct ggml_tensor * r   = nullptr;

    const char * ndims_env = getenv("NDIMS");
    const int n_dims = ndims_env ? atoi(ndims_env) : 64; // Qwen3.8 n_rot

    const float freq_base  = 10000000.0f; // Qwen3.8 rope.freq_base
    const float freq_scale = 1.0f;
    const float ext        = 0.0f;
    const float attn       = 1.0f;

    if (strcmp(g_mode, "mrope") == 0 || strcmp(g_mode, "imrope") == 0) {
        pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)nt*4);
        int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 }; // Qwen3.8
        const int mode = strcmp(g_mode, "imrope") == 0 ? GGML_ROPE_TYPE_IMROPE : GGML_ROPE_TYPE_MROPE;
        r = ggml_rope_multi(ctx, x, pos, nullptr, n_dims, sections, mode,
                            4096, freq_base, freq_scale, ext, attn, 0.0f, 0.0f);
    } else {
        const int mode = strcmp(g_mode, "normal") == 0 ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX;
        pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        r = ggml_rope_ext(ctx, x, pos, nullptr, n_dims, mode,
                          4096, freq_base, freq_scale, ext, attn, 0.0f, 0.0f);
    }

    struct ggml_tensor * view = ggml_view_2d(ctx, r, neg, nt, r->nb[2], 0);
    struct ggml_tensor * kvc  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, neg, kv);
    struct ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    struct ggml_tensor * out  = ggml_set_rows(ctx, kvc, view, idxs);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) { printf("alloc failed\n"); return 1; }

    // random-ish deterministic inputs
    std::vector<float> xd((size_t) neg*nt);
    uint32_t s = 12345;
    for (auto & e : xd) { s = s*1664525u + 1013904223u; e = ((int32_t)(s >> 8) % 4001 - 2000) / 1000.0f; }
    ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));

    if (strcmp(g_mode, "mrope") == 0 || strcmp(g_mode, "imrope") == 0) {
        std::vector<int32_t> pd((size_t) nt*4);
        for (int sec = 0; sec < 4; ++sec) for (int t = 0; t < nt; ++t) pd[sec*nt + t] = t;
        ggml_backend_tensor_set(pos, pd.data(), 0, ggml_nbytes(pos));
    } else {
        std::vector<int32_t> pd(nt);
        for (int t = 0; t < nt; ++t) pd[t] = t;
        ggml_backend_tensor_set(pos, pd.data(), 0, ggml_nbytes(pos));
    }

    std::vector<int64_t> id((size_t) nt);
    for (int t = 0; t < nt; ++t) id[t] = (t*3) % kv;
    ggml_backend_tensor_set(idxs, id.data(), 0, ggml_nbytes(idxs));

    std::vector<uint16_t> kvd((size_t) neg*kv, 0);
    ggml_backend_tensor_set(kvc, kvd.data(), 0, ggml_nbytes(kvc));

    printf("--- whole-graph compare, final cache only (backend1=GPU, backend2=CPU) ---\n");
    const struct ggml_tensor * tests[] = { out };
    ggml_backend_compare_graph_backend(gpu, cpu, gf, cb, nullptr, tests, 1);

    // also hash the final cache on the GPU side
    std::vector<uint8_t> od(ggml_nbytes(out));
    ggml_backend_tensor_get(out, od.data(), 0, od.size());
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : od) { h ^= b; h *= 1099511628211ULL; }
    printf("cache hash = %016llx\n", (unsigned long long) h);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
