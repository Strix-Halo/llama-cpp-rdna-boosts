// exp8: is the ~5.3 ms per-call cost in hipblasSgemm, in rocblas, or in the non-default stream?
// Shape matches the Qwen3.6 SSM/GDN GEMMs that llama.cpp sends to cuBLAS: A[K=256,M=256] (the F32
// weight), B[K,N] (the activation), C[M,N].  llama.cpp uses a NON-default stream.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void bench(const char * what, hipStream_t s, const int M, const int N, const int K,
                  hipblasHandle_t hb, rocblas_handle hr, bool use_roc,
                  const float * A, const float * B, float * C) {
    const float alpha = 1.0f, beta = 0.0f;
    const int reps = 60;
    for (int i = 0; i < 3; ++i) {   // warmup
        if (use_roc) rocblas_sgemm(hr, rocblas_operation_transpose, rocblas_operation_none, M, N, K, &alpha, A, K, B, K, &beta, C, M);
        else         hipblasSgemm(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &alpha, A, K, B, K, &beta, C, M);
    }
    hipDeviceSynchronize();
    std::vector<double> per;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now_ms();
        if (use_roc) {
            rocblas_sgemm(hr, rocblas_operation_transpose, rocblas_operation_none, M, N, K,
                          &alpha, A, K, B, K, &beta, C, M);
        } else {
            hipblasSgemm(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K,
                         &alpha, A, K, B, K, &beta, C, M);
        }
        per.push_back(now_ms() - t0);
    }
    hipDeviceSynchronize();
    std::sort(per.begin(), per.end());
    printf("%-34s M=%-5d N=%-6d K=%-5d  median=%7.3f ms  min=%7.3f  max=%7.3f\n",
           what, M, N, K, per[per.size()/2], per.front(), per.back());
}

int main(int argc, char ** argv) {
    hipSetDevice(0);
    float *A = nullptr, *B = nullptr, *C = nullptr;
    const int M = 256, K = 256;
    hipMalloc(&A, (size_t) K * K * 4);
    hipMalloc(&B, (size_t) K * 16384 * 4);
    hipMalloc(&C, (size_t) M * 16384 * 4);
    hipMemset(A, 1, (size_t) K * K * 4);
    hipMemset(B, 1, (size_t) K * 16384 * 4);
    hipMemset(C, 0, (size_t) M * 16384 * 4);
    hipDeviceSynchronize();

    const bool use_real = argc > 1 && std::string(argv[1]) == "real";
    if (!use_real) {
        printf("NOTE: no buffers passed (nullptr) -- this measures ONLY the call overhead, no kernel.\n");
    }
    (void) use_real;
    hipDeviceSynchronize();

    hipStream_t def = nullptr, s1 = nullptr;
    hipStreamCreate(&s1);
    hipblasHandle_t hb; hipblasCreate(&hb);
    rocblas_handle  hr; rocblas_create_handle(&hr);

    // The exact llama.cpp path: hipblasSgemm on a non-default stream.
    hipblasSetStream(hb, s1);
    rocblas_set_stream(hr, s1);
    bench("hipblasSgemm  (non-default stream)", s1, M, 8192, K, hb, hr, false, A, B, C);
    bench("hipblasSgemm  (non-default)",        s1, M, 1,    K, hb, hr, false, A, B, C);
    bench("hipblasSgemm  (non-default)",        s1, M, 64,   K, hb, hr, false, A, B, C);

    // Same, on the default stream (NULL).
    hipblasSetStream(hb, def);
    rocblas_set_stream(hr, def);
    bench("hipblasSgemm  (default stream)",     def, M, 8192, K, hb, hr, false, A, B, C);
    bench("hipblasSgemm  (default stream)",     def, M, 1,    K, hb, hr, false, A, B, C);

    // Bypass hipblas entirely: rocBLAS direct.
    rocblas_set_stream(hr, s1);
    bench("rocblas_sgemm (non-default stream)", s1, M, 8192, K, hb, hr, true, A, B, C);
    bench("rocblas_sgemm (non-default stream)", s1, M, 1,    K, hb, hr, true, A, B, C);
    return 0;
}
