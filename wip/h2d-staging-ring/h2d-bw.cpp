#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void spin(volatile float * p, long long iters) {
    long long t = clock64();
    float x = p[threadIdx.x];
    while (clock64() - t < iters) { x += 1.0f; }
    if (x == 12345.0f) p[threadIdx.x] = x;
}

static double ms_between(hipEvent_t a, hipEvent_t b, hipStream_t s) {
    float ms; CHECK(hipEventElapsedTime(&ms, a, b)); return ms;
}

int main() {
    CHECK(hipSetDevice(0));
    const size_t N = 2ull*1024*1024*1024;
    void * hp_page = malloc(N);
    void * hp_pin  = nullptr;
    CHECK(hipHostMalloc(&hp_pin, N));
    memset(hp_page, 1, N);
    memset(hp_pin,  1, N);
    void * d = nullptr;
    CHECK(hipMalloc(&d, N));
    hipStream_t sc, sk;
    CHECK(hipStreamCreate(&sc));
    CHECK(hipStreamCreate(&sk));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    float * dp = (float *) d;

    auto bw = [&](void * src, const char * name) {
        CHECK(hipMemcpyAsync(d, src, N, hipMemcpyHostToDevice, sc));
        CHECK(hipStreamSynchronize(sc));
        CHECK(hipEventRecord(e0, sc));
        for (int i = 0; i < 3; i++) CHECK(hipMemcpyAsync(d, src, N, hipMemcpyHostToDevice, sc));
        CHECK(hipEventRecord(e1, sc));
        CHECK(hipStreamSynchronize(sc));
        printf("%-10s H2D %.1f GB/s\n", name, 3.0*N/(ms_between(e0,e1,sc)/1000.0)/1e9);
    };
    bw(hp_page, "pageable");
    bw(hp_pin,  "pinned");

    // kernel-only time
    const long long iters = 200000000;
    hipLaunchKernelGGL(spin, dim3(1), dim3(256), 0, sk, dp, iters);
    CHECK(hipStreamSynchronize(sk));
    CHECK(hipEventRecord(e0, sk));
    hipLaunchKernelGGL(spin, dim3(1), dim3(256), 0, sk, dp, iters);
    CHECK(hipEventRecord(e1, sk));
    CHECK(hipStreamSynchronize(sk));
    double kms = ms_between(e0,e1,sk);
    printf("kernel-only: %.2f ms\n", kms);

    auto overlap = [&](void * src, const char * name) {
        CHECK(hipEventRecord(e0, sk));
        hipLaunchKernelGGL(spin, dim3(1), dim3(256), 0, sk, dp, iters);
        CHECK(hipMemcpyAsync(d, src, N, hipMemcpyHostToDevice, sc));
        CHECK(hipEventRecord(e1, sk));
        CHECK(hipStreamSynchronize(sk));
        CHECK(hipStreamSynchronize(sc));
        double t = ms_between(e0,e1,sk);
        printf("overlap %-8s: total %.2f ms (kernel %.2f + copy)\n", name, t, kms);
    };
    overlap(hp_page, "pageable");
    overlap(hp_pin,  "pinned");
    return 0;
}
