// Does an H2D on one stream overlap compute on another stream on this box?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstring>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void spin(float * out, long cycles) {
    long long t = clock64();
    while (clock64() - t < cycles) { }
    if (threadIdx.x == 0) *out = 1.0f;
}
int main(){
    hipSetDevice(0);
    const size_t sz = 512ull*1024*1024;         // 512 MiB
    void * dev;       hipMalloc(&dev, sz);
    void * pinned;    hipHostMalloc(&pinned, sz, hipHostMallocDefault);
    void * page = malloc(sz);
    memset(pinned,1,sz); memset(page,1,sz);
    hipStream_t s0, s1;
    hipStreamCreate(&s0); hipStreamCreate(&s1);
    float * scratch; hipMalloc(&scratch, 4);
    const long SPIN = 30000000;                  // ~30 ms of compute

    auto bench = [&](const char * lbl, void * src, bool overlap) {
        // warm
        spin<<<1,1,0,s0>>>(scratch, SPIN); hipStreamSynchronize(s0);
        hipMemcpy(dev, src, sz, hipMemcpyHostToDevice);
        double best = 1e9;
        for (int i = 0; i < 5; ++i) {
            const double a = now_ms();
            if (!overlap) {
                spin<<<1,1,0,s0>>>(scratch, SPIN);
                hipMemcpyAsync(dev, src, sz, hipMemcpyHostToDevice, s0);
                hipStreamSynchronize(s0);
            } else {
                spin<<<1,1,0,s0>>>(scratch, SPIN);
                hipMemcpyAsync(dev, src, sz, hipMemcpyHostToDevice, s1);
                hipStreamSynchronize(s1);
                hipStreamSynchronize(s0);
            }
            best = std::min(best, now_ms()-a);
        }
        printf("  %-30s %8.1f ms\n", lbl, best);
    };
    // baseline: copy alone, compute alone
    double a;
    a = now_ms(); hipMemcpy(dev, pinned, sz, hipMemcpyHostToDevice); printf("  %-30s %8.1f ms\n", "H2D alone (pinned, sync)", now_ms()-a);
    a = now_ms(); spin<<<1,1,0,s0>>>(scratch, SPIN); hipStreamSynchronize(s0); printf("  %-30s %8.1f ms\n", "compute alone", now_ms()-a);
    bench("same stream (serial)",      pinned, false);
    bench("two streams (overlap?)",    pinned, true);
    bench("pageable same stream",      page,   false);
    bench("pageable two streams",      page,   true);
    return 0;
}
