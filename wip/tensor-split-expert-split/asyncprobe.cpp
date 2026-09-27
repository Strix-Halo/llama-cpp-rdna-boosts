// exp12: hipMemcpyAsync from PAGEABLE memory is synchronous -- it blocks the host until the stream
// drains.  This is why the expert/input uploads serialize and why the 2-device case is 2x: the host
// cannot issue device B's copy until device A's copy AND all queued compute has completed.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void spin_kernel(float * out, long cycles) {
    long long t = clock64();
    while (clock64() - t < cycles) { }
    if (threadIdx.x == 0) *out = 1.0f;
}
static void trial(const char * lbl, void * src, size_t sz, void * dev, hipStream_t s, int reps) {
    float * scratch; hipMalloc(&scratch, 4);
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        spin_kernel<<<1, 1, 0, s>>>(scratch, 2000000);      // ~2 ms of queued GPU work
        const double a = now_ms();
        hipMemcpyAsync(dev, src, sz, hipMemcpyHostToDevice, s);
        v.push_back(now_ms() - a);                          // host time of the memcpyAsync CALL
        hipStreamSynchronize(s);
    }
    std::sort(v.begin(), v.end());
    printf("  %-26s %7.0f KiB  host-block median %7.3f ms  (min %.3f, max %.3f)\n",
        lbl, sz/1024.0, v[v.size()/2], v.front(), v.back());
    hipFree(scratch);
}
int main(){
    hipSetDevice(0);
    const size_t sz = 512*1024;
    void *dev; hipMalloc(&dev, sz);
    void *pinned; hipHostMalloc(&pinned, sz, hipHostMallocDefault);
    void *page = malloc(sz); memset(page,1,sz); memset(pinned,1,sz);
    hipStream_t s; hipStreamCreate(&s);
    printf("host wall time INSIDE the hipMemcpyAsync call, with ~2 ms of compute already queued:\n");
    trial("pageable source (malloc)", page, sz, dev, s, 100);
    trial("pinned source (hipHostMalloc)", pinned, sz, dev, s, 100);
    return 0;
}
