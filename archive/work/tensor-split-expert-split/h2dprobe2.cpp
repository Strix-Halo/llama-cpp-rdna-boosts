// exp11: per-COPY latency of the expert-range uploads: pageable (mmap'd model file / malloc) vs pinned.
// llama.cpp uploads each pruned expert range with one set_tensor call.  If the source is pageable the
// driver pins+unpins the range around each copy, which is a fixed ~1-2 ms per call regardless of size.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void one(const char * lbl, void * host, size_t sz, void * dev, hipStream_t s, int reps) {
    for (int i=0;i<3;i++) hipMemcpy(dev, host, sz, hipMemcpyHostToDevice);
    hipStreamSynchronize(s);
    std::vector<double> v;
    for (int i=0;i<reps;i++){ double a=now_ms(); hipMemcpy(dev, host, sz, hipMemcpyHostToDevice); v.push_back(now_ms()-a); }
    hipStreamSynchronize(s);
    std::sort(v.begin(),v.end());
    printf("  %-22s %6.0f KiB  median %7.3f ms  -> %7.1f MB/s   (min %.3f)\n",
        lbl, sz/1024.0, v[v.size()/2], sz/v[v.size()/2]/1000.0, v.front());
}
int main(){
    hipSetDevice(0);
    const size_t sizes[] = { 64*1024, 512*1024, 2*1024*1024, 8*1024*1024 };
    for (size_t k=0;k<4;k++){
        size_t sz=sizes[k];
        void *dev; hipMalloc(&dev, sz);
        void *pinned; hipHostMalloc(&pinned, sz, hipHostMallocDefault);
        void *page = malloc(sz);
        memset(page,1,sz); memset(pinned,1,sz);
        hipStream_t s; hipStreamCreate(&s);
        printf("size %.0f KiB\n", sz/1024.0);
        one("pageable (malloc)", page, sz, dev, s, 100);
        one("pinned (hipHostMalloc)", pinned, sz, dev, s, 100);
        hipStreamDestroy(s); hipFree(dev); hipHostFree(pinned); free(page);
    }
    return 0;
}
