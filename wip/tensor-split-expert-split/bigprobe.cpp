// exp14: at the expert-tensor size (~144 MiB) does hipMemcpyAsync from a pageable source BLOCK the
// host for the whole transfer?  If yes, the two devices' DMAs cannot overlap (the host can only be in
// one call at a time) and the mirrored upload costs 2x -- which is the tensor penalty.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void trial(const char * lbl, const void * src, size_t sz, void * dev, hipStream_t s, int reps) {
    for (int i=0;i<2;i++) hipMemcpy(dev, src, sz, hipMemcpyHostToDevice);
    hipStreamSynchronize(s);
    std::vector<double> v;
    for (int i=0;i<reps;i++){
        const double a=now_ms();
        hipMemcpyAsync(dev, src, sz, hipMemcpyHostToDevice, s);   // host time of the CALL
        v.push_back(now_ms()-a);
        hipStreamSynchronize(s);
    }
    std::sort(v.begin(),v.end());
    printf("  %-38s %6.0f MiB  host-block-in-call median %8.3f ms -> %6.1f GB/s\n",
        lbl, sz/1048576.0, v[v.size()/2], sz/(v[v.size()/2]*1e6));
}
int main(){
    hipSetDevice(0);
    const size_t sz = 144ull<<20;    // 150994944 B = one ffn_gate_exps.weight
    int fd = open("/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", O_RDONLY);
    struct stat st; fstat(fd,&st);
    void * map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    void *dev; hipMalloc(&dev, sz);
    void *pinned; hipHostMalloc(&pinned, sz, hipHostMallocDefault);
    void *page = malloc(sz);
    memset(page,1,sz); memset(pinned,1,sz);
    hipStream_t s; hipStreamCreate(&s);
    printf("host wall time INSIDE hipMemcpyAsync (a blocking call shows ~= the transfer time):\n");
    trial("malloc pageable", page, sz, dev, s, 30);
    trial("mmap model file", (const char*)map + (10ull<<30), sz, dev, s, 30);
    trial("PINNED (hipHostMalloc)", pinned, sz, dev, s, 30);
    return 0;
}
