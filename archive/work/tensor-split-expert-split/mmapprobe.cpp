// exp13: the expert uploads read from the mmap'd MODEL FILE.  Same hipMemcpyAsync, three sources:
// malloc (pageable, resident), mmap of the model file (pageable, file-backed).
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
    for (int i=0;i<3;i++) hipMemcpy(dev, src, sz, hipMemcpyHostToDevice);
    hipStreamSynchronize(s);
    std::vector<double> v;
    for (int i=0;i<reps;i++){
        const double a=now_ms();
        hipMemcpyAsync(dev, src, sz, hipMemcpyHostToDevice, s);
        v.push_back(now_ms()-a);
        hipStreamSynchronize(s);
    }
    std::sort(v.begin(),v.end());
    printf("  %-34s %7.0f KiB host-block median %7.3f ms (min %.3f max %.3f) -> %6.1f MB/s\n",
        lbl, sz/1024.0, v[v.size()/2], v.front(), v.back(), sz/v[v.size()/2]/1000.0);
}
int main(){
    hipSetDevice(0);
    const char * path = "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    void * map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("model %s: %.1f GiB, mmap %p\n", path, st.st_size/1073741824.0, map);
    const size_t sizes[] = { 512*1024, 2*1024*1024 };
    for (size_t k=0;k<2;k++){
        size_t sz=sizes[k];
        void *dev; hipMalloc(&dev, sz);
        void *page = malloc(sz); memset(page,1,sz);
        hipStream_t s; hipStreamCreate(&s);
        printf("size %.0f KiB\n", sz/1024.0);
        trial("malloc (pageable, resident)", page, sz, dev, s, 100);
        // a fresh, not-yet-touched region of the file (offset 12 GiB in)
        trial("mmap model file (cold-ish)", (const char*)map + (12ull<<30), sz, dev, s, 100);
        trial("mmap model file (same region again)", (const char*)map + (12ull<<30), sz, dev, s, 100);
        hipStreamDestroy(s); hipFree(dev); free(page);
    }
    return 0;
}
