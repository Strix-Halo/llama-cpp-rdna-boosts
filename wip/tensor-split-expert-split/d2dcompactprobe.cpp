#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void trial(const char*lbl, size_t width, size_t height, size_t spitch, size_t dpitch){
    size_t ssz=spitch*height, dsz=dpitch*height;
    void *s,*d; hipMalloc(&s,ssz); hipMalloc(&d,dsz);
    hipMemset(s,1,ssz);
    (void)0;
}
int main(){
    hipSetDevice(0);
    struct { const char* n; size_t w,h,sp,dp; } T[] = {
        {"down: 176 x 524288, spitch 352",  176, 524288, 352, 176},
        {"gate: 294912 x 256, spitch 589824", 294912, 256, 589824, 294912},
        {"contig half (down)", 92274688, 1, 92274688, 92274688},
    };
    printf("hipMemcpy2D D2D timing (10 iters, best):\n");
    for (auto &t : T) {
        size_t ssz=t.sp*t.h, dsz=t.dp*t.h;
        void *s,*d; if(hipMalloc(&s,ssz)||hipMalloc(&d,dsz)){printf("  %-40s alloc FAIL\n",t.n);continue;}
        hipMemset(s,1,ssz);
        hipMemcpy2D(d,t.dp,s,t.sp,t.w,t.h,hipMemcpyDeviceToDevice); hipDeviceSynchronize();
        double best=1e9;
        for(int i=0;i<10;i++){ double a=now_ms(); hipMemcpy2D(d,t.dp,s,t.sp,t.w,t.h,hipMemcpyDeviceToDevice); hipDeviceSynchronize(); best=best<now_ms()-a?best:now_ms()-a; }
        printf("  %-40s %8.2f ms  (%.1f GB/s)\n", t.n, best, (ssz+dsz)/best/1e6);
        hipFree(s); hipFree(d);
    }
    return 0;
}
