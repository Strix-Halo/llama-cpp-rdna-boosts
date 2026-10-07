#include <hip/hip_runtime.h>
#include <cstdio>
#include <algorithm>
int main(int argc, char** argv) {
    int dev = 0; hipSetDevice(dev);
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice; prop.location.id = dev;
    size_t gran = 0; hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
    const size_t n = (argc > 1) ? (size_t) atoi(argv[1]) : 6144;
    const size_t sz = gran * n;
    printf("gran=%zu units=%zu total=%.1f MiB\n", gran, n, sz/1048576.0);
    void * p = nullptr;
    hipError_t e = hipMemAddressReserve(&p, sz, gran, 0, 0);
    printf("reserve err=%d (%s)\n", (int)e, hipGetErrorString(e));
    hipMemAccessDesc acc = {};
    acc.location.type = hipMemLocationTypeDevice; acc.location.id = dev; acc.flags = hipMemAccessFlagsProtReadWrite;
    for (size_t i = 0; i < n; i++) {
        hipMemGenericAllocationHandle_t h;
        e = hipMemCreate(&h, gran, &prop, 0);
        if (e) { printf("CREATE unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 1; }
        e = hipMemMap((hipDeviceptr_t)((char*)p + i*gran), gran, 0, h, 0);
        if (e) { printf("MAP unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 2; }
        hipMemRelease(h);
        e = hipMemSetAccess((hipDeviceptr_t)((char*)p + i*gran), gran, &acc, 1);
        if (e) { printf("ACCESS unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 3; }
    }
    printf("=> mapped %zu units OK\n", n);
    printf("last error after loop: %s\n", hipGetErrorString(hipGetLastError()));
    return 0;
}
