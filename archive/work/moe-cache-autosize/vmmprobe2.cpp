// Does ROCm support unmapping a SUB-RANGE of a mapping?
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int dev = 0; hipSetDevice(dev);
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    size_t gran = 0;
    hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
    const size_t sz = 64ull << 20;   // 64 MiB
    void * p = nullptr;
    hipMemAddressReserve(&p, sz, gran, 0, 0);
    hipMemGenericAllocationHandle_t h;
    hipError_t e = hipMemCreate(&h, sz, &prop, 0);
    printf("create   err=%d\n", (int)e);
    e = hipMemMap((hipDeviceptr_t)p, sz, 0, h, 0);
    printf("map 64MiB err=%d (%s)\n", (int)e, hipGetErrorString(e));
    hipMemAccessDesc acc = {};
    acc.location.type = hipMemLocationTypeDevice; acc.location.id = dev; acc.flags = hipMemAccessFlagsProtReadWrite;
    hipMemSetAccess((hipDeviceptr_t)p, sz, &acc, 1);

    // partial unmap: give back the upper 32 MiB
    e = hipMemUnmap((hipDeviceptr_t)((char*)p + 32*1024*1024), 32ull<<20);
    printf("PARTIAL unmap (half) err=%d (%s)\n", (int)e, hipGetErrorString(e));

    // and the lower half?
    e = hipMemUnmap((hipDeviceptr_t)p, 32ull<<20);
    printf("PARTIAL unmap (other half) err=%d (%s)\n", (int)e, hipGetErrorString(e));
    return 0;
}
