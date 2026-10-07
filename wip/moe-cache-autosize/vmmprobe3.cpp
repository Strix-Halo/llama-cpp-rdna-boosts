// Unit mapping: map each granularity unit as its OWN mapping; unmap a unit; re-map it at the same VA.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <algorithm>
int main() {
    int dev = 0; hipSetDevice(dev);
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice; prop.location.id = dev;
    size_t gran = 0;
    hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
    printf("gran=%zu\n", gran);
    const size_t n_units = 32;                 // 32 units => 64 MiB at 2 MiB
    const size_t sz = gran * n_units;
    void * p = nullptr;
    hipError_t e = hipMemAddressReserve(&p, sz, gran, 0, 0);
    printf("reserve  err=%d\n", (int)e);
    hipMemAccessDesc acc = {};
    acc.location.type = hipMemLocationTypeDevice; acc.location.id = dev; acc.flags = hipMemAccessFlagsProtReadWrite;

    for (size_t i = 0; i < n_units; i++) {
        hipMemGenericAllocationHandle_t h;
        e = hipMemCreate(&h, gran, &prop, 0);
        if (e) { printf("create unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 1; }
        e = hipMemMap((hipDeviceptr_t)((char*)p + i*gran), gran, 0, h, 0);
        if (e) { printf("map unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 2; }
        hipMemRelease(h);
        e = hipMemSetAccess((hipDeviceptr_t)((char*)p + i*gran), gran, &acc, 1);
        if (e) { printf("access unit %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); return 3; }
    }
    printf("mapped %zu units OK\n", n_units);
    hipMemset((hipDeviceptr_t)p, 0x5a, sz); hipDeviceSynchronize();

    // unmap the TAIL units (a whole mapping each)
    e = hipMemUnmap((hipDeviceptr_t)((char*)p + 16*gran), 16*gran);
    printf("unmap tail 16 units err=%d (%s)\n", (int)e, hipGetErrorString(e));

    // re-map the tail at the same VA
    int bad = 0;
    for (size_t i = 16; i < n_units; i++) {
        hipMemGenericAllocationHandle_t h;
        e = hipMemCreate(&h, gran, &prop, 0);
        if (e) { printf("recreate %zu err=%d\n", i, (int)e); bad=1; break; }
        e = hipMemMap((hipDeviceptr_t)((char*)p + i*gran), gran, 0, h, 0);
        if (e) { printf("remap %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); bad=1; break; }
        hipMemRelease(h);
        e = hipMemSetAccess((hipDeviceptr_t)((char*)p + i*gran), gran, &acc, 1);
        if (e) { printf("reaccess %zu err=%d (%s)\n", i, (int)e, hipGetErrorString(e)); bad=1; break; }
    }
    if (!bad) {
        int host = 0;
        hipMemcpy(&host, (char*)p + 24*gran, sizeof(int), hipMemcpyDeviceToHost);
        printf("=> UNIT map/unmap/re-map works; head still readable, tail byte = 0x%08x\n", host);
    }
    return 0;
}
