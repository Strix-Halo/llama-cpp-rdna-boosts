// Minimal HIP VMM probe: does this ROCm/driver actually support VMM on this GPU?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define OK(e, what) do { hipError_t _e = (e); printf("%-42s err=%d (%s)\n", what, (int)_e, hipGetErrorString(_e)); } while (0)

int main() {
    int dev = 0;
    hipSetDevice(dev);
    hipDeviceProp_t pr;
    hipGetDeviceProperties(&pr, dev);
    printf("device %d: %s  gcnArch=%s\n", dev, pr.name, pr.gcnArchName);

    int vmm = -1;
    hipError_t e = hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev);
    printf("%-42s err=%d (%s) vmm=%d\n", "hipDeviceAttributeVirtualMemoryManagementSupported", (int)e, hipGetErrorString(e), vmm);

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;

    size_t gran = 0;
    e = hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
    printf("%-42s err=%d (%s) gran=%zu\n", "hipMemGetAllocationGranularity", (int)e, hipGetErrorString(e), gran);
    if (e != hipSuccess) { return 1; }

    const size_t sz = 64ull << 20;
    hipMemGenericAllocationHandle_t h = 0;
    e = hipMemCreate(&h, sz, &prop, 0);
    printf("%-42s err=%d (%s) handle=%llx\n", "hipMemCreate", (int)e, hipGetErrorString(e), (unsigned long long)h);
    if (e != hipSuccess) { printf("=> hipMemCreate FAILED: VMM physical allocation unsupported\n"); return 2; }

    void * p = nullptr;
    e = hipMemAddressReserve(&p, sz, gran, 0, 0);
    printf("%-42s err=%d (%s) p=%p\n", "hipMemAddressReserve", (int)e, hipGetErrorString(e), p);
    if (e != hipSuccess) { return 3; }

    e = hipMemMap((hipDeviceptr_t)p, sz, 0, h, 0);
    printf("%-42s err=%d (%s)\n", "hipMemMap", (int)e, hipGetErrorString(e));
    if (e != hipSuccess) { return 4; }

    hipMemAccessDesc acc = {};
    acc.location.type = hipMemLocationTypeDevice;
    acc.location.id = dev;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    e = hipMemSetAccess((hipDeviceptr_t)p, sz, &acc, 1);
    printf("%-42s err=%d (%s)\n", "hipMemSetAccess", (int)e, hipGetErrorString(e));

    // touch it from the device: a real kernel write followed by a readback
    int * d = (int *)p;
    hipMemset((hipDeviceptr_t)p, 0x5a, sz);
    hipDeviceSynchronize();
    int host = 0;
    hipMemcpy(&host, p, sizeof(int), hipMemcpyDeviceToHost);
    printf("%-42s host=0x%08x (expect 0x5a5a5a5a)\n", "readback after memset", host);

    // the operations the pool actually needs: unmap, then re-map AT THE SAME VA
    e = hipMemUnmap((hipDeviceptr_t)p, sz);
    printf("%-42s err=%d (%s)\n", "hipMemUnmap", (int)e, hipGetErrorString(e));
    hipMemGenericAllocationHandle_t h2 = 0;
    e = hipMemCreate(&h2, sz, &prop, 0);
    printf("%-42s err=%d (%s)\n", "hipMemCreate (again)", (int)e, hipGetErrorString(e));
    if (e == hipSuccess) {
        e = hipMemMap((hipDeviceptr_t)p, sz, 0, h2, 0);
        printf("%-42s err=%d (%s)\n", "hipMemMap RE-MAP at same VA", (int)e, hipGetErrorString(e));
        if (e == hipSuccess) {
            e = hipMemSetAccess((hipDeviceptr_t)p, sz, &acc, 1);
            printf("%-42s err=%d (%s)\n", "hipMemSetAccess (again)", (int)e, hipGetErrorString(e));
            hipMemset((hipDeviceptr_t)p, 0xa5, sz);
            hipDeviceSynchronize();
            hipMemcpy(&host, p, sizeof(int), hipMemcpyDeviceToHost);
            printf("%-42s host=0x%08x (expect 0xa5a5a5a5)\n", "readback after re-map", host);
            hipMemUnmap((hipDeviceptr_t)p, sz);
        }
    }
    printf("=> VMM appears FUNCTIONAL\n");
    return 0;
}
