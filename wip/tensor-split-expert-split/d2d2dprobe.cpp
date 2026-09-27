// exp28 probe (2026-09-27): does `hipMemcpy2DAsync` work when BOTH sides are device memory (D2D)?
// If yes, the tensor-split splice can avoid the broken pageable H2D 2-D copy by: (1) 1-D H2D the whole
// pruned expert range into a device staging buffer (the path the mirrored upload already uses), then
// (2) D2D 2-D compact it into the per-device split weight tensor.
//
// Build: hipcc -O2 --offload-arch=gfx1201 d2d2dprobe.cpp -o d2d2dprobe -lamdhip64
// Run:   ./d2d2dprobe h2d   # control: pageable H2D (expect fault)
//        ./d2d2dprobe d2d   # candidate: device-to-device (expect PASS)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(2); } } while (0)

int main(int argc, char ** argv) {
    const bool d2d = argc > 1 && strcmp(argv[1], "d2d") == 0;
    hipSetDevice(0);

    const size_t cf = 589824;
    const size_t cj = 294912;
    const int    nc = 10;
    const size_t chunks = 256;

    void * dst = nullptr;
    CK(hipMalloc(&dst, cj * chunks + 4096));

    void * src_dev = nullptr;
    CK(hipMalloc(&src_dev, cf * chunks));
    void * src_host = malloc(cf * chunks);
    memset(src_host, 0x5a, cf * chunks);
    CK(hipMemcpy(src_dev, src_host, cf * chunks, hipMemcpyHostToDevice));

    hipStream_t s;
    CK(hipStreamCreate(&s));

    fprintf(stderr, "mode=%s\n", d2d ? "d2d" : "h2d");

    for (int it = 0; it < 1000; ++it) {
        const size_t i_start = (size_t) (it % 40);
        for (int j = 0; j < 2; ++j) {
            const size_t off_j = (size_t) j * cj;
            const void * src = d2d ? (const void *) ((char *) src_dev + i_start * cf + off_j)
                                   : (const void *) ((char *) src_host + i_start * cf + off_j);
            CK(hipMemcpy2DAsync((char *) dst + i_start * cj, cj, src, cf, cj, nc,
                                d2d ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice, s));
        }
        if ((it % 50) == 0) {
            hipError_t e = hipStreamSynchronize(s);
            if (e != hipSuccess) { fprintf(stderr, "sync error it=%d: %s\n", it, hipGetErrorString(e)); return 3; }
        }
    }
    fprintf(stderr, "PASS mode=%s\n", d2d ? "d2d" : "h2d");
    return 0;
}
