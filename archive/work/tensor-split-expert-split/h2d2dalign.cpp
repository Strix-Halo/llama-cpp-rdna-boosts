// exp26 probe (2026-09-27): is `hipMemcpy2DAsync` (H2D) sensitive to the *destination* alignment?
// The tensor-split splice writes its compacted chunk at `simple_tensor->data + i_start*chunk_j`; on
// this box every simple tensor's data ends in 0x...2480 (128-byte aligned, NOT 256).  This probe runs
// the same 2-D copy shape at destination offsets 0/16/128/256/4096 and reports where it faults.
//
// Build: hipcc -O2 --offload-arch=gfx1201 h2d2dalign.cpp -o h2d2dalign -lamdhip64
// Run:   ./h2d2dalign <dst_off> [sdma]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(2); } } while (0)

int main(int argc, char ** argv) {
    const size_t doff = argc > 1 ? (size_t) strtoul(argv[1], nullptr, 0) : 0;
    hipSetDevice(0);

    const size_t cf = 589824;
    const size_t cj = 294912;
    const int    nc = 10;
    const size_t chunks = 256;

    void * src = malloc(cf * chunks);
    memset(src, 0x5a, cf * chunks);

    void * dst = nullptr;
    CK(hipMalloc(&dst, cj * chunks + doff + 4096));
    hipStream_t s;
    CK(hipStreamCreate(&s));

    fprintf(stderr, "dst_off=%zu src=%p dst=%p dst+off=%p (mod256=%zu)\n", doff, src, dst, (char *) dst + doff, ((size_t) dst + doff) % 256);

    for (int it = 0; it < 500; ++it) {
        const size_t i_start = (size_t) (it % 40);
        for (int j = 0; j < 2; ++j) {
            const size_t off_j = (size_t) j * cj;
            CK(hipMemcpy2DAsync((char *) dst + doff + i_start * cj, cj,
                                (const char *) src + i_start * cf + off_j, cf,
                                cj, nc, hipMemcpyHostToDevice, s));
        }
        if ((it % 50) == 0) {
            hipError_t e = hipStreamSynchronize(s);
            if (e != hipSuccess) { fprintf(stderr, "sync error it=%d: %s\n", it, hipGetErrorString(e)); return 3; }
        }
    }
    hipError_t e = hipStreamSynchronize(s);
    if (e != hipSuccess) { fprintf(stderr, "final sync error: %s\n", hipGetErrorString(e)); return 3; }
    fprintf(stderr, "PASS dst_off=%zu\n", doff);
    return 0;
}
