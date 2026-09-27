// exp24 probe (2026-09-27): is `hipMemcpy2DAsync` (H2D, pageable source) reliable on this platform?
// The tensor-split splice uses exactly this call; under HSA_ENABLE_SDMA=0 it faults in
// `__amd_rocclr_copyBufferRectAligned` with the fault address inside the *source* range, while the same
// ranges copied with 1-D `hipMemcpyAsync` (the mirrored path) are fine.  This probe isolates the call.
//
// Run `hipcc -O2 --offload-arch=gfx1201 h2d2dprobe.cpp -o h2d2dprobe` and then
//   ./h2d2dprobe pinned   # expect: completes
//   ./h2d2dprobe malloc   # expect: device fault (rc != 0)
//   ./h2d2dprobe mmap     # expect: device fault
// The pageable cases leak/abort the process on the first fault -- the exit code IS the result.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

int main(int argc, char ** argv) {
    const char * kind = argc > 1 ? argv[1] : "pinned";
    hipSetDevice(0);

    // Replicate the failing splice shape: the gate weight, chunk_full = 589824, chunk_j = 294912.
    const size_t cf = 589824;
    const size_t cj = 294912;
    const int    nc = 10;                 // one used-expert group
    const size_t src_bytes = cf * 64;     // 64 chunks of source
    const size_t dst_bytes = cj * 64 + cj;

    void * dst = nullptr;
    CK(hipMalloc(&dst, dst_bytes));

    void * src = nullptr;
    int fd = -1;
    switch (kind[0]) {
        case 'p': CK(hipHostMalloc(&src, src_bytes, hipHostMallocDefault)); break;
        case 'm': src = malloc(src_bytes); break;
        case 'f': {
            const char * path = "/tmp/h2d2dprobe.bin";
            fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd < 0 || ftruncate(fd, (off_t) src_bytes) != 0) { perror("file"); return 2; }
            src = mmap(nullptr, src_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (src == MAP_FAILED) { perror("mmap"); return 2; }
            break;
        }
        default: fprintf(stderr, "kind must be pinned|malloc|mmap\n"); return 2;
    }
    memset(src, 0x5a, src_bytes);

    hipStream_t s;
    CK(hipStreamCreate(&s));

    fprintf(stderr, "kind=%s cf=%zu cj=%zu nc=%d src=%p dst=%p\n", kind, cf, cj, nc, src, dst);

    // Mirror the splice: per device j, copy the j-th chunk half of each chunk, with the source row
    // base advanced by j*cj, into a compacted destination.
    for (int it = 0; it < 200; ++it) {
        const size_t i_start = (size_t) (it % 40);
        for (int j = 0; j < 2; ++j) {
            const size_t off_j   = (size_t) j * cj;
            const size_t dst_base = i_start * cj;
            CK(hipMemcpy2DAsync((char *) dst + dst_base, cj,
                                (const char *) src + i_start * cf + off_j, cf,
                                cj, nc, hipMemcpyHostToDevice, s));
        }
        // occasional drain so a fault surfaces with an iteration number
        if ((it % 20) == 0) {
            hipError_t e = hipStreamSynchronize(s);
            if (e != hipSuccess) { fprintf(stderr, "sync error it=%d: %s\n", it, hipGetErrorString(e)); return 3; }
            fprintf(stderr, "it=%d ok\n", it);
        }
    }
    hipError_t e = hipStreamSynchronize(s);
    if (e != hipSuccess) { fprintf(stderr, "final sync error: %s\n", hipGetErrorString(e)); return 3; }

    fprintf(stderr, "PASS kind=%s\n", kind);
    return 0;
}
