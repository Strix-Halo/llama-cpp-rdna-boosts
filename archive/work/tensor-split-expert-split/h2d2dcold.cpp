// exp25 probe (2026-09-27): does `hipMemcpy2DAsync` (H2D) fault when the source is a *cold*
// file-backed mmap (non-resident pages) -- i.e. the model file under `-ncmoe`?
//
// The tensor-split splice uses `hipMemcpy2DAsync` on the expert weight's slice of the model mmap.
// On the first prefill the expert pages have never been touched; under HSA_ENABLE_SDMA=0 the fault
// names `__amd_rocclr_copyBufferRectAligned` with the fault address inside the source range, and
// substituting 1-D `hipMemcpyAsync` (the mirrored path) fixes it.
//
// Build: hipcc -O2 --offload-arch=gfx1201 h2d2dcold.cpp -o h2d2dcold -lamdhip64
// Run:   ./h2d2dcold 2d   # expect fault if the cold-mmap hypothesis holds
//        ./h2d2dcold 1d   # control: same source, 1-D copies (the mirrored path)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <unistd.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

int main(int argc, char ** argv) {
    const bool use_2d = argc > 1 && strcmp(argv[1], "1d") != 0;
    hipSetDevice(0);

    const size_t cf = 589824;
    const size_t cj = 294912;
    const int    nc = 10;
    const size_t file_chunks = 4096;                 // ~2.4 GiB
    const size_t src_bytes   = cf * file_chunks;

    // A cold, file-backed, read-only mapping: pages are dropped with MADV_DONTNEED so the copy
    // pulls them from disk, exactly like the first read of the expert mmap.
    const char * path = "/tmp/h2d2dcold.bin";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, (off_t) src_bytes) != 0) { perror("file"); return 2; }
    // give the file real (sparse-zero) content
    const size_t blk = 1u << 20;
    std::vector<char> z(blk, (char) 0x5a);
    for (size_t off = 0; off < src_bytes; off += blk) {
        if (pwrite(fd, z.data(), std::min(blk, src_bytes - off), (off_t) off) < 0) { perror("pwrite"); return 2; }
    }
    void * src = mmap(nullptr, src_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (src == MAP_FAILED) { perror("mmap"); return 2; }

    void * dst = nullptr;
    CK(hipMalloc(&dst, cj * file_chunks + cj));
    hipStream_t s;
    CK(hipStreamCreate(&s));

    fprintf(stderr, "mode=%s src=%p dst=%p\n", use_2d ? "2d" : "1d", src, dst);

    for (int it = 0; it < 200; ++it) {
        const size_t i_start = (size_t) ((it * 13) % (file_chunks - nc - 2));
        // drop the pages so every iteration is cold
        madvise((char *) src + i_start * cf, nc * cf, MADV_DONTNEED);
        for (int j = 0; j < 2; ++j) {
            const size_t off_j = (size_t) j * cj;
            if (use_2d) {
                CK(hipMemcpy2DAsync((char *) dst + i_start * cj, cj,
                                    (const char *) src + i_start * cf + off_j, cf,
                                    cj, nc, hipMemcpyHostToDevice, s));
            } else {
                for (int k = 0; k < nc; ++k) {
                    CK(hipMemcpyAsync((char *) dst + (i_start + k) * cj, (const char *) src + (i_start + k) * cf + off_j,
                                      cj, hipMemcpyHostToDevice, s));
                }
            }
        }
        hipError_t e = hipStreamSynchronize(s);
        if (e != hipSuccess) { fprintf(stderr, "sync error it=%d: %s\n", it, hipGetErrorString(e)); return 3; }
        if ((it % 20) == 0) fprintf(stderr, "it=%d ok\n", it);
    }
    fprintf(stderr, "PASS mode=%s\n", use_2d ? "2d" : "1d");
    return 0;
}
