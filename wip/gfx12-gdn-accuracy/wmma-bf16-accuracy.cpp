// bf16 WMMA instruction accuracy microbenchmark.
//
// Computes C = A @ A^T for a 16x16 bf16 A with a SINGLE wmma_f32_16x16x16_bf16 per block,
// using the exact fragment helpers from the gdn kernels (so the arch-specific A/B and C
// layouts are handled), and compares against an fp64 CPU reference computed from the SAME
// bf16 inputs.  The inputs are generated on the host so gfx12 and gfx11 see identical bytes.
//
// Build (one arch per box):
//   hipcc -O3 -DGFX12 --offload-arch=gfx1201 wmma-bf16-accuracy.cpp -o /tmp/wmma-gfx12
//   hipcc -O3 -DGFX11 --offload-arch=gfx1100 wmma-bf16-accuracy.cpp -o /tmp/wmma-gfx11
//
// The bf16 inputs are already-rounded values, so every product is exact in fp32 and the only
// remaining error is the instruction's internal accumulation (16 terms + the C input).  If
// gfx12's instruction is coarser than gfx11's, the relative error will show it directly.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#if defined(GFX12)
typedef __bf16 gdn_v8bf __attribute__((ext_vector_type(8)));
#define GDN_BF16_KHALF 0
#elif defined(GFX11)
typedef __bf16 gdn_v8bf __attribute__((ext_vector_type(16)));
#define GDN_BF16_KHALF 0
#else
#error "define GFX12 or GFX11"
#endif
typedef float gdn_v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short gdn_f2bf(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ unsigned gdn_f2bf2(float a, float b) {
    unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
    unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ uint2 gdn_f2bf4(float a, float b, float c, float d) {
    return make_uint2(gdn_f2bf2(a, b), gdn_f2bf2(c, d));
}

// K-contiguous fragment from a row pointer already advanced to this lane's row and half-K.
__device__ __forceinline__ gdn_v8bf gdn_fragP(const unsigned short * p) {
#if defined(GFX12)
    const uint2 * pw = (const uint2 *) p;
    union { uint2 w[2]; gdn_v8bf f; } u;
    u.w[0] = pw[0]; u.w[1] = pw[2];
    return u.f;
#else
    const uint2 * pw = (const uint2 *) p;
    union { uint2 w[4]; gdn_v8bf f; } u;
    u.w[0] = pw[0]; u.w[1] = pw[1]; u.w[2] = pw[2]; u.w[3] = pw[3];
    return u.f;
#endif
}
__device__ __forceinline__ const unsigned short * gdn_rowP(const unsigned short * src,
                                                           int pitch, int base, int lane) {
    return src + (base + (lane & 15)) * pitch + GDN_BF16_KHALF * (lane >> 4)
#if defined(GFX12)
           + 4 * (lane >> 4)
#endif
           ;
}
__device__ __forceinline__ gdn_v8bf gdn_frag(const unsigned short * src, int pitch,
                                             int base, int kbase, int lane) {
    return gdn_fragP(gdn_rowP(src, pitch, base, lane) + kbase);
}
__device__ __forceinline__ gdn_v8f gdn_mma(gdn_v8bf a, gdn_v8bf b, gdn_v8f c) {
#if defined(GFX12)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
}

#define PITCH 16

__global__ void wmma_aat(const unsigned short * __restrict__ Aglob,
                         float * __restrict__ Cout, int warmup_depth) {
    __shared__ unsigned short As[16][PITCH];
    const int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 16 * PITCH; i += blockDim.x) {
        ((unsigned short *) &As[0][0])[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
        As[i / 16][i % 16] = Aglob[blockIdx.x * 256 + i];
    }
    __syncthreads();

    gdn_v8bf a = gdn_frag(&As[0][0], PITCH, 0, 0, lane);
    gdn_v8f acc = {};
    // Optional extra back-to-back MMAs into the same accumulator (accumulation-order probe):
    // each adds a fresh 16-term product to the running C.
    for (int w = 0; w <= warmup_depth; ++w) {
        acc = gdn_mma(a, a, acc);
    }

    const int n = lane & 15;
#if defined(GFX12)
    const int m0 = 8 * (lane >> 4);
#else
    const int m0 = lane >> 4;
#endif
#pragma unroll
    for (int e = 0; e < 8; ++e) {
#if defined(GFX12)
        const int m = m0 + e;
#else
        const int m = 2 * e + m0;
#endif
        Cout[blockIdx.x * 256 + m * 16 + n] = acc[e];
    }
}

static unsigned short host_f2bf(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x8000u;
    return (unsigned short)(u >> 16);
}
static float host_bf2f(unsigned short h) {
    unsigned u = (unsigned) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char ** argv) {
    const int ntrial  = argc > 1 ? atoi(argv[1]) : 20000;
    const int depth   = argc > 2 ? atoi(argv[2]) : 0;   // extra chained MMAs (0 = single)
    const float scale = argc > 3 ? atof(argv[3]) : 1.0f;

    std::vector<unsigned short> hA((size_t) ntrial * 256);
    srand(1234);
    for (size_t i = 0; i < hA.size(); ++i) {
        float f = ((rand() / (float) RAND_MAX) * 2.0f - 1.0f) * scale;
        hA[i] = host_f2bf(f);
    }

    unsigned short * dA = nullptr;
    float * dC = nullptr;
    hipMalloc(&dA, hA.size() * sizeof(unsigned short));
    hipMalloc(&dC, (size_t) ntrial * 256 * sizeof(float));
    hipMemcpy(dA, hA.data(), hA.size() * sizeof(unsigned short), hipMemcpyHostToDevice);
    hipMemset(dC, 0, (size_t) ntrial * 256 * sizeof(float));

    hipLaunchKernelGGL(wmma_aat, dim3(ntrial), dim3(32), 0, 0, dA, dC, depth);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("kernel error: %s\n", hipGetErrorString(err)); return 1; }

    std::vector<float> hC((size_t) ntrial * 256);
    hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost);

    // reference in double, from the same bf16 inputs
    double sum_rel2 = 0, max_rel = 0, sum_abs = 0, sum_ref = 0;
    double sum_abs_err = 0;
    for (int t = 0; t < ntrial; ++t) {
        double A[16][16];
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) A[i][j] = host_bf2f(hA[(size_t) t * 256 + i * 16 + j]);
        double num = 0, den = 0, mx = 0;
        for (int m = 0; m < 16; ++m) {
            for (int n = 0; n < 16; ++n) {
                double ref = 0;                      // (depth+1) chained products
                for (int w = 0; w <= depth; ++w) {
                    double s = 0;
                    for (int k = 0; k < 16; ++k) s += A[m][k] * A[n][k];
                    ref += s;
                }
                const double got = hC[(size_t) t * 256 + m * 16 + n];
                const double d = got - ref;
                num += d * d; den += ref * ref;
                mx = fmax(mx, fabs(d));
                sum_abs_err += fabs(d);
                sum_ref += fabs(ref);
            }
        }
        const double rel = sqrt(num / (den > 0 ? den : 1));
        sum_rel2 += rel * rel;
        if (rel > max_rel) max_rel = rel;
    }
    const int nelt = ntrial * 256;
    printf("trials=%d depth=%d scale=%.3g\n", ntrial, depth, scale);
    printf("  RMS relative error : %.3e\n", sqrt(sum_rel2 / ntrial));
    printf("  max relative error : %.3e\n", max_rel);
    printf("  mean |abs err|     : %.3e   (mean |ref| = %.3e)\n", sum_abs_err / nelt, sum_ref / nelt);
    return 0;
}
