/**
 * @file smc2_pcg32.cuh
 * @brief PCG32 RNG + Normal ICDF for GPU — replaces curand_kernel.h
 *
 * PCG32 state: 16 bytes (vs curandState Philox: ~52 bytes)
 * → 3× less register pressure → higher occupancy
 *
 * Normal generation: PCG32 uniform → normcdfinvf() CUDA intrinsic
 * → ~3-5× faster than curand_normal (Philox advance + internal ICDF)
 */

#ifndef SMC2_PCG32_CUH
#define SMC2_PCG32_CUH

#include <stdint.h>
#include <cuda_runtime.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * PCG32 State — 16 bytes vs curandState's ~52 bytes
 *═══════════════════════════════════════════════════════════════════════════════*/

struct PCG32State {
    uint64_t state;
    uint64_t inc;
};

/*═══════════════════════════════════════════════════════════════════════════════
 * PCG32 Core — O'Neill (2014)
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__
uint32_t pcg32_next(PCG32State* rng) {
    uint64_t old_state = rng->state;
    rng->state = old_state * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rot = (uint32_t)(old_state >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * @brief Seed PCG32 from a global seed + per-thread stream ID
 *
 * Each thread gets a unique stream (odd increment) for independent sequences.
 */
__device__ __forceinline__
void pcg32_seed(PCG32State* rng, uint64_t seed, uint64_t stream) {
    rng->state = 0;
    rng->inc = (stream << 1u) | 1u;  /* Must be odd */
    pcg32_next(rng);
    rng->state += seed;
    pcg32_next(rng);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Uniform float in (0, 1) — open interval, safe for log()
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__
float pcg32_uniformf(PCG32State* rng) {
    /* Use top 24 bits for float mantissa precision */
    uint32_t bits = pcg32_next(rng);
    /* Map to (0, 1) open interval: add 0.5 ULP to avoid exact 0 */
    return ((float)(bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Normal variate via CUDA intrinsic ICDF
 *
 * normcdfinvf() is NVIDIA's optimized single-precision Φ^{-1}(u).
 * Full float precision, hardware-accelerated on SM_50+.
 * Clamp output to [-6, +6] to avoid inf at extreme tails.
 *═══════════════════════════════════════════════════════════════════════════════*/

__device__ __forceinline__
float pcg32_normal(PCG32State* rng) {
    float u = pcg32_uniformf(rng);
    float z = normcdfinvf(u);
    /* Clamp extreme tails — irrelevant for Monte Carlo quality */
    return fmaxf(-6.0f, fminf(6.0f, z));
}

/*═══════════════════════════════════════════════════════════════════════════════
 * Host-side seeding helper — declared here, defined in smc2_rbpf_cuda.cu
 *═══════════════════════════════════════════════════════════════════════════════*/

__global__ void kernel_init_pcg32(PCG32State* states, uint64_t seed, int N);

#endif /* SMC2_PCG32_CUH */
