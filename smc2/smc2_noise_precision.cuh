/**
 * @file smc2_noise_precision.cuh
 * @brief Noise precision abstraction layer for SMC² CPMMH
 * 
 * Provides unified interface for FP16/FP32 noise storage with compile-time selection.
 * 
 * Usage:
 *   - Default (FP32): Just include this header (full precision, recommended)
 *   - Bandwidth-optimized (FP16): Define SMC2_NOISE_FP16 before including
 * 
 * Build examples:
 *   nvcc -o smc2 smc2_rbpf_cuda.cu                    # FP32 (default)
 *   nvcc -DSMC2_NOISE_FP16 -o smc2_fp16 smc2_rbpf_cuda.cu  # FP16
 */

#ifndef SMC2_NOISE_PRECISION_CUH
#define SMC2_NOISE_PRECISION_CUH

#include <cuda_fp16.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * PRECISION SELECTION
 * 
 * FP32: Full precision (DEFAULT), recommended for accuracy
 * FP16: ~50% memory bandwidth, ~0.1% relative error per step
 *       Use only when bandwidth is critical and T < 500
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef SMC2_NOISE_FP16

/*───────────────────────────────────────────────────────────────────────────────
 * FP16 Implementation (bandwidth-optimized, opt-in)
 *───────────────────────────────────────────────────────────────────────────────*/

typedef half noise_t;

#define NOISE_SIZEOF sizeof(half)

__device__ __forceinline__ float noise_load(const noise_t* ptr, int64_t idx) {
    return __half2float(ptr[idx]);
}

__device__ __forceinline__ void noise_store(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = __float2half(val);
}

/**
 * @brief Store value and return the quantized value actually stored
 * 
 * Critical for CPMMH correctness: the filter must use exactly what's stored,
 * including any quantization effects. This ensures:
 *   1. Forward filter and replay use identical noise
 *   2. Correlated noise z' = ρ*z_stored + √(1-ρ²)*ε works correctly
 */
__device__ __forceinline__ float noise_store_roundtrip(noise_t* ptr, int64_t idx, float val) {
    half h = __float2half(val);
    ptr[idx] = h;
    return __half2float(h);
}

#else  /* FP32 (default) */

/*───────────────────────────────────────────────────────────────────────────────
 * FP32 Implementation (full precision, default)
 *───────────────────────────────────────────────────────────────────────────────*/

typedef float noise_t;

#define NOISE_SIZEOF sizeof(float)

__device__ __forceinline__ float noise_load(const noise_t* ptr, int64_t idx) {
    return ptr[idx];
}

__device__ __forceinline__ void noise_store(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = val;
}

/**
 * @brief Store value and return what was actually stored
 * 
 * For FP32, this is identity (no quantization).
 * Critical for CPMMH correctness: the filter must use exactly what's stored.
 */
__device__ __forceinline__ float noise_store_roundtrip(noise_t* ptr, int64_t idx, float val) {
    ptr[idx] = val;
    return val;
}

#endif  /* SMC2_NOISE_FP16 */

/*═══════════════════════════════════════════════════════════════════════════════
 * HOST-SIDE HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Compute byte size for noise array allocation
 */
static inline size_t noise_array_bytes(int64_t count) {
    return (size_t)count * NOISE_SIZEOF;
}

/**
 * @brief Check if using FP32 precision at runtime
 */
static inline bool noise_is_fp32(void) {
#ifdef SMC2_NOISE_FP16
    return false;
#else
    return true;
#endif
}

/**
 * @brief Get precision string for diagnostics
 */
static inline const char* noise_precision_str(void) {
#ifdef SMC2_NOISE_FP16
    return "FP16";
#else
    return "FP32";
#endif
}

#endif /* SMC2_NOISE_PRECISION_CUH */
