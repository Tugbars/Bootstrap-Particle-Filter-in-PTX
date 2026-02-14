/*═══════════════════════════════════════════════════════════════════════════════
 * SMC² SORTING BACKENDS
 * 
 * Provides deterministic sorting for CPMMH coupling preservation.
 * 
 * Build options:
 *   -DSMC2_USE_CUB_SORT    Use CUB BlockRadixSort (deterministic, slower for small N)
 *   (default)              Use Bitonic sort (deterministic, fast for N≤1024)
 * 
 * Usage:
 *   cpmmh_sort<BLOCK_SIZE>(s_z, s_mu, s_var, s_idx, s_temp);
 *   size_t smem = cpmmh_sort_smem_size<BLOCK_SIZE>();
 *═══════════════════════════════════════════════════════════════════════════════*/

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*═══════════════════════════════════════════════════════════════════════════════
 * BITONIC SORT IMPLEMENTATION (Default)
 * 
 * Optimal for small N (≤1024):
 *   - Zero library overhead
 *   - Fully parallel compare-swap network
 *   - Deterministic (fixed network topology)
 *   - No extra shared memory beyond arrays being sorted
 * 
 * Complexity: O(N log²N) comparisons, log²N parallel stages
 * For N=256: 8 * 9 / 2 = 36 sync barriers
 *═══════════════════════════════════════════════════════════════════════════════*/

#ifndef SMC2_USE_CUB_SORT

/**
 * @brief Deterministic bitonic sort of particle state by μ_h
 * 
 * @tparam BLOCK_SIZE  Number of particles (must be power of 2, ≤1024)
 * @param s_z          Shared array: regime z̃ [BLOCK_SIZE]
 * @param s_mu         Shared array: Kalman mean μ_h [BLOCK_SIZE] (sort key)
 * @param s_var        Shared array: Kalman variance [BLOCK_SIZE]
 * @param s_idx        Shared array: scratch for indices [BLOCK_SIZE]
 * @param s_temp       Unused (API compatibility with CUB version)
 * 
 * After this call, particles are sorted by μ_h (ascending).
 */
template<int BLOCK_SIZE>
__device__ __forceinline__
void cpmmh_sort(
    float* __restrict__ s_z,
    float* __restrict__ s_mu,
    float* __restrict__ s_var,
    int* __restrict__ s_idx,
    void* s_temp  /* unused */
) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    
    (void)s_temp;  /* Unused in bitonic version */
    
    int tid = threadIdx.x;
    
    /* Initialize index array */
    s_idx[tid] = tid;
    __syncthreads();
    
    /* Bitonic sort network on (key=s_mu, value=s_idx) */
    #pragma unroll 1  /* Don't unroll outer loop - too much code bloat */
    for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
        #pragma unroll 1
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = tid ^ j;
            
            if (partner > tid) {
                /* Determine sort direction */
                bool ascending = ((tid & k) == 0);
                
                float key_lo = s_mu[tid];
                float key_hi = s_mu[partner];
                int idx_lo = s_idx[tid];
                int idx_hi = s_idx[partner];
                
                /* Swap if out of order */
                bool should_swap = ascending ? (key_lo > key_hi) : (key_lo < key_hi);
                
                if (should_swap) {
                    s_mu[tid] = key_hi;
                    s_mu[partner] = key_lo;
                    s_idx[tid] = idx_hi;
                    s_idx[partner] = idx_lo;
                }
            }
            __syncthreads();
        }
    }
    
    /* Gather z and var using sorted indices */
    int src_idx = s_idx[tid];
    float my_z = s_z[src_idx];
    float my_var = s_var[src_idx];
    __syncthreads();
    
    s_z[tid] = my_z;
    s_var[tid] = my_var;
    __syncthreads();
}

/**
 * @brief Extra shared memory needed for sort (0 for bitonic)
 */
template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_sort_smem_size() {
    return 0;  /* Bitonic uses no extra temp storage */
}

#else  /* SMC2_USE_CUB_SORT */

/*═══════════════════════════════════════════════════════════════════════════════
 * CUB BlockRadixSort IMPLEMENTATION
 * 
 * For comparison/testing. Deterministic but has overhead for small N.
 * 
 * Uses FP16 key to halve radix passes (16 bits vs 32 bits).
 *═══════════════════════════════════════════════════════════════════════════════*/

#include <cub/cub.cuh>

/**
 * @brief Deterministic CUB sort of particle state by μ_h
 * 
 * Uses FP16 key for faster sorting (rank order preserved).
 */
template<int BLOCK_SIZE>
__device__ __forceinline__
void cpmmh_sort(
    float* __restrict__ s_z,
    float* __restrict__ s_mu,
    float* __restrict__ s_var,
    int* __restrict__ s_idx,
    void* s_temp
) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    
    int tid = threadIdx.x;
    
    /* Convert float key to unsigned short for faster radix sort.
     * FP16 preserves rank order for reasonable μ_h ranges.
     * 
     * For proper negative number handling with radix sort:
     * - If sign bit set: flip all bits
     * - If sign bit clear: flip just sign bit
     * This makes the unsigned representation monotonic.
     */
    half h_key = __float2half(s_mu[tid]);
    unsigned short sort_key = *reinterpret_cast<unsigned short*>(&h_key);
    
    /* Fix sign bit for correct negative number sorting */
    unsigned short mask = -((sort_key >> 15) & 1) | 0x8000;
    sort_key ^= mask;
    
    /* Prepare key-value arrays (CUB wants arrays even for 1 item) */
    unsigned short keys[1] = { sort_key };
    int values[1] = { tid };
    
    /* CUB BlockRadixSort with 16-bit key */
    typedef cub::BlockRadixSort<unsigned short, BLOCK_SIZE, 1, int> BlockSortT;
    typename BlockSortT::TempStorage& temp_storage = 
        *reinterpret_cast<typename BlockSortT::TempStorage*>(s_temp);
    
    BlockSortT(temp_storage).Sort(keys, values);
    __syncthreads();
    
    /* Write sorted index to shared memory */
    s_idx[tid] = values[0];
    __syncthreads();
    
    /* Gather all three arrays using sorted indices */
    int src_idx = s_idx[tid];
    float gathered_z = s_z[src_idx];
    float gathered_mu = s_mu[src_idx];
    float gathered_var = s_var[src_idx];
    __syncthreads();
    
    s_z[tid] = gathered_z;
    s_mu[tid] = gathered_mu;
    s_var[tid] = gathered_var;
    __syncthreads();
}

/**
 * @brief Extra shared memory needed for CUB sort
 */
template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_sort_smem_size() {
    return sizeof(typename cub::BlockRadixSort<unsigned short, BLOCK_SIZE, 1, int>::TempStorage);
}

#endif  /* SMC2_USE_CUB_SORT */

/*═══════════════════════════════════════════════════════════════════════════════
 * SHARED MEMORY SIZE HELPERS
 *═══════════════════════════════════════════════════════════════════════════════*/

/**
 * @brief Total shared memory for RBPF forward step kernel
 * 
 * Layout (after race condition fix - s_cumsum is no longer aliased):
 *   [0..31]          : Warp reduction scratch (32 floats)
 *   [32..32+N-1]     : s_z (N floats)
 *   [32+N..32+2N-1]  : s_mu (N floats)
 *   [32+2N..32+3N-1] : s_var (N floats)
 *   [32+3N..32+4N-1] : s_cumsum (N floats) - DEDICATED, not aliased
 *   [32+4N..32+5N-1] : s_idx (N ints = N floats worth)
 *   [32+5N..]        : Sort temp (if using CUB)
 */
template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t rbpf_shared_mem_size() {
    size_t base = (32 + 5 * BLOCK_SIZE) * sizeof(float);  /* Changed from 4 to 5 */
    size_t sort_temp = cpmmh_sort_smem_size<BLOCK_SIZE>();
    return base + sort_temp;
}

/**
 * @brief Total shared memory for CPMMH rejuvenation kernel
 * 
 * Layout:
 *   [0..31]          : Warp reduction scratch
 *   [32..32+N-1]     : s_z
 *   [32+N..32+2N-1]  : s_mu
 *   [32+2N..32+3N-1] : s_var
 *   [32+3N..32+4N-1] : s_cdf
 *   [32+4N..32+5N-1] : s_idx
 *   [32+5N..]        : Sort temp (if using CUB)
 */
template<int BLOCK_SIZE>
__host__ __device__ __forceinline__
constexpr size_t cpmmh_shared_mem_size() {
    size_t base = (32 + 5 * BLOCK_SIZE) * sizeof(float);
    size_t sort_temp = cpmmh_sort_smem_size<BLOCK_SIZE>();
    return base + sort_temp;
}
